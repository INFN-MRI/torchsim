"""Direct kernel path for optimizing a batch of FSE refocusing trains.

The generic model stack rebuilds the sequence description, repacks the event
buffers and routes forward-mode differentiation through ``torch.func.jvp`` on
every call. For an optimizer that touches nothing but the refocusing flip
angles, all of that is fixed work: only the flip values change between
iterations. :class:`FseT2Plan` captures the fixed part once and rebuilds the
two buffers that vary, then calls the kernels directly.

The result is the T2 jacobian alone. The generic path also produces the primal
signal and so pays for a second forward kernel and a second adjoint; an
objective built on the jacobian never looks at those.
"""

from __future__ import annotations

__all__ = ["FseT2Plan"]

from typing import Any

import torch

from ..sequence._accelerators import (
    _pack_events,
    _run_packed_jvp,
    _run_packed_vjp_jvp,
    real_subspace_axis,
)
from ..sequence._builders import fse_description
from ..sequence._simulation import TissueProperties, _prepare_tissue

# Position of each differentiable buffer in the kernel gradient ordering
# ``(t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, duration, flip, phase)``.
_T2 = 1
_DURATION, _FLIP, _PHASE = 7, 8, 9


class FseT2Plan:
    """Reusable structure of an FSE echo train.

    Parameters
    ----------
    echo_train_length : int
        Number of refocusing pulses.
    echo_spacing_s : float
        Echo spacing in seconds.
    phases_rad : float
        Refocusing phase.
    excitation_flip_rad, excitation_phase_rad : float
        Excitation pulse.
    state_count : int
        EPG state count.
    device : torch.device
        Device the buffers live on.

    Notes
    -----
    Only the refocusing flip angles may change between calls; anything else
    needs a new plan.
    """

    def __init__(
        self,
        echo_train_length: int,
        echo_spacing_s: float,
        *,
        phases_rad: float = torch.pi / 2,
        excitation_flip_rad: float = torch.pi / 2,
        excitation_phase_rad: float = torch.pi / 2,
        state_count: int = 10,
        rf_raster_time_s: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        self.echo_train_length = int(echo_train_length)
        self.state_count = int(state_count)
        self.rf_raster_time_s = float(rf_raster_time_s)
        self.device = torch.device(device)

        reference = torch.full(
            (self.echo_train_length,), 0.5, dtype=torch.float32, device=self.device
        )
        description = fse_description(
            reference,
            echo_spacing_s,
            phases_rad=phases_rad,
            excitation_flip_rad=excitation_flip_rad,
            excitation_phase_rad=excitation_phase_rad,
        )
        packed = _pack_events(
            "fse",
            description,
            repetitions=1,
            record="all",
            device=self.device,
            rf_raster_time_s=self.rf_raster_time_s,
        )
        self.output_count = packed.output_count
        self.kind = packed.kind
        self.action = packed.action
        self.output_index = packed.output_index
        self._duration = packed.duration
        self._flip_template = packed.flip
        self._phase_template = packed.phase
        self._definition = description.rf_definitions[0]
        self._axis: int | None = None
        self._axis_key: tuple[int, ...] | None = None

        # fse_description lays events out as [excitation, (refocus, adc) * etl],
        # so the refocusing pulses are the odd positions. Verified against the
        # generic packer in the test suite.
        self._refocus = torch.arange(
            1, 1 + 2 * self.echo_train_length, 2, device=self.device
        )
        expected = 1 + 2 * self.echo_train_length
        if int(self.kind.numel()) != expected:
            raise ValueError(
                f"unexpected FSE event layout: {int(self.kind.numel())} events, "
                f"expected {expected}"
            )

    def buffers(self, flip_rad: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Packed event buffers for ``flip_rad``, shaped ``(n_trains, echoes)``.

        Returns the six-tuple the kernels take. Differentiable in ``flip_rad``.
        """
        flip_rad = torch.atleast_2d(flip_rad)
        trains = flip_rad.shape[0]
        if flip_rad.shape[-1] != self.echo_train_length:
            raise ValueError(
                f"expected {self.echo_train_length} flip angles, "
                f"got {flip_rad.shape[-1]}"
            )
        # One call on the whole batch replaces one call per event.
        magnitude, integral_phase = self._definition.flip_angle(
            flip_rad, rf_raster_time_s=self.rf_raster_time_s
        )
        zeros = torch.zeros_like(magnitude)
        refocus_flip = torch.stack((magnitude, zeros), dim=-1).reshape(trains, -1)
        base = self._phase_template[self._refocus]
        # The integral phase is the argument of a real multiple of the envelope
        # area, so it is piecewise constant in the flip angle and carries no
        # derivative. Detaching says so, which keeps the RF phase out of the
        # graph -- the real-subspace kernels cannot differentiate it, because a
        # per-pulse phase perturbation is precisely what leaves the subspace.
        refocus_phase = base + integral_phase.detach()
        adc_phase = self._phase_template[self._refocus + 1].expand(trains, -1)
        phase_pairs = torch.stack((refocus_phase, adc_phase), dim=-1).reshape(trains, -1)

        head_flip = self._flip_template[:1].expand(trains, 1)
        head_phase = self._phase_template[:1].expand(trains, 1)
        flip = torch.cat((head_flip, refocus_flip), dim=1).contiguous()
        phase = torch.cat((head_phase, phase_pairs), dim=1).contiguous()
        duration = self._duration.expand(trains, -1).contiguous()
        return (duration, self.kind, flip, phase, self.action, self.output_index)

    def t2_jacobian(
        self,
        flip_rad: torch.Tensor,
        tissue: TissueProperties,
        *,
        threads: int = 0,
    ) -> torch.Tensor:
        """Return ``d signal / d T2``, shaped ``(n_trains, n_atoms, n_echoes)``."""
        prepared, shape, _ = _prepare_tissue(tissue, self.device)
        # _prepare_tissue broadcasts, so a scalar property arrives as a stride-0
        # view; the kernels index raw pointers and would read past its one float.
        prepared = tuple(
            value.to(dtype=torch.float32).contiguous() for value in prepared
        )
        events = self.buffers(flip_rad)
        # A T2 seed stays inside the real subspace when the sequence qualifies,
        # so the cheaper kernel is exact here rather than an approximation. The
        # verdict depends on the RF phases and the tissue, never on the flip
        # angles, so it survives every iteration of an optimizer.
        key = tuple(value.data_ptr() for value in prepared)
        if key != self._axis_key:
            self._axis_key = key
            self._axis = real_subspace_axis(events, prepared)
        axis = self._axis
        jacobian = _T2Jacobian.apply(
            *events,
            *prepared,
            self.state_count,
            self.output_count,
            threads,
            -1 if axis is None else axis,
        )
        return jacobian.reshape(-1, *shape, self.output_count)


class _T2Jacobian(torch.autograd.Function):
    """``d signal / d T2`` with an analytic second-order adjoint."""

    @staticmethod
    def forward(*inputs: Any) -> torch.Tensor:
        events, tissue = inputs[:6], inputs[6:13]
        state_count, output_count, threads = inputs[13], inputs[14], inputs[15]
        tissue_tangents = _t2_direction(tissue)
        event_tangents = _zero_event_tangents(events)
        return _run_packed_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
            state_count,
            output_count,
            threads,
            inputs[16],
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        ctx.save_for_backward(*inputs[:13])
        ctx.state_count = inputs[13]
        ctx.output_count = inputs[14]
        ctx.threads = inputs[15]
        ctx.real_axis = inputs[16]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        events, tissue = saved[:6], saved[6:13]
        tangents = (*_t2_direction(tissue), *_zero_event_tangents(events))
        if ctx.real_axis == 1 and any(
            ctx.needs_input_grad[index] for index in (3, 10, 11)
        ):
            raise RuntimeError(
                "the real-subspace kernel cannot differentiate RF phase, "
                "transmit phase or off-resonance"
            )
        primal_grads, _ = _run_packed_vjp_jvp(
            tissue,
            events,
            tangents,
            grad_output,
            state_count=ctx.state_count,
            output_count=ctx.output_count,
            threads=ctx.threads,
            real_axis=ctx.real_axis,
        )
        gradients: list[torch.Tensor | None] = [
            primal_grads[_DURATION],
            None,  # kind
            primal_grads[_FLIP],
            primal_grads[_PHASE],
            None,  # action
            None,  # output_index
            *primal_grads[:7],
        ]
        result = [
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=False)
        ]
        return (*result, None, None, None, None)


def _t2_direction(tissue: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Unit forward-mode seed along T2."""
    return tuple(
        torch.ones_like(value) if index == _T2 else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    )


def _zero_event_tangents(
    events: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
