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

__all__ = ["FseT2Optimizer", "FseT2Plan"]

from typing import Any

import torch

from ..sequence._accelerators import (
    _pack_events,
    _pointers,
    _run_packed_jvp,
    _run_packed_vjp_jvp,
    real_subspace_axis,
)
from ..sequence._builders import fse_description
from ..sequence._parameters import TISSUE_COUNT as _TISSUE_COUNT
from ..sequence._parameters import TISSUE_NAMES as _TISSUE_NAMES
from ..sequence._simulation import TissueProperties, _prepare_tissue

# Position of each differentiable buffer in the kernel gradient ordering
# ``(t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, duration, flip, phase)``.
_T2 = 1
# Where the float event gradients sit in a gradient tuple, after the tissue.
_DURATION, _FLIP, _PHASE = (
    _TISSUE_COUNT,
    _TISSUE_COUNT + 1,
    _TISSUE_COUNT + 2,
)
# This function's own inputs: the packed events, then the tissue properties.
_EVENT_COUNT = 6
_TISSUE_END = _EVENT_COUNT + _TISSUE_COUNT
# Those same inputs, for the three directions a real kernel cannot follow:
# the RF phase among the events, transmit phase and off-resonance among the
# tissue properties.
_OUTSIDE_THE_SUBSPACE_INPUTS = (
    3,
    _EVENT_COUNT + _TISSUE_NAMES.index("b1_phase_rad"),
    _EVENT_COUNT + _TISSUE_NAMES.index("b0_hz"),
)


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
        self._axis_key: tuple[torch.Tensor, ...] | None = None
        self._seeds: tuple[tuple[torch.Tensor, ...], ...] | None = None

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
        axis = self._subspace_axis(events, prepared)
        jacobian = _T2Jacobian.apply(
            *events,
            *prepared,
            *self._tangents(prepared, events),
            self.state_count,
            self.output_count,
            threads,
            -1 if axis is None else axis,
        )
        return jacobian.reshape(-1, *shape, self.output_count)

    def _subspace_axis(
        self, events: tuple[torch.Tensor, ...], tissue: tuple[torch.Tensor, ...]
    ) -> int | None:
        """Whether a T2 seed stays inside a real subspace, reusing the verdict.

        A seed that stays inside makes the cheaper kernel exact rather than an
        approximation. The answer follows the RF phases and the tissue, so it is
        recomputed only when one of those changes -- comparing them costs far
        less than deciding again, and the caller passes freshly allocated
        tensors every time, so identity cannot stand in for equality.
        """
        signature = (events[3], tissue[4], tissue[5])  # phase, b1_phase, b0
        if self._axis_key is None or any(
            not torch.equal(new, old)
            for new, old in zip(signature, self._axis_key, strict=True)
        ):
            self._axis_key = tuple(value.clone() for value in signature)
            self._axis = real_subspace_axis(events, tissue)
        return self._axis

    def _tangents(
        self, tissue: tuple[torch.Tensor, ...], events: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """The forward-mode seed: one along T2, zero everywhere else.

        Shapes are fixed for the life of the plan and the kernels only read
        these, so they are built once instead of once per call.
        """
        if self._seeds is None:
            self._seeds = (
                tuple(
                    torch.ones_like(value) if index == _T2 else torch.zeros_like(value)
                    for index, value in enumerate(tissue)
                ),
                (
                    torch.zeros_like(events[0]),
                    torch.zeros_like(events[2]),
                    torch.zeros_like(events[3]),
                ),
            )
        return self._seeds


class _T2Jacobian(torch.autograd.Function):
    """``d signal / d T2`` with an analytic second-order adjoint."""

    @staticmethod
    def forward(*inputs: Any) -> torch.Tensor:
        events, tissue = inputs[:_EVENT_COUNT], inputs[_EVENT_COUNT:_TISSUE_END]
        tissue_tangents, event_tangents = inputs[_TISSUE_END], inputs[_TISSUE_END + 1]
        return _run_packed_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
            inputs[_TISSUE_END + 2],
            inputs[_TISSUE_END + 3],
            inputs[_TISSUE_END + 4],
            inputs[_TISSUE_END + 5],
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        ctx.save_for_backward(*inputs[:_TISSUE_END])
        ctx.tangents = (*inputs[_TISSUE_END], *inputs[_TISSUE_END + 1])
        ctx.state_count = inputs[_TISSUE_END + 2]
        ctx.output_count = inputs[_TISSUE_END + 3]
        ctx.threads = inputs[_TISSUE_END + 4]
        ctx.real_axis = inputs[_TISSUE_END + 5]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        events, tissue = saved[:_EVENT_COUNT], saved[_EVENT_COUNT:_TISSUE_END]
        tangents = ctx.tangents
        if ctx.real_axis == 1 and any(
            ctx.needs_input_grad[index] for index in _OUTSIDE_THE_SUBSPACE_INPUTS
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
            *primal_grads[:_TISSUE_COUNT],
        ]
        result = [
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad, strict=False)
        ]
        # tissue/event tangents, state_count, output_count, threads, real_axis
        return (*result, None, None, None, None, None, None)


class FseT2Optimizer:
    """Runs the whole A-optimal T2 optimization inside the kernel layer.

    Parameters
    ----------
    plan : FseT2Plan
        Structure of the echo train being optimized.
    t2_ms : torch.Tensor
        T2 design points, matching the tissue passed to :meth:`run`.
    smoothness_weight, curvature_weight, rf_power_weight : float
        Penalty weights, applied to flip angles normalized by 180 degrees.

    Notes
    -----
    :class:`FseT2Plan` still pays Python and autograd once per iteration, which
    costs more than the kernels do. This class hands the loop to C++ instead,
    so the interpreter sees one call rather than one per iteration. It computes
    the same objective and takes the same Adam steps.

    Only the default hard-pulse train qualifies: the fused loop writes flip
    angles straight into the event buffer, so a pulse whose flip angle is not
    its own nominal value has to use the plan.
    """

    def __init__(
        self,
        plan: FseT2Plan,
        t2_ms: torch.Tensor,
        *,
        smoothness_weight: float = 0.5,
        curvature_weight: float = 60.0,
        rf_power_weight: float = 0.05,
    ) -> None:
        self.plan = plan
        self.t2_ms = t2_ms
        self.smoothness_weight = float(smoothness_weight)
        self.curvature_weight = float(curvature_weight)
        self.rf_power_weight = float(rf_power_weight)

    def supports(self) -> bool:
        """Whether the plan's pulse lets flip angles go straight into events."""
        probe = torch.linspace(
            0.2, 3.0, self.plan.echo_train_length, device=self.plan.device
        ).unsqueeze(0)
        magnitude, integral_phase = self.plan._definition.flip_angle(
            probe, rf_raster_time_s=self.plan.rf_raster_time_s
        )
        return bool(
            torch.allclose(magnitude, probe) and torch.count_nonzero(integral_phase) == 0
        )

    def run(
        self,
        flip_deg: torch.Tensor,
        tissue: TissueProperties,
        *,
        iterations: int,
        learning_rate: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        threads: int = 0,
    ) -> tuple[torch.Tensor, float]:
        """Optimize ``flip_deg`` in place-free fashion; returns it and the loss."""
        from torchsim import _epg_cpu

        if not self.supports():
            raise RuntimeError(
                "the fused loop needs a pulse whose flip angle is its nominal "
                "value; use FseT2Plan with an autograd loop instead"
            )
        plan = self.plan
        flip_deg = torch.atleast_2d(flip_deg).to(torch.float32).contiguous().clone()
        trains, echoes = flip_deg.shape
        if echoes != plan.echo_train_length:
            raise ValueError(
                f"expected {plan.echo_train_length} flip angles, got {echoes}"
            )

        prepared, _, _ = _prepare_tissue(tissue, plan.device)
        prepared = tuple(
            value.to(dtype=torch.float32).contiguous() for value in prepared
        )
        atoms = prepared[0].numel()
        events = plan.buffers(torch.deg2rad(flip_deg))
        events = tuple(
            value.detach().contiguous() if value.is_floating_point() else value
            for value in events
        )
        axis = plan._subspace_axis(events, prepared)
        tissue_seed, event_seed = plan._tangents(prepared, events)

        # Every one of these must outlive the call: _pointers returns bare
        # addresses, so a temporary would be freed before the kernel writes it.
        signal_real = torch.zeros(
            (trains, atoms, plan.output_count), dtype=torch.float32
        )
        signal_imag = torch.zeros_like(signal_real)
        cotangent_real = torch.zeros_like(signal_real)
        cotangent_imag = torch.zeros_like(signal_real)
        value_grads = tuple(
            torch.zeros_like(value) for value in (*prepared, *event_seed)
        )
        tangent_grads = tuple(torch.zeros_like(value) for value in value_grads)
        moment = torch.zeros_like(flip_deg)
        velocity = torch.zeros_like(flip_deg)
        t2 = self.t2_ms.to(torch.float32).contiguous()

        pointers = _pointers(
            (
                *prepared,
                *events,
                *tissue_seed,
                *event_seed,
                signal_real,
                signal_imag,
                cotangent_real,
                cotangent_imag,
                *value_grads,
                *tangent_grads,
                flip_deg,
                moment,
                velocity,
                t2,
            )
        )
        loss = _epg_cpu.optimize_fse_t2(
            pointers,
            (
                atoms,
                trains,
                int(events[1].numel()),
                plan.state_count,
                plan.output_count,
                echoes,
                int(iterations),
                int(threads),
                -1 if axis is None else axis,
            ),
            (
                float(learning_rate),
                float(betas[0]),
                float(betas[1]),
                float(eps),
                self.smoothness_weight,
                self.curvature_weight,
                self.rf_power_weight,
            ),
        )
        return flip_deg, loss
