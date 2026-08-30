"""Differentiable EPG interpreter for sequence descriptions."""

from __future__ import annotations

__all__ = [
    "EpgEngine",
    "SimulationResult",
    "TissueProperties",
    "simulate_subspace",
]

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from .._subspace import Subspace
from ._accelerators import (
    largest_pulse_offset,
    simulate_native,
)
from ._description import (
    EventAction,
    RfMode,
    SequenceDescription,
)
from ._lineshape import lineshape_reaching
from ._parameters import (
    TISSUE_NAMES,
    features_of,
    wants_bound_pool,
    wants_exchange_pool,
)
from ._transition import ExactSliceProfile
from ._transmit import shim_rows, transmit_field

RecordMode = Literal["all", "acquired", "echo"]

# Relaxation times enter every backend as rates ``1000 / T``. A non-positive
# time therefore yields an infinite rate, and zero-duration events (an RF pulse
# at the same timestamp as its neighbour) turn ``inf * 0`` into NaN, which the
# state matrix then propagates to every echo. Air voxels in a measured map are
# routinely exactly zero, so clamp the times to a small positive value: the
# resulting rate is large enough to null the signal over any real interval
# while keeping ``exp(-R * 0) == 1``. Clamping (rather than ``1 / (T + eps)``)
# also keeps the gradient finite, since clamped entries simply stop
# contributing to the backward pass.
MINIMUM_RELAXATION_TIME_MS = 1e-6

# The two tissue buffers a shim gives a row of its own.
_TRANSMIT = frozenset((TISSUE_NAMES.index("b1"), TISSUE_NAMES.index("b1_phase_rad")))

# The tissue buffers the kernels invert into a rate.
_RELAXATION_TIMES = tuple(
    TISSUE_NAMES.index(name)
    for name in ("t1_ms", "t2_ms", "t1_bound_ms", "t1_pool_b_ms", "t2_pool_b_ms")
)


def _absorption_table(
    description: SequenceDescription,
    b0_hz: torch.Tensor,
    device: torch.device | str | None,
) -> Any:
    """The bound pool's lineshape, reaching as far off centre as this run goes.

    The read is the pulse's frequency less the voxel's own off-resonance, so
    the sequence fixes one half of it and the tissue the other. Sizing the
    table to their sum is what keeps a pulse played far off resonance from
    reading the last knot -- which would saturate the pool by the value at the
    table's edge rather than by the far smaller one out where the pulse is.
    """
    voxel = float(b0_hz.abs().max()) if b0_hz.numel() else 0.0
    return lineshape_reaching(largest_pulse_offset(description) + voxel, device=device)


@dataclass(frozen=True)
class TissueProperties:
    """Tissue and transmit-field properties.

    Values may be scalars or broadcastable tensors. Relaxation times use
    milliseconds; off-resonance uses Hz. The apparent diffusion coefficient
    uses um**2/ms, so free water at body temperature is about 3; it damps the
    states only where the sequence also declares the gradient that dephases
    them, and costs nothing at its default of zero.

    Velocity uses m/s and drives two terms through the geometry the sequence
    declares. Across an unbalanced gradient it turns each dephasing order
    through a phase rather than damping it, which is what takes the states out
    of any real subspace. Across the voxel itself it washes the spins out and
    replaces them with unexcited magnetization, which needs no gradient at all.

    ``b1`` and ``b1_phase_rad`` are the transmit field a voxel sees. Where the
    sequence declares a shim they carry one transmit channel each along a
    leading axis and are combined into that field; see :mod:`._transmit` for
    what the array model assumes.

    ``bound_fraction`` is the share of the magnetization held by the
    macromolecule-bound proton pool, which RF saturates and which exchanges
    with the free water at ``bound_exchange_hz``. It is the gate on the whole
    second pool: at its default of zero the exchange conserves nothing across
    pools and the bound pool starts empty, so the simulation is single-pool.
    ``t1_bound_ms`` is that pool's own longitudinal relaxation time; it has no
    transverse magnetization to relax, so no T2 goes with it.

    ``pool_b_fraction`` gates a second pool of a different kind: one that
    exchanges chemically rather than by saturation, and that carries
    transverse magnetization of its own. It therefore has both a
    ``t1_pool_b_ms`` and a ``t2_pool_b_ms``, and it sits ``pool_b_shift_hz``
    off the free water -- which itself sits at ``b0_hz``. Fat beside water,
    myelin water beside free water, a metabolite beside its solvent.

    A tissue may declare both, which is a three-pool system: the free water
    exchanges with each second pool and the two do not exchange with each
    other. Only the longitudinal axis sees all three, and the fractions share
    the voxel with the free water, so they cannot sum past one.

    Attributes
    ----------
    t1_ms, t2_ms : float or array-like
        Longitudinal and transverse relaxation times, in milliseconds.
    m0 : float or array-like, optional
        Equilibrium magnetization, as a scaling on the signal.
    b1 : float or array-like, optional
        Transmit efficiency, as a fraction of the prescribed flip angle.
    b1_phase_rad : float or array-like, optional
        Transmit phase, in radians.
    b0_hz : float or array-like, optional
        Off-resonance, in Hz.
    inversion_efficiency : float or array-like, optional
        How much of a perfect inversion an inversion pulse achieves.
    diffusion_um2_per_ms : float or array-like, optional
        Apparent diffusion coefficient. Free water at body temperature is
        about 3.
    velocity_m_per_s : float or array-like, optional
        Spin velocity, in m/s.
    bound_fraction : float or array-like, optional
        Share of the magnetization held by the macromolecule-bound pool, and
        the gate on that whole pool.
    bound_exchange_hz : float or array-like, optional
        Exchange rate between the bound pool and the free water, in Hz.
    t1_bound_ms : float or array-like, optional
        Longitudinal relaxation time of the bound pool, in milliseconds.
    pool_b_fraction : float or array-like, optional
        Share of the magnetization held by a chemically exchanging second
        pool, and the gate on it.
    pool_b_exchange_hz : float or array-like, optional
        Exchange rate between that pool and the free water, in Hz.
    t1_pool_b_ms, t2_pool_b_ms : float or array-like, optional
        Its relaxation times, in milliseconds.
    pool_b_shift_hz : float or array-like, optional
        How far it sits off the free water, in Hz.
    """

    t1_ms: Any
    t2_ms: Any
    m0: Any = 1.0
    b1: Any = 1.0
    b1_phase_rad: Any = 0.0
    b0_hz: Any = 0.0
    inversion_efficiency: Any = 1.0
    diffusion_um2_per_ms: Any = 0.0
    velocity_m_per_s: Any = 0.0
    bound_fraction: Any = 0.0
    bound_exchange_hz: Any = 0.0
    t1_bound_ms: Any = 1000.0
    pool_b_fraction: Any = 0.0
    pool_b_exchange_hz: Any = 0.0
    t1_pool_b_ms: Any = 1000.0
    t2_pool_b_ms: Any = 100.0
    pool_b_shift_hz: Any = 0.0


@dataclass(frozen=True)
class SimulationResult:
    """What one run of :meth:`EpgEngine.simulate` recorded.

    The signal, and the labels saying where in the sequence each sample of it
    came from -- which is what lets a caller pick out the echoes it wanted
    without counting events itself.

    Attributes
    ----------
    signal : torch.Tensor
        ``(..., samples)`` -- what was recorded, complex.
    time_us : torch.Tensor
        When each sample was taken, in microseconds.
    event_index : torch.Tensor
        Which event of the stream produced each sample.
    repetition : torch.Tensor
        Which playing of the stream each sample belongs to.
    echo : torch.Tensor
        Which echo within its repetition each sample is.
    """

    signal: torch.Tensor
    time_us: torch.Tensor
    event_index: torch.Tensor
    repetition: torch.Tensor
    echo: torch.Tensor


class EpgEngine:
    """Run a :class:`SequenceDescription` on the fused EPG kernels.

    This is the direct route, for when what you hold is a description rather
    than a simulator: one a builder such as
    :func:`~torchsim.fse_description` emitted, one assembled from operators by
    hand, or one that arrived from a scanner. A
    :class:`~torchsim.model.Simulator` wraps this and is what most sequences
    are written as -- it fixes tissue and protocol on itself, and adds
    :meth:`~torchsim.model.SignalModel.jacobian`, binding and device
    placement. Nothing here does any of that: give it a description and a
    tissue, get a :class:`SimulationResult` back.

    What a sequence plays around each event is carried by the events
    themselves, so every description runs here whatever assembled it.
    """

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        """How far the sequence winds the states on in one repetition.

        Read off the actions the description carries, so a state matrix is
        sized by what the sequence plays rather than by which policy is
        driving it.
        """
        winding = EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER
        winding |= EventAction.SHIFT_AFTER
        return sum(
            bin(int(event.action & winding)).count("1") for event in description.events
        )

    def simulate(
        self,
        description: SequenceDescription,
        tissue: TissueProperties,
        *,
        repetitions: int = 1,
        ss_iter: int = 1,
        record: RecordMode = "all",
        nstates: int | None = None,
        slice_profile: ExactSliceProfile | None = None,
        rf_raster_time_s: float = 1e-6,
        device: torch.device | str | None = None,
        events: Any = None,
    ) -> SimulationResult:
        """Walk an event stream and record selected ADC signals.

        ``events`` is the description's stream already packed into buffers, for
        a caller that holds the structure fixed and rebuilds only what varies;
        see :mod:`torchsim.optim`. It must be the packing of ``description``.

        Raises
        ------
            RuntimeError: if no fused kernel can take this tissue -- a device
                that has none, or a build without the extension.
        """
        repetitions = _as_integer(repetitions, "repetitions")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        ss_iter = _as_integer(ss_iter, "ss_iter")
        if ss_iter < 1:
            raise ValueError("ss_iter must be positive")
        # Every playing but the last is a settling one, played into the state
        # the next inherits and recorded from none of them.
        settle = ss_iter - 1
        if record not in {"all", "acquired", "echo"}:
            raise ValueError("record must be 'all', 'acquired', or 'echo'")

        tissue, sensitivities = _dynamic_transmit(tissue, description, device)
        tissue, shims = _resolve_transmit(tissue, description, device)
        prepared, output_shape, target_device = _prepare_tissue(tissue, device, shims)
        (
            t1,
            t2,
            m0,
            b1,
            b1_phase,
            b0,
            inversion_efficiency,
            diffusion,
            velocity,
            _bound_fraction,
            _bound_exchange,
            _t1_bound,
            _pool_b_fraction,
            _pool_b_exchange,
            _t1_pool_b,
            _t2_pool_b,
            _pool_b_shift,
        ) = prepared
        features = features_of(tissue)
        bound_pool = wants_bound_pool(tissue.bound_fraction)
        exchange_pool = wants_exchange_pool(tissue.pool_b_fraction)
        _within_one_voxel(tissue.bound_fraction, tissue.pool_b_fraction)

        if nstates is None:
            winding = (settle + repetitions) * self.shifts_per_repetition(description)
            nstates = max(8, min(64, 1 + winding))
        else:
            nstates = _as_integer(nstates, "nstates")
        if nstates < 1:
            raise ValueError("nstates must be positive")

        if slice_profile is not None and not isinstance(
            slice_profile, ExactSliceProfile
        ):
            raise TypeError(
                "slice_profile says where across the slice to integrate the "
                "sequence's own pulse -- ask for it with exact_slice_profile(). "
                "A tensor of flip-angle scalings treats a Bloch response as "
                "proportional to the pulse driving it, which it is not; give "
                "the RF definition its waveform instead"
            )
        accelerated = simulate_native(
            description,
            prepared,
            output_shape,
            repetitions=repetitions,
            settle=settle,
            record=record,
            nstates=nstates,
            slice_profile=slice_profile,
            rf_raster_time_s=rf_raster_time_s,
            lineshape=_absorption_table(description, b0, target_device)
            if bound_pool
            else None,
            exchanging=exchange_pool,
            features=features,
            transmit=sensitivities,
            packed=events,
        )
        if accelerated is None:
            raise RuntimeError(
                f"no fused EPG kernel is available for a tissue on {target_device}"
            )
        return SimulationResult(*accelerated)


def simulate_subspace(
    description: SequenceDescription,
    tissue: TissueProperties,
    *,
    rank: int,
    repetitions: int = 1,
    record: RecordMode = "all",
    nstates: int | None = None,
    slice_profile: ExactSliceProfile | None = None,
    rf_raster_time_s: float = 1e-6,
    device: torch.device | str | None = None,
) -> Subspace:
    """Simulate a dictionary and return its leading temporal basis."""
    result = EpgEngine().simulate(
        description,
        tissue,
        repetitions=repetitions,
        record=record,
        nstates=nstates,
        slice_profile=slice_profile,
        rf_raster_time_s=rf_raster_time_s,
        device=device,
    )
    fitted = Subspace.fit(result.signal, rank)
    return replace(fitted, dictionary=result.signal, simulation=result)


# %% private module subroutines


def _dynamic_transmit(
    tissue: TissueProperties,
    description: SequenceDescription,
    device: torch.device | str | None,
) -> tuple[TissueProperties, torch.Tensor | None]:
    """The per-channel sensitivities a dynamically shimmed pulse integrates.

    A pulse whose channels each carry their own waveform does not reduce to a
    flip and a phase, so its rotation is worked out per voxel from the complex
    sensitivities themselves. What is left on the tissue is the magnitude of
    the field a voxel sees when every channel drives unit weight -- which is
    what the flip angle is read against and what a bound pool's saturation
    reads -- and no transmit phase at all, since the rotation already turned by
    it and the kernels would turn by it again.

    Returns the tissue and the sensitivities, or the tissue unchanged and
    ``None`` where no pulse asks for this.

    Raises
    ------
        NotImplementedError: if the sequence also declares a transmit shim.
        ValueError: if the pulses disagree on how many channels they drive, or
            if the transmit maps do not lead with that many.
    """
    widths = {
        definition.channel_count
        for definition in description.rf_definitions.values()
        if definition.rf_mode() is RfMode.DYNAMIC
    }
    if not widths:
        return tissue, None
    if len(widths) > 1:
        raise ValueError(
            f"the sequence's pulses drive {sorted(widths)} channels, and a "
            f"voxel has one set of sensitivities"
        )
    (width,) = widths
    if description.shim_definitions:
        raise NotImplementedError(
            "a pulse driving its own waveform per channel already says how "
            "hard each channel plays; a static shim beside it is a second "
            "answer to the same question"
        )
    resolved = torch.device(
        device
        if device is not None
        else next(
            (
                value.device
                for value in (tissue.b1, tissue.b1_phase_rad)
                if isinstance(value, torch.Tensor)
            ),
            torch.device("cpu"),
        )
    )
    magnitude, phase = torch.broadcast_tensors(
        _as_float_tensor(tissue.b1, resolved),
        _as_float_tensor(tissue.b1_phase_rad, resolved),
    )
    if magnitude.ndim == 0:
        magnitude = magnitude.expand(width, 1)
        phase = phase.expand(width, 1)
    elif magnitude.shape[0] != width:
        raise ValueError(
            f"a pulse driving {width} channels reads a transmit map apiece, "
            f"so b1 leads with {width} and not {magnitude.shape[0]}"
        )
    sensitivities = torch.polar(magnitude, phase).reshape(width, -1).mT.contiguous()
    return (
        replace(
            tissue,
            b1=sensitivities.sum(dim=-1).abs().reshape(magnitude.shape[1:]),
            b1_phase_rad=0.0,
        ),
        sensitivities.to(torch.complex128),
    )


def _resolve_transmit(
    tissue: TissueProperties,
    description: SequenceDescription,
    device: torch.device | str | None,
) -> tuple[TissueProperties, int]:
    """Reduce a transmit array to the field it puts in each voxel per shim.

    Leaves a single-channel sequence exactly as it was, so it reaches the
    kernels through the same buffers and the same arithmetic. Returns the
    tissue alongside how many shim rows its transmit buffers hold.
    """
    if not description.shim_definitions:
        return tissue, 1
    resolved = torch.device(
        device
        if device is not None
        else next(
            (
                value.device
                for value in (tissue.b1, tissue.b1_phase_rad)
                if isinstance(value, torch.Tensor)
            ),
            torch.device("cpu"),
        )
    )
    magnitude, phase = transmit_field(
        description, tissue.b1, tissue.b1_phase_rad, resolved
    )
    return (
        replace(tissue, b1=magnitude, b1_phase_rad=phase),
        max(1, len(shim_rows(description))),
    )


def target_device(
    tissue: TissueProperties, device: torch.device | str | None = None
) -> torch.device:
    """Where a run's per-voxel buffers will live.

    A device named by the caller, otherwise the one the tissue is already on,
    otherwise the host.
    """
    if device is not None:
        return torch.device(device)
    return next(
        (
            getattr(tissue, name).device
            for name in TISSUE_NAMES
            if isinstance(getattr(tissue, name), torch.Tensor)
        ),
        torch.device("cpu"),
    )


def _prepare_tissue(
    tissue: TissueProperties,
    device: torch.device | str | None,
    shims: int = 1,
) -> tuple[tuple[torch.Tensor, ...], torch.Size, torch.device]:
    values = tuple(getattr(tissue, name) for name in TISSUE_NAMES)
    device = target_device(tissue, device)
    given = [_as_float_tensor(value, device) for value in values]
    # The transmit buffers may lead with a shim axis, which is not a voxel axis
    # and must stay out of the broadcast that decides how many voxels there are.
    tensors = torch.broadcast_tensors(
        *(
            value[0] if shims > 1 and index in _TRANSMIT else value
            for index, value in enumerate(given)
        )
    )
    shape = tensors[0].shape
    # Broadcasting leaves a scalar property as a stride-0 view. The kernels index
    # raw pointers, so materialize before anyone hands one to them.
    flat = [
        given[index].expand(shims, *shape).reshape(-1).contiguous()
        if shims > 1 and index in _TRANSMIT
        else value.reshape(-1).contiguous()
        for index, value in enumerate(tensors)
    ]
    # The relaxation times are the entries used as denominators downstream.
    for index in _RELAXATION_TIMES:
        flat[index] = flat[index].clamp_min(MINIMUM_RELAXATION_TIME_MS)
    return tuple(flat), shape, device


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _within_one_voxel(bound_fraction: Any, pool_b_fraction: Any) -> None:
    """Refuse fractions that give away more of the voxel than there is.

    The free water is whatever the two second pools leave, so their fractions
    have to sum to at most one. Past that the free pool starts at a negative
    magnetization, which every pass afterwards would carry as though it meant
    something.

    Raises
    ------
        ValueError: if the two fractions sum past one anywhere.
    """
    total = torch.as_tensor(bound_fraction) + torch.as_tensor(pool_b_fraction)
    if bool((total > 1.0).any()):
        raise ValueError(
            "bound_fraction and pool_b_fraction share the voxel with the free "
            "water, so they cannot sum past one"
        )


def _as_integer(value: Any, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())
