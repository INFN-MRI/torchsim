"""Composable modules a sequence description is assembled from.

An operator is one module -- a pulse, a readout, a delay, a preparation --
that knows what it plays and how long it holds the timeline. :func:`compose`
lays modules end to end and turns their spans into the absolute timestamps a
description carries, which is the accumulator a builder would otherwise keep
for itself. A sequence whose timing is computed rather than accumulated calls
:meth:`Operator.emit` at the time it wants instead.

An operator names a thing rather than an action -- ``Readout``, ``Dephase``,
``SSFPFidReadout`` -- and is capitalized to say so, even where it is a function
rather than a class. ``bSSFPReadout`` keeps the lowercase ``b`` the sequence is
written with everywhere else.

Operators speak only the vocabulary the fused kernels already implement: the
three event types, the four dephasing actions, and the RF and ADC roles. A
module that needs nothing beyond that -- a T2 preparation, a
magnetization-transfer preparation, a shaped or per-channel pulse -- is
written here and reaches the kernels with no change to them.

What an operator cannot say is *how much* a gradient dephases. A description
carries one crusher moment for the whole sequence and dephasing is quantized
to whole configuration orders, so a bipolar pair, a b-value of its own, or a
crusher of twice its neighbour's area have no representation here.
"""

from __future__ import annotations

__all__ = [
    "Delay",
    "Dephase",
    "Excitation",
    "FSEReadout",
    "Inversion",
    "Operator",
    "Readout",
    "Refocusing",
    "SPGRReadout",
    "SSFPEchoReadout",
    "SSFPFidReadout",
    "SampledPulse",
    "Saturation",
    "Spoil",
    "bSSFPReadout",
    "compose",
    "module",
    "operator",
    "operator_names",
    "register_operator",
]

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ._description import (
    AdcRole,
    EventAction,
    EventType,
    RfUse,
    SequenceEvent,
)

# Seconds to the microseconds a description timestamps in.
_TO_US = 1e6


@dataclass(frozen=True)
class Operator:
    """One module of a sequence: the events it plays and the time it spans.

    ``a @ b`` plays two operators back to back as one operator.

    Parameters
    ----------
    emit : callable
        Maps the module's start time in seconds to the tuple of
        ``SequenceEvent`` it plays.
    duration_s : float or torch.Tensor, default=0.0
        Time the module holds the timeline, in seconds. Zero leaves the next
        module to start at the same time.
    """

    emit: Callable[[Any], tuple[SequenceEvent, ...]]
    duration_s: Any = 0.0

    def __matmul__(self, other: Operator) -> Operator:
        """Play ``self`` then ``other`` as one operator; durations add."""
        return module(self, other, duration_s=self.duration_s + other.duration_s)


def compose(
    *parts: Operator | tuple[Any, Operator], start_s: Any = 0.0
) -> tuple[tuple[SequenceEvent, ...], Any]:
    """Lay operators out in time and return their events and end time.

    A bare operator starts where the previous one ended. An ``(offset_s,
    operator)`` pair starts ``offset_s`` after ``start_s``.

    Parameters
    ----------
    *parts : Operator or tuple
        Operators in play order, or ``(offset_s, operator)`` pairs, in seconds.
    start_s : float or torch.Tensor, default=0.0
        Start time of the first operator, in seconds.

    Returns
    -------
    events : tuple of SequenceEvent
        Events at absolute times.
    end_s : float or torch.Tensor
        End time of the last operator, in seconds.
    """
    events: list[SequenceEvent] = []
    cursor = start_s
    for part in parts:
        offset, played = part if isinstance(part, tuple) else (None, part)
        when = cursor if offset is None else start_s + offset
        events.extend(played.emit(when))
        cursor = when + played.duration_s
    return tuple(events), cursor


def module(*parts: Operator | tuple[Any, Operator], duration_s: Any) -> Operator:
    """Package operators as one operator.

    Offsets inside ``parts`` are relative to the start of the module.

    Parameters
    ----------
    *parts : Operator or tuple
        Operators in play order, or ``(offset_s, operator)`` pairs.
    duration_s : float or torch.Tensor
        Time the module holds the timeline, in seconds. Not derived from
        ``parts``, so an offset part may extend past it.

    Returns
    -------
    Operator
        The parts as one operator that spans ``duration_s``.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        events, _ = compose(*parts, start_s=start_s)
        return events

    return Operator(emit, duration_s)


def Excitation(
    flip_rad: Any,
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    definition_id: int = 0,
    frequency_hz: Any = 0.0,
    shim_id: int = 0,
    action: EventAction = EventAction.NONE,
) -> Operator:
    """Excitation pulse.

    Plays one RF event with the excitation tag at the start of the module. The
    module holds the timeline for ``duration_s`` without rotating over it: an
    ideal pulse (``definition_id=0``) turns instantly, and a shaped one turns
    through the integral of its definition.

    Parameters
    ----------
    flip_rad : float or torch.Tensor
        Flip angle, in radians.
    phase_rad : float or torch.Tensor, default=0.0
        RF phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Time held on the timeline, in seconds.
    definition_id : int, default=0
        RF definition played; 0 is the ideal pulse.
    frequency_hz : float or torch.Tensor, default=0.0
        Frequency offset, in Hz.
    shim_id : int, default=0
        Transmit shim definition; 0 for none.
    action : EventAction, default=EventAction.NONE
        Gradient action played around the pulse.

    Returns
    -------
    Operator
        One RF event, tagged as an excitation, at the start of the module.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.EXCITATION,
                flip_rad,
                phase_rad,
                frequency_hz,
                shim_id,
                action=action,
            ),
        )

    return Operator(emit, duration_s)


def Refocusing(
    flip_rad: Any,
    phase_rad: Any = 0.0,
    *,
    crushed: bool = True,
    duration_s: Any = 0.0,
    definition_id: int = 0,
    frequency_hz: Any = 0.0,
    shim_id: int = 0,
) -> Operator:
    """Refocusing pulse, between crusher gradients by default.

    Plays one RF event with the refocusing tag at the start of the module. With
    ``crushed`` an unbalanced gradient is played before and after it
    (``EventAction.CRUSH_BEFORE | CRUSH_AFTER``).

    Parameters
    ----------
    flip_rad : float or torch.Tensor
        Flip angle, in radians.
    phase_rad : float or torch.Tensor, default=0.0
        RF phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Time held on the timeline, in seconds.
    definition_id : int, default=0
        RF definition played; 0 is the ideal pulse.
    frequency_hz : float or torch.Tensor, default=0.0
        Frequency offset, in Hz.
    shim_id : int, default=0
        Transmit shim definition; 0 for none.
    crushed : bool, default=True
        Play an unbalanced gradient before and after the pulse.

    Returns
    -------
    Operator
        One RF event, tagged as a refocusing pulse, at the start of the module.
    """
    action = (
        EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER
        if crushed
        else EventAction.NONE
    )

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.REFOCUSING,
                flip_rad,
                phase_rad,
                frequency_hz,
                shim_id,
                action=action,
            ),
        )

    return Operator(emit, duration_s)


def Inversion(
    *,
    duration_s: Any = 0.0,
    flip_rad: Any = torch.pi,
    phase_rad: Any = 0.0,
    definition_id: int = 0,
) -> Operator:
    """Ideal inversion pulse.

    The kernels apply ``Mz -> -efficiency * Mz`` on every order with no
    rotation, so the pulse deposits no power in a bound pool. Use
    :func:`Saturation` for a preparation whose saturation matters.

    Parameters
    ----------
    duration_s : float or torch.Tensor, default=0.0
        Time held on the timeline, in seconds.
    flip_rad : float or torch.Tensor, default=torch.pi
        Nominal flip angle, in radians.
    phase_rad : float or torch.Tensor, default=0.0
        RF phase, in radians.
    definition_id : int, default=0
        RF definition played; 0 is the ideal pulse.

    Returns
    -------
    Operator
        One RF event, tagged as an inversion, at the start of the module.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.INVERSION,
                flip_rad,
                phase_rad,
            ),
        )

    return Operator(emit, duration_s)


def Saturation(
    definition_id: int,
    flip_rad: Any,
    *,
    offset_hz: Any = 0.0,
    phase_rad: Any = 0.0,
    duration_s: Any = 0.0,
    shim_id: int = 0,
) -> Operator:
    """Off-resonance saturation pulse that deposits power in a semisolid pool.

    The deposit follows the waveform of ``definition_id`` and ``offset_hz``,
    read against the tissue lineshape.

    Parameters
    ----------
    definition_id : int
        RF definition played.
    flip_rad : float or torch.Tensor
        Flip angle, in radians.
    offset_hz : float or torch.Tensor, default=0.0
        Frequency offset from the centre frequency, in Hz.
    phase_rad : float or torch.Tensor, default=0.0
        RF phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Time held on the timeline, in seconds.
    shim_id : int, default=0
        Transmit shim definition; 0 for none.

    Returns
    -------
    Operator
        One RF event at the start of the module, played at ``offset_hz``.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.EXCITATION,
                flip_rad,
                phase_rad,
                offset_hz,
                shim_id,
            ),
        )

    return Operator(emit, duration_s)


def SampledPulse(
    waveform_rad_per_s: Any,
    dwell_s: float,
    *,
    phase_rad: Any = 0.0,
    offset_hz: Any = 0.0,
    definition_id: int = 0,
    shim_id: int = 0,
) -> Operator:
    """Shaped pulse played as one hard pulse per sample, with free evolution between.

    Each sample rotates the magnetization through its area at the middle of
    its dwell; between samples each pool precesses, relaxes and exchanges
    (hard-pulse approximation [1]_). A complex waveform rotates about the axis
    given by its phase; a real one rotates through its signed area. An
    ``offset_hz`` advances the rotation axis by ``-2 pi offset_hz`` rad/s from
    the start of the pulse. A ``phase_rad`` or ``offset_hz`` with one value per
    train plays one train per value.

    Parameters
    ----------
    waveform_rad_per_s : array-like
        Nutation rate (gamma times B1), one sample per dwell, in rad/s.
    dwell_s : float
        Sample duration, in seconds.
    phase_rad : float or torch.Tensor, default=0.0
        Phase of the whole pulse, in radians.
    offset_hz : float or torch.Tensor, default=0.0
        Frequency offset from the centre frequency, in Hz.
    definition_id : int, default=0
        RF definition every sample names; sets the power deposited in a
        semisolid pool. Non-zero requires a single offset.
    shim_id : int, default=0
        Transmit shim definition; 0 for none.

    Returns
    -------
    Operator
        Holds the timeline for ``len(waveform) * dwell_s``.

    Raises
    ------
    ValueError
        If the waveform is empty or not one-dimensional, ``dwell_s`` is not
        positive, or a non-zero ``definition_id`` is played at more than one
        offset.

    References
    ----------
    .. [1] Pauly, J., Le Roux, P., Nishimura, D., Macovski, A., "Parameter
       relations for the Shinnar-Le Roux selective excitation pulse design
       algorithm", IEEE Transactions on Medical Imaging 10.1 (1991),
       pp. 53-65. https://doi.org/10.1109/42.75611
    """
    samples = torch.as_tensor(waveform_rad_per_s)
    if samples.dim() != 1 or samples.numel() == 0:
        raise ValueError("a sampled pulse is a non-empty one-dimensional waveform")
    if not float(dwell_s) > 0.0:
        raise ValueError(f"dwell_s must be positive, not {dwell_s!r}")
    offset = torch.as_tensor(offset_hz, dtype=torch.float64)
    if definition_id != 0 and offset.numel() != 1:
        raise ValueError(
            "a semisolid pool reads its lineshape at one offset, so a sampled "
            "pulse naming a definition is played at one offset for every train"
        )
    frequency_hz = float(offset.reshape(-1)[0]) if offset.numel() == 1 else 0.0
    if samples.is_complex():
        areas = samples.abs() * dwell_s
        turns = samples.angle()
    else:
        areas = samples * dwell_s
        turns = torch.zeros_like(samples, dtype=torch.float64)
    middle_s = (torch.arange(samples.numel(), dtype=torch.float64) + 0.5) * dwell_s
    phases = (
        torch.as_tensor(phase_rad)[..., None]
        + turns
        - 2.0 * torch.pi * offset[..., None] * middle_s
    )
    # A value per train leads the axis a train count is read from, and a
    # plain number stays one so a train of a thousand samples packs as fast
    # as a train of a thousand ideal pulses.
    flips = list(areas.unbind(0)) if areas.requires_grad else areas.tolist()
    if phases.requires_grad:
        per_sample = list(phases.movedim(-1, 0).unbind(0))
    elif phases.dim() == 1:
        per_sample = phases.tolist()
    else:
        per_sample = list(phases.movedim(-1, 0).unbind(0))
    offsets_s = middle_s.tolist()

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return tuple(
            SequenceEvent.rf(
                _TO_US * (start_s + at_s),
                definition_id,
                RfUse.EXCITATION,
                flip,
                phase,
                frequency_hz,
                shim_id,
            )
            for at_s, flip, phase in zip(offsets_s, flips, per_sample, strict=True)
        )

    return Operator(emit, samples.numel() * dwell_s)


def Readout(
    phase_rad: Any = 0.0,
    *,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
    duration_s: Any = 0.0,
    action: EventAction = EventAction.NONE,
) -> Operator:
    """One ADC sample of the transverse magnetization.

    The sample is the transverse magnetization at the start of the module,
    demodulated by ``phase_rad``; ``duration_s`` only sets how long the module
    holds the timeline.

    Parameters
    ----------
    phase_rad : float or torch.Tensor, default=0.0
        Demodulation phase, in radians.
    role : AdcRole, default=AdcRole.SINGLE
        Role of the sample in the event stream.
    is_echo : bool, default=True
        Whether the sample is at the k-space origin.
    duration_s : float or torch.Tensor, default=0.0
        Time held on the timeline, in seconds.
    action : EventAction, default=EventAction.NONE
        Gradient action played after the sample, e.g. ``SPOIL_AFTER``.

    Returns
    -------
    Operator
        One ADC event at the start of the module.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.adc(
                _TO_US * start_s,
                role,
                phase_rad,
                is_echo=is_echo,
                action=action,
            ),
        )

    return Operator(emit, duration_s)


def Delay(duration_s: Any, *, action: EventAction = EventAction.NONE) -> Operator:
    """Delay, optionally carrying a gradient action.

    Free precession, relaxation and exchange act over ``duration_s``.

    Parameters
    ----------
    duration_s : float or torch.Tensor
        Time held on the timeline, in seconds.
    action : EventAction, default=EventAction.NONE
        Gradient action played at the start of the delay.

    Returns
    -------
    Operator
        One wait event at the start of the module, spanning ``duration_s``.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        if action is EventAction.NONE:
            return (SequenceEvent.wait(_TO_US * start_s),)
        return (SequenceEvent(EventType.WAIT, _TO_US * start_s, (), action),)

    return Operator(emit, duration_s)


def Dephase() -> Operator:
    """Unbalanced gradient of zero duration.

    Every configuration order advances by one. The dephasing across a voxel
    is the description's ``crusher_dephasing_rad``.
    """
    return Operator(
        lambda start_s: (
            SequenceEvent(
                EventType.WAIT, _TO_US * start_s, (), EventAction.CRUSH_AFTER
            ),
        )
    )


def Spoil() -> Operator:
    """Ideal transverse spoiling of zero duration.

    Every transverse order is discarded.
    """
    return Operator(
        lambda start_s: (
            SequenceEvent(
                EventType.WAIT, _TO_US * start_s, (), EventAction.SPOIL_AFTER
            ),
        )
    )


def bSSFPReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Balanced SSFP readout: a sample, then the rest of the repetition time.

    No gradient action is played, so the dephasing over a repetition is zero.

    Parameters
    ----------
    phase_rad : float or torch.Tensor, default=0.0
        Demodulation phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Repetition time remaining after the sample, in seconds.
    role : AdcRole, default=AdcRole.SINGLE
        Role of the sample in the event stream.
    is_echo : bool, default=True
        Whether the sample is at the k-space origin.

    Returns
    -------
    Operator
        Sample, then a delay of ``duration_s``; the module spans ``duration_s``.
    """
    return module(
        Readout(phase_rad, role=role, is_echo=is_echo),
        Delay(duration_s),
        duration_s=duration_s,
    )


def SSFPFidReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Unbalanced SSFP readout (FISP): a sample, then one unbalanced gradient.

    The gradient advances every configuration order by one, so the free
    induction decay is sampled before the order shift.

    Parameters
    ----------
    phase_rad : float or torch.Tensor, default=0.0
        Demodulation phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Repetition time remaining after the sample, in seconds.
    role : AdcRole, default=AdcRole.SINGLE
        Role of the sample in the event stream.
    is_echo : bool, default=True
        Whether the sample is at the k-space origin.

    Returns
    -------
    Operator
        Sample, one unbalanced gradient, then a delay of ``duration_s``; the module spans ``duration_s``.
    """
    return module(
        Readout(phase_rad, role=role, is_echo=is_echo),
        Dephase(),
        Delay(duration_s),
        duration_s=duration_s,
    )


def SSFPEchoReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Reversed SSFP readout (PSIF): one unbalanced gradient, then a sample.

    The sample reads the echo of the previous excitation (the T2-weighted
    half of a reversed-FISP pair). The order shift is that
    of :func:`SSFPFidReadout`, applied before the sample.

    Parameters
    ----------
    phase_rad : float or torch.Tensor, default=0.0
        Demodulation phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Repetition time remaining after the sample, in seconds.
    role : AdcRole, default=AdcRole.SINGLE
        Role of the sample in the event stream.
    is_echo : bool, default=True
        Whether the sample is at the k-space origin.

    Returns
    -------
    Operator
        Unbalanced gradient, sample, then a delay of ``duration_s``; the module spans ``duration_s``.
    """
    return module(
        Dephase(),
        Readout(phase_rad, role=role, is_echo=is_echo),
        Delay(duration_s),
        duration_s=duration_s,
    )


def SPGRReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Spoiled gradient-echo readout: a sample, then ideal transverse spoiling.

    Parameters
    ----------
    phase_rad : float or torch.Tensor, default=0.0
        Demodulation phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Repetition time remaining after the sample, in seconds.
    role : AdcRole, default=AdcRole.SINGLE
        Role of the sample in the event stream.
    is_echo : bool, default=True
        Whether the sample is at the k-space origin.

    Returns
    -------
    Operator
        Sample, ideal spoiling, then a delay of ``duration_s``; the module spans ``duration_s``.
    """
    return module(
        Readout(phase_rad, role=role, is_echo=is_echo),
        Spoil(),
        Delay(duration_s),
        duration_s=duration_s,
    )


def FSEReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    is_echo: bool = True,
) -> Operator:
    """Fast spin-echo readout: the sample at a spin echo, marked as the echo centre.

    No gradient action is played; dephasing is carried by the refocusing
    pulses.

    Parameters
    ----------
    phase_rad : float or torch.Tensor, default=0.0
        Demodulation phase, in radians.
    duration_s : float or torch.Tensor, default=0.0
        Echo spacing remaining after the sample, in seconds.
    is_echo : bool, default=True
        Whether the sample is at the k-space origin.

    Returns
    -------
    Operator
        Sample, then a delay of ``duration_s``; the module spans ``duration_s``.
    """
    return module(
        Readout(phase_rad, role=AdcRole.ECHO_CENTER, is_echo=is_echo),
        Delay(duration_s),
        duration_s=duration_s,
    )


_REGISTRY: dict[str, Callable[..., Operator]] = {
    "excitation": Excitation,
    "refocusing": Refocusing,
    "inversion": Inversion,
    "saturation": Saturation,
    "sampled-pulse": SampledPulse,
    "readout": Readout,
    "delay": Delay,
    "dephase": Dephase,
    "spoil": Spoil,
    "bssfp-readout": bSSFPReadout,
    "ssfp-fid-readout": SSFPFidReadout,
    "ssfp-echo-readout": SSFPEchoReadout,
    "spgr-readout": SPGRReadout,
    "fse-readout": FSEReadout,
}


def register_operator(name: str, factory: Callable[..., Operator]) -> None:
    """Register an operator factory under a name.

    Parameters
    ----------
    name : str
        Case-insensitive; underscores are read as hyphens.
    factory : callable
        Called with the operator's own parameters; returns an ``Operator``.

    Raises
    ------
    ValueError
        If the name is already registered.
    """
    normalized = name.lower().replace("_", "-")
    if normalized in _REGISTRY:
        raise ValueError(f"operator {name!r} is already registered")
    _REGISTRY[normalized] = factory


def operator(name: str) -> Callable[..., Operator]:
    """Return the registered operator factory for a name.

    Parameters
    ----------
    name : str
        Case-insensitive; underscores are read as hyphens.

    Returns
    -------
    callable
        The factory, e.g. :func:`Excitation` for ``"excitation"``.

    Raises
    ------
    ValueError
        If no operator of that name is registered.
    """
    normalized = name.lower().replace("_", "-")
    try:
        return _REGISTRY[normalized]
    except KeyError as error:
        raise ValueError(f"unknown operator {name!r}") from error


def operator_names() -> tuple[str, ...]:
    """Return every registered operator name, in registration order.

    Returns
    -------
    tuple of str
        Lowercase hyphenated names, e.g. ``"ssfp-fid-readout"``.
    """
    return tuple(_REGISTRY)
