"""Composable modules a sequence description is assembled from.

An operator is one module -- a pulse, a readout, a delay, a preparation --
that knows what it plays and how long it holds the timeline. :func:`compose`
lays modules end to end and turns their spans into the absolute timestamps a
description carries, which is the accumulator a builder would otherwise keep
for itself. A sequence whose timing is computed rather than accumulated calls
:meth:`Operator.emit` at the time it wants instead.

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
    "Operator",
    "compose",
    "delay",
    "excitation",
    "inversion",
    "module",
    "operator",
    "operator_names",
    "readout",
    "refocusing",
    "register_operator",
    "saturation",
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
    """One module of a sequence: what it plays, and how long it lasts.

    Parameters
    ----------
    emit:
        Given the absolute time the module starts, in seconds, the events it
        plays.
    duration_s:
        What the module holds the timeline for. Zero for a module that plays
        instantaneously and leaves the next one to start where it did.
    """

    emit: Callable[[Any], tuple[SequenceEvent, ...]]
    duration_s: Any = 0.0


def compose(
    *parts: Operator | tuple[Any, Operator], start_s: Any = 0.0
) -> tuple[tuple[SequenceEvent, ...], Any]:
    """Lay operators out and return their events and the span they cover.

    A bare operator starts where the one before it ended. One given as
    ``(offset_s, operator)`` starts that far after ``start_s`` instead, which
    is how a sequence that times itself from an echo rather than by
    accumulating says so.

    Parameters
    ----------
    parts:
        The operators, in the order they play.
    start_s:
        When the first one starts.

    Returns
    -------
    tuple
        The events at absolute times, and where the last one ended.
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
    """Package several operators as one that can be placed and repeated.

    Offsets inside are relative to wherever the module itself is placed, which
    is what lets a shot -- a pulse, a sample some way into the repetition, and
    the rest of the repetition after it -- be one thing a caller lays down
    once.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        events, _ = compose(*parts, start_s=start_s)
        return events

    return Operator(emit, duration_s)


def excitation(
    flip_rad: Any,
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    definition_id: int = 0,
    frequency_hz: Any = 0.0,
    shim_id: int = 0,
    action: EventAction = EventAction.NONE,
) -> Operator:
    """Return a pulse that tips magnetization into the transverse plane."""

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


def refocusing(
    flip_rad: Any,
    phase_rad: Any = 0.0,
    *,
    crushed: bool = True,
    duration_s: Any = 0.0,
    definition_id: int = 0,
    frequency_hz: Any = 0.0,
    shim_id: int = 0,
) -> Operator:
    """Return a pulse the sequence sits between crushers, unless told otherwise.

    ``crushed`` is what makes the pair of unbalanced gradients around the
    pulse part of the module rather than something the caller remembers to
    add.
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


def inversion(
    *,
    duration_s: Any = 0.0,
    flip_rad: Any = torch.pi,
    phase_rad: Any = 0.0,
    definition_id: int = 0,
) -> Operator:
    """Return an ideal inversion, scaled by the tissue's inversion efficiency.

    The kernels answer this with ``Mz -> -efficiency * Mz`` on every order and
    no rotation at all, so the pulse deposits nothing in a bound pool. A
    preparation whose saturation matters is an :func:`excitation` or a
    :func:`saturation` carrying a real waveform instead.
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


def saturation(
    definition_id: int,
    flip_rad: Any,
    *,
    offset_hz: Any = 0.0,
    phase_rad: Any = 0.0,
    duration_s: Any = 0.0,
    shim_id: int = 0,
) -> Operator:
    """Return an off-resonance pulse that deposits power in a semisolid pool.

    The deposit follows the waveform ``definition_id`` names and the offset it
    is played at, read against the tissue's lineshape, so this is an
    :class:`RfUse.EXCITATION` carrying a real shape rather than an ideal
    rotation.
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


def readout(
    phase_rad: Any = 0.0,
    *,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
    duration_s: Any = 0.0,
    action: EventAction = EventAction.NONE,
) -> Operator:
    """Return one sample of the transverse magnetization, demodulated by a phase.

    ``action`` is what the sequence plays after the sample -- ``SPOIL_AFTER``
    for a spoiled gradient echo, ``SHIFT_AFTER`` for an unbalanced one, and
    nothing for a train that rewinds.
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


def delay(
    duration_s: Any, *, action: EventAction = EventAction.NONE
) -> Operator:
    """Return time passing, and whatever the sequence plays across it.

    :meth:`SequenceEvent.wait` takes no action, so a delay that spoils or
    winds the states is built here rather than by the caller reaching for the
    raw constructor.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        if action is EventAction.NONE:
            return (SequenceEvent.wait(_TO_US * start_s),)
        return (
            SequenceEvent(EventType.WAIT, _TO_US * start_s, (), action),
        )

    return Operator(emit, duration_s)


_REGISTRY: dict[str, Callable[..., Operator]] = {
    "excitation": excitation,
    "refocusing": refocusing,
    "inversion": inversion,
    "saturation": saturation,
    "readout": readout,
    "delay": delay,
}


def register_operator(name: str, factory: Callable[..., Operator]) -> None:
    """Make an operator reachable by name.

    A stream that arrives already labelled -- from an MRD file, or from any
    generator that names what it plays -- dispatches through this rather than
    through the caller's own mapping.

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
    """Look up an operator factory by name.

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
    """Every registered operator name, in registration order."""
    return tuple(_REGISTRY)
