"""Resolving a protocol's structure once, and binding its values per call.

A simulator called many times with the same protocol and different values --
a design loop moving flip angles, a dictionary sweeping a schedule -- rebuilds
the same event stream every time. Walking the layout, assembling the events
and packing them into buffers is per-event Python, and on a small problem it
costs an order of magnitude more than the kernels it feeds.

Nothing about the stream changes except the numbers in four of its buffers.
:class:`Packing` is that observation made usable: the structure is resolved
once, and each call rebuilds ``duration``, ``flip``, ``phase`` and the sample
times with whole-tensor arithmetic instead of one operation per event.

**Why an affine map is exact here.** A layout turns the values it is given
into event parameters by scaling and offsetting them: degrees become radians,
a flip angle is divided by the pulse's envelope integral and multiplied back
by it, a phase adds the integral's argument, a timestamp accumulates a
spacing. So each entry of a packed buffer is ``offset + scale * value`` in one
element of one argument, and two forward-mode passes recover both -- a tangent
of ones gives the scale, a tangent of ``1, 2, 3, ...`` gives the scale times
the index.

**What refuses.** A ratio that is not a whole number says an entry draws on
more than one element, and a rebuild that disagrees with a fresh packing at a
point the map never saw says the layout is not affine after all. Either one
gives no binding, and the caller keeps the ordinary path -- slower, and the
same answer.
"""

from __future__ import annotations

__all__ = ["Packing", "bind", "run_key"]

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..sequence._accelerators import _PackedEvents, pack_description
from ..sequence._description import SequenceDescription

#: The packed buffers whose entries are values rather than structure.
_VALUES = ("duration", "flip", "phase", "time_us")

#: How far from a whole number a recovered source index may read before the
#: entry is taken to draw on more than one element at once.
_WHOLE = 1e-3

#: Where the map is checked against a fresh packing. Chosen to share no factor
#: with the values it was read at, so an entry that only happens to agree
#: there does not agree here.
_ELSEWHERE = (1.37, 0.11)


@dataclass(frozen=True)
class _Term:
    """One argument's contribution to one packed buffer."""

    scale: torch.Tensor
    index: torch.Tensor


@dataclass(frozen=True)
class _Map:
    """One packed buffer, as an affine function of the arguments that vary."""

    offset: torch.Tensor
    terms: Mapping[str, _Term]

    def rebuild(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        built = self.offset
        for name, term in self.terms.items():
            built = built + term.scale * _drawn(term, values[name])
        return built.to(torch.float32).contiguous()


class Packing:
    """One protocol's event stream, ready to take a new set of values.

    Built by :func:`bind`, which is also what decides whether it can be built.

    Attributes
    ----------
    description:
        The sequence the structure was resolved from. Everything a run reads
        that is not an event value -- the RF definitions, the gradient
        geometry, how far the states wind -- comes from here.
    varying:
        The protocol arguments this rebinds, which are the ones that arrived
        as arrays.
    """

    def __init__(
        self,
        description: SequenceDescription,
        varying: tuple[str, ...],
        key: tuple[Any, ...],
        structure: _PackedEvents,
        maps: Mapping[str, _Map],
    ) -> None:
        self.description = description
        self.varying = varying
        self._key = key
        self._structure = structure
        self._maps = maps

    def matches(self, key: tuple[Any, ...]) -> bool:
        """Whether this packing was resolved for the run ``key`` describes."""
        return self._key == key

    def pack(self, values: Mapping[str, Any]) -> _PackedEvents:
        """Return the event buffers for ``values``, differentiable in them."""
        return replace(
            self._structure,
            **{name: self._maps[name].rebuild(values) for name in _VALUES},
        )


def run_key(
    values: Mapping[str, Any],
    *,
    repetitions: int,
    record: str,
    rf_raster_time_s: float,
    device: torch.device,
) -> tuple[Any, ...]:
    """What a packing is resolved for, and so what a later call must match.

    A floating-point array is keyed by its shape, dtype and device: its values
    are what the packing exists to rebind. Everything else is keyed by value,
    because a plain number reaches the timestamps and the structure with them.
    """
    return (
        tuple((name, _keyed(value)) for name, value in sorted(values.items())),
        repetitions,
        record,
        rf_raster_time_s,
        torch.device(device),
    )


def bind(
    simulator: Any,
    values: Mapping[str, Any],
    *,
    repetitions: int,
    record: str,
    rf_raster_time_s: float = 1e-6,
    device: torch.device | str = "cpu",
) -> Packing | None:
    """Resolve ``simulator``'s structure for ``values``, or return ``None``.

    Parameters
    ----------
    simulator:
        The protocol, whose :meth:`~torchsim.model.Simulator.describe`
        is walked here and, if this succeeds, not again.
    values:
        The protocol arguments as
        :meth:`~torchsim.model.Simulator.played` returns them. The
        floating-point arrays among them are the ones rebound.
    repetitions, record, rf_raster_time_s, device:
        The run this is for. A call differing in any of them gets no binding.

    Returns
    -------
    Packing or None
        ``None`` when one buffer entry draws on more than one element, or when
        the map does not reproduce a fresh packing away from where it was
        read -- both of which say the layout is not affine in what varies.
    """
    device = torch.device(device)
    names = tuple(
        name
        for name in sorted(values)
        if torch.is_tensor(values[name]) and values[name].is_floating_point()
    )
    held = {name: value for name, value in values.items() if name not in names}

    def described(given: tuple[torch.Tensor, ...]) -> SequenceDescription:
        return simulator.describe(**held, **dict(zip(names, given, strict=True)))

    def packed(description: SequenceDescription) -> _PackedEvents:
        return pack_description(
            description,
            repetitions=repetitions,
            record=record,
            device=device,
            rf_raster_time_s=rf_raster_time_s,
        )

    def buffers(given: tuple[torch.Tensor, ...]) -> _PackedEvents:
        return packed(described(given))

    def floats(*given: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(buffers(given), buffer) for buffer in _VALUES)

    primals = tuple(values[name].detach() for name in names)
    reference = described(primals)
    structure = packed(reference)

    terms: dict[str, dict[str, _Term]] = {buffer: {} for buffer in _VALUES}
    for position, name in enumerate(names):
        seeds = _seeds(primals, position)
        if seeds is None:
            continue
        along = tuple(torch.func.jvp(floats, primals, seed)[1] for seed in seeds)
        for buffer, scale, ramp in zip(_VALUES, *along, strict=True):
            term = _term(scale, ramp, primals[position].numel())
            if term is None:
                return None
            if bool(term.scale.any()):
                terms[buffer][name] = term

    maps = {}
    for buffer in _VALUES:
        offset = getattr(structure, buffer)
        for name, term in terms[buffer].items():
            offset = offset - term.scale * _drawn(term, primals[names.index(name)])
        maps[buffer] = _Map(offset=offset.detach(), terms=dict(terms[buffer]))

    packing = Packing(
        reference,
        names,
        run_key(
            values,
            repetitions=repetitions,
            record=record,
            rf_raster_time_s=rf_raster_time_s,
            device=device,
        ),
        structure,
        maps,
    )
    if not _agrees(buffers, names, primals, packing):
        return None
    return packing


# %% private module subroutines


def _keyed(value: Any) -> Any:
    """A comparable stand-in for one protocol argument.

    A floating-point array stands for its shape alone -- its values are what
    is rebound. Anything else stands for itself, an integer array included:
    what it holds decides the structure, so a change to it is a change of
    sequence.
    """
    if not torch.is_tensor(value):
        # A snapshot rather than the object: a list edited in place would
        # otherwise compare equal to itself, and the packing it keys would
        # be one built from different numbers.
        return tuple(value) if isinstance(value, list) else value
    fingerprint = (tuple(value.shape), value.dtype, value.device)
    if value.is_floating_point():
        return fingerprint
    return (*fingerprint, tuple(value.flatten().tolist()))


def _drawn(term: _Term, value: torch.Tensor) -> torch.Tensor:
    """The elements one term draws on, put where its scale is.

    The gather belongs where the values are and the product where the buffer
    is, and those are two places whenever a design holds its parameters on the
    host while the run is on a card.
    """
    source = value.reshape(-1)
    return source[term.index.to(source.device)].to(term.scale.device)


def _seeds(
    primals: tuple[torch.Tensor, ...], position: int
) -> tuple[tuple[torch.Tensor, ...], ...] | None:
    """A tangent of ones and one of ``1, 2, 3, ...`` along one argument."""
    chosen = primals[position]
    if chosen.numel() == 0:
        return None
    ramp = torch.arange(
        1, chosen.numel() + 1, dtype=chosen.dtype, device=chosen.device
    ).reshape(chosen.shape)
    return tuple(
        tuple(
            seed if index == position else torch.zeros_like(value)
            for index, value in enumerate(primals)
        )
        for seed in (torch.ones_like(chosen), ramp)
    )


def _term(scale: torch.Tensor, ramp: torch.Tensor, elements: int) -> _Term | None:
    """Read one buffer's source index off the two directional derivatives.

    ``None`` when an entry moves under the ramp by something that is not a
    whole multiple of what it moves under ones, which is what an entry drawing
    on two elements at once looks like.
    """
    live = scale != 0
    empty = _Term(
        torch.zeros_like(scale),
        torch.zeros(scale.shape, dtype=torch.long, device=scale.device),
    )
    if not bool(live.any()):
        return empty
    ratio = ramp / torch.where(live, scale, torch.ones_like(scale))
    whole = ratio.round()
    if bool((live & ((ratio - whole).abs() > _WHOLE)).any()):
        return None
    if bool((live & ((whole < 1) | (whole > elements))).any()):
        return None
    index = (whole.long() - 1).clamp(0, max(elements - 1, 0))
    return _Term(
        scale=torch.where(live, scale, torch.zeros_like(scale)).detach(),
        index=torch.where(live, index, torch.zeros_like(index)).detach(),
    )


def _agrees(
    buffers: Any,
    names: tuple[str, ...],
    primals: tuple[torch.Tensor, ...],
    packing: Packing,
) -> bool:
    """Whether the map reproduces a fresh packing at a point it never saw."""
    gain, shift = _ELSEWHERE
    moved = tuple(value * gain + shift for value in primals)
    fresh = buffers(moved)
    rebuilt = packing.pack(dict(zip(names, moved, strict=True)))
    return all(
        torch.allclose(
            getattr(rebuilt, buffer),
            getattr(fresh, buffer),
            rtol=1e-5,
            atol=1e-7,
        )
        if getattr(fresh, buffer).is_floating_point()
        else torch.equal(getattr(rebuilt, buffer), getattr(fresh, buffer))
        for buffer in _PackedEvents.__dataclass_fields__
    )
