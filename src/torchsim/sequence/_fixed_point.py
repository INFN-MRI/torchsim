"""The settled state of a train that never winds, solved rather than reached.

A description played over and over is an affine recursion on its states,
``x(n) = T x(n-1) + b``, whose settled state is ``(I - T)^-1 b``. Forming ``T``
means running the sequence from a set of known states and reading what it
leaves, which for a train carrying many configuration orders is a dense matrix
per voxel and not worth having.

A train that winds nothing carries one order, and one order is three numbers: a
transverse magnetization, which is one complex number since ``F-`` is the
conjugate of ``F+`` there, and a longitudinal one, which is real. Both ends of
the map are then reachable with nothing but the sequence itself. A state is
prepared by turning equilibrium through a pulse, and read by an ADC for the
transverse part followed by a pulse that tips the longitudinal part into it.
Four prepared states -- equilibrium and three turns of it -- give twelve
equations for the nine entries of ``T`` and the three of ``b``, and the same
four give the map from the entering state to what the ADCs record.

Every quantity here is a recorded signal, so the settled answer differentiates
like any other.
"""

from __future__ import annotations

__all__: list[str] = []

import math
from dataclasses import replace
from typing import Any

import torch

from ._description import EventAction, SequenceDescription
from ._operators import compose, operator

# Equilibrium, and three turns of it. Their augmented rows have to span four
# dimensions for the map to be determined, which these do: two quarter turns
# about axes a quarter turn apart, and one half turn.
_STARTS = (
    (0.0, 0.0),
    (0.5 * math.pi, 0.0),
    (0.5 * math.pi, 0.5 * math.pi),
    (math.pi, 0.0),
)

_WINDING = int(EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER)
_WINDING |= int(EventAction.SHIFT_AFTER)


def carries_one_order(description: SequenceDescription) -> bool:
    """Whether the description ever winds a state off order zero.

    A spoiler is not winding: it empties the transverse states rather than
    moving them along, and what it leaves is still order zero.
    """
    return not any(int(event.action) & _WINDING for event in description.events)


def settled_state(
    simulate: Any,
    description: SequenceDescription,
    **run: Any,
) -> torch.Tensor:
    """What the ADCs record once the description has been played to its limit.

    ``simulate`` runs one description and returns its samples; everything else
    is linear algebra on three numbers per voxel.
    """
    probe = max(description.rf_definitions, default=-1) + 1
    definitions = dict(description.rf_definitions)
    definitions[probe] = _ideal()

    entering, leaving, recorded = [], [], []
    for flip_rad, phase_rad in _STARTS:
        before, _ = _read(
            simulate, description, definitions, probe, flip_rad, phase_rad, 0, **run
        )
        after, samples = _read(
            simulate, description, definitions, probe, flip_rad, phase_rad, 1, **run
        )
        entering.append(before)
        leaving.append(after)
        recorded.append(samples)

    # [x 1] @ map = [leaving | recorded], one row per prepared state.
    ones = torch.ones_like(entering[0][..., :1])
    rows = torch.stack([torch.cat([state, ones], dim=-1) for state in entering], dim=-2)
    answers = torch.stack(
        [
            torch.cat([state.to(sample.dtype), sample], dim=-1)
            for state, sample in zip(leaving, recorded, strict=True)
        ],
        dim=-2,
    )
    resolved = torch.linalg.solve(rows.to(answers.dtype), answers)

    turn, offset = resolved[..., :3, :3], resolved[..., 3:4, :3]
    read, level = resolved[..., :3, 3:], resolved[..., 3:4, 3:]
    # x = x @ turn + offset, solved as (I - turn)^T x^T = offset^T.
    identity = torch.eye(3, dtype=resolved.dtype, device=resolved.device)
    fixed = torch.linalg.solve(
        (identity - turn).transpose(-1, -2), offset.transpose(-1, -2)
    ).transpose(-1, -2)
    return (fixed @ read + level).squeeze(-2)


def _ideal() -> Any:
    from ._description import ideal_rf_definition

    return ideal_rf_definition()


def _read(
    simulate: Any,
    description: SequenceDescription,
    definitions: dict[int, Any],
    probe: int,
    flip_rad: float,
    phase_rad: float,
    playings: int,
    **run: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The state after ``playings`` of the description, and what it recorded.

    The state comes back in the coordinates the probe reads it in -- the two
    parts of what an ADC sees, and the part of what it sees after the tip that
    the longitudinal magnetization reaches. A pulse about the axis the states
    already lie on sends ``Z`` into the same component the transverse part
    reports, so the third coordinate is that one and not the other, which is
    structurally empty.

    They are a fixed linear function of the magnetization for a given voxel,
    and an invertible one, which is all the map has to be affine in.
    """
    samples = simulate(
        _probed(description, definitions, probe, flip_rad, phase_rad, playings), **run
    )
    transverse, tipped = samples[..., -2], samples[..., -1]
    state = torch.stack([transverse.real, transverse.imag, tipped.imag], dim=-1)
    return state, samples[..., :-2]


def _probed(
    description: SequenceDescription,
    definitions: dict[int, Any],
    probe: int,
    flip_rad: float,
    phase_rad: float,
    playings: int,
) -> SequenceDescription:
    """The description behind a prepared state and in front of a probe."""
    excitation, readout = operator("excitation"), operator("readout")
    events: list[Any] = []
    at: Any = 0.0
    if flip_rad:
        prepared, _ = compose(excitation(flip_rad, phase_rad, definition_id=probe))
        events.extend(prepared)
    for _ in range(playings):
        events.extend(
            replace(event, timestamp_us=event.timestamp_us + at)
            for event in description.events
        )
        at = at + description.tr_duration_us
    probed, _ = compose(
        readout(0.0),
        excitation(0.5 * math.pi, 0.0, definition_id=probe),
        readout(0.0),
    )
    events.extend(
        replace(event, timestamp_us=event.timestamp_us + at) for event in probed
    )
    return SequenceDescription(
        subsequence_index=description.subsequence_index,
        tr_duration_us=at,
        events=tuple(events),
        rf_definitions=definitions,
        shim_definitions=description.shim_definitions,
        crusher_dephasing_rad=description.crusher_dephasing_rad,
        voxel_size_m=description.voxel_size_m,
    )
