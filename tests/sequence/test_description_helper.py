"""Assembling a description out of operators in one call.

The claim is only that :func:`~torchsim.description` is
:func:`~torchsim.compose` with the surrounding fields filled in, so what is
checked is that a description built the short way simulates to the same signal
as one built the long way -- and that the fields a caller does not name take
values that mean "nothing here" rather than being quietly dropped.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsim import (
    Delay,
    EpgEngine,
    Excitation,
    Readout,
    Refocusing,
    SequenceDescription,
    ShimDefinition,
    TissueProperties,
    compose,
    description,
    ideal_rf_definition,
)

ECHOES = 6


def _modules():
    parts = [Excitation(0.5 * math.pi, 0.5 * math.pi)]
    for _ in range(ECHOES):
        parts += [
            Delay(2.5e-3),
            Refocusing(math.pi, 0.5 * math.pi),
            Delay(2.5e-3),
            Readout(0.5 * math.pi),
        ]
    return parts


def _tissue():
    return TissueProperties(t1_ms=torch.tensor([830.0]), t2_ms=torch.tensor([80.0]))


def test_the_short_way_is_the_long_way() -> None:
    """One call against compose plus the constructor, through the signal."""
    parts = _modules()
    events, duration_s = compose(*parts)
    written_out = SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration_s,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )

    engine = EpgEngine()
    short = engine.simulate(description(*parts), _tissue(), nstates=8).signal
    long = engine.simulate(written_out, _tissue(), nstates=8).signal

    assert short.shape == (1, ECHOES)
    assert float((short - long).abs().max()) == 0.0


def test_what_is_not_named_means_nothing_is_there() -> None:
    """An empty shim table, no gradient moment, and one ideal pulse."""
    built = description(*_modules())

    assert built.shim_definitions == {}
    assert built.crusher_dephasing_rad == 0.0
    assert built.voxel_size_m is None
    assert set(built.rf_definitions) == {0}
    assert built.subsequence_index == 0


def test_the_repetition_lasts_as_long_as_the_operators_do() -> None:
    """The span compose reports, in the microseconds a description carries."""
    parts = _modules()
    _events, duration_s = compose(*parts)

    assert description(*parts).tr_duration_us == pytest.approx(1e6 * duration_s)


def test_what_is_named_is_carried() -> None:
    """A pulse, a shim and a gradient moment reach the description."""
    shim = ShimDefinition(3, (1.0, 1.0), (0.0, math.pi))
    built = description(
        *_modules(),
        rf_definitions={7: ideal_rf_definition(7)},
        shim_definitions={3: shim},
        subsequence_index=2,
        crusher_dephasing_rad=2.0 * math.pi,
        voxel_size_m=1e-3,
    )

    assert set(built.rf_definitions) == {7}
    assert built.shim_definitions == {3: shim}
    assert built.subsequence_index == 2
    assert built.crusher_dephasing_rad == pytest.approx(2.0 * math.pi)
    assert built.voxel_size_m == pytest.approx(1e-3)
