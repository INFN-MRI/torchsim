"""Naming a dephasing gradient must not change what it does.

The kernels read one action word per event: a winding before it, a winding
after it, or ideal spoiling. Writing that winding as an operator of its own,
so a Readout followed by a crusher is a composite rather than a keyword, is a
change of spelling and nothing else -- which is checkable to the bit, and is
checked that way here. ``allclose`` would pass for two different sequences
that happen to agree; only the same events give the same bits.
"""

from __future__ import annotations

import pytest
import torch

from torchsim.sequence import (
    AdcRole,
    Delay,
    Dephase,
    EpgEngine,
    EventAction,
    Excitation,
    Readout,
    Refocusing,
    SequenceDescription,
    SPGRReadout,
    Spoil,
    SSFPFidReadout,
    TissueProperties,
    bSSFPReadout,
    compose,
    ideal_rf_definition,
    operator,
)

TISSUE = TissueProperties(
    t1_ms=torch.tensor([600.0, 1000.0]), t2_ms=torch.tensor([40.0, 80.0])
)
FLIP = torch.deg2rad(torch.tensor(40.0))
REFOCUS = torch.deg2rad(torch.tensor(150.0))
TR = 5e-3
SHOTS = 6


def _described(modules):
    """A description over the modules given, with the ideal hard pulse."""
    events, duration_s = compose(*modules)
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration_s,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
        # Without a gradient to wind across, a dephasing order turns through
        # nothing and every spelling would agree by carrying nothing.
        crusher_dephasing_rad=torch.pi,
        voxel_size_m=1e-3,
    )


def _run(modules, nstates=10):
    return EpgEngine().simulate(_described(modules), TISSUE, nstates=nstates).signal


@pytest.mark.parametrize(
    ("action", "composite"),
    [
        (EventAction.SHIFT_AFTER, SSFPFidReadout),
        (EventAction.SPOIL_AFTER, SPGRReadout),
        (EventAction.NONE, bSSFPReadout),
    ],
    ids=["ssfp-fid", "spgr", "bssfp"],
)
def test_a_composite_readout_is_its_action_word(action, composite) -> None:
    """The composite and the keyword are the same events in the same order."""
    keyword = [
        part
        for _ in range(SHOTS)
        for part in (Excitation(FLIP), Readout(action=action, duration_s=TR))
    ]
    composed = [
        part
        for _ in range(SHOTS)
        for part in (Excitation(FLIP), composite(duration_s=TR))
    ]
    assert torch.equal(_run(keyword), _run(composed))


def test_a_crusher_pair_is_the_refocusing_keyword() -> None:
    """``crushed=True`` is the pulse between two windings, written short."""
    keyword = [Excitation(FLIP, torch.pi / 2)]
    composed = [Excitation(FLIP, torch.pi / 2)]
    for _ in range(SHOTS):
        keyword += [Delay(TR / 2), Refocusing(REFOCUS), Delay(TR / 2), Readout()]
        composed += [
            Delay(TR / 2),
            Dephase(),
            Refocusing(REFOCUS, crushed=False),
            Dephase(),
            Delay(TR / 2),
            Readout(),
        ]
    assert torch.equal(_run(keyword), _run(composed))


def test_the_spelling_is_a_choice_and_not_a_coincidence() -> None:
    """Three readouts that differ must give three different answers.

    Otherwise the agreements above would be three ways of carrying nothing.
    """
    answers = [
        _run(
            [
                part
                for _ in range(SHOTS)
                for part in (Excitation(FLIP), composite(duration_s=TR))
            ]
        )
        for composite in (bSSFPReadout, SSFPFidReadout, SPGRReadout)
    ]
    scale = max(float(answer.abs().max()) for answer in answers)
    for first in range(len(answers)):
        for second in range(first + 1, len(answers)):
            apart = float((answers[first] - answers[second]).abs().max())
            assert apart > 1e-3 * scale


def test_spoiling_and_winding_are_different_things() -> None:
    """One discards the transverse orders, the other moves them along."""
    wound = Dephase().emit(0.0)[0]
    spoiled = Spoil().emit(0.0)[0]
    assert wound.action is EventAction.CRUSH_AFTER
    assert spoiled.action is EventAction.SPOIL_AFTER


@pytest.mark.parametrize(
    "name",
    ["dephase", "spoil", "bssfp-readout", "ssfp-fid-readout", "spgr-readout"],
)
def test_every_new_operator_is_reachable_by_name(name) -> None:
    """A stream that arrives labelled dispatches through the registry."""
    assert operator(name) is not None


def test_a_composite_holds_the_repetition_it_is_given() -> None:
    """The sample is at the start of the module and the rest is the wait."""
    played = SSFPFidReadout(duration_s=TR)
    assert played.duration_s == TR
    events = played.emit(0.0)
    assert [event.type.name for event in events] == ["ADC", "WAIT", "WAIT"]
    assert events[0].params[0] is AdcRole.SINGLE
