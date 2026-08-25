"""Naming a dephasing gradient must not change what it does.

The kernels read one action word per event: a winding before it, a winding
after it, or ideal spoiling. Writing that winding as an operator of its own,
so a readout followed by a crusher is a composite rather than a keyword, is a
change of spelling and nothing else -- which is checkable to the bit, and is
checked that way here. ``allclose`` would pass for two different sequences
that happen to agree; only the same events give the same bits.
"""

from __future__ import annotations

import pytest
import torch

from torchsim.sequence import (
    FSE,
    SPGR,
    SSFPFID,
    AdcRole,
    EventAction,
    SequenceDescription,
    TissueProperties,
    bssfp_readout,
    compose,
    delay,
    dephase,
    excitation,
    ideal_rf_definition,
    operator,
    readout,
    refocusing,
    spgr_readout,
    spoil,
    ssfp_fid_readout,
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


def _run(policy, modules, nstates=10):
    return policy().simulate(_described(modules), TISSUE, nstates=nstates).signal


@pytest.mark.parametrize(
    ("policy", "action", "composite"),
    [
        (SSFPFID, EventAction.SHIFT_AFTER, ssfp_fid_readout),
        (SPGR, EventAction.SPOIL_AFTER, spgr_readout),
        (SSFPFID, EventAction.NONE, bssfp_readout),
    ],
    ids=["ssfp-fid", "spgr", "bssfp"],
)
def test_a_composite_readout_is_its_action_word(policy, action, composite) -> None:
    """The composite and the keyword are the same events in the same order."""
    keyword = [
        part
        for _ in range(SHOTS)
        for part in (excitation(FLIP), readout(action=action, duration_s=TR))
    ]
    composed = [
        part
        for _ in range(SHOTS)
        for part in (excitation(FLIP), composite(duration_s=TR))
    ]
    assert torch.equal(_run(policy, keyword), _run(policy, composed))


def test_a_crusher_pair_is_the_refocusing_keyword() -> None:
    """``crushed=True`` is the pulse between two windings, written short."""
    keyword = [excitation(FLIP, torch.pi / 2)]
    composed = [excitation(FLIP, torch.pi / 2)]
    for _ in range(SHOTS):
        keyword += [delay(TR / 2), refocusing(REFOCUS), delay(TR / 2), readout()]
        composed += [
            delay(TR / 2),
            dephase(),
            refocusing(REFOCUS, crushed=False),
            dephase(),
            delay(TR / 2),
            readout(),
        ]
    assert torch.equal(_run(FSE, keyword), _run(FSE, composed))


def test_the_spelling_is_a_choice_and_not_a_coincidence() -> None:
    """Three readouts that differ must give three different answers.

    Otherwise the agreements above would be three ways of carrying nothing.
    """
    answers = [
        _run(
            SSFPFID,
            [
                part
                for _ in range(SHOTS)
                for part in (excitation(FLIP), composite(duration_s=TR))
            ],
        )
        for composite in (bssfp_readout, ssfp_fid_readout, spgr_readout)
    ]
    scale = max(float(answer.abs().max()) for answer in answers)
    for first in range(len(answers)):
        for second in range(first + 1, len(answers)):
            apart = float((answers[first] - answers[second]).abs().max())
            assert apart > 1e-3 * scale


def test_spoiling_and_winding_are_different_things() -> None:
    """One discards the transverse orders, the other moves them along."""
    wound = dephase().emit(0.0)[0]
    spoiled = spoil().emit(0.0)[0]
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
    played = ssfp_fid_readout(duration_s=TR)
    assert played.duration_s == TR
    events = played.emit(0.0)
    assert [event.type.name for event in events] == ["ADC", "WAIT", "WAIT"]
    assert events[0].params[0] is AdcRole.SINGLE
