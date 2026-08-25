"""The split between what an event does and what order events play in.

A trigger is a construction-time binding from a kind of event to the operator
that realizes it. What a protocol then produces is an ordinary description
whose events carry their own action word, so the fused kernels answer for it
exactly as they answer for a builder's -- which is what these tests hold it
to, at the bit where they can.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchsim.model import (
    BALANCED,
    REFOCUSED,
    SPOILED,
    UNBALANCED,
    AbstractSimulator,
    StateMachineModel,
    Triggers,
)
from torchsim.sequence import EpgEngine, EventAction, EventType, ideal_rf_definition
from torchsim.simulators import MRFSimulator

FLIP = torch.linspace(5.0, 60.0, 12)
T1 = torch.tensor([600.0, 1000.0, 1400.0])
T2 = torch.tensor([40.0, 80.0, 120.0])
TISSUE = {"T1": T1, "T2": T2}


class Train(AbstractSimulator):
    """One Excitation and one sample per repetition, however it is realized."""

    model = StateMachineModel(properties={"T1": "t1_ms", "T2": "t2_ms"})
    states = 8

    def layout(self, *, flip, TR):
        """Return one excitation and one sample per repetition."""
        angles = torch.deg2rad(torch.as_tensor(flip))
        parts = []
        for index in range(angles.numel()):
            parts.append(self.triggers.excitation(angles[index]))
            parts.append(self.triggers.readout(duration_s=TR * 1e-3))
        return parts


def _with(triggers):
    """Return the same protocol, realized by a different trigger table."""
    return Train(
        model=replace(Train.model, triggers=triggers),
        flip=FLIP,
        TR=10.0,
        crusher_dephasing_rad=torch.pi,
        voxel_size_m=1e-3,
    )


def test_a_trigger_is_a_choice() -> None:
    """Three readouts that differ must give three different answers.

    This is what the retired policy classes could not do: they named a
    sequence without deciding anything about it.
    """
    answers = [
        _with(table).simulate(**TISSUE)
        for table in (BALANCED, UNBALANCED, SPOILED)
    ]
    scale = max(float(answer.abs().max()) for answer in answers)
    assert scale > 0.0
    for first in range(len(answers)):
        for second in range(first + 1, len(answers)):
            assert float((answers[first] - answers[second]).abs().max()) > 1e-3 * scale


def test_swapping_a_trigger_back_restores_the_answer_to_the_bit() -> None:
    """The binding is the whole of the difference; nothing else is carried."""
    first = _with(UNBALANCED).simulate(**TISSUE)
    _ = _with(SPOILED).simulate(**TISSUE)
    again = _with(UNBALANCED).simulate(**TISSUE)
    assert torch.equal(first, again)


def test_the_trigger_leaves_the_description_ordinary() -> None:
    """After construction a trigger has no representation.

    What reaches the kernels is events carrying action words, which is what
    keeps the whole fused path -- packing, the feature mask, the subspace
    verdict, offload -- available to a protocol nobody shipped.
    """
    described = _with(UNBALANCED).describe(flip=FLIP, TR=10.0)
    kinds = {event.type for event in described.events}
    assert kinds <= {EventType.RF, EventType.ADC, EventType.WAIT}
    wound = [
        event for event in described.events
        if event.action & EventAction.CRUSH_AFTER
    ]
    assert len(wound) == FLIP.numel()


def test_a_shipped_simulator_reaches_the_fused_kernels(monkeypatch) -> None:
    """A fast path has to be shown taken, not inferred from agreement."""
    from torchsim.sequence import _accelerators

    seen: list[int] = []
    original = _accelerators._run_packed

    def watched(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_accelerators, "_run_packed", watched)
    MRFSimulator(flip=FLIP, TR=10.0).simulate(**TISSUE)
    assert seen, "the protocol did not reach the packed launcher"


def test_a_protocol_with_no_layout_says_so() -> None:
    """The message names both ways of saying what a sequence plays."""

    class Empty(AbstractSimulator):
        model = StateMachineModel(properties={"T1": "t1_ms", "T2": "t2_ms"})

    with pytest.raises(NotImplementedError, match="layout|describe"):
        Empty().simulate(**TISSUE)


def test_a_description_handed_over_whole_skips_the_layout() -> None:
    """The path a stream from a scanner takes: the events are already concrete."""
    protocol = _with(UNBALANCED)
    described = protocol.describe(flip=FLIP, TR=10.0)

    handed = AbstractSimulator.from_description(described, protocol.model, states=8)
    assert torch.equal(handed.simulate(**TISSUE), protocol.simulate(**TISSUE))

    # And it is the stream that was given, not one rebuilt from a layout.
    assert handed.describe() is described


def test_a_handed_over_description_agrees_with_the_engine_directly() -> None:
    """Nothing is added between a stream and the kernels."""
    protocol = _with(SPOILED)
    described = protocol.describe(flip=FLIP, TR=10.0)
    handed = AbstractSimulator.from_description(described, protocol.model, states=8)

    from torchsim.sequence import TissueProperties

    direct = EpgEngine().simulate(
        described, TissueProperties(t1_ms=T1, t2_ms=T2), nstates=8
    ).signal
    assert torch.equal(handed.simulate(**TISSUE), direct)


def test_the_protocol_may_be_overridden_per_call() -> None:
    """A sequence optimizer changes the schedule without rebuilding the object."""
    protocol = MRFSimulator(flip=FLIP, TR=10.0)
    first = protocol.simulate(**TISSUE)
    other = protocol.simulate(flip=0.5 * FLIP, **TISSUE)
    assert not torch.equal(first, other)
    assert torch.equal(protocol.simulate(**TISSUE), first)


def test_a_trigger_table_defaults_to_the_bare_operators() -> None:
    """An unassigned slot is the plain event, not a surprise."""
    from torchsim.sequence import Excitation, Readout

    table = Triggers()
    assert table.excitation is Excitation
    assert table.readout is Readout


def test_every_shipped_table_says_something() -> None:
    """A preset that equalled the default would be a name and nothing else.

    That is what the retired policy classes were, so each table here has to
    differ from the bare one and from its siblings.
    """
    tables = {"balanced": BALANCED, "unbalanced": UNBALANCED,
              "spoiled": SPOILED, "refocused": REFOCUSED}
    for name, table in tables.items():
        assert table != Triggers(), f"{name} decides nothing"
    assert len({table.readout for table in tables.values()}) == len(tables)


def test_the_definitions_come_from_the_model() -> None:
    """A pulse is physics, so the RF resources travel with the model."""
    described = _with(UNBALANCED).describe(flip=FLIP, TR=10.0)
    assert set(described.rf_definitions) == {0}
    assert described.rf_definitions[0] == ideal_rf_definition()
