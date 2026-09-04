"""The split between what an event does and what order events play in.

A trigger is a construction-time binding from a kind of event to the operator
that realizes it. What a protocol then produces is an ordinary description
whose events carry their own action word, so the fused kernels answer for it
exactly as they answer for a builder's -- which is what these tests hold it
to, at the bit where they can.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from torchsim.model import (
    BALANCED,
    REFOCUSED,
    SPOILED,
    UNBALANCED,
    EventOperators,
    Simulator,
    SpinPhysics,
)
from torchsim.sequence import EpgEngine, EventAction, EventType, ideal_rf_definition
from torchsim.simulators import MRFSimulator

FLIP = torch.linspace(5.0, 60.0, 12)
T1 = torch.tensor([600.0, 1000.0, 1400.0])
T2 = torch.tensor([40.0, 80.0, 120.0])
TISSUE = {"T1": T1, "T2": T2}


class Train(Simulator):
    """One Excitation and one sample per repetition, however it is realized."""

    model = SpinPhysics(properties={"T1": "t1_ms", "T2": "t2_ms"})
    states = 8

    def layout(self, *, flip, TR):
        """Return one excitation and one sample per repetition."""
        angles = torch.deg2rad(torch.as_tensor(flip))
        parts = []
        for index in range(angles.numel()):
            parts.append(self.operators.excitation(angles[index]))
            parts.append(self.operators.readout(duration_s=TR * 1e-3))
        return parts


def _with(operators):
    """Return the same protocol, realized by a different trigger table."""
    return Train(
        model=replace(Train.model, operators=operators),
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
        _with(table).simulate(**TISSUE) for table in (BALANCED, UNBALANCED, SPOILED)
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
        event for event in described.events if event.action & EventAction.CRUSH_AFTER
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

    class Empty(Simulator):
        model = SpinPhysics(properties={"T1": "t1_ms", "T2": "t2_ms"})

    with pytest.raises(NotImplementedError, match="layout|describe"):
        Empty().simulate(**TISSUE)


def test_a_description_handed_over_whole_skips_the_layout() -> None:
    """The path a stream from a scanner takes.

    No layout is walked and no sequence parameter is named again: what arrives
    is played back through the same handlers that would have laid it down, so
    the answer is the one the protocol itself gives.
    """
    protocol = _with(UNBALANCED)
    described = protocol.describe(flip=FLIP, TR=10.0)

    handed = Simulator.from_description(described, protocol.model, states=8)
    assert torch.equal(handed.simulate(**TISSUE), protocol.simulate(**TISSUE))

    # Re-emitted through the model's own operators, and the same stream comes
    # back: every event of the same kind, at the same instant, dephasing alike.
    again = handed.describe()
    assert len(again.events) == len(described.events)
    for was, now in zip(described.events, again.events, strict=True):
        assert was.type is now.type
        assert was.action is now.action
        assert float(was.timestamp_us) == pytest.approx(float(now.timestamp_us))


def test_which_simulator_reads_a_stream_is_what_decides_the_dephasing() -> None:
    """Because the transport carries none of it.

    A description says a pulse was played and a window was opened; the
    gradients between them are not on the wire. They belong to the sequence
    family, and the family is the simulator the stream is handed to -- so the
    same events read as a refocused train and as an unbalanced one have to
    give different answers, or the choice was doing nothing.
    """
    refocused = _with(REFOCUSED)
    described = refocused.describe(flip=FLIP, TR=10.0)

    as_refocused = Simulator.from_description(described, refocused.model, states=8)
    as_unbalanced = Simulator.from_description(
        described, _with(UNBALANCED).model, states=8
    )

    one = as_refocused.simulate(**TISSUE)
    other = as_unbalanced.simulate(**TISSUE)
    assert float((one - other).abs().max() / one.abs().max()) > 0.1


def test_a_handed_over_description_agrees_with_the_engine_directly() -> None:
    """Nothing is added between a stream and the kernels."""
    protocol = _with(SPOILED)
    described = protocol.describe(flip=FLIP, TR=10.0)
    handed = Simulator.from_description(described, protocol.model, states=8)

    from torchsim.sequence import TissueProperties

    direct = (
        EpgEngine()
        .simulate(described, TissueProperties(t1_ms=T1, t2_ms=T2), nstates=8)
        .signal
    )
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

    table = EventOperators()
    assert table.excitation is Excitation
    assert table.readout is Readout


def test_every_shipped_table_says_something() -> None:
    """A preset that equalled the default would be a name and nothing else.

    That is what the retired policy classes were, so each table here has to
    differ from the bare one and from its siblings.
    """
    tables = {
        "balanced": BALANCED,
        "unbalanced": UNBALANCED,
        "spoiled": SPOILED,
        "refocused": REFOCUSED,
    }
    for name, table in tables.items():
        assert table != EventOperators(), f"{name} decides nothing"
    assert len({table.readout for table in tables.values()}) == len(tables)


def test_the_definitions_come_from_the_model() -> None:
    """A pulse is physics, so the RF resources travel with the model."""
    described = _with(UNBALANCED).describe(flip=FLIP, TR=10.0)
    assert set(described.rf_definitions) == {0}
    assert described.rf_definitions[0] == ideal_rf_definition()


def test_a_gradient_standing_on_its_own_does_not_survive_the_round_trip() -> None:
    """The limit of reading a stream through handlers.

    A handler reinstates what a pulse or a sample implies -- crushers around a
    refocusing pulse, a winding after an unbalanced sample -- which is exactly
    what the transport drops. A preparation's own spoiler is neither a pulse
    nor a sample, so nothing reinstates it, and a stream arriving from a
    scanner could not have carried it either.
    """
    from torchsim import Delay, Excitation, Spoil
    from torchsim.model._state_machine import realised
    from torchsim.sequence import EventAction

    class Prepared(Simulator):
        model = _with(SPOILED).model
        states = 1

        def layout(self, *, TS, flip):
            parts = []
            for wait in torch.atleast_1d(torch.as_tensor(TS)) * 1e-3:
                parts += [
                    Excitation(0.5 * torch.pi) @ Spoil(),
                    Delay(wait),
                    self.operators.excitation(torch.deg2rad(torch.as_tensor(flip))),
                    self.operators.readout(0.0),
                ]
            return parts

    protocol = Prepared(TS=torch.tensor([100.0, 400.0]), flip=10.0, **TISSUE)
    described = protocol.describe(TS=torch.tensor([100.0, 400.0]), flip=10.0)

    def spoilers(stream):
        return [e for e in stream.events if e.action is EventAction.SPOIL_AFTER]

    # Two per block: the preparation's own, and the one the spoiled readout
    # brings with it. Only the readout's is reinstated.
    assert len(spoilers(described)) == 4
    assert len(spoilers(realised(described, protocol.model))) == 2


def test_shim_definition_reaches_the_pulses():
    """Two channels driven in anti-phase put nothing on a voxel that sums them.

    The shim is a run setting rather than a property, so the check is that it
    reaches the description a layout builds and, from there, the kernels: an
    array whose sensitivities are alike cancels exactly when the drive is a
    half turn apart, and does not when it is not.
    """
    from torchsim import ShimDefinition
    from torchsim.simulators import FSESimulator

    array = dict(
        T1=torch.tensor([1000.0]),
        T2=torch.tensor([100.0]),
        B1=torch.ones(2, 1),
        B1phase=torch.zeros(2, 1),
    )

    def played(phase_rad):
        train = FSESimulator(
            ESP=5.0,
            flip=torch.full((4,), 150.0),
            states=12,
            shims={0: ShimDefinition(0, (0.5, 0.5), (0.0, phase_rad))},
        )
        return train.simulate(**array).abs()

    assert torch.allclose(played(math.pi), torch.zeros(1, 4), atol=1e-6)
    assert played(0.0).max() > 0.5
