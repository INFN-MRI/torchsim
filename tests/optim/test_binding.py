"""What a resolved simulator promises, and what it refuses to promise.

A binding exists for speed, so every test here is really the same question in
a different place: does going fast change the answer. The two that matter most
are the ones asserting a *route* -- that the packer was not reached, and that
it was -- because agreement alone cannot tell a binding that worked from one
that silently was not built.
"""

from __future__ import annotations

import pytest
import torch

from torchsim.model import Simulator, SpinPhysics
from torchsim.model._binding import bind
from torchsim.sequence import _accelerators
from torchsim.sequence._accelerators import pack_description
from torchsim.simulators import (
    FSESimulator,
    MPnRAGESimulator,
    MPRAGESimulator,
    MRFSimulator,
)

TISSUE = {"T1": [800.0, 1400.0], "T2": [45.0, 120.0]}
#: What an inversion-prepared spoiled train exposes, which is no T2 at all.
LONGITUDINAL = {"T1": [800.0, 1400.0]}

#: One case per shipped state-machine protocol, each with the arguments a
#: design loop would move and the tissue it exposes. Four layouts, so a map
#: read off one of them is not what is being tested.
PROTOCOLS = [
    pytest.param(
        FSESimulator(ESP=5.0, TR=3000.0, states=10),
        {"flip": torch.full((4, 24), 120.0)},
        TISSUE,
        id="fse-batched",
    ),
    pytest.param(
        FSESimulator(ESP=5.0, TR=3000.0, states=10),
        {"flip": torch.full((24,), 120.0), "phases": torch.zeros(24)},
        TISSUE,
        id="fse-phases",
    ),
    pytest.param(
        MRFSimulator(TR=10.0, TI=20.0, states=10),
        {"flip": torch.linspace(5.0, 60.0, 32)},
        TISSUE,
        id="mrf",
    ),
    pytest.param(
        MPRAGESimulator(TI=1000.0, TRspgr=8.0, nshots=16, states=10),
        {"flip": torch.full((16,), 8.0)},
        LONGITUDINAL,
        id="mprage",
    ),
    pytest.param(
        MPnRAGESimulator(nshots=8, TR=10.0, TI=500.0, states=10),
        {"flip": torch.full((8,), 6.0)},
        LONGITUDINAL,
        id="mpnrage",
    ),
]


@pytest.fixture
def packings(monkeypatch):
    """Every packing this test performs, so a skipped one can be asserted."""
    seen: list[int] = []
    original = _accelerators._pack_for

    def watched(description, shapes, **settings):
        seen.append(len(description.events))
        return original(description, shapes, **settings)

    monkeypatch.setattr(_accelerators, "_pack_for", watched)
    return seen


@pytest.mark.parametrize("simulator,design,tissue", PROTOCOLS)
def test_the_bound_buffers_are_the_packed_ones(simulator, design, tissue) -> None:
    """The whole claim, at the values the structure was resolved at."""
    played = simulator.played(**design)
    packing = bind(
        simulator, played, repetitions=simulator.repetitions, record=simulator.record
    )
    assert packing is not None
    fresh = pack_description(
        packing.description,
        repetitions=simulator.repetitions,
        record=simulator.record,
        device=torch.device("cpu"),
    )
    bound = packing.pack(played)
    for buffer in ("duration", "flip", "phase", "time_us", "kind", "action"):
        assert torch.equal(getattr(bound, buffer), getattr(fresh, buffer)), buffer


@pytest.mark.parametrize("simulator,design,tissue", PROTOCOLS)
def test_the_bound_buffers_follow_values_the_map_never_saw(
    simulator, design, tissue
) -> None:
    """And away from them, where the map is doing the work rather than a copy.

    Held to float32 round-off rather than to the bit: the same product formed
    in a different order lands within an ulp, and the packer's order is not
    something a map read off its derivative can reproduce.
    """
    played = simulator.played(**design)
    packing = bind(
        simulator, played, repetitions=simulator.repetitions, record=simulator.record
    )
    assert packing is not None
    elsewhere = {
        **played,
        **{name: played[name] * 0.63 + 4.0 for name in packing.varying},
    }
    fresh = pack_description(
        simulator.describe(**elsewhere),
        repetitions=simulator.repetitions,
        record=simulator.record,
        device=torch.device("cpu"),
    )
    bound = packing.pack(elsewhere)
    for buffer in ("duration", "flip", "phase", "time_us"):
        torch.testing.assert_close(
            getattr(bound, buffer), getattr(fresh, buffer), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("simulator,design,tissue", PROTOCOLS)
def test_a_resolved_simulator_answers_what_the_plain_one_does(
    simulator, design, tissue
) -> None:
    """Forward and backward, at values neither one was built at."""
    moved = {name: value * 0.9 for name, value in design.items()}
    resolved = simulator.resolved()
    resolved.simulate(**design, **tissue)

    plain_design = {
        name: value.clone().requires_grad_(True) for name, value in moved.items()
    }
    bound_design = {
        name: value.clone().requires_grad_(True) for name, value in moved.items()
    }
    plain = simulator.simulate(**plain_design, **tissue)
    bound = resolved.simulate(**bound_design, **tissue)
    plain.abs().square().sum().backward()
    bound.abs().square().sum().backward()

    torch.testing.assert_close(bound, plain, rtol=1e-6, atol=1e-7)
    for name, value in plain_design.items():
        torch.testing.assert_close(
            bound_design[name].grad, value.grad, rtol=1e-5, atol=1e-6
        )


def test_a_resolved_simulator_stops_packing(packings) -> None:
    """The route, not the answer: a bound call must not reach the packer.

    Agreement cannot tell a working binding from one that was never built,
    and one that was never built is exactly what a regression here looks
    like.
    """
    resolved = FSESimulator(ESP=5.0, TR=3000.0, states=10).resolved()
    flip = torch.full((16,), 120.0)
    resolved.simulate(flip=flip, **TISSUE)
    packings.clear()

    resolved.simulate(flip=flip * 0.8, **TISSUE)

    assert packings == []


def test_a_simulator_that_refuses_to_resolve_keeps_packing(packings) -> None:
    """The companion: turning resolution off really turns it off."""
    simulator = FSESimulator(ESP=5.0, TR=3000.0, states=10, resolve=False)
    flip = torch.full((16,), 120.0)
    simulator.simulate(flip=flip, **TISSUE)
    packings.clear()

    simulator.simulate(flip=flip * 0.8, **TISSUE)

    assert packings != []


def test_a_protocol_value_that_is_not_an_array_rebuilds(packings) -> None:
    """A plain number reaches the timestamps, so changing one is a new sequence.

    Keyed by shape alone it would be reused, and the answer would be the old
    echo spacing's.
    """
    resolved = FSESimulator(ESP=5.0, TR=3000.0, states=10).resolved()
    flip = torch.full((16,), 120.0)
    resolved.simulate(flip=flip, **TISSUE)
    packings.clear()

    stretched = resolved.simulate(flip=flip, ESP=12.0, **TISSUE)

    assert packings != []
    reference = FSESimulator(ESP=12.0, TR=3000.0, states=10).simulate(
        flip=flip, **TISSUE
    )
    torch.testing.assert_close(stretched, reference, rtol=1e-6, atol=1e-7)


def test_a_train_of_a_different_length_rebuilds() -> None:
    """A shape is part of the key, so a new one is a new structure."""
    resolved = FSESimulator(ESP=5.0, TR=3000.0, states=10).resolved()
    plain = FSESimulator(ESP=5.0, TR=3000.0, states=10)
    resolved.simulate(flip=torch.full((16,), 120.0), **TISSUE)

    for echoes in (8, 24, 16):
        flip = torch.full((echoes,), 110.0)
        torch.testing.assert_close(
            resolved.simulate(flip=flip, **TISSUE),
            plain.simulate(flip=flip, **TISSUE),
            rtol=1e-6,
            atol=1e-7,
        )


# --- what cannot be followed ------------------------------------------------


class _Squared(FSESimulator):
    """A layout whose flip angles are not affine in what it is given."""

    def layout(self, *, flip, **protocol):
        return super().layout(flip=torch.as_tensor(flip) ** 2 / 180.0, **protocol)


class _Shared(FSESimulator):
    """A layout where one event's angle draws on two of the given values."""

    def layout(self, *, flip, **protocol):
        angles = torch.as_tensor(flip)
        return super().layout(flip=angles + angles.flip(-1), **protocol)


@pytest.mark.parametrize("protocol", [_Squared, _Shared], ids=["nonlinear", "shared"])
def test_a_layout_the_map_cannot_follow_is_refused(protocol, packings) -> None:
    """No binding, the ordinary path, and the right answer.

    Both refusals are checked by the route as well as the answer: a map that
    was quietly built and quietly wrong would agree at the point it was read.
    """
    resolved = protocol(ESP=5.0, TR=3000.0, states=10).resolved()
    plain = protocol(ESP=5.0, TR=3000.0, states=10)
    flip = torch.linspace(90.0, 150.0, 16)
    resolved.simulate(flip=flip, **TISSUE)
    assert resolved._packing is None
    packings.clear()

    answer = resolved.simulate(flip=flip * 0.8, **TISSUE)

    assert packings != []
    torch.testing.assert_close(
        answer, plain.simulate(flip=flip * 0.8, **TISSUE), rtol=1e-6, atol=1e-7
    )


def test_a_refusal_is_not_retried_every_call(packings) -> None:
    """Discovery costs three packings, so a layout it cannot follow pays once."""
    resolved = _Squared(ESP=5.0, TR=3000.0, states=10).resolved()
    flip = torch.full((16,), 120.0)
    resolved.simulate(flip=flip, **TISSUE)
    packings.clear()

    resolved.simulate(flip=flip * 0.8, **TISSUE)

    assert len(packings) == 1


def test_a_simulator_holding_a_description_is_left_alone(packings) -> None:
    """A stream handed over whole has no layout to resolve.

    There is nothing to rebind -- the events are already concrete -- so asking
    for a binding must change neither the answer nor the route.
    """

    class Bare(Simulator):
        model = SpinPhysics(properties={"T1": "t1_ms", "T2": "t2_ms"})

    builder = FSESimulator(ESP=5.0, TR=3000.0, states=10)
    description = builder.describe(**builder.played(flip=torch.full((8,), 120.0)))
    settings = {"model": builder.model, "states": 10}
    plain = Bare.from_description(description, **settings)
    handed = Bare.from_description(description, **settings).resolved()

    answer = handed.simulate(**TISSUE)

    assert handed._packing is None
    assert torch.equal(answer, plain.simulate(**TISSUE))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_design_on_the_host_runs_against_a_card() -> None:
    """The values and the buffers may live in two places.

    A design loop keeps its parameters wherever the optimizer put them, and
    the run goes wherever the tissue is. The map gathers from one and
    multiplies in the other, and the gradient comes back to the host.
    """
    tissue = {name: torch.tensor(value).cuda() for name, value in TISSUE.items()}
    simulator = FSESimulator(ESP=5.0, TR=3000.0, states=10, device="cuda")
    flip = torch.full((4, 24), 120.0, requires_grad=True)

    resolved = simulator.resolved()
    signal = resolved.simulate(flip=flip, **tissue)
    signal.abs().sum().backward()

    assert signal.device.type == "cuda"
    assert flip.grad is not None and flip.grad.device.type == "cpu"
    assert torch.isfinite(flip.grad).all()
    assert resolved._packing is not None
    plain = simulator.simulate(flip=flip.detach(), **tissue)
    assert torch.allclose(signal.detach(), plain, rtol=1e-5, atol=1e-7)


def test_a_train_masked_to_zero_is_refused(packings) -> None:
    """A flip angle of exactly zero is a corner, not a point.

    What reaches the kernel is a magnitude, so the packed flip is ``|x|`` and
    a design that touches zero is not affine there. Masking a train to end
    early has to floor at a negligible angle rather than at zero, and the two
    routes are what this asserts -- refusal falls back to the packer.
    """
    grid = torch.arange(1, 25, dtype=torch.float32)
    ends = torch.tensor([[8.0], [16.0], [24.0]])
    simulator = FSESimulator(ESP=5.0, TR=3000.0, states=10)
    acquired = (grid <= ends).to(torch.float32)

    zeroed = simulator.resolved()
    zeroed.simulate(flip=torch.full((3, 24), 120.0) * acquired, **TISSUE)
    assert zeroed._packing is None
    assert len(zeroed._refused) == 1

    floored = simulator.resolved()
    answer = floored.simulate(
        flip=torch.full((3, 24), 120.0) * acquired.clamp_min(1e-6), **TISSUE
    )
    assert floored._packing is not None
    packed = len(packings)

    floored.simulate(
        flip=torch.full((3, 24), 110.0) * acquired.clamp_min(1e-6), **TISSUE
    )
    assert len(packings) == packed, "a bound rerun must not reach the packer"
    assert torch.isfinite(answer).all()
