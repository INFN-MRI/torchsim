"""What a signal model declares, and what the engine is told because of it.

The layer exists to keep two promises: that a model reaches the fastest kernel
its physics allows without the author arranging it, and that derivatives are
taken in the mode the problem calls for. Both are checked here by looking at
what the engine is *told*, not only at the answer -- a run that declares
physics it does not have gives the right number slowly, which no comparison of
signals would show.
"""

from __future__ import annotations

import pytest
import torch

from torchsim.model import REFOCUSED, Simulator, SpinPhysics
from torchsim.sequence import _simulation


class Relaxation(Simulator):
    """T1 and T2 alone -- the narrowest a model can be."""

    model = SpinPhysics(properties={"T1": "t1_ms", "T2": "t2_ms"}, operators=REFOCUSED)

    def layout(self, *, flip, ESP, phases=0.0):
        """Return a refocused train at the flip angles given."""
        angles = torch.deg2rad(torch.atleast_1d(torch.as_tensor(flip)))
        turns = torch.deg2rad(torch.as_tensor(phases, dtype=angles.dtype))
        turns = turns.expand_as(angles) if turns.numel() == 1 else turns
        spacing_s = ESP * 1e-3
        parts = [(0.0, self.operators.excitation(torch.pi / 2, torch.pi / 2))]
        for index in range(angles.numel()):
            echo_s = (index + 1) * spacing_s
            parts.append(
                (
                    echo_s - 0.5 * spacing_s,
                    self.operators.refocusing(angles[index], turns[index]),
                )
            )
            parts.append((echo_s, self.operators.readout(turns[index])))
        return parts


class Transmit(Relaxation):
    """The same sequence, with a transmit map declared."""

    model = SpinPhysics(
        properties={"T1": "t1_ms", "T2": "t2_ms", "B1": "b1"}, operators=REFOCUSED
    )


SEQUENCE = {"flip": torch.full((8,), 120.0), "ESP": 5.0}
"""The protocol, given to the constructor rather than to the call."""
T1 = [800.0, 1000.0, 1200.0]
T2 = [50.0, 70.0, 90.0]


@pytest.fixture
def declared(monkeypatch):
    """Every feature set the simulations in a test declare, in order."""
    seen: list[frozenset[str]] = []
    original = _simulation.features_of

    def watched(tissue):
        got = original(tissue)
        seen.append(got)
        return got

    monkeypatch.setattr(_simulation, "features_of", watched)
    return seen


def test_a_property_the_model_does_not_declare_stays_out_of_the_kernel(
    declared,
) -> None:
    """The point of the layer.

    A default broadcast to the voxel count before the tissue is built reads as
    a live map, and the run pays for a term it does not have. Asserting the
    signal would not show it: the answer is right either way.
    """
    Relaxation(**SEQUENCE).simulate(T1=T1, T2=T2)
    assert sorted(declared[-1]) == ["T1", "T2"]


def test_a_declared_property_left_at_its_default_still_stays_out(
    declared,
) -> None:
    """Declaring a property is not using it; a scalar default is still absent."""
    Transmit(**SEQUENCE).simulate(T1=T1, T2=T2, B1=1.0)
    assert sorted(declared[-1]) == ["T1", "T2"]


def test_a_declared_property_given_a_map_reaches_the_kernel(declared) -> None:
    """And the gate is a choice rather than a coincidence."""
    Transmit(**SEQUENCE).simulate(T1=T1, T2=T2, B1=[0.9, 1.0, 1.1])
    assert sorted(declared[-1]) == ["B1", "T1", "T2"]


def test_an_undeclared_property_cannot_change_the_answer() -> None:
    """A model that does not expose a property is not quietly given one."""
    narrow = Relaxation(**SEQUENCE).simulate(T1=T1, T2=T2)
    wide = Transmit(**SEQUENCE).simulate(T1=T1, T2=T2, B1=1.0)
    assert torch.allclose(narrow, wide, atol=1e-6)


def test_a_model_declaring_unknown_tissue_says_so() -> None:
    """A typo in a tissue field is caught where it is written, not in a kernel."""

    class Wrong(Relaxation):
        model = SpinPhysics(properties={"T1": "t1_ms", "T2": "not_a_tissue_field"})

    with pytest.raises(ValueError, match="unknown tissue"):
        Wrong(**SEQUENCE).simulate(T1=T1, T2=T2)


def test_differentiating_something_the_model_does_not_expose_says_so() -> None:
    """``diff`` names a property, so a name it does not expose is an error."""
    with pytest.raises(ValueError, match="is not a property"):
        Relaxation(**SEQUENCE).jacobian("B1", T1=T1, T2=T2)


@pytest.mark.parametrize(
    "t1,t2,diff,signal_shape,jacobian_shape",
    [
        (1000.0, 80.0, "T2", (8,), (8,)),
        (T1, T2, "T2", (3, 8), (3, 8)),
        (1000.0, 80.0, ("T1", "T2"), (8,), (2, 8)),
        (T1, T2, ("T1", "T2"), (3, 8), (3, 2, 8)),
    ],
)
def test_the_jacobian_shape_contract(
    t1, t2, diff, signal_shape, jacobian_shape
) -> None:
    """A bare name collapses the parameter axis; a sequence keeps it."""
    signal, jacobian = Relaxation(**SEQUENCE).jacobian(diff, T1=t1, T2=t2)
    assert signal.shape == signal_shape
    assert jacobian.shape == jacobian_shape


def test_the_jacobian_matches_finite_differences() -> None:
    """Forward mode is exact where differencing is only close."""
    model = Relaxation(**SEQUENCE)
    step = 1e-2
    up = model.simulate(T1=T1, T2=[value + step for value in T2])
    down = model.simulate(T1=T1, T2=[value - step for value in T2])
    _, analytic = model.jacobian("T2", T1=T1, T2=T2)
    assert torch.allclose(analytic, (up - down) / (2 * step), atol=1e-4)


def test_the_primal_beside_the_jacobian_is_the_ordinary_signal() -> None:
    """One pass gives both, so they cannot be allowed to disagree."""
    model = Relaxation(**SEQUENCE)
    signal, _ = model.jacobian("T2", T1=T1, T2=T2)
    assert torch.allclose(signal, model.simulate(T1=T1, T2=T2), atol=0.0)


def test_a_cost_differentiates_back_to_the_sequence() -> None:
    """The other mode: reverse, for the parameters an acquisition optimizes.

    Nothing here wraps it -- the engine reads which inputs carry a gradient and
    chooses its kernel from that, so the model layer only has to stay out of
    the way.
    """
    flip = torch.full((8,), 120.0, requires_grad=True)
    signal = Relaxation(flip=flip, ESP=5.0).simulate(T1=T1, T2=T2)
    signal.abs().square().sum().backward()
    assert flip.grad is not None
    assert float(flip.grad.abs().max()) > 0.0
