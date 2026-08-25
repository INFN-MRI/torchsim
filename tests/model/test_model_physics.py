"""What a generic model can reach, and what the engine is told when it does.

A model is where the physics is chosen, so the two things worth pinning here
are that choosing more of it works -- a second pool declared by name and
reached through the same call -- and that choosing less of it is *told* to the
engine, since a kernel that computes gradients nobody asked for gives the
right answer at twice the price.
"""

from __future__ import annotations

import pytest
import torch

from torchsim.model import EpgModel
from torchsim.sequence import (
    EpgEngine,
    mrf_description,
    TissueProperties,
)
from torchsim.sequence import _accelerators
from torchsim.sequence._parameters import FLOAT_NAMES, TISSUE_NAMES

FLIP = torch.linspace(5.0, 60.0, 24)
T1 = torch.tensor([600.0, 1000.0, 1400.0])
T2 = torch.tensor([40.0, 80.0, 120.0])
BOUND = torch.tensor([0.05, 0.10, 0.15])


class Anything(EpgModel):
    """A model over the whole tissue: whatever is named is what is carried."""

    properties = {
        "T1": "t1_ms",
        "T2": "t2_ms",
        "bound_fraction": "bound_fraction",
        "bound_exchange": "bound_exchange_hz",
        "T1_bound": "t1_bound_ms",
    }
    simulator = EpgEngine()
    states = 8

    def describe(self, *, flip, TR):
        """Return an unbalanced SSFP train at the flip angles given."""
        return mrf_description(torch.deg2rad(torch.as_tensor(flip)), TR * 1e-3)


SEQUENCE = {"flip": FLIP, "TR": 10.0}
POOLED = {"bound_fraction": BOUND, "bound_exchange": 30.0, "T1_bound": 1000.0}


def _by_hand(**pool) -> torch.Tensor:
    """The same run, with the tissue and the description written out."""
    described = mrf_description(torch.deg2rad(FLIP), 10e-3)
    tissue = TissueProperties(t1_ms=T1, t2_ms=T2, **pool)
    return EpgEngine().simulate(described, tissue, nstates=8).signal


def test_a_model_reaches_the_semisolid_pool() -> None:
    """The generic model is the whole tissue, not a chosen corner of it."""
    through_the_model = Anything().simulate(T1=T1, T2=T2, **POOLED, **SEQUENCE)
    assert torch.allclose(
        through_the_model,
        _by_hand(bound_fraction=BOUND, bound_exchange_hz=30.0, t1_bound_ms=1000.0),
        atol=1e-6,
    )


def test_the_pool_genuinely_moves_the_answer() -> None:
    """Otherwise the agreement above is two ways of carrying nothing."""
    pooled = Anything().simulate(T1=T1, T2=T2, **POOLED, **SEQUENCE)
    free = Anything().simulate(T1=T1, T2=T2, **SEQUENCE)
    assert float((pooled - free).abs().max()) > 1e-3


def test_a_pool_parameter_carries_a_real_derivative() -> None:
    """A physics a model can reach but not differentiate is half reachable."""
    _, jacobian = Anything().jacobian(
        "bound_fraction", T1=T1, T2=T2, **POOLED, **SEQUENCE
    )
    assert float(jacobian.abs().max()) > 0.0


@pytest.fixture
def asked(monkeypatch):
    """Which tissue gradients each adjoint launch was told the caller reads."""
    seen: list[set[str]] = []
    original = _accelerators._wanted

    def watched(needs_input_grad):
        got = original(needs_input_grad)
        seen.append(
            {
                name
                for name, read in zip(FLOAT_NAMES, got, strict=True)
                if read and name in TISSUE_NAMES
            }
        )
        return got

    monkeypatch.setattr(_accelerators, "_wanted", watched)
    return seen


class Relaxation(EpgModel):
    """T1 and T2 over an unbalanced train."""

    properties = {"T1": "t1_ms", "T2": "t2_ms"}
    simulator = EpgEngine()
    states = 8

    def describe(self, *, flip, TR):
        """Return an unbalanced SSFP train at the flip angles given."""
        return mrf_description(torch.deg2rad(torch.as_tensor(flip)), TR * 1e-3)


def test_a_cost_on_the_sequence_alone_leaves_the_tissue_unwanted(asked) -> None:
    """``wanted`` is not a per-gradient gate -- it chooses the kernel.

    A caller that will not read the four gradients outside the real subspace
    can be given the real reverse kernel, which is worth about twice. That
    choice is made from what the autograd graph says the caller needs, so what
    matters here is that the model layer passes the question through rather
    than answering it.
    """
    flip = FLIP.clone().requires_grad_(True)
    signal = Relaxation().simulate(T1=T1, T2=T2, flip=flip, TR=10.0)
    signal.abs().square().sum().backward()

    assert asked, "the adjoint never ran"
    assert asked[-1] == set(), "a tissue nobody differentiated was still wanted"
    assert float(flip.grad.abs().max()) > 0.0


def test_differentiating_the_tissue_asks_for_it(asked) -> None:
    """And the answer changes with the question, so it is read rather than fixed."""
    t1 = T1.clone().requires_grad_(True)
    flip = FLIP.clone().requires_grad_(True)
    signal = Relaxation().simulate(T1=t1, T2=T2, flip=flip, TR=10.0)
    signal.abs().square().sum().backward()

    assert asked[-1] == {"t1_ms"}, "the question was not read from the graph"
    assert float(t1.grad.abs().max()) > 0.0
