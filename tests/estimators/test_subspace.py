"""What a temporal basis keeps, and whether it says so honestly.

The reason to compress is speed, and the reason it is safe is that the error
is known before it is paid. So the test that matters is not that the basis
exists but that ``retained`` is the approximation itself.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import (
    Subspace,
)
from torchsim.simulators import FSESimulator

RANKS = (1, 2, 4, 8)


def _dictionary(count: int = 240, echoes: int = 64) -> torch.Tensor:
    """An FSE dictionary, which is what a real subspace is fitted to."""
    generator = torch.Generator().manual_seed(0)
    t1 = 200.0 + 2800.0 * torch.rand(count, generator=generator)
    t2 = 20.0 + 250.0 * torch.rand(count, generator=generator)
    simulator = FSESimulator(ESP=5.0, TR=3000.0, states=10)
    return simulator.simulate(flip=torch.full((echoes,), 150.0), T1=t1, T2=t2)


@pytest.mark.parametrize("rank", RANKS)
def test_retained_is_the_error_it_will_cost(rank: int) -> None:
    """One minus the retained energy is the relative squared error, exactly.

    Not approximately and not a bound: the fraction the basis reports is the
    same number as projecting the dictionary through it and measuring.
    """
    dictionary = _dictionary()
    subspace = Subspace.fit(dictionary, rank)

    through = subspace.expand(subspace.project(dictionary))
    measured = (
        dictionary - through
    ).abs().square().sum() / dictionary.abs().square().sum()

    assert subspace.retained == pytest.approx(1.0 - float(measured), abs=1e-6)


def test_a_full_rank_basis_loses_nothing() -> None:
    """At the rank the signals actually span, the round trip is the identity."""
    dictionary = _dictionary(count=32, echoes=16)
    subspace = Subspace.fit(dictionary, 16)

    through = subspace.expand(subspace.project(dictionary))

    assert subspace.retained == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(through, dictionary, atol=1e-5)


def test_the_basis_is_orthonormal() -> None:
    """Projection is a rotation onto the kept directions, so norms mean what
    they mean in the full space."""
    subspace = Subspace.fit(_dictionary(), 6)

    gram = subspace.basis.mH @ subspace.basis

    assert torch.allclose(gram, torch.eye(6, dtype=gram.dtype), atol=1e-5)


def test_an_echo_train_needs_very_few_directions() -> None:
    """A relaxation-driven train is why any of this is worth doing."""
    subspace = Subspace.fit(_dictionary(), 8)

    assert subspace.retained > 0.9999
    assert subspace.rank == 8
    assert subspace.contrasts == 64


@pytest.mark.parametrize("rank", [0, -1])
def test_a_rank_must_be_positive(rank: int) -> None:
    with pytest.raises(ValueError, match="rank must be positive"):
        Subspace.fit(_dictionary(), rank)


def test_a_rank_beyond_the_signals_is_refused() -> None:
    """Asking for more directions than the signals span is a mistake, not a
    silently padded basis."""
    with pytest.raises(ValueError, match="exceeds"):
        Subspace.fit(_dictionary(count=4, echoes=8), 6)


def test_the_leading_axes_are_flattened() -> None:
    """A dictionary and a training set are the same input to a fit."""
    dictionary = _dictionary(count=60, echoes=32)

    flat = Subspace.fit(dictionary, 5)
    shaped = Subspace.fit(dictionary.reshape(6, 10, 32), 5)

    assert torch.allclose(flat.basis, shaped.basis, atol=1e-6)
