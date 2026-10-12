"""Saying what a parameter may be: bounds and equality constraints on the model operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from blochsim import ModelOperator, Simulator
from blochsim.simulators import MultiEchoSimulator

TE_MS = torch.linspace(1.2, 12.0, 10)
#: Where the main fat peak sits at 3 T, in hertz.
FAT_HZ = -434.0


class FatWater(Simulator):
    """A gradient echo whose water and fat fractions must sum to one.

    Written on the fat fraction alone: water is ``1 - f`` inside the model, so
    the sum is one at every iterate by construction rather than by correction.
    That is how an equality constraint is imposed -- by removing the degree of
    freedom, not by restoring it afterwards.
    """

    properties = ("fat_fraction", "R2s")

    def __init__(self) -> None:
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def evaluate(self, properties: Mapping[str, Any], *, TE: Any) -> torch.Tensor:
        echo = torch.as_tensor(TE) * 1e-3
        fat = properties["fat_fraction"][..., None]
        water = 1.0 - fat
        self.seen.append((water.squeeze(-1).clone(), fat.squeeze(-1).clone()))
        rate = properties["R2s"][..., None]
        phase = torch.exp(2j * torch.pi * FAT_HZ * echo)
        return (water + fat * phase) * torch.exp(-rate * echo)


# %% equality constraints


def test_an_equality_constraint_is_the_model_written_on_what_is_free() -> None:
    """One channel for the pair, because there is one degree of freedom.

    The operator never learns about the constraint and never enforces it. The
    model has no way to violate it, wherever the free variable goes.
    """
    model = FatWater()
    operator = ModelOperator(
        model.bind(TE=TE_MS),
        "fat_fraction",
        "R2s",
        bounds={"fat_fraction": (0.0, 1.0), "R2s": (0.0, 200.0)},
    )
    assert operator.channels == 4  # f, R2s, and the two of the amplitude
    x = operator.initial((5,), fat_fraction=0.5, R2s=50.0)
    x = x + 10.0 * torch.randn(x.shape, generator=torch.Generator().manual_seed(3))

    operator.A(x)

    assert model.seen
    for water, fat in model.seen:
        assert bool((water == 1.0 - fat).all())
        assert float(fat.min()) >= 0.0 and float(fat.max()) <= 1.0


def test_the_constraint_is_what_makes_the_pair_identifiable() -> None:
    """Freeing both fractions leaves the amplitude able to absorb their sum.

    Water, fat and a complex scale are three amplitudes for two measurements'
    worth of information, so the recast is not a convenience -- without it the
    problem the solver is handed is a different, degenerate one.
    """
    model = FatWater()
    operator = ModelOperator(
        model.bind(TE=TE_MS),
        "fat_fraction",
        "R2s",
        bounds={"fat_fraction": (0.0, 1.0), "R2s": (0.0, 200.0)},
    )
    x = operator.initial((4,), fat_fraction=0.3, R2s=40.0)

    blocks = operator.jacobian(x)

    rows = torch.cat((blocks.real, blocks.imag), dim=-1)
    curvature = rows @ rows.mT
    # Four unknowns, and the model is written so all four are separable.
    assert int(torch.linalg.matrix_rank(curvature).min()) == operator.channels


# %% a state built from known maps


def test_initial_takes_a_map_per_voxel() -> None:
    """``initial`` is the inverse of ``split``, so it has to accept known maps.

    A dictionary is a state whose voxels carry the parameters it was built
    over -- what a subspace basis is estimated from, and what a matching table
    is -- and one starting scalar cannot express it.
    """
    operator = ModelOperator(
        MultiEchoSimulator(TE=TE_MS), "T2", bounds={"T2": (10.0, 500.0)}
    )
    values = torch.tensor([20.0, 60.0, 180.0])

    recovered = operator.split(operator.initial((3,), T2=values))["T2"]

    torch.testing.assert_close(recovered, values, rtol=1e-4, atol=1e-2)


def test_initial_rejects_a_map_that_leaves_the_bound() -> None:
    operator = ModelOperator(
        MultiEchoSimulator(TE=TE_MS), "T2", bounds={"T2": (10.0, 500.0)}
    )
    with pytest.raises(ValueError, match="not strictly inside its bound"):
        operator.initial((2,), T2=torch.tensor([50.0, 900.0]))
