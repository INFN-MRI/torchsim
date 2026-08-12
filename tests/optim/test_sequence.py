"""Tests for generic sequence optimization."""

from __future__ import annotations

import torch

from torchsim.optim import FSET2Precision, SequenceOptimizer


def test_sequence_optimizer_handles_multiple_named_parameters() -> None:
    objective = lambda values: (values["x"] - 0.25).square().mean() + (
        values["y"] + 0.5
    ).square().mean()
    optimizer = SequenceOptimizer(
        objective,
        bounds={"x": (0.0, 1.0)},
        iterations=20,
        learning_rate=0.2,
    )

    result = optimizer.optimize({"x": torch.tensor([0.8]), "y": torch.tensor([0.2])})

    assert isinstance(result.parameters, dict)
    assert 0.0 <= result.parameters["x"].item() <= 1.0
    assert result.loss[-1] < result.loss[0]


def test_fse_objective_composes_with_generic_optimizer() -> None:
    objective = FSET2Precision(
        torch.tensor([900.0, 1300.0]),
        torch.tensor([50.0, 100.0]),
        5.0,
    )
    optimizer = SequenceOptimizer(
        objective,
        bounds=(40.0, 160.0),
        iterations=2,
    )

    result = optimizer.optimize(torch.full((4,), 120.0))

    assert isinstance(result.parameters, torch.Tensor)
    assert result.parameters.shape == (4,)
    assert result.loss.shape == (2,)
    assert torch.isfinite(result.loss).all()


def test_sequence_optimizer_callback_can_stop_early() -> None:
    optimizer = SequenceOptimizer(
        lambda value: value.square().mean(),
        iterations=10,
    )
    iterations = []

    def callback(
        iteration: int,
        parameters: torch.Tensor | dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> bool:
        iterations.append((iteration, parameters, loss))
        return iteration == 2

    result = optimizer.optimize(torch.ones(2), callback=callback)

    assert len(iterations) == 3
    assert result.loss.shape == (3,)
