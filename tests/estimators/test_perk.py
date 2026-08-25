"""Tests for the PERK estimator."""

from __future__ import annotations

import math

import pytest
import torch

from torchsim.estimators import PERK
from torchsim.model import SignalModel


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_perk_learns_smooth_nonlinear_inverse(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(4)
    train_parameter = 0.1 + 2.8 * torch.rand(2048, 1, generator=generator, device=device)
    train_signal = torch.cat(
        (torch.sin(train_parameter), torch.cos(train_parameter)), dim=-1
    )
    test_parameter = torch.linspace(0.15, 2.85, 128, device=device)[:, None]
    test_signal = torch.cat(
        (torch.sin(test_parameter), torch.cos(test_parameter)), dim=-1
    )
    estimator = PERK(
        n_features=256,
        regularization=1e-5,
        chunk_size=257,
        seed=8,
    ).to(device)

    estimator.fit(train_signal, train_parameter)
    actual = estimator(test_signal)

    assert torch.mean((actual - test_parameter) ** 2) < 2e-4


def test_perk_accepts_complex_signals_and_known_parameters() -> None:
    parameter = torch.linspace(0.1, 1.0, 512)[:, None]
    known = torch.linspace(0.8, 1.2, 512)[:, None]
    signal = known * torch.exp(1j * parameter)
    estimator = PERK(n_features=128, regularization=1e-5, seed=2)

    estimator.fit(signal, parameter, known)
    actual = estimator(signal[:16], known[:16])

    assert actual.shape == (16, 1)
    torch.testing.assert_close(actual, parameter[:16], atol=2e-2, rtol=2e-2)


def test_perk_estimation_is_differentiable() -> None:
    parameter = torch.linspace(0.1, 1.0, 256)[:, None]
    signal = torch.cat((parameter, parameter.square()), dim=-1)
    estimator = PERK(n_features=64, regularization=1e-4, seed=1).fit(
        signal, parameter
    )
    measured = signal[:8].clone().requires_grad_()

    estimator(measured).sum().backward()

    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()


class _Counted(SignalModel):
    """A model whose signal is its property and its square, and that counts."""

    properties = ("x",)

    def __init__(self, seen: list[int]) -> None:
        super().__init__()
        self._seen = seen

    def evaluate(self, properties, **sequence):
        values = properties["x"].reshape(-1, 1)
        self._seen.append(values.shape[0])
        return torch.cat((values, values.square()), dim=-1)


def test_perk_fit_simulator_chunks_generation() -> None:
    """Memory follows the chunk, and the dictionary is never retained."""
    parameter = torch.linspace(0.1, 1.0, 65)
    seen: list[int] = []

    estimator = PERK(n_features=32, seed=1).fit_simulator(
        _Counted(seen),
        {"x": parameter},
        simulation_chunk_size=16,
    )

    assert estimator.fitted
    # One pass to read the kernel width off the inputs, one to fit.
    assert seen == [16, 16, 16, 16, 1] * 2


def test_a_given_length_scale_costs_one_pass() -> None:
    """The kernel width is the only reason to look at the data twice.

    Nothing can be accumulated until the random features exist, and they
    cannot be drawn until the width is known. Say what it is and the training
    set is walked once -- which for a source that simulates is the difference
    between simulating it once and simulating it twice.
    """
    parameter = torch.linspace(0.1, 1.0, 65)
    seen: list[int] = []

    estimator = PERK(n_features=32, seed=1, length_scale=0.5).fit_simulator(
        _Counted(seen),
        {"x": parameter},
        simulation_chunk_size=16,
    )

    assert estimator.fitted
    assert seen == [16, 16, 16, 16, 1]


def test_the_merged_pass_gives_the_covariance_it_would_have_centred() -> None:
    """Means and products come out of one pass, not a pass each.

    A covariance can be accumulated centred, which needs the mean first, or as
    a raw second moment the mean is subtracted from afterwards. The second
    reads the data once. This asserts the two agree where it matters -- in the
    weights that come out.
    """
    generator = torch.Generator().manual_seed(0)
    signals = torch.randn(2000, 12, generator=generator)
    parameters = torch.stack(
        (signals.square().sum(-1), signals.abs().mean(-1)), dim=-1
    )
    estimator = PERK(n_features=64, seed=3).fit(signals, parameters)

    features = _reference_features(estimator, signals).to(torch.float64)
    targets = parameters.to(torch.float64)
    centred = features - features.mean(0)
    covariance = centred.mT @ centred / (signals.shape[0] - 1)
    covariance.diagonal().add_(estimator.regularization)
    cross = (targets - targets.mean(0)).mT @ centred / (signals.shape[0] - 1)
    expected = torch.linalg.solve(covariance, cross.mT).mT

    assert torch.allclose(estimator.weight, expected.to(torch.float32), rtol=1e-4)


def _reference_features(estimator, signals):
    """The feature map, written out rather than called."""
    scale = math.sqrt(2.0 / estimator.frequency.shape[0])
    return scale * torch.cos(signals @ estimator.frequency.mT + estimator.phase)
