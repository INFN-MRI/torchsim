"""Tests for the PERK estimator."""

from __future__ import annotations

import pytest
import torch

from torchsim.estimators import PERK


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


def test_perk_fit_simulator_chunks_generation() -> None:
    parameter = torch.linspace(0.1, 1.0, 65)[:, None]
    seen: list[int] = []

    def simulator(values: torch.Tensor, _known: torch.Tensor | None) -> torch.Tensor:
        seen.append(values.shape[0])
        return torch.cat((values, values.square()), dim=-1)

    estimator = PERK(n_features=32, seed=1).fit_simulator(
        simulator,
        parameter,
        simulation_chunk_size=16,
    )

    assert estimator.fitted
    # PERK makes bounded passes for input statistics, feature statistics, and
    # covariance accumulation; it never retains the simulated dictionary.
    assert seen == [16, 16, 16, 16, 1] * 3
