"""Whether the fused kernels compute the same function as the plain one.

A fused kernel that quietly did not run agrees perfectly, so the tests that
matter here come in pairs: one that the answers match, and one that asserts
which code actually produced them.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsim.estimators import PERK
import torchsim.estimators._perk as _perk

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


@pytest.fixture
def composed(monkeypatch):
    """Force the plain Torch path, whatever backends are loaded."""

    def only_composed():
        monkeypatch.setattr(_perk, "_TRITON", None)
        monkeypatch.setattr(_perk, "_NATIVE", None)

    return only_composed


def _fitted(device: str, *, contrasts: int = 24, complex_signals: bool = False,
            **settings) -> tuple[PERK, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(11)
    signals = torch.randn(3000, contrasts, generator=generator, device=device)
    if complex_signals:
        signals = torch.complex(
            signals, torch.randn(3000, contrasts, generator=generator, device=device)
        )
    parameters = torch.stack(
        (signals.abs().sum(-1), signals.abs().square().mean(-1)), dim=-1
    )
    estimator = PERK(n_features=256, seed=5, **settings).to(device)
    estimator.fit(signals, parameters)
    return estimator, signals[:512]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "settings, complex_signals",
    [
        ({}, False),
        ({"normalize": True}, False),
        ({}, True),
        ({"complex_mode": "magnitude"}, True),
        ({"normalize": True, "complex_mode": "magnitude"}, True),
    ],
)
def test_the_fused_answer_is_the_composed_answer(
    device, settings, complex_signals, composed
) -> None:
    """Every combination of how a signal is turned into features."""
    estimator, measured = _fitted(
        device, complex_signals=complex_signals, **settings
    )
    fused = estimator(measured)

    composed()
    plain = estimator(measured)

    assert torch.allclose(fused, plain, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_the_fused_path_is_the_one_that_ran(device, monkeypatch) -> None:
    """Agreement cannot tell a fused kernel from a fallback, so ask directly."""
    estimator, measured = _fitted(device)
    reached: list[str] = []
    backend = _perk._TRITON if device == "cuda" else _perk._NATIVE
    original = backend.regress
    monkeypatch.setattr(
        backend,
        "regress",
        lambda *args: (reached.append(device), original(*args))[1],
    )

    estimator(measured)

    assert reached == [device]


@pytest.mark.parametrize("device", DEVICES)
def test_a_known_parameter_block_goes_through_the_kernel(device, composed) -> None:
    """Known parameters are extra feature columns, and the kernel sees them."""
    generator = torch.Generator(device=device).manual_seed(3)
    signals = torch.randn(2000, 16, generator=generator, device=device)
    known = torch.rand(2000, 2, generator=generator, device=device)
    parameters = (signals.sum(-1) + known.sum(-1))[:, None]
    estimator = PERK(n_features=128, seed=1).to(device)
    estimator.fit(signals, parameters, known)

    fused = estimator(signals[:64], known[:64])
    composed()
    plain = estimator(signals[:64], known[:64])

    assert torch.allclose(fused, plain, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_the_adjoint_is_the_composed_gradient(device, monkeypatch) -> None:
    """A PERK inside a reconstruction keeps its gradient and its speed."""
    estimator, measured = _fitted(device)

    fused_input = measured.clone().requires_grad_()
    cotangent = torch.randn_like(estimator(fused_input))
    estimator(fused_input).backward(cotangent)

    monkeypatch.setattr(_perk, "_TRITON", None)
    monkeypatch.setattr(_perk, "_NATIVE", None)
    plain_input = measured.clone().requires_grad_()
    estimator(plain_input).backward(cotangent)

    assert torch.allclose(fused_input.grad, plain_input.grad, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_a_gradient_wanted_for_a_fitted_tensor_falls_back(device, monkeypatch) -> None:
    """The kernel differentiates the signals only, so anything else takes the
    path that differentiates everything -- and the route is what is asserted."""
    estimator, measured = _fitted(device)
    estimator.weight.requires_grad_(True)
    backend = _perk._TRITON if device == "cuda" else _perk._NATIVE
    monkeypatch.setattr(
        backend, "regress", lambda *args: pytest.fail("the kernel ran")
    )

    values = estimator(measured)

    assert values.requires_grad


def _through_the_polynomial(angles: torch.Tensor) -> torch.Tensor:
    """``sqrt(2) * cos(angle)``, computed by the host kernel and nothing else."""
    native = pytest.importorskip("torchsim.estimators._perk_native")
    one = torch.ones(1, 1)
    return native.regress(
        angles, one, one, torch.zeros(1), torch.zeros(1), one, torch.zeros(1)
    )


def test_the_polynomial_cosine_is_a_cosine() -> None:
    """The host kernel does not call libm.

    A vectorizing compiler cannot put a call in a register, so the cosine is a
    polynomial -- and a polynomial is only a cosine to the accuracy it is
    written to. Over the angles a fitted feature map produces, which reach
    about twenty, that accuracy is float32.
    """
    angles = torch.linspace(-20.0, 20.0, 200_001, dtype=torch.float32)[:, None]

    error = (
        _through_the_polynomial(angles) - math.sqrt(2.0) * torch.cos(angles)
    ).abs().max()

    assert error < 1e-6


def test_a_far_larger_angle_degrades_rather_than_breaks() -> None:
    """Folding an angle into one turn loses the bits the angle needed to say
    which turn it was on, so accuracy falls away from zero. It should fall
    slowly, and a length scale chosen badly enough to get here is the real
    problem."""
    angles = torch.linspace(-200.0, 200.0, 200_001, dtype=torch.float32)[:, None]

    error = (
        _through_the_polynomial(angles) - math.sqrt(2.0) * torch.cos(angles)
    ).abs().max()

    assert error < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_a_single_voxel_goes_through_a_kernel_tiled_for_many(device) -> None:
    """The tiles are wider than one voxel, so the edge is worth asserting."""
    estimator, measured = _fitted(device)

    assert estimator(measured[:1]).shape == (1, 2)
