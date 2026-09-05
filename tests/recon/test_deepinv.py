"""Composing the model operator with an encoding operator someone else wrote."""

from __future__ import annotations

import pytest
import torch

from torchsim import (
    execution,
)
from torchsim.recon import ModelOperator
from torchsim.simulators import MultiEchoSimulator

deepinv = pytest.importorskip("deepinv")

TE_MS = torch.tensor([10.0, 20.0, 40.0, 80.0, 160.0])


@pytest.fixture
def operator():
    return ModelOperator(
        MultiEchoSimulator(TE=TE_MS),
        "T2",
        bounds={"T2": (10.0, 300.0)},
    )


@pytest.fixture
def maps(operator):
    """Channel-first, which is the convention on deepinv's side of the line."""
    generator = torch.Generator().manual_seed(0)
    x = operator.initial((16, 16), T2=80.0)
    x = x + 0.2 * torch.randn(x.shape, generator=generator)
    return x.movedim(-1, 0)[None]


def test_the_wrapper_moves_the_map_axis_and_nothing_else(operator, maps) -> None:
    """One convention on each side, and the boundary is where they meet."""
    physics = operator.physics()

    recorded = physics.A(maps)

    assert recorded.shape == (1, TE_MS.numel(), 16, 16)
    torch.testing.assert_close(recorded, operator.A(maps.movedim(1, -1)).movedim(-1, 1))


def test_our_adjoint_is_the_one_autograd_would_have_found(operator, maps) -> None:
    """The analytic derivative is the derivative, not an approximation of it."""
    physics = operator.physics()
    cotangent = torch.randn(
        (1, TE_MS.numel(), 16, 16),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.complex64,
    )

    ours = physics.A_vjp(maps, cotangent)

    theirs = deepinv.physics.Physics.A_vjp(physics, maps, cotangent)
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-5)


def test_differentiating_through_the_operator_does_not_stream_it(
    operator, maps, monkeypatch
) -> None:
    """A chunk crossing a pinned buffer cannot carry a derivative back.

    An outer ``torch.func`` transform is how a composed operator is
    differentiated, and the streamed path would silently give it nothing, so
    a call under one runs where it stands however loud the policy is.
    """
    from torchsim.recon import _operator as module

    entered = []
    real = module.per_voxel
    monkeypatch.setattr(
        module,
        "per_voxel",
        lambda *args, **kwargs: (entered.append(1), real(*args, **kwargs))[1],
    )
    physics = operator.physics()
    cotangent = torch.ones((1, TE_MS.numel(), 16, 16), dtype=torch.complex64)

    with execution("cpu"):
        through = deepinv.physics.Physics.A_vjp(physics, maps, cotangent)

    assert not entered
    torch.testing.assert_close(
        through, physics.A_vjp(maps, cotangent), atol=1e-5, rtol=1e-5
    )


def test_it_composes_with_a_linear_encoding_operator(operator, maps) -> None:
    """``encoding * model`` is the chain the whole approach is written as."""
    encoding = deepinv.physics.Inpainting(
        img_size=(TE_MS.numel(), 16, 16), mask=0.5, device="cpu"
    )

    chain = encoding * operator.physics()

    measured = chain.A(maps)
    assert measured.shape == (1, TE_MS.numel(), 16, 16)
    torch.testing.assert_close(measured, encoding.A(operator.physics().A(maps)))


def test_a_deepinv_optimizer_drives_the_composed_operator(operator) -> None:
    """End to end through somebody else's solver, on the nonlinear chain.

    Composing this way gives up the analytic derivative -- deepinv takes the
    Jacobian products by differentiating the whole chain -- which is the trade
    for reaching every optimizer and prior it has.
    :class:`~torchsim.recon.GaussNewton` chains the two operators' own
    products instead and keeps it.
    """
    from deepinv.optim import L2

    # Every voxel keeps at least half its echoes: one that lost all of them
    # would have no data behind it, and gradient descent would leave it where
    # it started for reasons that say nothing about the composition.
    mask = torch.ones(TE_MS.numel(), 8, 8)
    mask[::2, ::2, ::2] = 0.0
    encoding = deepinv.physics.Inpainting(
        img_size=(TE_MS.numel(), 8, 8), mask=mask, device="cpu"
    )
    chain = encoding * operator.physics()
    truth = operator.initial((1, 8, 8), T2=60.0).movedim(-1, 1)
    measured = chain.A(truth)

    fidelity = L2()
    x = operator.initial((1, 8, 8), T2=150.0).movedim(-1, 1)
    before = float(fidelity(x, measured, chain))
    for _ in range(200):
        x = x - 0.5 * fidelity.grad(x, measured, chain)

    assert float(fidelity(x, measured, chain)) < 0.01 * before
    found = operator.split(x.movedim(1, -1))["T2"]
    assert float((found - 60.0).abs().mean()) < 10.0
