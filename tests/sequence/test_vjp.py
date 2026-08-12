"""The fused adjoint must agree with the differentiable fallback."""

from __future__ import annotations

import pytest
import torch

from torchsim import (
    FSE,
    TissueProperties,
    fse_description,
    mprage_description,
    mrf_description,
    spgr_description,
)
from torchsim.sequence import _accelerators
from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed_vjp,
    _simulate_packed_torch,
)
from torchsim.sequence._simulation import _prepare_tissue

# (t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, duration, flip, phase)
PARAMETER_NAMES = [
    "t1",
    "t2",
    "m0",
    "b1",
    "b1_phase",
    "b0",
    "inversion_efficiency",
    "duration",
    "flip",
    "phase",
]
ECHOES = 6


def _tissue():
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 90.0]),
        m0=torch.tensor([1.0, 0.8]),
        b1=torch.tensor([1.0, 0.9]),
        b1_phase_rad=torch.tensor([0.05, -0.1]),
        b0_hz=torch.tensor([13.0, -27.0]),
        inversion_efficiency=torch.tensor([0.95, 0.9]),
    )


def _descriptions():
    flip = torch.deg2rad(torch.full((ECHOES,), 140.0))
    return {
        "fse": (
            "fse",
            fse_description(flip, echo_spacing_s=5e-3, phases_rad=torch.pi / 2),
        ),
        "spgr": (
            "spgr",
            spgr_description(torch.deg2rad(torch.full((ECHOES,), 15.0)), 8e-3, 3e-3),
        ),
        "mrf": (
            "ssfpfid",
            mrf_description(
                torch.deg2rad(torch.linspace(5.0, 60.0, ECHOES)),
                torch.full((ECHOES,), 10e-3),
                inversion_time_s=20e-3,
            ),
        ),
        "mprage": (
            "spgr",
            mprage_description(
                2, ECHOES - 2, torch.deg2rad(torch.tensor(8.0)), 8e-3, 20e-3
            ),
        ),
    }


@pytest.mark.parametrize("name", sorted(_descriptions()))
@pytest.mark.parametrize("threads", [1, 4])
def test_fused_vjp_matches_fallback(name: str, threads: int) -> None:
    policy, description = _descriptions()[name]
    prepared, _, device = _prepare_tissue(_tissue(), "cpu")
    packed = _pack_events(
        policy,
        description,
        repetitions=1,
        record="all",
        device=device,
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )
    state_count = 8
    output_count = packed.output_count

    leaves = tuple(value.detach().clone().requires_grad_(True) for value in prepared)
    differentiable_events = (
        events[0].detach().clone().requires_grad_(True),
        events[1],
        events[2].detach().clone().requires_grad_(True),
        events[3].detach().clone().requires_grad_(True),
        events[4],
        events[5],
    )
    output = _simulate_packed_torch(
        leaves,
        differentiable_events,
        state_count=state_count,
        output_count=output_count,
    )
    torch.manual_seed(0)
    seed = torch.randn(output.shape, dtype=torch.complex64)
    reference = torch.autograd.grad(
        output,
        (
            *leaves,
            differentiable_events[0],
            differentiable_events[2],
            differentiable_events[3],
        ),
        grad_outputs=seed,
        allow_unused=True,
    )
    fused = _run_packed_vjp(
        prepared,
        events,
        seed,
        state_count=state_count,
        output_count=output_count,
        threads=threads,
    )

    for parameter, expected, actual in zip(
        PARAMETER_NAMES, reference, fused, strict=True
    ):
        if expected is None:
            continue
        scale = expected.abs().max().item()
        if scale < 1e-7:
            # gradient is numerically zero; nothing meaningful to compare
            continue
        error = (expected - actual).abs().max().item() / scale
        assert error < 1e-4, f"{parameter} gradient differs by {error:.2e}"


def _gradients():
    description = fse_description(
        torch.deg2rad(torch.full((12,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )
    t1 = torch.full((64,), 1000.0, requires_grad=True)
    t2 = torch.linspace(40.0, 120.0, 64).requires_grad_(True)
    signal = FSE().simulate(
        description, TissueProperties(t1_ms=t1, t2_ms=t2), nstates=10
    ).signal
    return torch.autograd.grad(signal.abs().square().sum(), (t1, t2))


def test_fused_vjp_is_bitwise_deterministic(monkeypatch) -> None:
    first = _gradients()
    for _ in range(3):
        for expected, actual in zip(first, _gradients(), strict=True):
            assert torch.equal(expected, actual)

    monkeypatch.setenv("TORCHSIM_NUM_THREADS", "1")
    single = _gradients()
    monkeypatch.setenv("TORCHSIM_NUM_THREADS", "8")
    for expected, actual in zip(single, _gradients(), strict=True):
        assert torch.equal(expected, actual)


def _objective_gradient():
    from torchsim import FSET2Precision

    objective = FSET2Precision(
        torch.tensor([800.0, 1400.0]), torch.tensor([45.0, 120.0]), 5.0
    )
    flip = torch.linspace(150.0, 90.0, 12).requires_grad_(True)
    loss = objective(flip)
    (gradient,) = torch.autograd.grad(loss, flip)
    return loss.detach(), gradient


def test_forward_over_reverse_matches_fallback(monkeypatch) -> None:
    """The second-order path drives sequence optimization, so it must agree."""
    monkeypatch.setattr(_accelerators, "_vjp_available", lambda device: False)
    reference_loss, reference_gradient = _objective_gradient()
    monkeypatch.undo()
    fused_loss, fused_gradient = _objective_gradient()

    assert torch.allclose(reference_loss, fused_loss, rtol=1e-5)
    scale = reference_gradient.abs().max().item()
    error = (reference_gradient - fused_gradient).abs().max().item() / scale
    assert error < 1e-4, f"objective gradient differs by {error:.2e}"


def test_forward_over_reverse_is_deterministic() -> None:
    _, first = _objective_gradient()
    for _ in range(3):
        assert torch.equal(first, _objective_gradient()[1])


def test_double_backward_falls_back() -> None:
    """The fused kernel is not differentiable, so graph building must fall back.

    Torch disables grad mode while running backward functions unless
    ``create_graph=True``, which is exactly the condition selected on here.
    """
    with torch.no_grad():  # how a plain backward pass runs
        assert _accelerators._vjp_available(torch.device("cpu"))
    with torch.enable_grad():  # how create_graph=True runs
        assert not _accelerators._vjp_available(torch.device("cpu"))
    with torch.no_grad():
        assert not _accelerators._vjp_available(torch.device("cuda"))
