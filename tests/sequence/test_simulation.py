"""Tests for differentiable sequence-description simulation."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from torchsim.sequence import (
    FSE,
    SPGR,
    SSFPFID,
    TissueProperties,
    fse_description,
    mrf_description,
    simulate_subspace,
    spgr_description,
)


def test_fse_simulates_many_atoms_in_one_state_machine() -> None:
    description = fse_description(torch.deg2rad(torch.full((4,), 160.0)), 5e-3)
    tissue = TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0, 1600.0]),
        t2_ms=torch.tensor([50.0, 80.0, 120.0]),
    )

    result = FSE().simulate(description, tissue)

    assert result.signal.shape == (3, 4)
    assert result.time_us.shape == (4,)
    assert torch.isfinite(result.signal).all()


def test_state_machine_supports_forward_mode_for_tissue() -> None:
    description = fse_description(torch.deg2rad(torch.full((3,), 140.0)), 5e-3)

    def signal(t2: torch.Tensor) -> torch.Tensor:
        result = FSE().simulate(
            description,
            TissueProperties(t1_ms=1000.0, t2_ms=t2),
        )
        return torch.view_as_real(result.signal).reshape(-1)

    derivative = torch.func.jacfwd(signal)(torch.tensor(80.0))

    assert derivative.shape == (6,)
    assert torch.isfinite(derivative).all()
    assert torch.any(derivative != 0)


def test_state_machine_supports_reverse_mode_for_flip_train() -> None:
    flip = torch.deg2rad(torch.full((3,), 150.0)).requires_grad_()

    result = FSE().simulate(
        fse_description(flip, 5e-3),
        TissueProperties(t1_ms=1000.0, t2_ms=80.0),
    )
    result.signal.abs().sum().backward()

    assert flip.grad is not None
    assert torch.isfinite(flip.grad).all()


def test_mrf_and_spgr_policies_have_expected_shapes() -> None:
    tissue = TissueProperties(t1_ms=torch.tensor([700.0, 1200.0]), t2_ms=80.0)
    mrf = SSFPFID().simulate(
        mrf_description(torch.deg2rad(torch.tensor([5.0, 10.0, 15.0])), 8e-3),
        tissue,
    )
    spgr = SPGR().simulate(
        spgr_description(torch.deg2rad(torch.tensor([5.0, 10.0])), 8e-3, 3e-3),
        tissue,
    )

    assert mrf.signal.shape == (2, 3)
    assert spgr.signal.shape == (2, 2)


def test_subspace_basis_uses_time_as_leading_dimension() -> None:
    description = fse_description(torch.deg2rad(torch.full((6,), 160.0)), 5e-3)
    tissue = TissueProperties(
        t1_ms=1000.0,
        t2_ms=torch.linspace(30.0, 180.0, 8),
    )

    result = simulate_subspace(description, "fse", tissue, rank=3)

    assert result.dictionary.shape == (8, 6)
    assert result.basis.shape == (6, 3)
    assert result.singular_values.shape == (6,)


@pytest.mark.skipif(
    importlib.util.find_spec("torchsim._epg_cpu") is None,
    reason="CPU extension has not been built",
)
@pytest.mark.parametrize("policy_name", ["fse", "ssfp-fid", "spgr"])
def test_cpu_native_backend_matches_torch(policy_name: str) -> None:
    policy, description = _case(policy_name, "cpu")
    tissue = TissueProperties(
        t1_ms=torch.tensor([700.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 100.0]),
        b1=torch.tensor([0.9, 1.1]),
        b0_hz=torch.tensor([0.0, 20.0]),
        inversion_efficiency=0.95,
    )

    expected = policy.simulate(description, tissue, nstates=10, backend="torch")
    with torch.no_grad():
        actual = policy.simulate(description, tissue, nstates=10, backend="native")

    torch.testing.assert_close(actual.signal, expected.signal, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.time_us, expected.time_us)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("policy_name", ["fse", "ssfp-fid", "spgr"])
def test_triton_backend_matches_torch(policy_name: str) -> None:
    policy, description = _case(policy_name, "cuda")
    tissue = TissueProperties(
        t1_ms=torch.tensor([700.0, 1200.0], device="cuda"),
        t2_ms=torch.tensor([50.0, 100.0], device="cuda"),
        b1=torch.tensor([0.9, 1.1], device="cuda"),
        b0_hz=torch.tensor([0.0, 20.0], device="cuda"),
        inversion_efficiency=0.95,
    )

    expected = policy.simulate(description, tissue, nstates=10, backend="torch")
    with torch.no_grad():
        actual = policy.simulate(description, tissue, nstates=10, backend="native")

    torch.testing.assert_close(actual.signal, expected.signal, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.time_us, expected.time_us)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_native_forward_mode_matches_torch_for_tissue_and_flip(device: str) -> None:
    flip = torch.deg2rad(torch.tensor([135.0, 150.0, 165.0], device=device))
    t2 = torch.tensor([45.0, 90.0], device=device)

    def simulate(
        t2_ms: torch.Tensor,
        flip_rad: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        description = fse_description(flip_rad, 5e-3, phases_rad=torch.pi / 2)
        return FSE().simulate(
            description,
            TissueProperties(t1_ms=1000.0, t2_ms=t2_ms),
            nstates=10,
            backend=backend,
        ).signal

    t2_tangent = torch.tensor([0.5, -0.25], device=device)
    flip_tangent = torch.tensor([0.1, -0.2, 0.3], device=device)
    native = torch.func.jvp(
        lambda current_t2, current_flip: simulate(
            current_t2, current_flip, "native"
        ),
        (t2, flip),
        (t2_tangent, flip_tangent),
    )
    reference = torch.func.jvp(
        lambda current_t2, current_flip: simulate(current_t2, current_flip, "torch"),
        (t2, flip),
        (t2_tangent, flip_tangent),
    )

    torch.testing.assert_close(native[0], reference[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(native[1], reference[1], rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_native_reverse_over_forward_supports_sequence_design(device: str) -> None:
    flip = torch.deg2rad(
        torch.tensor([135.0, 150.0, 165.0], device=device)
    ).requires_grad_()
    description = fse_description(flip, 5e-3, phases_rad=torch.pi / 2)

    def signal(t2_ms: torch.Tensor) -> torch.Tensor:
        return FSE().simulate(
            description,
            TissueProperties(t1_ms=1000.0, t2_ms=t2_ms),
            nstates=10,
            backend="native",
        ).signal

    _, derivative = torch.func.jvp(
        signal,
        (torch.tensor(80.0, device=device),),
        (torch.tensor(1.0, device=device),),
    )
    gradient = torch.autograd.grad(derivative.abs().square().sum(), flip)[0]

    assert torch.isfinite(gradient).all()
    assert torch.any(gradient != 0)


# %% private module subroutines


def _case(name: str, device: str):
    flip = torch.deg2rad(torch.tensor([15.0, 60.0, 120.0], device=device))
    if name == "fse":
        return FSE(), fse_description(flip, 5e-3, phases_rad=torch.pi / 2)
    if name == "ssfp-fid":
        return SSFPFID(), mrf_description(flip, 8e-3, inversion_time_s=20e-3)
    return SPGR(), spgr_description(flip, 8e-3, 3e-3)
