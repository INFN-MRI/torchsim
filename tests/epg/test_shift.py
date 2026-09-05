"""Test EPG shift."""

from types import SimpleNamespace

import pytest
import torch

from utils.epg import shift


@pytest.fixture
def sample_states():
    """Fixture to provide a sample EPG states object."""
    dtype = torch.complex64
    device = torch.device("cpu")
    nstates = 5
    nlocs = 3
    ntrans_pools = 2

    Fplus = (
        torch.arange(nstates * nlocs * ntrans_pools, device=device)
        .reshape(nstates, nlocs, ntrans_pools)
        .to(dtype)
        + 1j
    )
    Fminus = (
        torch.arange(nstates * nlocs * ntrans_pools, device=device)
        .reshape(nstates, nlocs, ntrans_pools)
        .to(dtype)
        - 1j
    )

    Z = torch.zeros(
        (nstates, nlocs, ntrans_pools), dtype=dtype, device=device
    )  # Longitudinal state unaffected by shift

    return SimpleNamespace(Fplus=Fplus.clone(), Fminus=Fminus.clone(), Z=Z.clone())


# def test_shift_default_delta(sample_states):
#     delta = 1
#     original_states = sample_states
#     shifted_states = shift(sample_states, delta)

#     # Check Fplus has been shifted forward
#     expected_Fplus = torch.roll(original_states.Fplus, delta, -3)
#     expected_Fplus[0] = original_states.Fminus[0].conj()
#     assert torch.allclose(
#         shifted_states.Fplus, expected_Fplus
#     ), "Fplus shift mismatch with delta=1"

#     # Check Fminus has been shifted backward
#     expected_Fminus = torch.roll(original_states.Fminus, -delta, -3)
#     expected_Fminus[-1] = 0.0
#     assert torch.allclose(
#         shifted_states.Fminus, expected_Fminus
#     ), "Fminus shift mismatch with delta=1"


# def test_shift_custom_delta(sample_states):
#     delta = 2
#     original_states = sample_states
#     shifted_states = shift(sample_states, delta)

#     # Check Fplus has been shifted forward
#     expected_Fplus = torch.roll(original_states.Fplus, delta, -3)
#     expected_Fplus[0] = original_states.Fminus[0].conj()
#     assert torch.allclose(
#         shifted_states.Fplus, expected_Fplus
#     ), f"Fplus shift mismatch with delta={delta}"

#     # Check Fminus has been shifted backward
#     expected_Fminus = torch.roll(original_states.Fminus, -delta, -3)
#     expected_Fminus[-1] = 0.0
#     assert torch.allclose(
#         shifted_states.Fminus, expected_Fminus
#     ), f"Fminus shift mismatch with delta={delta}"


def test_shift_no_delta(sample_states):
    delta = 0
    original_states = sample_states
    shifted_states = shift(sample_states, delta)

    # No shift should occur
    assert torch.allclose(shifted_states.Fplus, original_states.Fplus), (
        "Fplus should remain unchanged with delta=0"
    )
    assert torch.allclose(shifted_states.Fminus, original_states.Fminus), (
        "Fminus should remain unchanged with delta=0"
    )


def test_shift_boundary_conditions(sample_states):
    delta = sample_states.Fplus.shape[-3]  # Maximum shift equal to number of states
    original_states = sample_states
    shifted_states = shift(sample_states, delta)

    # Check Fplus is circularly shifted by delta (full cycle returns to original)
    expected_Fplus = torch.roll(original_states.Fplus, delta, -3)
    expected_Fplus[0] = original_states.Fminus[0].conj()
    assert torch.allclose(shifted_states.Fplus, expected_Fplus), (
        "Fplus boundary shift mismatch"
    )

    # Check Fminus is circularly shifted backward
    expected_Fminus = torch.roll(original_states.Fminus, -delta, -3)
    expected_Fminus[-1] = 0.0
    assert torch.allclose(shifted_states.Fminus, expected_Fminus), (
        "Fminus boundary shift mismatch"
    )


def test_shift_shape_and_device_preservation(sample_states):
    original_states = sample_states
    shifted_states = shift(sample_states)

    # Ensure shape and device are preserved
    assert shifted_states.Fplus.shape == original_states.Fplus.shape, (
        "Fplus shape mismatch after shift"
    )
    assert shifted_states.Fminus.shape == original_states.Fminus.shape, (
        "Fminus shape mismatch after shift"
    )
    assert shifted_states.Fplus.device == original_states.Fplus.device, (
        "Fplus device mismatch after shift"
    )
    assert shifted_states.Fminus.device == original_states.Fminus.device, (
        "Fminus device mismatch after shift"
    )
    assert shifted_states.Z.shape == original_states.Z.shape, (
        "Z shape should remain unchanged"
    )
    assert shifted_states.Z.device == original_states.Z.device, (
        "Z device should remain unchanged"
    )


def test_shift_in_place(sample_states):
    original_states = sample_states
    shifted_states = shift(sample_states)

    # Ensure the operation is in-place
    assert shifted_states is original_states, "Shift should modify states in place"
