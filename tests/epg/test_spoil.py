"""Test perfect spoiling."""

from types import SimpleNamespace

import pytest
import torch

from utils.epg import spoil


@pytest.fixture
def sample_states():
    """Fixture to provide a sample EPG states object."""
    dtype = torch.float32
    device = torch.device("cpu")
    nstates = 5
    nlocs = 3
    ntrans_pools = 2
    nlong_pools = 2

    Fplus = torch.rand((nstates, nlocs, ntrans_pools), dtype=dtype, device=device)
    Fminus = torch.rand((nstates, nlocs, ntrans_pools), dtype=dtype, device=device)
    Z = torch.rand((nstates, nlocs, nlong_pools), dtype=dtype, device=device)

    return SimpleNamespace(Fplus=Fplus.clone(), Fminus=Fminus.clone(), Z=Z.clone())


def test_spoil_transverse_magnetization(sample_states):
    original_states = sample_states
    spoiled_states = spoil(sample_states)

    # Ensure Fplus and Fminus are zeroed
    assert torch.all(spoiled_states.Fplus == 0), "Fplus should be zeroed after spoil"
    assert torch.all(spoiled_states.Fminus == 0), "Fminus should be zeroed after spoil"

    # Ensure Z is unchanged
    assert torch.all(spoiled_states.Z == original_states.Z), (
        "Z should remain unchanged after spoil"
    )


def test_spoil_in_place(sample_states):
    original_states = sample_states
    spoiled_states = spoil(sample_states)

    # Ensure the operation is in-place
    assert spoiled_states is original_states, "Spoil should modify states in place"


def test_spoil_no_shape_change(sample_states):
    original_states = sample_states
    spoiled_states = spoil(sample_states)

    # Ensure shapes of Fplus, Fminus, and Z remain unchanged
    assert spoiled_states.Fplus.shape == original_states.Fplus.shape, (
        "Fplus shape should not change after spoil"
    )
    assert spoiled_states.Fminus.shape == original_states.Fminus.shape, (
        "Fminus shape should not change after spoil"
    )
    assert spoiled_states.Z.shape == original_states.Z.shape, (
        "Z shape should not change after spoil"
    )


def test_spoil_does_not_affect_device(sample_states):
    original_states = sample_states
    spoiled_states = spoil(sample_states)

    # Ensure device of the tensors is preserved
    assert spoiled_states.Fplus.device == original_states.Fplus.device, (
        "Device of Fplus should remain unchanged"
    )
    assert spoiled_states.Fminus.device == original_states.Fminus.device, (
        "Device of Fminus should remain unchanged"
    )
    assert spoiled_states.Z.device == original_states.Z.device, (
        "Device of Z should remain unchanged"
    )
