"""Test diffusion damping operators."""

import pytest

import torch
from types import SimpleNamespace

from utils import epg


@pytest.fixture
def diffusion_fixture():
    """Fixture for creating a common set of input EPG states."""
    return SimpleNamespace(
        Fplus=torch.tensor([1.0, 0.5, 0.2]),
        Fminus=torch.tensor([0.9, 0.4, 0.1]),
        Z=torch.tensor([0.8, 0.6, 0.3]),
    )


# Test for diffusion_op
def test_diffusion_op():
    D = torch.tensor(1e-9)  # Diffusion coefficient
    time = torch.tensor(10.0)  # Time interval in seconds
    nstates = 5
    total_dephasing = torch.tensor(3.0)  # Total gradient dephasing in radians
    voxelsize = 1.0  # Default voxel size

    # Call the diffusion_op function
    D1, D2 = epg.diffusion_op(D, time, nstates, total_dephasing, voxelsize)

    # Assert output shapes
    assert D1.shape == (nstates, 1, 1)
    assert D2.shape == (nstates, 1, 1)

    # Assert values
    assert torch.all(D1 <= 1) and torch.all(D1 >= 0), "D1 must be between 0 and 1"
    assert torch.all(D2 <= 1) and torch.all(D2 >= 0), "D2 must be between 0 and 1"

    # Check boundary case: D=0 should produce all ones
    D_zero = torch.tensor(0.0)
    D1_zero, D2_zero = epg.diffusion_op(
        D_zero, time, nstates, total_dephasing, voxelsize
    )
    assert torch.allclose(
        D1_zero, torch.ones_like(D1_zero)
    ), "D1 should be all ones for D=0"
    assert torch.allclose(
        D2_zero, torch.ones_like(D2_zero)
    ), "D2 should be all ones for D=0"


# Test for diffusion
def test_diffusion(diffusion_fixture):
    states = diffusion_fixture
    Fplus = states.Fplus.clone()
    Fminus = states.Fminus.clone()
    Z = states.Z.clone()

    D1 = torch.tensor([1.0, 0.9, 0.8])  # Longitudinal damping
    D2 = torch.tensor([0.9, 0.7, 0.5])  # Transverse damping

    # Call the diffusion function
    output_states = epg.diffusion(states, D1, D2)

    # Assert output shapes are the same as input
    assert output_states.Fplus.shape == Fplus.shape
    assert output_states.Fminus.shape == Fminus.shape
    assert output_states.Z.shape == Z.shape

    # Assert correct damping
    assert torch.allclose(output_states.Fplus, Fplus * D2), "Fplus damping incorrect"
    assert torch.allclose(output_states.Fminus, Fminus * D2), "Fminus damping incorrect"
    assert torch.allclose(output_states.Z, Z * D1), "Z damping incorrect"


def test_diffusion_edge_cases(diffusion_fixture):
    states = diffusion_fixture
    nstates = len(states.Fplus)
    D1 = torch.ones(nstates)  # No damping longitudinally
    D2 = torch.zeros(nstates)  # Complete transverse damping

    # Call the diffusion function
    output_states = epg.diffusion(states, D1, D2)

    # Check edge cases
    assert torch.all(output_states.Fplus == 0), "Fplus should be zero for D2=0"
    assert torch.all(output_states.Fminus == 0), "Fminus should be zero for D2=0"
    assert torch.all(output_states.Z == states.Z), "Z should remain unchanged for D1=1"
