"""Test flow dephasing and washout operators."""

import pytest

import torch
from types import SimpleNamespace

from utils import epg


@pytest.fixture
def flow_fixture():
    """Fixture for creating a common set of input EPG states."""
    return SimpleNamespace(
        Fplus=torch.tensor([1.0 + 0j, 0.5 + 0.1j, 0.2 - 0.1j]),
        Fminus=torch.tensor([0.9 + 0.2j, 0.4 - 0.3j, 0.1 + 0.1j]),
        Z=torch.tensor([0.8 + 0j, 0.6 + 0.1j, 0.3 - 0.1j]),
    )


# Test for flow_op
def test_flow_op():
    v = torch.tensor(0.5)  # Spin velocity in m/s
    time = torch.tensor(2.0)  # Time interval in seconds
    nstates = 3
    total_dephasing = torch.tensor(4.0)  # Gradient dephasing in radians
    voxelsize = 1.0

    # Call the flow_op function
    J1, J2 = epg.flow_op(v, time, nstates, total_dephasing, voxelsize)

    # Assert output shapes
    assert J1.shape == (nstates, 1, 1)
    assert J2.shape == (nstates, 1, 1)

    # Assert values (complex exponentials)
    assert torch.allclose(
        torch.abs(J1), torch.ones_like(J1).real
    ), "J1 magnitude should be 1"
    assert torch.allclose(
        torch.abs(J2), torch.ones_like(J2).real
    ), "J2 magnitude should be 1"


# Test for flow
def test_flow(flow_fixture):
    states = flow_fixture
    Fplus = states.Fplus.clone()
    Fminus = states.Fminus.clone()
    Z = states.Z.clone()

    nstates = len(states.Fplus)
    J1 = torch.exp(
        -1j * torch.arange(nstates, dtype=torch.float32)
    )  # Longitudinal dephasing
    J2 = torch.exp(
        -1j * (torch.arange(nstates, dtype=torch.float32) + 0.5)
    )  # Transverse dephasing

    # Call the flow function
    output_states = epg.flow(states, J1, J2)

    # Assert output shapes are the same as input
    assert output_states.Fplus.shape == Fplus.shape
    assert output_states.Fminus.shape == Fminus.shape
    assert output_states.Z.shape == Z.shape

    # Assert correct dephasing
    assert torch.allclose(output_states.Fplus, Fplus * J2), "Fplus dephasing incorrect"
    assert torch.allclose(
        output_states.Fminus, Fminus * J2.conj()
    ), "Fminus dephasing incorrect"
    assert torch.allclose(output_states.Z, Z * J1), "Z dephasing incorrect"


# Test for washout_op
def test_washout_op():
    v = torch.tensor(1.0)  # Spin velocity in m/s
    time = torch.tensor(2.0)  # Time interval in seconds
    voxelsize = 1.0

    # Call the washout_op function
    Win, Wout = epg.washout_op(v, time, voxelsize)

    # Assert output shapes
    assert Win.shape == Wout.shape, "Win and Wout should have the same shape"

    # Assert values
    assert torch.all(Win >= 0) and torch.all(Win <= 1), "Win should be between 0 and 1"
    assert torch.all(Wout >= 0) and torch.all(
        Wout <= 1
    ), "Wout should be between 0 and 1"
    assert torch.allclose(Win + Wout, torch.ones_like(Win)), "Win + Wout should equal 1"

    # Check boundary case: v=0 should produce Wout=1 and Win=0
    Win_zero, Wout_zero = epg.washout_op(torch.tensor(0.0), time, voxelsize)
    assert torch.allclose(
        Win_zero, torch.zeros_like(Win_zero)
    ), "Win should be 0 for v=0"
    assert torch.allclose(
        Wout_zero, torch.ones_like(Wout_zero)
    ), "Wout should be 1 for v=0"


# Test for washout
def test_washout(flow_fixture):
    states = flow_fixture
    Fplus = states.Fplus.clone()
    Fminus = states.Fminus.clone()
    Z = states.Z.clone()

    moving_states = SimpleNamespace(
        Fplus=torch.tensor([0.2 + 0j, 0.3 + 0.1j, 0.4 - 0.2j]),
        Fminus=torch.tensor([0.1 + 0.2j, 0.2 - 0.3j, 0.3 + 0.3j]),
        Z=torch.tensor([0.5 + 0j, 0.4 + 0.2j, 0.2 - 0.1j]),
    )
    Win = torch.tensor([0.1, 0.2, 0.3])  # Inflow operator
    Wout = torch.tensor([0.9, 0.8, 0.7])  # Wash-out operator

    # Call the washout function
    output_states = epg.washout(states, moving_states, Win, Wout)

    # Assert output shapes are the same as input
    assert output_states.Fplus.shape == Fplus.shape
    assert output_states.Fminus.shape == Fminus.shape
    assert output_states.Z.shape == Z.shape

    # Assert correct washout/inflow
    assert torch.allclose(
        output_states.Fplus, Wout * Fplus + Win * moving_states.Fplus
    ), "Fplus washout/inflow incorrect"
    assert torch.allclose(
        output_states.Fminus, Wout * Fminus + Win * moving_states.Fminus
    ), "Fminus washout/inflow incorrect"
    assert torch.allclose(
        output_states.Z, Wout * Z + Win * moving_states.Z
    ), "Z washout/inflow incorrect"
