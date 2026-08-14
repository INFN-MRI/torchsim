"""Test RF operators."""

import pytest
import torch
from types import SimpleNamespace

from torchsim import epg


# Fixtures for common test inputs
@pytest.fixture
def states_fixture():
    return SimpleNamespace(
        Fplus=torch.tensor([0.0, 0.0, 0.0]),
        Fminus=torch.tensor([0.0, 0.0, 0.0]),
        Z=torch.tensor([1.0, 1.0, 1.0]),
    )


@pytest.fixture
def RF_fixture():
    T = torch.eye(3, dtype=torch.float32)
    RF = [
        [T[0][0][..., None], T[0][1][..., None], T[0][2][..., None]],
        [T[1][0][..., None], T[1][1][..., None], T[1][2][..., None]],
        [T[2][0][..., None], T[2][1][..., None], T[2][2][..., None]],
    ]
    return RF  # Identity RF operation


# Test functions
def test_rf_pulse_op():
    fa = torch.tensor(0.5)  # 0.5 radians
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)

    RF = epg.rf_pulse_op(fa, slice_prof, B1)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)


def test_phased_rf_pulse_op():
    fa = torch.tensor(0.5)
    phi = torch.tensor(0.2)
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)
    B1phase = torch.tensor(0.1)

    RF = epg.phased_rf_pulse_op(fa, phi, slice_prof, B1, B1phase)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)


def test_multidrive_rf_pulse_op():
    fa = torch.tensor([0.3, 0.4])
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor([0.8, 0.9])

    RF = epg.multidrive_rf_pulse_op(fa, slice_prof, B1)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)


def test_an_in_phase_array_drives_the_sum_of_its_channels():
    """Two channels are one channel at their combined weight."""
    fa = torch.tensor([0.3, 0.4])
    B1 = torch.tensor([0.8, 0.9])
    together = epg.multidrive_rf_pulse_op(fa, torch.tensor(1.0), B1)
    alone = epg.rf_pulse_op((B1 * fa).sum(), torch.tensor(1.0), 1.0)

    for left, right in zip(together, alone):
        for one, other in zip(left, right):
            assert torch.allclose(one, other, atol=1e-6)


def test_channels_in_antiphase_leave_the_voxel_untouched():
    """The check that separates a complex sum from two independent ones.

    Summing the magnitudes and the phases apart from each other cannot
    cancel, and would turn this into a rotation of the full flip angle.
    """
    fa = torch.tensor([0.5, 0.5])
    phi = torch.tensor([0.0, torch.pi])
    B1 = torch.tensor([1.0, 1.0])
    B1phase = torch.zeros(2)

    RF, _phi = epg.phased_multidrive_rf_pulse_op(
        fa, phi, torch.tensor(1.0), B1, B1phase
    )

    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for row, reference in zip(RF, identity):
        for element, value in zip(row, reference):
            assert torch.allclose(
                element, torch.as_tensor(value, dtype=element.dtype), atol=1e-6
            )


def test_a_transmit_phase_turns_the_axis_without_changing_the_flip():
    """A phase common to every channel rotates the drive, nothing more."""
    fa = torch.tensor([0.3, 0.4])
    B1 = torch.tensor([0.8, 0.9])
    turn = 0.7

    RF, _ = epg.phased_multidrive_rf_pulse_op(
        fa, torch.full((2,), turn), torch.tensor(1.0), B1, torch.zeros(2)
    )
    reference = epg.phased_rf_pulse_op(
        (B1 * fa).sum(), torch.tensor(turn), torch.tensor(1.0), 1.0
    )

    for left, right in zip(RF, reference):
        for one, other in zip(left, right):
            assert torch.allclose(one, other, atol=1e-6)


def test_a_single_channel_array_is_the_single_channel_pulse():
    fa = torch.tensor([0.4])
    phi = torch.tensor([0.25])
    B1 = torch.tensor([0.9])
    B1phase = torch.tensor([0.15])

    RF, net = epg.phased_multidrive_rf_pulse_op(
        fa, phi, torch.tensor(1.0), B1, B1phase
    )
    reference = epg.phased_rf_pulse_op(
        B1 * fa, phi + B1phase, torch.tensor(1.0), 1.0
    )

    for left, right in zip(RF, reference):
        for one, other in zip(left, right):
            assert torch.allclose(one, other, atol=1e-6)
    # The demodulation reference is the phase that was asked for, which the
    # transmit field's own phase does not enter.
    assert torch.allclose(net, phi.reshape(()), atol=1e-6)


def test_phased_multidrive_rf_pulse_op():
    fa = torch.tensor([0.3, 0.4])
    phi = torch.tensor([0.1, 0.2])
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor([0.8, 0.9])
    B1phase = torch.tensor([0.05, 0.07])

    RF, phi_out = epg.phased_multidrive_rf_pulse_op(fa, phi, slice_prof, B1, B1phase)

    assert len(RF) == 3
    assert len(RF[0]) == 3
    assert isinstance(RF[0][0], torch.Tensor)
    commanded = (fa * torch.exp(1j * phi)).sum()
    assert torch.allclose(phi_out, commanded.angle())


def test_initialize_mt_sat():
    duration = torch.tensor(0.001)  # 1 ms
    b1rms = torch.tensor(0.05)  # Tesla
    df = torch.tensor(0.0)
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)

    WT = epg.initialize_mt_sat(duration, b1rms, df, slice_prof, B1)

    assert isinstance(WT, torch.Tensor)


def test_mt_sat_op():
    WT = torch.tensor(-0.01)
    fa = torch.tensor(0.5)
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor(0.9)

    exp_WT = epg.mt_sat_op(WT, fa, slice_prof, B1)

    assert isinstance(exp_WT, torch.Tensor)


def test_multidrive_mt_sat_op():
    WT = torch.tensor(-0.01)
    fa = torch.tensor([0.3, 0.4])
    slice_prof = torch.tensor(1.0)
    B1 = torch.tensor([0.8, 0.9])

    exp_WT = epg.multidrive_mt_sat_op(WT, fa, slice_prof, B1)

    assert isinstance(exp_WT, torch.Tensor)


def test_rf_pulse(states_fixture, RF_fixture):
    states = states_fixture
    RF = RF_fixture

    states_out = epg.rf_pulse(states, RF)

    assert torch.allclose(states_out.Fplus, states.Fplus)
    assert torch.allclose(states_out.Fminus, states.Fminus)
    assert torch.allclose(states_out.Z, states.Z)


def test_mt_sat(states_fixture):
    states = states_fixture
    Z = states.Z.clone()
    S = torch.tensor(0.9)

    states_out = epg.mt_sat(states, S)

    expected_Z = Z * S
    assert torch.allclose(states_out.Z, expected_Z)
