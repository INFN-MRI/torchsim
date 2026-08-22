"""Test RF operators."""

import pytest
import torch
from types import SimpleNamespace

from utils import epg


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


def _matrix(operator):
    """The operator as a plain 3x3, for comparing whole rotations."""
    return torch.stack(
        [torch.stack([entry.reshape(()) for entry in row]) for row in operator]
    )


def _spinor(fa, phi):
    """The Cayley-Klein pair of an instantaneous pulse."""
    fa = torch.as_tensor(fa, dtype=torch.float32)
    phi = torch.as_tensor(phi, dtype=torch.float32)
    a = torch.cos(fa / 2).to(torch.complex64)
    b = -1j * torch.exp(-1j * phi.to(torch.complex64)) * torch.sin(fa / 2)
    return a, b


@pytest.mark.parametrize("fa_deg", [0.0, 17.0, 90.0, 180.0, 250.0])
@pytest.mark.parametrize("phi_deg", [0.0, 33.0, 90.0, 200.0])
def test_the_spinor_operator_is_the_phased_one_written_another_way(fa_deg, phi_deg):
    """The general form has to contain the instantaneous pulse as a case."""
    fa = torch.deg2rad(torch.tensor(fa_deg))
    phi = torch.deg2rad(torch.tensor(phi_deg))
    expected = _matrix(epg.phased_rf_pulse_op(fa, phi))
    actual = _matrix(epg.spinor_rf_pulse_op(*_spinor(fa, phi)))

    assert torch.allclose(expected, actual, atol=1e-6)


def test_the_spinor_operator_conserves_the_magnetization():
    """A rotation is unitary in the EPG basis under its own inner product.

    ``|F+|^2 + |F-|^2 + 2|Z|^2`` is what the (F+, F-, Z) basis makes of the
    length of the magnetization vector, so a genuine rotation preserves it and
    a matrix with a sign wrong in it does not.
    """
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        pair = torch.randn(4, generator=generator)
        a = torch.complex(pair[0], pair[1])
        b = torch.complex(pair[2], pair[3])
        scale = torch.sqrt(a.abs() ** 2 + b.abs() ** 2)
        a, b = a / scale, b / scale

        rotation = _matrix(epg.spinor_rf_pulse_op(a, b))
        state = torch.complex(
            torch.randn(3, generator=generator), torch.randn(3, generator=generator)
        )
        # F- is the conjugate of F+ for a physical state.
        state[1] = state[0].conj()
        state[2] = state[2].real + 0j
        turned = rotation @ state

        def length(value):
            return (
                value[0].abs() ** 2 + value[1].abs() ** 2 + 2.0 * value[2].abs() ** 2
            )

        assert torch.allclose(length(turned), length(state), atol=1e-4)


def test_a_pulse_split_in_two_is_the_two_pulses_composed():
    """SU(2) composes, so the operator built from the product is the product.

    This is what makes a shaped pulse expressible at all: its rotation is the
    ordered product of the rotations of its samples.
    """
    first, second = _spinor(0.7, 0.3), _spinor(1.1, -0.9)
    # (a, b) of the product of the two SU(2) elements, second acting last.
    a = second[0] * first[0] - second[1] * first[1].conj()
    b = second[0] * first[1] + second[1] * first[0].conj()

    composed = _matrix(epg.spinor_rf_pulse_op(a, b))
    stepwise = _matrix(epg.spinor_rf_pulse_op(*second)) @ _matrix(
        epg.spinor_rf_pulse_op(*first)
    )

    assert torch.allclose(composed, stepwise, atol=1e-6)


def _pair_cotangents(a, b, seed, state):
    """What a cotangent on the rotated states makes of the pair.

    Every entry of the operator is a product of two factors drawn from the pair
    and its conjugate, so the two Wirtinger halves are read off the outer
    product of the seed and the state the rotation acted on. The kernels'
    adjoints have to reproduce this, and a sign wrong anywhere in it is a
    gradient that is wrong only for shaped pulses -- which is exactly where
    nothing else would catch it.
    """
    m = torch.outer(seed.conj(), state)
    ca, cb = a.conj(), b.conj()
    holding_conj_a = 2 * a * m[1, 1] - 2 * b * m[1, 2] + cb * m[2, 1] + ca * m[2, 2]
    holding_a = 2 * ca * m[0, 0] - 2 * cb * m[0, 2] + b * m[2, 0] + a * m[2, 2]
    holding_conj_b = -2 * b * m[1, 0] - 2 * a * m[1, 2] + ca * m[2, 0] - cb * m[2, 2]
    holding_b = -2 * cb * m[0, 1] - 2 * ca * m[0, 2] + a * m[2, 1] - b * m[2, 2]
    return holding_conj_a.conj() + holding_a, holding_conj_b.conj() + holding_b


def _random_rotation(generator):
    pair = torch.randn(4, generator=generator, dtype=torch.float64)
    a = torch.complex(pair[0], pair[1])
    b = torch.complex(pair[2], pair[3])
    scale = torch.sqrt(a.abs() ** 2 + b.abs() ** 2)
    return (a / scale).detach(), (b / scale).detach()


def test_the_adjoint_sends_the_states_back_through_the_conjugate_transpose():
    generator = torch.Generator().manual_seed(3)
    for _ in range(10):
        a, b = _random_rotation(generator)
        state = torch.complex(
            torch.randn(3, generator=generator, dtype=torch.float64),
            torch.randn(3, generator=generator, dtype=torch.float64),
        ).requires_grad_(True)
        seed = torch.complex(
            torch.randn(3, generator=generator, dtype=torch.float64),
            torch.randn(3, generator=generator, dtype=torch.float64),
        )
        rotation = _matrix(epg.spinor_rf_pulse_op(a, b)).to(torch.complex128)
        loss = ((rotation @ state).conj() * seed).real.sum()

        assert torch.allclose(
            torch.autograd.grad(loss, state)[0], rotation.conj().T @ seed, atol=1e-12
        )


def test_the_adjoint_reaches_the_pair_in_closed_form():
    generator = torch.Generator().manual_seed(4)
    for _ in range(10):
        a0, b0 = _random_rotation(generator)
        state = torch.complex(
            torch.randn(3, generator=generator, dtype=torch.float64),
            torch.randn(3, generator=generator, dtype=torch.float64),
        )
        seed = torch.complex(
            torch.randn(3, generator=generator, dtype=torch.float64),
            torch.randn(3, generator=generator, dtype=torch.float64),
        )
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        rotation = _matrix(epg.spinor_rf_pulse_op(a, b)).to(torch.complex128)
        loss = ((rotation @ state).conj() * seed).real.sum()
        grad_a, grad_b = torch.autograd.grad(loss, (a, b))

        expected_a, expected_b = _pair_cotangents(a0, b0, seed, state)
        assert torch.allclose(grad_a, expected_a, atol=1e-12)
        assert torch.allclose(grad_b, expected_b, atol=1e-12)


def test_a_parameter_behind_the_pair_reaches_it_through_the_slope():
    """A flip angle moves both halves of the pair, and this is the chain."""
    generator = torch.Generator().manual_seed(5)
    for _ in range(10):
        a0, b0 = _random_rotation(generator)
        slope_a, slope_b = _random_rotation(generator)
        state = torch.complex(
            torch.randn(3, generator=generator, dtype=torch.float64),
            torch.randn(3, generator=generator, dtype=torch.float64),
        )
        seed = torch.complex(
            torch.randn(3, generator=generator, dtype=torch.float64),
            torch.randn(3, generator=generator, dtype=torch.float64),
        )
        theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
        rotation = _matrix(
            epg.spinor_rf_pulse_op(a0 + slope_a * theta, b0 + slope_b * theta)
        ).to(torch.complex128)
        loss = ((rotation @ state).conj() * seed).real.sum()

        grad_a, grad_b = _pair_cotangents(a0, b0, seed, state)
        expected = (grad_a.conj() * slope_a).real + (grad_b.conj() * slope_b).real
        assert torch.allclose(torch.autograd.grad(loss, theta)[0], expected, atol=1e-12)


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
