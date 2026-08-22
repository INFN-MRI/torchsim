"""Test relaxation operators."""

import torch
from types import SimpleNamespace

from utils import epg


# %% longitudinal
def test_longitudinal_relaxation_op():
    R1 = torch.tensor(0.5)
    time = torch.tensor(2.0)
    E1, rE1 = epg.longitudinal_relaxation_op(R1, time)

    # Expected results
    expected_E1 = torch.exp(-R1 * time)
    expected_rE1 = 1 - expected_E1

    assert torch.allclose(E1, expected_E1, atol=1e-6)
    assert torch.allclose(rE1, expected_rE1, atol=1e-6)

    # Test with batch inputs
    R1_batch = torch.tensor([0.5, 1.0])
    time_batch = torch.tensor([2.0, 3.0])
    E1_batch, rE1_batch = epg.longitudinal_relaxation_op(R1_batch, time_batch)

    expected_E1_batch = torch.exp(-R1_batch * time_batch)
    expected_rE1_batch = 1 - expected_E1_batch

    assert torch.allclose(E1_batch, expected_E1_batch, atol=1e-6)
    assert torch.allclose(rE1_batch, expected_rE1_batch, atol=1e-6)


def test_longitudinal_relaxation_exchange_op():
    weight = torch.tensor([0.5, 0.5])
    k = torch.tensor([[0, 0.2], [0.1, 0]])
    R1 = torch.tensor([0.5, 1.0])
    time = torch.tensor(2.0)
    E1, rE1 = epg.longitudinal_relaxation_exchange_op(weight, k, R1, time)

    # Shape validation
    assert E1.shape == k.shape
    assert rE1.shape == weight.shape

    # Numerical correctness can be tested with precomputed values or expected behavior
    assert E1.dtype == torch.complex64
    assert rE1.dtype == torch.complex64

    # Check with zero exchange rate
    k_zero = torch.zeros_like(k)
    E1_zero, rE1_zero = epg.longitudinal_relaxation_exchange_op(
        weight, k_zero, R1, time
    )
    expected_E1_zero, expected_rE1_zero = epg.longitudinal_relaxation_op(R1, time)

    assert torch.allclose(torch.diag(E1_zero.real), expected_E1_zero, atol=1e-6)
    assert torch.allclose(rE1_zero.real, weight * expected_rE1_zero, atol=1e-6)


def test_longitudinal_relaxation():
    Z = torch.tensor([1.0, 0.5])[None, :]
    states = SimpleNamespace(Z=Z.clone())
    E1 = torch.tensor(0.8)
    rE1 = torch.tensor(0.2)

    updated_states = epg.longitudinal_relaxation(states, E1, rE1)

    # Expected results
    expected_Z = Z.clone() * E1
    expected_Z[0] += rE1

    assert torch.allclose(updated_states.Z, expected_Z, atol=1e-6)


def test_longitudinal_relaxation_exchange():
    Z = torch.tensor([1.0, 0.5])[None, :]
    states = SimpleNamespace(Z=Z.clone())
    E1 = torch.tensor([[0.8, 0.1], [0.2, 0.9]])
    rE1 = torch.tensor([0.2, 0.1])

    updated_states = epg.longitudinal_relaxation_exchange(states, E1, rE1)

    # Expected results
    expected_Z = torch.einsum("...ij,...j->...i", E1, Z.clone())
    expected_Z[0] += rE1

    assert torch.allclose(updated_states.Z, expected_Z, atol=1e-6)


def test_longitudinal_edge_cases():
    # Test with zero R1
    R1_zero = torch.tensor(0.0)
    time = torch.tensor(1.0)
    E1, rE1 = epg.longitudinal_relaxation_op(R1_zero, time)

    assert torch.allclose(E1, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(rE1, torch.tensor(0.0), atol=1e-6)

    # Test with zero time
    R1 = torch.tensor(0.5)
    time_zero = torch.tensor(0.0)
    E1, rE1 = epg.longitudinal_relaxation_op(R1, time_zero)

    assert torch.allclose(E1, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(rE1, torch.tensor(0.0), atol=1e-6)

    # Test with large time (approaching steady-state)
    large_time = torch.tensor(1e6)
    E1, rE1 = epg.longitudinal_relaxation_op(R1, large_time)

    assert torch.allclose(E1, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(rE1, torch.tensor(1.0), atol=1e-6)


# %% transverse
def test_transverse_relaxation_op():
    R2 = torch.tensor(0.5)
    time = torch.tensor(2.0)
    E2 = epg.transverse_relaxation_op(R2, time)

    # Expected results
    expected_E2 = torch.exp(-R2 * time)

    assert torch.allclose(E2, expected_E2, atol=1e-6)

    # Test with batch inputs
    R2_batch = torch.tensor([0.5, 1.0])
    time_batch = torch.tensor([2.0, 3.0])
    E2_batch = epg.transverse_relaxation_op(R2_batch, time_batch)

    expected_E2_batch = torch.exp(-R2_batch * time_batch)

    assert torch.allclose(E2_batch, expected_E2_batch, atol=1e-6)


def test_transverse_relaxation_exchange_op():
    k = torch.tensor([[0, 0.2], [0.1, 0]])
    R2 = torch.tensor([0.5, 1.0])
    time = torch.tensor(2.0)
    df = torch.tensor([10.0, 20.0])  # Chemical shifts
    E2 = epg.transverse_relaxation_exchange_op(k, R2, time, df)

    # Shape validation
    assert E2.shape == k.shape

    # Numerical correctness can be tested against known results or expected behavior
    assert E2.dtype == torch.complex64

    # Test with zero exchange rate and zero frequency offset
    k_zero = torch.zeros_like(k)
    df_zero = torch.zeros_like(df)
    E2_zero = epg.transverse_relaxation_exchange_op(k_zero, R2, time, df_zero)
    expected_E2_zero = torch.exp(-R2 * time)

    assert torch.allclose(torch.diag(E2_zero.real), expected_E2_zero, atol=1e-6)
    assert torch.allclose(
        torch.diag(E2_zero.imag), torch.zeros_like(E2_zero.imag), atol=1e-6
    )


def test_transverse_relaxation():
    states = SimpleNamespace(
        Fplus=torch.tensor([1.0, 0.5]), Fminus=torch.tensor([0.8, 0.4])
    )
    E2 = torch.tensor(0.8)

    updated_states = epg.transverse_relaxation(states, E2)

    # Expected results
    expected_Fplus = torch.tensor([1.0, 0.5]) * E2
    expected_Fminus = torch.tensor([0.8, 0.4]) * E2

    assert torch.allclose(updated_states.Fplus, expected_Fplus, atol=1e-6)
    assert torch.allclose(updated_states.Fminus, expected_Fminus, atol=1e-6)


def test_transverse_relaxation_exchange():
    Fplus = torch.tensor([1.0, 0.5])[None, ...]
    Fminus = torch.tensor([0.8, 0.4])[None, ...]
    states = SimpleNamespace(Fplus=Fplus.clone(), Fminus=Fminus.clone())
    E2 = torch.tensor([[0.8, 0.1], [0.2, 0.9]])

    updated_states = epg.transverse_relaxation_exchange(states, E2)

    # Expected results
    expected_Fplus = torch.einsum("...ij,...j->...i", E2, Fplus.clone())
    expected_Fminus = torch.einsum("...ij,...j->...i", E2.conj(), Fminus.clone())

    assert torch.allclose(updated_states.Fplus, expected_Fplus, atol=1e-6)
    assert torch.allclose(updated_states.Fminus, expected_Fminus, atol=1e-6)


def test_transverse_edge_cases():
    # Test with zero R2
    R2_zero = torch.tensor(0.0)
    time = torch.tensor(1.0)
    E2 = epg.transverse_relaxation_op(R2_zero, time)

    assert torch.allclose(E2, torch.tensor(1.0), atol=1e-6)

    # Test with zero time
    R2 = torch.tensor(0.5)
    time_zero = torch.tensor(0.0)
    E2 = epg.transverse_relaxation_op(R2, time_zero)

    assert torch.allclose(E2, torch.tensor(1.0), atol=1e-6)

    # Test with large time (approaching steady-state)
    large_time = torch.tensor(1e6)
    E2 = epg.transverse_relaxation_op(R2, large_time)

    assert torch.allclose(E2, torch.tensor(0.0), atol=1e-6)

    # Test with zero exchange rate and frequency offset
    k = torch.zeros((2, 2))
    df = torch.zeros(2)
    time = torch.tensor(2.0)
    R2 = torch.tensor([0.5, 1.0])

    E2 = epg.transverse_relaxation_exchange_op(k, R2, time, df)
    expected_E2 = torch.diag(torch.exp(-R2 * time)).to(torch.complex64)

    assert torch.allclose(E2, expected_E2, atol=1e-6)


def test_dtype_and_shapes():
    # Validate data types and shapes
    R2 = torch.tensor([0.5, 1.0])
    time = torch.tensor(2.0)
    k = torch.zeros((2, 2))
    df = torch.tensor([10.0, 20.0])

    E2 = epg.transverse_relaxation_exchange_op(k, R2, time, df)

    assert E2.dtype == torch.complex64
    assert E2.shape == k.shape
