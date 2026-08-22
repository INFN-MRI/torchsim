"""Test ADC operations."""

import pytest
import torch
from types import SimpleNamespace

from utils.epg import get_signal, get_demodulated_signal


@pytest.fixture
def sample_states():
    """Fixture to provide a sample EPG states object."""
    dtype = torch.complex64
    device = torch.device("cpu")
    Fplus = torch.tensor(
        [
            [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]],
            [[0 + 1j, 1 + 0j], [1 + 2j, 2 + 1j], [2 + 3j, 3 + 2j]],
            [[0 + 0j, 0 + 0j], [1 + 1j, 1 + 1j], [2 + 2j, 2 + 2j]],
            [[1 + 1j, 1 + 1j], [2 + 2j, 2 + 2j], [3 + 3j, 3 + 3j]],
            [[4 + 4j, 4 + 4j], [5 + 5j, 5 + 5j], [6 + 6j, 6 + 6j]],
        ],
        dtype=dtype,
        device=device,
    )

    Fminus = torch.zeros_like(Fplus)
    Z = torch.zeros_like(Fplus)

    return SimpleNamespace(Fplus=Fplus, Fminus=Fminus, Z=Z)


def test_get_signal_single_order(sample_states):
    order = 0
    result = get_signal(sample_states, order)
    expected = (3 + 3j + 7 + 7j + 11 + 11j) / 3
    assert result == pytest.approx(expected), "Signal mismatch for single order"


def test_get_signal_multiple_orders(sample_states):
    orders = [0, 1]
    result = get_signal(sample_states, orders)
    expected_order_0 = (3 + 3j + 7 + 7j + 11 + 11j) / 3
    expected_order_1 = (1 + 1j + 3 + 3j + 5 + 5j) / 3
    expected = expected_order_0 + expected_order_1
    assert result == pytest.approx(expected), "Signal mismatch for multiple orders"


def test_get_signal_invalid_order(sample_states):
    with pytest.raises(IndexError):
        get_signal(sample_states, 10)  # Out-of-bounds order


@pytest.mark.parametrize(
    "phi,expected_phase",
    [
        (torch.tensor(0.0), 1),  # No phase shift
        (torch.tensor(torch.pi / 2), -1j),  # 90-degree phase shift
        (torch.tensor(torch.pi), -1),  # 180-degree phase shift
    ],
)
def test_get_demodulated_signal_single_order(sample_states, phi, expected_phase):
    order = 0
    result = get_demodulated_signal(sample_states, phi, order)
    expected = (
        +(3 + 3j) * expected_phase
        + (7 + 7j) * expected_phase
        + (11 + 11j) * expected_phase
    ) / 3
    assert result == pytest.approx(
        expected
    ), f"Demodulated signal mismatch for phi={phi}"


@pytest.mark.parametrize(
    "phi,expected_phase",
    [
        (torch.tensor(0.0), 1),  # No phase shift
        (torch.tensor(torch.pi / 2), -1j),  # 90-degree phase shift
        (torch.tensor(torch.pi), -1),  # 180-degree phase shift
    ],
)
def test_get_demodulated_signal_multiple_orders(sample_states, phi, expected_phase):
    orders = [0, 1]
    result = get_demodulated_signal(sample_states, phi, orders)

    expected_order_0 = (
        +(3 + 3j) * expected_phase
        + (7 + 7j) * expected_phase
        + (11 + 11j) * expected_phase
    ) / 3

    expected_order_1 = (
        +(1 + 1j) * expected_phase
        + (3 + 3j) * expected_phase
        + (5 + 5j) * expected_phase
    ) / 3

    expected = expected_order_0 + expected_order_1
    assert result == pytest.approx(
        expected
    ), "Demodulated signal mismatch for multiple orders"


def test_get_demodulated_signal_invalid_order(sample_states):
    phi = torch.tensor(torch.pi / 4)
    with pytest.raises(IndexError):
        get_demodulated_signal(sample_states, phi, 10)  # Out-of-bounds order
