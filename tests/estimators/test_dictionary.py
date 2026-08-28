"""Tests for memory-bounded dictionary matching."""

from __future__ import annotations

import pytest
import torch

from torchsim.estimators import DictionaryMatcher
from torchsim.simulators import MultiEchoSimulator


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
def test_dictionary_matcher_recovers_complex_scaled_atoms(device: str) -> None:
    parameter = torch.linspace(0.1, 2.0, 67, device=device)[:, None]
    dictionary = torch.cat(
        (torch.exp(1j * parameter), torch.exp(2j * parameter)), dim=-1
    )
    scale = torch.tensor([2.0 - 0.5j, -0.2 + 1.1j], device=device)
    selected = torch.tensor([3, 52], device=device)
    signals = scale[:, None] * dictionary[selected]
    matcher = DictionaryMatcher(
        dictionary=dictionary,
        parameters=parameter,
        query_chunk_size=1,
        dictionary_chunk_size=11,
        top_k=2,
    )

    result = matcher.match(signals)

    torch.testing.assert_close(result.indices[:, 0], selected)
    torch.testing.assert_close(result.parameters[:, 0, 0], parameter[selected, 0])
    torch.testing.assert_close(result.scales[:, 0], scale)
    torch.testing.assert_close(result.scores[:, 0], torch.ones(2, device=device))


def test_dictionary_matcher_forward_returns_indices_without_parameters() -> None:
    dictionary = torch.eye(4)
    matcher = DictionaryMatcher(dictionary=dictionary, dictionary_chunk_size=2)

    actual = matcher(dictionary[[2, 0]])

    torch.testing.assert_close(actual, torch.tensor([2, 0]))


def test_a_complex_measurement_of_a_real_model_keeps_both_its_parts() -> None:
    """The phase a voxel carries must not decide whether it matches.

    A model that is real is stored real, and half the arithmetic of a complex
    one. The measurement of it is not: proton density and receive phase put a
    complex scale on every voxel. Reading only the real part leaves
    ``Re(rho)`` times the signal, and normalizing puts the size back -- so on
    noiseless data it matches anyway, and the mistake does not show. What it
    also multiplies by ``Re(rho)`` is the signal-to-noise ratio, and a voxel
    whose phase approaches a quarter turn is then matching noise.
    """
    grid = torch.linspace(20.0, 300.0, 128)
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 16))
    atoms = torch.as_tensor(acquisition.simulate(T2=grid))
    assert not torch.is_complex(atoms)
    matcher = DictionaryMatcher(dictionary=atoms, parameters=grid[:, None])
    truth = grid[::7]
    generator = torch.Generator().manual_seed(0)
    clean = torch.as_tensor(acquisition.simulate(T2=truth)).to(torch.complex64)
    noise = 0.01 * torch.complex(
        torch.randn(clean.shape, generator=generator),
        torch.randn(clean.shape, generator=generator),
    )
    step = float(torch.diff(grid).max())

    for turn in (0.0, 0.15, 0.25, 0.5):
        phase = torch.exp(2j * torch.pi * torch.tensor(turn))

        found = matcher(clean * phase + noise)[:, 0]

        error = float((found - truth).abs().mean())
        assert error < step, f"at {turn} turns the match drifted {error:.1f} ms"
