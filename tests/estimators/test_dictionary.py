"""Tests for memory-bounded dictionary matching."""

from __future__ import annotations

import pytest
import torch

from torchsim.estimators import DictionaryMatcher


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
        dictionary,
        parameter,
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
    matcher = DictionaryMatcher(dictionary, dictionary_chunk_size=2)

    actual = matcher(dictionary[[2, 0]])

    torch.testing.assert_close(actual, torch.tensor([2, 0]))
