"""End-to-end comparison of PERK and dictionary-based FSE mapping."""

from __future__ import annotations

import torch

from torchsim.estimators import DictionaryMatcher, PERK
from torchsim.simulators import FSESimulator


def test_perk_is_comparable_to_dictionary_matching_under_noise() -> None:
    generator = torch.Generator().manual_seed(4)
    sequence = FSESimulator(
        flip=torch.full((16,), 150.0), ESP=5.0, phases=90.0, states=10
    )

    def simulate(t2_ms: torch.Tensor) -> torch.Tensor:
        return sequence.simulate(T1=1000.0, T2=t2_ms.reshape(-1))

    training_t2 = 20.0 + 280.0 * torch.rand(3000, 1, generator=generator)
    test_t2 = 20.0 + 280.0 * torch.rand(512, 1, generator=generator)
    clean = simulate(test_t2)
    noise_std = 0.02
    measured = clean + noise_std * torch.complex(
        torch.randn(clean.shape, generator=generator),
        torch.randn(clean.shape, generator=generator),
    )

    grid = torch.linspace(20.0, 300.0, 1024)[:, None]
    matcher = DictionaryMatcher(
        dictionary=simulate(grid),
        parameters=grid,
        dictionary_chunk_size=256,
    )
    perk = PERK(
        sequence.bind(T1=1000.0),
        n_features=256,
        chunk_size=512,
        feature_seed=4,
        complex_mode="magnitude",
        stream=True,
    ).fit(T2=training_t2, noise_std=noise_std)

    perk_rmse = (perk.map(measured)["T2"] - test_t2.reshape(-1)).square().mean().sqrt()
    dictionary_rmse = (matcher(measured) - test_t2).square().mean().sqrt()

    assert perk_rmse <= dictionary_rmse
