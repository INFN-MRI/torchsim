"""End-to-end comparison of PERK and dictionary-based FSE mapping."""

from __future__ import annotations

import torch

from torchsim import EpgEngine, fse_description, TissueProperties
from torchsim.estimators import DictionaryMatcher, PERK


def test_perk_is_comparable_to_dictionary_matching_under_noise() -> None:
    generator = torch.Generator().manual_seed(4)
    description = fse_description(
        torch.deg2rad(torch.full((16,), 150.0)),
        5e-3,
        phases_rad=torch.pi / 2,
    )

    def simulate(t2_ms: torch.Tensor) -> torch.Tensor:
        return EpgEngine().simulate(
            description,
            TissueProperties(t1_ms=1000.0, t2_ms=t2_ms.reshape(-1)),
            nstates=10,
        ).signal

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
        simulate(grid),
        grid,
        dictionary_chunk_size=256,
    )
    perk = PERK(
        n_features=256,
        chunk_size=512,
        seed=4,
        complex_mode="magnitude",
    ).fit_simulator(
        lambda parameters, _known: simulate(parameters[:, 0]),
        training_t2,
        noise_std=noise_std,
    )

    perk_rmse = (perk(measured) - test_t2).square().mean().sqrt()
    dictionary_rmse = (matcher(measured) - test_t2).square().mean().sqrt()

    assert perk_rmse <= dictionary_rmse
