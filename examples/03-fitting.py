"""
===========================
Fast parameter mapping/PERK
===========================

This example trains a compact PERK estimator on TorchSim-generated FSE
signals. Unlike exhaustive dictionary matching, inference has a fixed memory
footprint and consists only of a random-feature projection and matrix product.
"""

# %%
import matplotlib.pyplot as plt
import torch

from torchsim import (
    DictionaryMatcher,
    FSE,
    PERK,
    TissueProperties,
    fse_description,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(11)

# %%
# Build the acquisition once. SequenceDescription is the same representation
# accepted by the generic nonlinear simulator and Pulserver's scanner adapter.
echo_train_length = 32
description = fse_description(
    torch.deg2rad(torch.full((echo_train_length,), 150.0, device=device)),
    echo_spacing_s=5e-3,
    phases_rad=torch.pi / 2,
)


def signal_model(parameters: torch.Tensor, _known: torch.Tensor | None) -> torch.Tensor:
    """Vectorized nonlinear signal model used to synthesize PERK training data."""

    return FSE().simulate(
        description,
        TissueProperties(t1_ms=1000.0, t2_ms=parameters[:, 0]),
        nstates=12,
    ).signal


# %%
# Sample the prior over the unknown T2 parameter and fit PERK. Complex Gaussian
# noise is injected during training so the estimator learns the noisy inverse.
n_training = 8000
t2_training = 20.0 + 280.0 * torch.rand(
    n_training, 1, generator=generator, device=device
)
estimator = PERK(
    n_features=512,
    regularization=1e-5,
    chunk_size=2048,
    seed=4,
    complex_mode="magnitude",
).to(device)
estimator.fit_simulator(
    signal_model,
    t2_training,
    simulation_chunk_size=2048,
    noise_std=0.01,
)

# %%
# Parameter mapping is now a batched feed-forward operation. This is directly
# usable on a reconstructed image with shape ``(..., echoes)``.
t2_true = torch.linspace(25.0, 280.0, 256, device=device)[:, None]
signal = signal_model(t2_true, None)
signal += 0.01 * torch.complex(
    torch.randn(signal.shape, generator=generator, device=device),
    torch.randn(signal.shape, generator=generator, device=device),
)
t2_estimated = estimator(signal)

# Exhaustive matching remains useful as a transparent reference. Its score
# matrix is chunked, while each chunk is evaluated by BLAS/cuBLAS.
t2_dictionary = torch.linspace(20.0, 300.0, 4096, device=device)[:, None]
matcher = DictionaryMatcher(
    signal_model(t2_dictionary, None),
    t2_dictionary,
    dictionary_chunk_size=2048,
).to(device)
t2_matched = matcher(signal)

plt.plot(t2_true.cpu(), t2_estimated.detach().cpu(), label="PERK")
plt.plot(t2_true.cpu(), t2_matched.cpu(), label="dictionary")
plt.plot(t2_true.cpu(), t2_true.cpu(), "--", label="identity")
plt.xlabel("True T2 [ms]")
plt.ylabel("Estimated T2 [ms]")
plt.legend()
plt.tight_layout()
