"""
=================
Parameter Fitting
=================

This example shows how to use TorchSim to perform parameter inference.

We will build on the previous example: we synthesize an FSE echo series from
realistic maps, then recover T2 from it with two complementary estimators
shipped with TorchSim:

* :class:`torchsim.DictionaryMatcher`, the familiar exhaustive search;
* :class:`torchsim.PERK`, a kernel-regression estimator whose inference cost
  and memory footprint do not grow with the size of the parameter grid.

Both consume signals produced by the very same simulator, so the forward model
is written once and reused for training, matching, and validation.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim torchio sigpy

# %%
#
# We'll generate an FSE dataset from IXI database.
# We will neglect encoding and assume single coil for this case.
#
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import torchio as tio
import torchsim
from torchsim.simulators import FSESimulator

path = os.path.realpath("data")
ixi_dataset = tio.datasets.IXI(
    path,
    modalities=("PD", "T2"),
    download=False,
)

# get subject 0
sample_subject = ixi_dataset[0]

M0 = sample_subject.PD.numpy().astype(np.float32).squeeze()[:, :, 60].T
T2w = sample_subject.T2.numpy().astype(np.float32).squeeze()[:, :, 60].T

T2 = -92.0 / np.log(T2w / M0)
T2 = np.nan_to_num(T2, neginf=0.0, posinf=0.0)
T2 = np.clip(T2, a_min=0.0, a_max=np.inf)

M0 = np.flip(M0)
T2 = np.flip(T2)

# %%
#
# The maps contain air voxels with ``T2 == 0``. TorchSim maps a non-positive
# relaxation time to a fully dephased signal, so those voxels simply produce
# zero rather than a NaN that would spread through the reconstruction.
#
device = "cuda" if torch.cuda.is_available() else "cpu"
flip = 180.0 * np.ones(32, dtype=np.float32)
ESP = 5.0
T1 = 1000.0


def simulate(T2, flip, ESP, device="cpu"):
    ishape = T2.shape
    output = torchsim.fse_sim(
        flip=flip, ESP=ESP, T1=T1, T2=T2.flatten(), device=device
    )

    # T2 arrived as a NumPy array, so the signal comes back as one -- even
    # when the simulation itself ran on the GPU.
    return abs(output.T.reshape(-1, *ishape))


# simulate acquisition
echo_series = M0 * simulate(T2, flip.copy(), ESP, device=device)

# %%
#
# Measured data are noisy, and an estimator is only interesting insofar as it
# tolerates that noise. We add Gaussian noise at roughly 2% of the peak signal:
#
rng = np.random.default_rng(42)
noise_level = 0.02 * echo_series.max()
echo_series = echo_series + noise_level * rng.standard_normal(echo_series.shape)
echo_series = echo_series.astype(np.float32)

# display
img = np.concatenate((echo_series[0], echo_series[16], echo_series[-1]), axis=1)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(abs(img), cmap="gray"), plt.axis("image"), plt.axis("off")
plt.title("echoes 1, 17 and 32")

# %%
#
# One problem, stated once
# ------------------------
#
# What is being estimated, from what acquisition, at what noise level is one
# statement -- a :class:`torchsim.ParameterMapping` over an
# :class:`torchsim.Acquisition`. The method that fills it in is a separate
# choice, and swapping one for another is a word.
#
import time

acquisition = torchsim.Acquisition(
    FSESimulator(ESP=ESP, flip=flip), T1=T1
)

# %%
#
# Dictionary matching
# -------------------
#
# The reference approach evaluates every atom and keeps the best normalized
# inner product. A dictionary wants a *grid*, so that is what this mapping is
# trained over. :class:`torchsim.DictionaryMatcher` chunks the score matrix so
# memory stays bounded, while each chunk is a BLAS/cuBLAS matrix product.
#
t2_grid = torch.linspace(1.0, 350.0, 1000)

by_dictionary = torchsim.ParameterMapping(
    acquisition, T2=t2_grid, seed=0
).train(torchsim.DictionaryMatcher(), samples=t2_grid.numel())
matcher = by_dictionary.method

# %%
#
# The measured series is shaped ``(nechoes, ny, nx)``; a mapping expects the
# contrast axis last, and returns one map per unknown, under its own name.
#
signals = torch.as_tensor(echo_series, device=device).permute(1, 2, 0)

start = time.perf_counter()
T2_dict = by_dictionary(signals)["T2"].numpy(force=True)
dictionary_time = time.perf_counter() - start

# The scale is the complex least-squares fit between the measured signal and
# the matched atom, i.e. the proton density.
M0_dict = abs(matcher.match(signals).scales[..., 0]).numpy(force=True)

# %%
#
# PERK
# ----
#
# PERK regresses the parameter directly from a random Fourier feature map of
# the signal. It wants a *prior* rather than a grid: training draws T2 from
# it, simulates the corresponding signals, and adds noise so that the
# estimator learns the *noisy* inverse mapping. ``normalize=True`` makes the
# estimate invariant to the unknown proton density.
#
# The prior is sampled *log*-uniformly. Sampling it uniformly spends most of
# the training budget on long T2, where the echo train is nearly flat and
# carries little information, and leaves the short-T2 end underdetermined.
#
import math

generator = torch.Generator().manual_seed(11)
log_low, log_high = math.log(5.0), math.log(400.0)
t2_train = torch.exp(
    log_low + (log_high - log_low) * torch.rand(20000, generator=generator)
)

start = time.perf_counter()
by_regression = torchsim.ParameterMapping(
    acquisition, T2=t2_train, noise_std=0.02, seed=4
).train(
    torchsim.PERK(
        n_features=1000,
        regularization=1e-6,
        complex_mode="magnitude",
        normalize=True,
        seed=4,
    ),
    samples=t2_train.numel(),
)
estimator = by_regression.method
training_time = time.perf_counter() - start

# %%
#
# Inference is a fixed-cost feed-forward pass, independent of how finely the
# training prior was sampled.
#
# PERK is an unconstrained regression, so nothing stops it from returning a
# negative T2 in voxels where the parameter is not identifiable. Clamping to a
# strictly positive range is not cosmetic: the proton density below divides by
# the atom energy, which collapses to zero as T2 does.
#
start = time.perf_counter()
T2_estimate = by_regression(signals)["T2"].clamp(5.0, 350.0)

# %%
#
# PERK returns T2 only, but the proton density costs nothing extra. Since the
# atom for a given T2 is already tabulated in the dictionary above, we gather
# it by index and take the same least-squares scale that dictionary matching
# reports. This is a lookup plus one dot product per voxel, so no part of the
# forward model is simulated a second time.
#
index = torch.searchsorted(
    t2_grid.contiguous().to(T2_estimate.device),
    T2_estimate.reshape(-1).contiguous(),
).clamp(0, t2_grid.numel() - 1)
atoms = matcher.dictionary.to(T2_estimate.device)
atom = atoms[index]
measured = signals.reshape(-1, len(flip)).to(atom.device).to(atom.dtype)
M0_perk = (
    ((atom.conj() * measured).sum(-1) / atom.abs().square().sum(-1))
    .abs()
    .reshape(T2_estimate.shape)
    .numpy(force=True)
)
perk_time = time.perf_counter() - start
T2_perk = T2_estimate.numpy(force=True)

print(f"dictionary : {dictionary_time:.3f} s inference")
print(f"PERK       : {training_time:.3f} s training, {perk_time:.3f} s inference")

# %%
#
# Comparison
# ----------
#
# We restrict the display to voxels with signal, since T2 is not identifiable
# in air:
#
mask = M0 > 0.05 * M0.max()


def masked(x):
    return np.where(mask, x, np.nan)


# The reference T2 comes from a two-point log ratio, which is itself unstable:
# a few voxels (mostly CSF and partial-volume edges) land far outside the
# 1-350 ms range the estimators can represent. Quantitative comparison is only
# meaningful where the reference is inside that range.
valid = mask & (T2 > 1.0) & (T2 < 350.0)


figure, axes = plt.subplots(2, 3, figsize=(9.5, 6))
for ax, (data, title) in zip(
    axes[0],
    [(T2, "true T2 [ms]"), (T2_dict, "dictionary T2"), (T2_perk, "PERK T2")],
):
    handle = ax.imshow(masked(data), vmin=0.0, vmax=350.0, cmap="viridis")
    ax.set_title(title), ax.axis("off")
    figure.colorbar(handle, ax=ax, fraction=0.046)

# a shared scale, so the three proton-density panels are actually comparable
m0_max = np.nanpercentile(masked(M0), 99.5)
for ax, (data, title) in zip(
    axes[1],
    [(M0, "true M0"), (M0_dict, "dictionary M0"), (M0_perk, "PERK M0")],
):
    handle = ax.imshow(masked(data), cmap="gray", vmin=0.0, vmax=m0_max)
    ax.set_title(title), ax.axis("off")
    figure.colorbar(handle, ax=ax, fraction=0.046)
figure.tight_layout()

# %%
#
# Both estimators recover the same structure. Quantitatively, we compare them
# against the ground truth over the masked region:
#
for name, t2_estimate, m0_estimate in [
    ("dictionary", T2_dict, M0_dict),
    ("PERK", T2_perk, M0_perk),
]:
    t2_error = t2_estimate[valid] - T2[valid]
    m0_error = m0_estimate[mask] - M0[mask]
    print(
        f"{name:>10}:  T2 bias {t2_error.mean():+6.2f} ms "
        f"RMSE {np.sqrt((t2_error**2).mean()):5.2f} ms  |  "
        f"M0 bias {m0_error.mean():+6.2f} RMSE {np.sqrt((m0_error**2).mean()):6.2f}"
    )
relative = np.abs(M0_perk[mask] - M0_dict[mask]).mean() / np.abs(M0_dict[mask]).mean()
print(f"PERK vs dictionary M0: {100 * relative:.2f}% mean relative difference")

# %%
#
# Exhaustive matching is the more accurate of the two here, and on a
# thousand-atom grid it is also the faster one. Its cost, however, grows with
# the size of the grid, while PERK's does not: the cost of a finer or
# higher-dimensional parameter space is paid once, during training.
#
reference = T2[valid]
plt.figure(figsize=(5, 4))
plt.plot(reference[::37], T2_dict[valid][::37], ".", markersize=2, label="dictionary")
plt.plot(reference[::37], T2_perk[valid][::37], ".", markersize=2, label="PERK")
plt.plot([0, 350], [0, 350], "k--", linewidth=1, label="identity")
plt.xlabel("true T2 [ms]"), plt.ylabel("estimated T2 [ms]")
plt.xlim([0, 350]), plt.ylim([0, 350])
plt.legend(), plt.tight_layout()

# %%
#
# That trade-off is worth measuring rather than asserting. Matching the same
# image against progressively finer grids, against PERK's constant cost:
#
print(f"{'atoms':>8} {'matching':>10} {'vs PERK':>9}")
for n_atoms in [1000, 4000, 16000, 64000]:
    grid = torch.linspace(1.0, 350.0, n_atoms)
    scaling_matcher = torchsim.ParameterMapping(
        acquisition, T2=grid, seed=0
    ).train(torchsim.DictionaryMatcher(), samples=n_atoms).method
    scaling_matcher.match(signals[:1])  # warm up
    start = time.perf_counter()
    scaling_matcher.match(signals)
    elapsed = time.perf_counter() - start
    print(f"{n_atoms:>8} {elapsed:9.3f}s {elapsed / perk_time:8.2f}x")

# %%
#
# Matching stays flat while the score matrix still fits in cache and only then
# starts to scale with the grid, so where exactly the two cross depends on the
# machine and its BLAS. The shape of the two curves is the durable part: one
# grows with the parameter grid, the other does not.
#
# A single relaxation time is the case least favourable to PERK, since a
# thousand atoms is a small matrix product. The argument strengthens for joint
# T1/T2/B1 dictionaries, where the grid grows multiplicatively and exhaustive
# search stops being affordable at all.
