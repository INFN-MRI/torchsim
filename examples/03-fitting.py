"""
===============================================
MR fingerprinting, and the subspace it lives in
===============================================

MR fingerprinting drives the sequence hard on purpose. The flip angle changes
every repetition, so a voxel never reaches a steady state and its signal over
the train is a trajectory rather than a contrast -- one that depends on T1 and
T2 differently enough that both can be read from it at once.

Reading them back is where the cost is. A dictionary has to span every
combination of the parameters, so its size is the *product* of the grids, and
each atom is as long as the train. This example maps a brain slice from a
four-hundred-contrast train two ways -- exhaustive matching and kernel
regression -- and compresses both onto a low-rank temporal basis first.

The compression is the interesting part. Four hundred contrasts of a
relaxation-driven train do not span four hundred directions; they span about a
dozen. What that costs is a number the basis reports before anything is
projected through it.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim torchio

# %%
#
# The imports:
#
import warnings

warnings.filterwarnings("ignore")

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio

import torchsim
from torchsim import (
    Acquisition,
    DictionaryMatcher,
    PERK,
    ParameterMapping,
    Subspace,
)
from torchsim.simulators import MRFSimulator

# %%
#
# A brain to map
# --------------
#
# The anatomy is a slice from the IXI database; the relaxation times are not.
# A proton-density and a T2-weighted image give the shape of the head and a
# rough T2, and the voxels are sorted by it into three classes that then carry
# literature relaxation times at 3 T. That makes a digital phantom with real
# anatomy and known answers, which is what a mapping example needs: the truth
# has to come from somewhere other than the estimator being tested.
#
sample = tio.datasets.IXI(
    os.path.realpath("data"), modalities=("PD", "T2"), download=False
)[0]
proton_density = sample.PD.numpy().astype(np.float32).squeeze()[:, :, 60].T
t2_weighted = sample.T2.numpy().astype(np.float32).squeeze()[:, :, 60].T

with np.errstate(divide="ignore", invalid="ignore"):
    rough_t2 = np.nan_to_num(
        -92.0 / np.log(t2_weighted / proton_density), neginf=0.0, posinf=0.0
    ).clip(0.0, None)

proton_density = np.flip(proton_density)
rough_t2 = np.flip(rough_t2)

#                    T1 [ms]  T2 [ms]
TISSUE_CLASSES = {
    "white matter": (830.0, 80.0),
    "grey matter": (1330.0, 110.0),
    "CSF": (4000.0, 500.0),
}
EDGES = (1.0, 70.0, 150.0, np.inf)

T1_true = np.zeros_like(rough_t2)
T2_true = np.zeros_like(rough_t2)
for (name, (t1, t2)), low, high in zip(
    TISSUE_CLASSES.items(), EDGES[:-1], EDGES[1:], strict=True
):
    inside = (rough_t2 >= low) & (rough_t2 < high)
    T1_true[inside], T2_true[inside] = t1, t2
    print(f"{name:>13}: {100 * inside.mean():4.1f}% of the slice")

mask = T1_true > 0
M0_true = np.where(mask, proton_density / proton_density.max(), 0.0)

# %%
#
# The train
# ---------
#
# Four hundred repetitions after an inversion, at a fixed repetition time and a
# flip angle that varies smoothly along the train. Smooth is the point: a
# schedule that jumped about would give trajectories that differ by noise
# rather than by physics.
#
CONTRASTS = 400
TR_MS = 10.0
TI_MS = 20.0

repetition = torch.arange(CONTRASTS, dtype=torch.float32)
flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / CONTRASTS) ** 2

acquisition = Acquisition(
    MRFSimulator(flip=flip, TR=TR_MS, TI=TI_MS, states=20), M0=1.0
)

# %%
#
# What the classes look like through it. Three trajectories, three tissues, and
# the separation between them is what any estimator has to live on.
#
fingerprints = acquisition.simulate(
    T1=torch.tensor([t1 for t1, _ in TISSUE_CLASSES.values()]),
    T2=torch.tensor([t2 for _, t2 in TISSUE_CLASSES.values()]),
).abs()

figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(repetition.numpy(), flip.numpy())
axes[0].set(xlabel="Repetition", ylabel="Flip angle [deg]", title="the schedule")
axes[0].grid(alpha=0.3)
for row, name in enumerate(TISSUE_CLASSES):
    axes[1].plot(repetition.numpy(), fingerprints[row].numpy(force=True), label=name)
axes[1].set(xlabel="Repetition", ylabel="|signal|", title="fingerprints")
axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3)
figure.tight_layout()

# %%
#
# The measurement, with noise at 2% of the peak fingerprint. One number sets
# it, and the same number is what the estimators are told to expect -- an
# estimator trained for more noise than the scan has learns to distrust the
# data and answers with the prior instead.
#
NOISE_STD = float(0.02 * fingerprints.max())

truth = {
    "T1": torch.as_tensor(T1_true[mask].copy()),
    "T2": torch.as_tensor(T2_true[mask].copy()),
}
clean = acquisition.simulate(**truth)
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(
    clean.shape, generator=generator, dtype=clean.dtype
)
measured = measured * torch.as_tensor(M0_true[mask].copy())[:, None]

print(f"\n{measured.shape[0]} voxels of {CONTRASTS} contrasts")
print(f"noise standard deviation {NOISE_STD:.5f}")

# %%
#
# The problem, stated once
# ------------------------
#
# What is unknown, over what range, from what acquisition, at what noise level.
# The method that fills it in is a separate choice, and the only thing that
# changes between the three answers below.
#
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)

# Both relaxation times span more than a decade, so the prior is sampled
# logarithmically. Sampling uniformly would spend most of the budget on long
# T1, where the trajectories are nearly parallel and carry little information,
# and leave the short end underdetermined. A mapping takes values as readily as
# a range, which is how a prior like this one is given.
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)


def log_uniform(low, high, count):
    """``count`` draws spread evenly in the logarithm."""
    span = torch.rand(count, generator=prior)
    return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))


def mapping(**extra):
    """The same problem every time, with only the method left to choose."""
    return ParameterMapping(
        acquisition,
        T1=log_uniform(*T1_RANGE, SAMPLES),
        T2=log_uniform(*T2_RANGE, SAMPLES),
        noise_std=NOISE_STD,
        seed=0,
        **extra,
    )


# %%
#
# How many directions does the train actually span?
# -------------------------------------------------
#
# Fit a basis to a set of simulated trajectories and read off how much of their
# energy each rank keeps. This is not an estimate: one minus the fraction is
# the relative squared error of projecting those trajectories through the basis
# and back.
#
training_signals, _, _ = mapping().training_set(SAMPLES)

print("\n rank   retained energy")
for rank in (2, 4, 8, 16, 32):
    print(f"{rank:>5}   {Subspace.fit(training_signals, rank).retained:.9f}")

RANK = 16

# %%
#
# Sixteen directions out of four hundred, and what is left outside them is well
# under the noise the measurement already carries. Every contrast dropped is
# arithmetic that neither matching nor training has to do.
#
# Dictionary matching
# -------------------
#
# A dictionary must span the parameters jointly, so its size is the product of
# the grids. Here that is a modest two-dimensional grid; a third parameter --
# transmit, say, or a diffusion coefficient -- would multiply it again, which
# is the pressure the compression relieves.
#
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 60)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 40)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")
ATOMS = grid_t1.numel()

print(f"\ndictionary: {T1_GRID.numel()} T1 x {T2_GRID.numel()} T2 = {ATOMS} atoms")


def matched(rank):
    """Fit a matcher over the grid, then map the slice."""
    problem = ParameterMapping(
        acquisition,
        T1=grid_t1.reshape(-1),
        T2=grid_t2.reshape(-1),
        seed=0,
        **({"rank": rank} if rank else {}),
    )
    start = time.perf_counter()
    problem.train(DictionaryMatcher(), samples=ATOMS)
    training = time.perf_counter() - start
    start = time.perf_counter()
    with torchsim.execution():
        maps = problem(measured)
    return maps, training, time.perf_counter() - start


full_maps, full_training, full_matching = matched(None)
low_maps, low_training, low_matching = matched(RANK)

# %%
#
# PERK
# ----
#
# A kernel regression trained over the same ranges, drawn from a prior rather
# than a grid. It never sees a candidate parameter pair at inference: it maps
# the signal onto a fixed set of random Fourier features and reads the answer
# off a linear combination of them, so its cost per voxel is the same whatever
# the parameter space looks like.
#
# What that cost buys is set by how many features there are, and unlike the
# grid it is paid once, during training. Worth showing rather than asserting:
#


def regressed(features):
    """Train a regression of this size, then map the slice."""
    start = time.perf_counter()
    problem = mapping(rank=RANK).train(
        PERK(
            n_features=features, regularization=1e-6, normalize=True, seed=4
        ),
        samples=SAMPLES,
    )
    training = time.perf_counter() - start
    start = time.perf_counter()
    with torchsim.execution():
        maps = problem(measured)
    return maps, training, time.perf_counter() - start


regressions = {features: regressed(features) for features in (500, 1000, 4000)}
perk_maps, perk_training, perk_mapping = regressions[4000]

# %%
#
# What it cost, and what it got
# -----------------------------
#


def error(maps):
    """Median absolute percentage error against the phantom, per parameter."""
    return {
        name: float(((maps[name] - truth[name]).abs() / truth[name] * 100).median())
        for name in truth
    }


rows = [
    (f"matching, {CONTRASTS} contrasts", full_maps, full_training, full_matching),
    (f"matching, rank {RANK}", low_maps, low_training, low_matching),
]
rows += [
    (f"PERK, rank {RANK}, {features} features", *result)
    for features, result in regressions.items()
]

print(f"\n{'method':<34} {'train':>8} {'map':>8}   median error")
for label, maps, training, inference in rows:
    wrong = error(maps)
    print(
        f"{label:<34} {training:7.2f}s {inference:7.2f}s   "
        f"T1 {wrong['T1']:5.1f}%   T2 {wrong['T2']:5.1f}%"
    )

print(
    f"\ncompressing the match: {full_matching / low_matching:.1f}x faster, "
    f"on {CONTRASTS // RANK}x fewer numbers per atom"
)

# %%
#
# The maps
# --------
#


def unpack(values):
    """One flat map back into the slice it came from."""
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = values.numpy(force=True)
    return out


figure, axes = plt.subplots(2, 4, figsize=(13, 6.2))
panels = (
    ("truth", truth),
    (f"matching, {CONTRASTS}", full_maps),
    (f"matching, rank {RANK}", low_maps),
    (f"PERK, rank {RANK}", perk_maps),
)
for column, (title, maps) in enumerate(panels):
    for row, (name, limit) in enumerate((("T1", 4500.0), ("T2", 600.0))):
        drawn = axes[row, column].imshow(
            unpack(maps[name]), cmap="magma", vmin=0.0, vmax=limit
        )
        axes[row, column].set_title(f"{name}, {title}", fontsize=9)
        axes[row, column].axis("off")
        figure.colorbar(drawn, ax=axes[row, column], fraction=0.046)
figure.tight_layout()

# %%
#
# Reading the result honestly
# ---------------------------
#
# The compressed match is not an approximation of the full one that happens to
# come close. It is the same match taken in a basis that keeps essentially all
# of the signal, and the errors above say so.
#
# What it is *not* is free of the phantom's own structure. Three tissue classes
# make an easy problem, because every voxel's answer is one of three points and
# the dictionary contains all three almost exactly. A phantom with a continuum
# of relaxation times would separate the two methods further, and the grid
# spacing would be what limited matching.
#
# The regression is the less accurate of the two here, and the table says what
# buys that back: its error falls with the number of features while its cost
# per voxel barely moves, and every one of those features is paid for once,
# during training. Matching has no such knob -- its accuracy is the grid
# spacing, and the grid is also its cost.
#
# That is the durable part. Matching costs one inner product per atom per
# voxel, so it grows with the product of the parameter grids *and* with the
# length of the train. The subspace removes the second factor once and for
# all; nothing removes the first. Add a third parameter and the grid is
# multiplied again while the regression is not, which is why the argument for
# it strengthens with every parameter rather than weakening.
#
