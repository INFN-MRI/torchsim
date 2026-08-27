"""
=========================================
Kernel regression, with no grid at all
=========================================

A dictionary spans the parameters jointly, so its size is the product of the
grids and a third parameter multiplies it again. PERK never builds one. It is a
kernel regression, trained on signals drawn from a prior rather than laid on a
grid, and at inference it maps a signal onto a fixed set of random Fourier
features and reads the answer off a linear combination of them -- so its cost
per voxel is the same whatever the parameter space looks like.

What that cost buys is set by how many features there are, and unlike a grid it
is paid once, during training.

This example maps a brain slice from a four-hundred-contrast MR fingerprinting
train, sweeps the size of the regression, and puts a compressed dictionary match
beside it so that the trade is a table rather than an argument.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim brainweb-dl

# %%
#
# The phantom is BrainWeb's, reached through ``brainweb-dl``: ``get_mri``
# fetches the fuzzy tissue memberships, and the package ships the table of
# relaxation times that goes with them -- which is what the two standard
# library imports read.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# sphinx_gallery_end_ignore
import csv
from pathlib import Path

import brainweb_dl
from brainweb_dl import get_mri

# %%
#
# The problem is stated over a simulator carrying the sequence and filled in
# by an estimator. :func:`~torchsim.execution` decides where that work runs,
# and the timings below are taken inside it.
#
import time

import numpy as np
import torch

import torchsim
from torchsim import ParameterMapping, Subspace
from torchsim.estimators import PERK, DictionaryMatcher
from torchsim.simulators import MRFSimulator

# %%
#
# A brain to map
# --------------
#
# BrainWeb publishes its phantom as *fuzzy* tissue memberships: every voxel
# carries a fraction of each tissue rather than a label. Those fractions,
# weighted by the relaxation times BrainWeb tabulates for each tissue, give
# maps with the structure of a brain and answers that are known.
#
# The fractions matter more than the anatomy. Roughly a third of the brain
# voxels in this slice are mixtures of two tissues or more, so the truth is a
# continuum rather than a handful of values, and an estimator cannot do well
# merely by having seen the right three answers.
#
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
tissues = list(csv.DictReader(table.open()))
tissue_T1 = np.array([float(row["T1 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_T2 = np.array([float(row["T2 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(row["PD (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior, and an image is
# drawn from its first row down. Flipping here puts anterior at the top of
# every figure below rather than in each one of them.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5

# A mixed voxel is given the relaxation times its tissues average to. That is
# the parameter a fit can actually return: no single T1 explains a voxel that
# is half one tissue and half another, and pretending otherwise would make the
# error being reported partly the phantom's.
share = np.maximum(occupancy, 1e-6)
T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
T2_true = np.where(mask, fractions @ tissue_T2 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

mixed = ((fractions.max(-1) < 0.99) & mask).sum()
print(f"slice {fractions.shape[:2]}: {mask.sum()} brain voxels")
print(f"{100 * mixed / mask.sum():.0f}% of them are mixtures of two tissues or more")
print(f"T1 {T1_true[mask].min():.0f}-{T1_true[mask].max():.0f} ms, "
      f"T2 {T2_true[mask].min():.0f}-{T2_true[mask].max():.0f} ms")

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

acquisition = MRFSimulator(flip=flip, TR=TR_MS, TI=TI_MS, states=20, M0=1.0)

# %%
#
# The readouts wind the states on rather than rewinding them, so nothing
# returns transverse magnetization to the imaginary axis and the trajectory is
# real. That is worth checking rather than assuming, because it halves both
# the dictionary and the arithmetic that searches it.
#
probe = acquisition.simulate(
    T1=torch.tensor([500.0, 833.0, 2569.0]), T2=torch.tensor([70.0, 83.0, 329.0])
)
print(f"\nsimulated dtype {probe.dtype}")
print(f"largest imaginary part: {float(probe.imag.abs().max()):.3g}")

fingerprints = probe.real

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(repetition.numpy(), flip.numpy())
axes[0].set(xlabel="Repetition", ylabel="Flip angle [deg]", title="the schedule")
axes[0].grid(alpha=0.3)
for row, name in enumerate(("white matter", "grey matter", "CSF")):
    axes[1].plot(repetition.numpy(), fingerprints[row].numpy(force=True), label=name)
axes[1].set(xlabel="Repetition", ylabel="signal", title="fingerprints")
axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3)
figure.tight_layout()
# sphinx_gallery_end_ignore

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
density = torch.as_tensor(M0_true[mask].copy())
clean = acquisition.simulate(**truth).real * density[:, None]
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)

print(f"\n{measured.shape[0]} voxels of {CONTRASTS} contrasts, "
      f"{measured.dtype}")
print(f"noise standard deviation {NOISE_STD:.5f}")

# %%
#
# What a route costs
# ------------------
#
# Timings below are the best of three passes, and exclude the one-off
# measurement :func:`~torchsim.execution` makes the first time it meets a
# workload -- that is amortized over every volume a protocol is ever used on,
# rather than paid per slice.
#
# Two memory numbers are worth separating. The **model** is what the fitted
# estimator holds and carries between volumes. The **peak** is the high-water
# mark of what the card held while the slice was being mapped, which is what
# decides whether a whole volume fits or has to be streamed -- and a route that
# :func:`~torchsim.execution` judged not worth a launch never crosses to the
# card at all, so its peak is near zero and its cost is on the host.
#


def footprint(problem):
    """MiB the fitted estimator itself holds -- a dictionary, or a regression."""
    held = sum(t.numel() * t.element_size() for t in problem.method.buffers())
    return held / 2**20


def mapped(problem, passes=3):
    """Map the slice a few times: the quickest pass, and what it held."""
    on_device = torch.cuda.is_available()
    with torchsim.execution():
        problem(measured[:64])
        if on_device:
            torch.cuda.reset_peak_memory_stats()
        best = float("inf")
        for _ in range(passes):
            start = time.perf_counter()
            maps = problem(measured)
            best = min(best, time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated() / 2**20 if on_device else float("nan")
    return maps, best, peak


# %%
#
# The problem, stated once
# ------------------------
#
# What is unknown, over what range, from what acquisition, at what noise level.
# The method that fills it in is a separate choice, and the only thing that
# changes between the answers below.
#
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)

# Both relaxation times span more than a decade, so the prior is sampled
# logarithmically. Sampling uniformly would spend most of the budget on long
# T1, where the trajectories are nearly parallel and carry little information,
# and leave the short end underdetermined.
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)


def log_uniform(low, high, count):
    """``count`` draws spread evenly in the logarithm."""
    span = torch.rand(count, generator=prior)
    return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))


# %%
#
# The basis the regression works in
# ---------------------------------
#
# Four hundred contrasts of a relaxation-driven train do not span four hundred
# directions. Fitting a basis to simulated trajectories says how many they do
# span, and one minus the energy it keeps is the relative squared error of
# projecting a trajectory through it and back -- a number, not an estimate.
#
training_signals, _, _ = ParameterMapping(
    acquisition,
    T1=log_uniform(*T1_RANGE, SAMPLES),
    T2=log_uniform(*T2_RANGE, SAMPLES),
    noise_std=NOISE_STD,
    seed=0,
).training_set(SAMPLES)
training_signals = training_signals.real

print("\n rank   retained energy   left outside")
for rank in (2, 4, 8, 16, 32):
    retained = Subspace.fit(training_signals, rank).retained
    print(f"{rank:>5}   {retained:.9f}   {1 - retained:.2e}")

RANK = 4

# %%
#
# Four directions already leave less outside the basis than the noise puts in.
# Both methods below are given that same basis, so what the table compares is
# the estimator rather than the compression.
#
# Training a regression
# ---------------------
#
# The training set is drawn from the prior, not from a grid: twenty thousand
# parameter pairs spread logarithmically over the ranges, simulated and given
# the noise the scan has. Fitting is a linear solve against the features, and
# it is the whole of what the method costs before it ever sees a voxel.
#


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


def regressed(features):
    """Train a regression of this size, then map the slice."""
    start = time.perf_counter()
    problem = mapping(rank=RANK).train(
        PERK(n_features=features, regularization=1e-6, normalize=True, seed=4),
        samples=SAMPLES,
    )
    training = time.perf_counter() - start
    maps, timing, peak = mapped(problem)
    return maps, training, timing, footprint(problem), peak


regressions = {features: regressed(features) for features in (500, 1000, 4000)}
FEATURES = 1000
perk_maps, perk_training, perk_mapping, perk_model, perk_peak = regressions[FEATURES]

# %%
#
# A dictionary, for scale
# -----------------------
#
# The same problem given to a compressed dictionary match over a grid fine
# enough that the grid spacing is not what limits the answer. Two parameters is
# the case least favourable to a regression, because it is the case where a
# grid is still small.
#
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 200)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 100)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")

matched = ParameterMapping(
    acquisition,
    T1=grid_t1.reshape(-1),
    T2=grid_t2.reshape(-1),
    rank=RANK,
    seed=0,
)
start = time.perf_counter()
matched.train(DictionaryMatcher())
match_training = time.perf_counter() - start
match_maps, match_mapping, match_peak = mapped(matched)
match_model = footprint(matched)

print(f"\ndictionary: {grid_t1.numel()} atoms at rank {RANK}")
print(f"regression: {FEATURES} features, {SAMPLES} training draws")

# %%
#
# Proton density, for nothing extra
# ---------------------------------
#
# Neither method estimates M0, and neither has to. Both answer with relaxation
# times, and a fingerprint at those times is a shape the measurement is some
# multiple of -- so the multiple is a projection, one inner product per voxel.
#


def proton_density(maps):
    """The scale the measurement is, of the fingerprint the answer predicts."""
    predicted = acquisition.simulate(T1=maps["T1"], T2=maps["T2"]).real
    return (predicted * measured).sum(-1) / predicted.square().sum(-1).clamp_min(1e-12)


estimates = {
    f"match, rank {RANK}": (match_maps, proton_density(match_maps)),
    f"PERK, {FEATURES} features": (perk_maps, proton_density(perk_maps)),
}

# %%
#
# What it cost, and what it got
# -----------------------------
#


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


print(f"\n{'method':<26}{'train':>8}{'map':>8}{'model':>10}{'peak':>10}"
      f"{'T1':>8}{'T2':>8}")
print("-" * 78)
rows = [(f"match, rank {RANK}", match_training, match_mapping,
         match_model, match_peak, match_maps)]
for features, (maps, training, timing, model, peak) in regressions.items():
    rows.append((f"PERK, {features} features", training, timing, model, peak, maps))

for name, training, timing, model, peak, maps in rows:
    print(f"{name:<26}{training:7.1f}s{timing:7.2f}s"
          f"{model:6.1f} MiB{peak:6.0f} MiB"
          f"{error(maps['T1'], truth['T1']):7.1f}%"
          f"{error(maps['T2'], truth['T2']):7.1f}%")

# %%
#
# The maps
# --------
#


def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


panels = [
    ("T1 [ms]", T1_true, {name: maps["T1"] for name, (maps, _) in estimates.items()},
     (0, 3000)),
    ("T2 [ms]", T2_true, {name: maps["T2"] for name, (maps, _) in estimates.items()},
     (0, 350)),
    ("M0", M0_true, {name: m0 for name, (_, m0) in estimates.items()}, (0, 1.1)),
]

# sphinx_gallery_start_ignore
def panel(axis, values, cmap, limits, label=None):
    """One map, with a colorbar every panel loses the same width to.

    A figure where only some panels carry a colorbar has panels of different
    sizes. Giving every one a colorbar and hiding the ones that would repeat a
    scale keeps the images comparable.
    """
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    bar = axis.figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
    if label is None:
        bar.ax.set_visible(False)
    else:
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


columns = 1 + len(estimates)
rows = len(panels)
figure, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.3 * rows))
for row, (label, reference, found, limits) in enumerate(panels):
    panel(axes[row, 0], reference, "magma", limits)
    axes[row, 0].set_ylabel(label, fontsize=11)
    axes[row, 0].set_title("truth" if row == 0 else "", fontsize=10)
    for column, (name, values) in enumerate(found.items(), start=1):
        panel(
            axes[row, column],
            painted(values),
            "magma",
            limits,
            label=label if column == columns - 1 else None,
        )
        if row == 0:
            axes[row, column].set_title(name, fontsize=9)
figure.tight_layout()

# The errors, each parameter on a scale of its own: an error map read at the
# scale of the map it came from is a black rectangle.
figure, axes = plt.subplots(
    rows, len(estimates), figsize=(3.3 * len(estimates), 3.3 * rows), squeeze=False
)
for row, (label, reference, found, limits) in enumerate(panels):
    residuals = {
        name: np.abs(painted(values) - reference) for name, values in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
    for column, (name, values) in enumerate(residuals.items()):
        panel(
            axes[row, column],
            values,
            "inferno",
            (0.0, top or 1.0),
            label=f"|error|, {label}" if column == len(residuals) - 1 else None,
        )
        if row == 0:
            axes[row, column].set_title(name, fontsize=9)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# Reading the result honestly
# ---------------------------
#
# On two parameters the regression is the less accurate of the two and has no
# speed to offer in exchange: a compressed match is both quicker and closer.
# That is the case a grid is best at, and the comparison is being run where it
# favours the grid.
#
# What the sweep shows is the part that carries over. The cost per voxel is
# almost unmoved by the size of the model -- eight times the features changes
# the mapping time by less than the measurement is worth -- and that cost is
# paid once, during training.
#
# The sweep also shows where a regression stops improving. From five hundred
# features to a thousand the error roughly halves; from a thousand to four
# thousand it does not fall at all. Twenty thousand training samples are being
# asked to determine a four-thousand-by-four-thousand covariance, which is
# about five samples per direction, and no amount of regularization makes that
# a well-posed estimate. More features want more training, and training is
# simulation.
#
# The scaling is what decides between them. A match costs one inner product per
# atom per voxel and the atom count is the product of the parameter grids, so a
# third parameter multiplies it; a regression's inference cost does not move at
# all. The argument for regression strengthens with every parameter added
# rather than weakening.
#
