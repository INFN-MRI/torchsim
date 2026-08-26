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
four-hundred-contrast train three ways -- exhaustive matching, matching with
the dictionary clustered, and kernel regression -- and gives all three the same
low-rank temporal basis to work in.

The compression is where to start. Four hundred contrasts of a
relaxation-driven train do not span four hundred directions; they span a
handful. What that costs is a number the basis reports before anything is
projected through it.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim brainweb-dl

# %%
#
# The imports:
#
import warnings

warnings.filterwarnings("ignore")

import csv
import time
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
import numpy as np
import torch
from brainweb_dl import get_mri

import torchsim
from torchsim import Acquisition, ParameterMapping, Subspace
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

acquisition = Acquisition(
    MRFSimulator(flip=flip, TR=TR_MS, TI=TI_MS, states=20), M0=1.0
)

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

figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(repetition.numpy(), flip.numpy())
axes[0].set(xlabel="Repetition", ylabel="Flip angle [deg]", title="the schedule")
axes[0].grid(alpha=0.3)
for row, name in enumerate(("white matter", "grey matter", "CSF")):
    axes[1].plot(repetition.numpy(), fingerprints[row].numpy(force=True), label=name)
axes[1].set(xlabel="Repetition", ylabel="signal", title="fingerprints")
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
density = torch.as_tensor(M0_true[mask].copy())
clean = acquisition.simulate(**truth).real * density[:, None]
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)

print(f"\n{measured.shape[0]} voxels of {CONTRASTS} contrasts, "
      f"{measured.dtype}")
print(f"noise standard deviation {NOISE_STD:.5f}")

# %%
#
# Timings below are the best of three passes, and exclude the one-off
# measurement :func:`~torchsim.execution` makes the first time it meets a
# workload -- that is amortized over every volume a protocol is ever used on,
# rather than paid per slice.
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


def fastest(problem, passes=3):
    """Map the slice a few times and keep the quickest, with the maps."""
    with torchsim.execution():
        problem(measured[:64])
        best = float("inf")
        for _ in range(passes):
            start = time.perf_counter()
            maps = problem(measured)
            best = min(best, time.perf_counter() - start)
    return maps, best


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
training_signals = training_signals.real

print("\n rank   retained energy   left outside")
for rank in (2, 4, 8, 16, 32):
    retained = Subspace.fit(training_signals, rank).retained
    print(f"{rank:>5}   {retained:.9f}   {1 - retained:.2e}")

RANK = 4

# %%
#
# Four directions out of four hundred already leave less outside the basis than
# the noise puts in. There is nothing to be gained by keeping more, and every
# contrast dropped is arithmetic that neither training nor matching has to do.
#
# Dictionary matching
# -------------------
#
# A dictionary must span the parameters jointly, so its size is the product of
# the grids -- twenty thousand atoms for two parameters on a grid fine enough
# that the grid spacing is not what limits the answer. A third parameter would
# multiply it again, which is the pressure everything below is relieving.
#
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 200)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 100)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")
ATOMS = grid_t1.numel()

print(f"\ndictionary: {T1_GRID.numel()} T1 x {T2_GRID.numel()} T2 = {ATOMS} atoms")
print(f"stored at full length: {ATOMS * CONTRASTS * 4 / 2**20:7.2f} MiB")
print(f"stored at rank {RANK}:      {ATOMS * RANK * 4 / 2**20:7.2f} MiB")


def matched(rank=None, groups=None):
    """Fit a matcher over the grid, then map the slice."""
    problem = ParameterMapping(
        acquisition,
        T1=grid_t1.reshape(-1),
        T2=grid_t2.reshape(-1),
        seed=0,
        **({"rank": rank} if rank else {}),
    )
    start = time.perf_counter()
    problem.train(DictionaryMatcher(groups=groups))
    training = time.perf_counter() - start
    maps, timing = fastest(problem)
    return problem, maps, training, timing


_, full_maps, full_training, full_matching = matched()
low_problem, low_maps, low_training, low_matching = matched(rank=RANK)

# %%
#
# Clustering the dictionary
# -------------------------
#
# Compressing shortened every inner product. Grouping cuts how many are taken:
# neighbouring tissues make nearly parallel signals, so the atoms cluster, and
# a voxel matched against one representative signal per group can rule out most
# of the groups before scoring a single atom inside them.
#
# The clustering happens *in the compressed basis* -- compress first, then
# cluster -- so a group is entered without leaving the space the measurement is
# already in. The two savings are independent and multiply.
#
GROUPS = 32
group_problem, group_maps, group_training, group_matching = matched(
    rank=RANK, groups=GROUPS
)

matcher = group_problem.method
grouping = matcher.grouping
normalized = low_problem.subspace.project(measured)
normalized = normalized / normalized.norm(dim=-1, keepdim=True)
survivors = grouping.survivors(normalized, matcher.prune)
print(f"\n{GROUPS} groups of {grouping.sizes[0]} atoms")
print(f"condition number of the representatives: {grouping.condition:.1f}")
print(f"groups still open per voxel: {float(survivors.sum(-1).float().mean()):.1f}"
      f" of {GROUPS}")

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
# grid it is paid once, during training.
#


def regressed(features):
    """Train a regression of this size, then map the slice."""
    start = time.perf_counter()
    problem = mapping(rank=RANK).train(
        PERK(n_features=features, regularization=1e-6, normalize=True, seed=4),
        samples=SAMPLES,
    )
    training = time.perf_counter() - start
    maps, timing = fastest(problem)
    return maps, training, timing


regressions = {features: regressed(features) for features in (500, 1000, 4000)}
FEATURES = 1000
perk_maps, perk_training, perk_mapping = regressions[FEATURES]

# %%
#
# Proton density, for nothing extra
# ---------------------------------
#
# Neither method estimates M0, and neither has to. Both answer with relaxation
# times, and a fingerprint at those times is a shape the measurement is some
# multiple of -- so the multiple is a projection, one inner product per voxel.
# It is the same step that turns an MP2RAGE T1 map into an M0 map.
#


def proton_density(maps):
    """The scale the measurement is, of the fingerprint the answer predicts."""
    predicted = acquisition.simulate(T1=maps["T1"], T2=maps["T2"]).real
    return (predicted * measured).sum(-1) / predicted.square().sum(-1).clamp_min(1e-12)


estimates = {
    "matched": (low_maps, proton_density(low_maps)),
    "grouped": (group_maps, proton_density(group_maps)),
    "PERK": (perk_maps, proton_density(perk_maps)),
}

# %%
#
# What it cost, and what it got
# -----------------------------
#


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


print(f"\n{'method':<22}{'train':>8}{'map':>8}{'T1':>8}{'T2':>8}{'M0':>8}")
print("-" * 62)
rows = [
    ("match, 400 contrasts", full_training, full_matching, full_maps, None),
    (f"match, rank {RANK}", low_training, low_matching, low_maps, estimates["matched"][1]),
    (f"match, rank {RANK} + groups", group_training, group_matching, group_maps,
     estimates["grouped"][1]),
]
for features, (maps, training, timing) in regressions.items():
    rows.append((f"PERK, {features} features", training, timing, maps,
                 proton_density(maps) if features == FEATURES else None))

for name, training, timing, maps, m0 in rows:
    line = (f"{name:<22}{training:7.1f}s{timing:7.2f}s"
            f"{error(maps['T1'], truth['T1']):7.1f}%"
            f"{error(maps['T2'], truth['T2']):7.1f}%")
    print(line + (f"{error(m0, density):7.1f}%" if m0 is not None else f"{'--':>8}"))

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

figure, axes = plt.subplots(3, 4, figsize=(12, 9.5))
for row, (label, reference, found, limits) in enumerate(panels):
    axes[row, 0].imshow(reference.T, cmap="magma", vmin=limits[0], vmax=limits[1])
    axes[row, 0].set_ylabel(label, fontsize=11)
    axes[row, 0].set_title("truth" if row == 0 else "", fontsize=10)
    for column, (name, values) in enumerate(found.items(), start=1):
        axes[row, column].imshow(
            painted(values).T, cmap="magma", vmin=limits[0], vmax=limits[1]
        )
        if row == 0:
            axes[row, column].set_title(name, fontsize=10)
for axis in axes.ravel():
    axis.set_xticks([]), axis.set_yticks([])
figure.tight_layout()

# %%
#
# Reading the result honestly
# ---------------------------
#
# The compressed match is not an approximation of the full one that happens to
# come close. It is the same match taken in a basis that keeps essentially all
# of the signal, and the errors above say so. Clustering that basis is a
# further claim -- that the groups ruled out could not have held the match --
# and it is the one of the three that can be wrong: a voxel at the edge of the
# parameter grid can have the group holding its answer pruned away. Widening
# ``prune`` is what buys it back, and costs time.
#
# T2 is the harder of the two here, and the reason is the phantom rather than
# the estimators. A third of these voxels are tissue mixtures, and a mixture's
# averaged T2 sits between two values that the train separates well, in a part
# of the range where the trajectories are close together. That is a real
# feature of partial volume, not an artefact of the method: no fit of a
# single-compartment model returns the average of a mixture exactly.
#
# The regression is the less accurate of the two families here, and on two
# parameters it has no speed to offer in exchange: a compressed, clustered
# match is both quicker and closer. Its cost per voxel is, though, almost
# unmoved by the size of the model -- eight times the features changes the
# mapping time by less than the measurement is worth -- and that cost is paid
# once, during training.
#
# The feature sweep also shows where a regression stops improving. From five
# hundred features to a thousand the error roughly halves; from a thousand to
# four thousand it does not fall at all. Twenty thousand training samples are
# being asked to determine a four-thousand-by-four-thousand covariance, which
# is about five samples per direction, and no amount of regularization makes
# that a well-posed estimate. More features want more training, and training
# is simulation.
#
# That is the durable part. Matching costs one inner product per atom per
# voxel, so it grows with the product of the parameter grids *and* with the
# length of the train. The subspace removes the second factor once and for all
# and the grouping removes most of the first; nothing removes the grid itself.
# Add a third parameter and it is multiplied again while the regression is not,
# which is why the argument for regression strengthens with every parameter
# rather than weakening -- and why the comparison above, on two, is the case
# least favourable to it.
#
