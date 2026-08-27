"""
=====================================================
Dictionary matching, compressed and clustered
=====================================================

MR fingerprinting drives the sequence hard on purpose. The flip angle changes
every repetition, so a voxel never reaches a steady state and its signal over
the train is a trajectory rather than a contrast -- one that depends on T1 and
T2 differently enough that both can be read from it at once.

Reading them back is where the cost is. A dictionary has to span every
combination of the parameters, so its size is the *product* of the grids, and
each atom is as long as the train. This example maps a brain slice from a
four-hundred-contrast train by exhaustive matching, and then relieves that cost
twice over: once by working in the low-rank basis the train actually spans, and
once by clustering the dictionary so that most atoms are never scored.

The two savings are independent and they multiply. What each costs in time and
in memory, and what each gets wrong, is read off the same slice.
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
from torchsim.estimators import DictionaryMatcher
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
# How many directions does the train actually span?
# -------------------------------------------------
#
# Fit a basis to a set of simulated trajectories and read off how much of their
# energy each rank keeps. This is not an estimate: one minus the fraction is
# the relative squared error of projecting those trajectories through the basis
# and back.
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
# Four directions out of four hundred already leave less outside the basis than
# the noise puts in. There is nothing to be gained by keeping more, and every
# contrast dropped is arithmetic that neither training nor matching has to do.
#
# The dictionary
# --------------
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
    maps, timing, peak = mapped(problem)
    return problem, maps, training, timing, footprint(problem), peak


_, full_maps, full_training, full_matching, full_model, full_peak = matched()
low_problem, low_maps, low_training, low_matching, low_model, low_peak = matched(
    rank=RANK
)

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
(
    group_problem,
    group_maps,
    group_training,
    group_matching,
    group_model,
    group_peak,
) = matched(rank=RANK, groups=GROUPS)

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
# Proton density, for nothing extra
# ---------------------------------
#
# The match answers with relaxation times, and a fingerprint at those times is
# a shape the measurement is some multiple of -- so the multiple is a
# projection, one inner product per voxel. It is the same step that turns an
# MP2RAGE T1 map into an M0 map.
#


def proton_density(maps):
    """The scale the measurement is, of the fingerprint the answer predicts."""
    predicted = acquisition.simulate(T1=maps["T1"], T2=maps["T2"]).real
    return (predicted * measured).sum(-1) / predicted.square().sum(-1).clamp_min(1e-12)


estimates = {
    "match, 400 contrasts": (full_maps, proton_density(full_maps)),
    f"match, rank {RANK}": (low_maps, proton_density(low_maps)),
    f"match, rank {RANK} + groups": (group_maps, proton_density(group_maps)),
}

# %%
#
# What it cost, and what it got
# -----------------------------
#


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


print(f"\n{'method':<28}{'train':>8}{'map':>8}{'model':>10}{'peak':>10}"
      f"{'T1':>8}{'T2':>8}{'M0':>8}")
print("-" * 88)
rows = [
    ("match, 400 contrasts", full_training, full_matching, full_model, full_peak),
    (f"match, rank {RANK}", low_training, low_matching, low_model, low_peak),
    (f"match, rank {RANK} + groups", group_training, group_matching,
     group_model, group_peak),
]
for name, training, timing, model, peak in rows:
    maps, m0 = estimates[name]
    print(f"{name:<28}{training:7.1f}s{timing:7.2f}s"
          f"{model:6.1f} MiB{peak:6.0f} MiB"
          f"{error(maps['T1'], truth['T1']):7.1f}%"
          f"{error(maps['T2'], truth['T2']):7.1f}%"
          f"{error(m0, density):7.1f}%")

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
            axes[row, column].set_title(name, fontsize=9)
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
# the estimator. A third of these voxels are tissue mixtures, and a mixture's
# averaged T2 sits between two values that the train separates well, in a part
# of the range where the trajectories are close together. That is a real
# feature of partial volume, not an artefact of the method: no fit of a
# single-compartment model returns the average of a mixture exactly.
#
# The memory column is not simply the dictionary's size. Streaming sizes its
# chunk against a budget, so a shorter atom mostly buys a larger chunk rather
# than a smaller high-water mark; what does move the mark is the grouping,
# because a pruned voxel never materializes the scores of the atoms it ruled
# out.
#
# The durable part is the scaling. Matching costs one inner product per atom
# per voxel, so it grows with the product of the parameter grids *and* with the
# length of the train. The subspace removes the second factor once and for all
# and the grouping removes most of the first; nothing removes the grid itself.
# Add a third parameter and it is multiplied again -- which is the argument for
# the methods that never build a grid at all.
#
