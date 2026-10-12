"""
===================
Dictionary matching
===================

You map T1, T2 and proton density in a brain slice from a magnetic resonance
fingerprinting (MRF) train by dictionary matching, then reduce the cost of the
match in two independent ways: by working in the low-rank subspace spanned by
the train, and by clustering the dictionary so that most atoms are not scored.

Prerequisites: Course lesson :doc:`../01-framework/01-first-simulation`.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install blochsim matplotlib brainweb-dl cmap
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# sphinx_gallery_start_ignore
import csv
import time
import warnings
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
from brainweb_dl import get_mri
from cmap import Colormap

import blochsim
from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")

# One perceptually uniform colormap per relaxation parameter (Fuderer et al.,
# Magn. Reson. Med. 2025), so that a T1 map is not read as a T2 map. Both
# relaxation windows stop short of CSF: white and grey matter take up most of
# the scale and CSF saturates.
STYLE = {
    "T1": (Colormap("crameri:lipari").to_matplotlib(), (0.0, 1200.0), "T1 (ms)"),
    "T2": (Colormap("crameri:navia").to_matplotlib(), (0.0, 120.0), "T2 (ms)"),
    "M0": ("gray", (0.0, 1.0), "M0 (a.u.)"),
}
BAR_WIDTH = 0.8  # inches taken by one colorbar
PANEL = (PAGE_WIDTH - BAR_WIDTH) / 4  # side of one image panel, inches


def panel(axis, values, cmap, limits, title=None, ylabel=None):
    """One map without ticks; the handle is what a row shares a colorbar from."""
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    return handle


def scalebar(handle, axes, label):
    """One colorbar for a group of panels."""
    axes = list(np.ravel(axes))
    axes[0].figure.colorbar(handle, ax=axes, label=label, shrink=0.92, aspect=20)


def canvas(rows, columns, shape, *, bars=1, extra=0.6):
    """A grid of image panels in the proportion of the images."""
    return plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(
            columns * PANEL + bars * BAR_WIDTH,
            PANEL * shape[0] / shape[1] * rows + extra,
        ),
    )


def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask] = values.numpy(force=True)
    return image


def footprint(problem):
    """MiB the fitted estimator holds."""
    held = sum(t.numel() * t.element_size() for t in problem.buffers())
    return held / 2**20


def mapped(problem, passes=3):
    """Map the slice a few times: the quickest pass, and the peak GPU memory."""
    on_device = torch.cuda.is_available()
    with blochsim.execution():
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


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


def held(megabytes):
    """Memory in MiB, or a dash on a machine without a GPU."""
    return f"{megabytes:6.0f} MiB" if np.isfinite(megabytes) else f"{'--':>10}"


def log_uniform(low, high, count):
    """``count`` draws uniform in the logarithm, from the generator ``prior``."""
    span = torch.rand(count, generator=prior)
    return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))


# sphinx_gallery_end_ignore
import numpy as np
import torch

from blochsim import Subspace
from blochsim.estimators import DictionaryMatcher
from blochsim.simulators import MRFSimulator

# %%
# Phantom
# -------
#
# The phantom is BrainWeb subject 0, slice 90: an axial 1 mm slice through the
# lateral ventricles. BrainWeb provides fuzzy tissue memberships, so each voxel
# holds a fraction of each tissue and its relaxation times are the
# fraction-weighted average. About a third of the voxels are mixtures, so the
# ground truth is a continuum of values rather than four.

# sphinx_gallery_start_ignore
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
tissues = list(csv.DictReader(table.open()))
tissue_T1 = np.array([float(row["T1 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_T2 = np.array([float(row["T2 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(row["PD (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior; flipping puts
# anterior at the top of every figure.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5

# A mixed voxel takes the average of its tissues' parameters, the value a fit
# can return: no single T1 describes a voxel that is half one tissue and half
# another.
share = np.maximum(occupancy, 1e-6)
T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
T2_true = np.where(mask, fractions @ tissue_T2 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

MAPS = (
    ("T1", T1_true),
    ("T2", T2_true),
    ("M0", M0_true),
)
figure, axes = canvas(1, 3, mask.shape, bars=3, extra=1.1)
for axis, (name, values) in zip(axes[0], MAPS, strict=True):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Sequence
# --------
#
# The sequence is an inversion followed by 400 repetitions at TR = 10 ms. The
# flip angle varies smoothly along the train, so that the signal evolutions of
# different tissues differ by physics and not by the noise of an irregular
# schedule.
CONTRASTS = 400
TR_MS = 10.0
TI_MS = 20.0

repetition = torch.arange(CONTRASTS, dtype=torch.float32)
flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / CONTRASTS) ** 2

simulator = MRFSimulator(flip=flip, TR=TR_MS, TI=TI_MS, states=20, M0=1.0)

# %%
# The readouts advance the configuration states and do not rewind them, so the
# signal evolution is real to float32 precision. The dictionary and the inner
# products that search it therefore need only the real part. These are the
# evolutions of white matter, grey matter and CSF:
fingerprints = simulator.simulate(
    T1=torch.tensor([500.0, 833.0, 2569.0]), T2=torch.tensor([70.0, 83.0, 329.0])
)
print(f"largest imaginary part: {fingerprints.imag.abs().max():.1e}")
fingerprints = fingerprints.real

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.38 * PAGE_WIDTH))
axes[0].plot(repetition, flip, color=SERIES[0])
axes[0].set(xlabel="repetition", ylabel="flip angle (deg)", title="flip angles")
for row, name in enumerate(("white matter", "grey matter", "CSF")):
    axes[1].plot(repetition, fingerprints[row], color=SERIES[row], label=name)
axes[1].set(xlabel="repetition", ylabel="signal (a.u.)", title="signal evolutions")
legend_outside(axes[1])
plt.show()
# sphinx_gallery_end_ignore

# %%
# Measurement
# -----------
#
# The measurement is the evolution of every voxel at its true T1 and T2, scaled
# by its proton density, with Gaussian noise of 2% of the peak signal. The
# estimators are given the same noise level: one trained for more noise than
# the scan has trusts the data less and returns values closer to the prior.
NOISE_STD = float(0.02 * fingerprints.max())

truth = {
    "T1": torch.as_tensor(T1_true[mask].copy()),
    "T2": torch.as_tensor(T2_true[mask].copy()),
}
density = torch.as_tensor(M0_true[mask].copy())
clean = simulator.simulate(**truth).real * density[:, None]
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)

# %%
# Problem statement
# -----------------
#
# The unknowns are T1 and T2. Both span more than a decade, so the parameter
# grids and the training draws are logarithmic: uniform spacing would spend most
# of the atoms on long T1, where the evolutions are nearly parallel.
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)

# %%
# Subspace rank
# -------------
#
# The 400 contrasts span far fewer than 400 independent directions.
# ``Subspace.fit`` computes a basis from simulated evolutions and reports the
# fraction of their energy that a given rank retains; one minus that fraction
# is the relative squared error of projecting onto the basis and back.
training_signals, _, _ = (
    DictionaryMatcher(simulator)
    .fit(
        T1=log_uniform(*T1_RANGE, SAMPLES),
        T2=log_uniform(*T2_RANGE, SAMPLES),
        noise_std=NOISE_STD,
        seed=0,
    )
    .training_set(SAMPLES)
)
training_signals = training_signals.real

RANK = 4

# sphinx_gallery_start_ignore
ranks = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
outside = [1 - Subspace.fit(training_signals, rank).retained for rank in ranks]
noise_energy = (
    NOISE_STD**2 * CONTRASTS / float(training_signals.square().sum(-1).mean())
)

fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH))
ax.semilogy(ranks, outside, "o-", color=SERIES[0], label="outside the basis")
ax.axhline(noise_energy, color=MUTED, ls="--", label="added by the noise")
ax.axvline(RANK, color=MUTED, lw=1)
ax.set(xlabel="rank", ylabel="relative energy")
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# At rank 4 the energy left outside the basis is below the energy the noise
# adds, so projecting onto four directions loses less than the noise already
# contributes. Every contrast dropped also removes arithmetic from the match.
#
# Dictionary
# ----------
#
# A dictionary spans the parameters jointly, so its size is the product of the
# grid sizes: here 200 values of T1 by 100 of T2, 20 000 atoms, fine enough
# that the grid spacing does not limit the result. Each additional parameter
# multiplies the size again. ``fit`` simulates the atoms and ``map`` returns one
# value per voxel for each parameter, and for M0, which follows from the scale
# between the measurement and the best-matching atom:
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 200)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 100)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")

# sphinx_gallery_start_ignore
start = time.perf_counter()
# sphinx_gallery_end_ignore
full = DictionaryMatcher(simulator).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), seed=0
)

maps = full.map(measured)  # {"T1": ..., "T2": ..., "M0": ...}

# sphinx_gallery_start_ignore
full_training = time.perf_counter() - start
full_maps, full_matching, full_peak = mapped(full)
full_model = footprint(full)
# sphinx_gallery_end_ignore

# %%
# Matching in the subspace
# ------------------------
#
# Passing ``rank`` fits the basis, projects the dictionary onto it and stores
# it at that rank. The measurement is projected onto the same basis before the
# atoms are scored.

# sphinx_gallery_start_ignore
start = time.perf_counter()
# sphinx_gallery_end_ignore
low = DictionaryMatcher(simulator).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), seed=0, rank=RANK
)

# sphinx_gallery_start_ignore
low_training = time.perf_counter() - start
low_maps, low_matching, low_peak = mapped(low)
low_model = footprint(low)
# sphinx_gallery_end_ignore

# %%
# Clustered dictionary
# --------------------
#
# Compression shortens every inner product; clustering reduces how many are
# computed. Atoms of neighbouring tissues have nearly parallel signal
# evolutions, so they fall into groups. Each voxel is scored against one
# representative per group, and groups that cannot contain the best match are
# excluded before any of their atoms is scored. The grouping is computed in the
# compressed basis, the space the measurement is already in.
GROUPS = 32

# sphinx_gallery_start_ignore
start = time.perf_counter()
# sphinx_gallery_end_ignore
grouped = DictionaryMatcher(simulator, groups=GROUPS).fit(
    T1=grid_t1.reshape(-1),
    T2=grid_t2.reshape(-1),
    seed=0,
    rank=RANK,
)

# sphinx_gallery_start_ignore
group_training = time.perf_counter() - start
group_maps, group_matching, group_peak = mapped(grouped)
group_model = footprint(grouped)

grouping = grouped.grouping
normalized = low.subspace.project(measured)
normalized = normalized / normalized.norm(dim=-1, keepdim=True)
survivors = grouping.survivors(normalized, grouped.prune)
print(
    f"{GROUPS} groups of {grouping.sizes[0]} atoms; "
    f"{float(survivors.sum(-1).float().mean()):.1f} still open per voxel"
)
estimates = {
    "full": (full_maps, full_maps["M0"]),
    f"rank {RANK}": (low_maps, low_maps["M0"]),
    "+ groups": (group_maps, group_maps["M0"]),
}
# sphinx_gallery_end_ignore

# %%
# Cost and accuracy
# -----------------
#
# Times are the best of three passes after a warm-up. *model* is the memory the
# fitted estimator holds between volumes. *peak* is the maximum GPU memory
# allocated while mapping the slice, which decides whether a volume fits or
# has to be processed in pieces; it is blank on a machine without a GPU.
# Errors are medians over the brain voxels, relative to the truth.

# sphinx_gallery_start_ignore
print(
    f"\n{'method':<28}{'train':>8}{'map':>8}{'model':>10}{'peak':>10}"
    f"{'T1':>8}{'T2':>8}{'M0':>8}"
)
print("-" * 88)
rows = [
    (
        "full",
        "match, 400 contrasts",
        full_training,
        full_matching,
        full_model,
        full_peak,
    ),
    (
        f"rank {RANK}",
        f"match, rank {RANK}",
        low_training,
        low_matching,
        low_model,
        low_peak,
    ),
    (
        "+ groups",
        f"match, rank {RANK} + groups",
        group_training,
        group_matching,
        group_model,
        group_peak,
    ),
]
for label, name, training, timing, model, peak in rows:
    found, m0 = estimates[label]
    print(
        f"{name:<28}{training:7.1f}s{timing:7.2f}s"
        f"{model:6.1f} MiB{held(peak)}"
        f"{error(found['T1'], truth['T1']):7.1f}%"
        f"{error(found['T2'], truth['T2']):7.1f}%"
        f"{error(m0, density):7.1f}%"
    )
# sphinx_gallery_end_ignore

# %%
# Maps
# ----
#
# The maps of each method, and below them the absolute error on a scale of its
# own for each parameter.

# sphinx_gallery_start_ignore
panels = [
    ("T1", T1_true, {k: maps_["T1"] for k, (maps_, _) in estimates.items()}),
    ("T2", T2_true, {k: maps_["T2"] for k, (maps_, _) in estimates.items()}),
    ("M0", M0_true, {k: m0 for k, (_, m0) in estimates.items()}),
]

fig, axes = canvas(len(panels), 1 + len(estimates), mask.shape)
for row, (name, reference, found) in enumerate(panels):
    cmap, limits, label = STYLE[name]
    panel(
        axes[row, 0],
        reference,
        cmap,
        limits,
        ylabel=label,
        title="truth" if row == 0 else None,
    )
    for column, (method, values) in enumerate(found.items(), start=1):
        handle = panel(
            axes[row, column],
            painted(values),
            cmap,
            limits,
            title=method if row == 0 else None,
        )
    scalebar(handle, axes[row], "")
plt.show()

fig, axes = canvas(len(panels), len(estimates), mask.shape)
for row, (name, reference, found) in enumerate(panels):
    label = STYLE[name][2]
    residuals = {
        method: np.abs(painted(values) - reference) for method, values in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
    for column, (method, values) in enumerate(residuals.items()):
        handle = panel(
            axes[row, column],
            values,
            "inferno",
            (0.0, top or 1.0),
            title=f"Δ {method}" if row == 0 else None,
        )
    unit = label[label.find(" (") :] if "(" in label else ""
    scalebar(handle, axes[row], f"|error|{unit}")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Specification
# -------------
#
# What this page did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, map T1, T2 and M0 in BrainWeb subject 0, slice 90, from an
#    inversion-prepared MRF train of 400 repetitions (TR 10 ms, TI 20 ms,
#    flip angle 10 + 50 sin^2(pi n / 400) deg) at 2% noise. Use a
#    DictionaryMatcher over 200 log-spaced T1 by 100 log-spaced T2 values.
#    Compare the full match, the match at subspace rank 4 and the match at
#    rank 4 with 32 groups, reporting training time, mapping time, estimator
#    memory and the median relative error of each map.
