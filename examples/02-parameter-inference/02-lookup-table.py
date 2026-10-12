"""
====================
MP2RAGE lookup table
====================

You map T1 from a two-block MP2RAGE acquisition in two ways, by interpolating
along the signal curve and by dictionary matching, and sweep the number of
points on the curve to see which method depends on it.

With a single unknown, the dictionary atoms lie on a curve instead of filling
a space. Interpolating between the two nearest points on the curve costs
nothing and removes the grid spacing from the error; this is what
:class:`~blochsim.LookupTable` does.

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


# sphinx_gallery_end_ignore
import numpy as np
import torch

from blochsim.estimators import DictionaryMatcher, LookupTable
from blochsim.simulators import MP2RAGESimulator

# %%
# Phantom
# -------
#
# The phantom is BrainWeb subject 0, slice 90: an axial 1 mm slice through the
# lateral ventricles. BrainWeb provides fuzzy tissue memberships, so each voxel
# holds a fraction of each tissue and its T1 is the fraction-weighted average.
# About a third of the voxels are mixtures, so the ground truth is a continuum
# of values rather than four.

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
    ("M0", M0_true),
)
figure, axes = canvas(1, 2, mask.shape, bars=2, extra=1.1)
for axis, (name, values) in zip(axes[0], MAPS, strict=True):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
plt.show()
# sphinx_gallery_end_ignore
truth = torch.as_tensor(T1_true[mask].copy())
density = torch.as_tensor(M0_true[mask].copy())

# %%
# Protocol
# --------
#
# MP2RAGE applies one inversion and reads two spoiled gradient-echo blocks at
# two inversion times. Each block samples the centre of k-space of its own
# shot train, so a voxel contributes two values. The train is spoiled after
# every readout, so T1 is the only tissue property that changes them.
PROTOCOL = dict(
    TI=(800.0, 2700.0),
    flip=(4.0, 5.0),
    TRspgr=6.7,
    TRmp2rage=6000.0,
    nshots=128,
)
INVERSION_EFFICIENCY = 0.96

simulator = MP2RAGESimulator(**PROTOCOL, inv_efficiency=INVERSION_EFFICIENCY)

# %%
# Signal curve
# ------------
#
# Neither block alone determines T1, since both are scaled by the proton
# density and the receive gain. The unified combination divides this scale
# out and leaves a value between -0.5 and 0.5 that depends on T1 alone:


def unified(blocks):
    """The MP2RAGE unified image: independent of the scale, a function of T1."""
    return (blocks[..., 0] * blocks[..., 1]) / blocks.square().sum(-1).clamp_min(1e-12)


# %%
# Whether the unified image is monotonic in T1 depends on the protocol. Where
# it turns back it has no inverse, so the table keeps the longest monotonic run
# and reports the range it spans. The dashed line marks the turning point of
# the unified image at this protocol, beyond which T1 is not invertible.
sweep = torch.arange(50.0, 6000.0, 10.0)
curve = unified(simulator.simulate(T1=sweep, M0=1.0))

# sphinx_gallery_start_ignore
turning = int(curve.argmin()) if curve[0] > curve[-1] else int(curve.argmax())
blocks = simulator.simulate(T1=sweep, M0=1.0)

fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))
for column in range(2):
    axes[0].plot(
        sweep,
        blocks[:, column],
        color=SERIES[column],
        label=f"TI = {PROTOCOL['TI'][column]:.0f} ms",
    )
axes[0].set(xlabel="T1 (ms)", ylabel="magnetization (a.u.)", title="the two blocks")
legend_outside(axes[0])

axes[1].plot(sweep, curve, color=SERIES[2])
axes[1].axvline(float(sweep[turning]), color=MUTED, ls="--", lw=1)
axes[1].set(xlabel="T1 (ms)", ylabel="unified image", title="unified image")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Measurement
# -----------
#
# The measurement is both blocks at the true T1 and proton density of every
# brain voxel, with Gaussian noise of 0.5% of the peak magnetization.
clean = simulator.simulate(T1=truth, M0=density)
NOISE_STD = float(0.005 * clean.abs().max())

generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)


# sphinx_gallery_start_ignore
def estimated(make, points):
    """Fit this method over a grid of this many points, then map the slice."""
    grid = torch.linspace(50.0, 6000.0, points)
    problem = make(simulator.bind(M0=1.0))
    start = time.perf_counter()
    problem.fit(T1=grid, seed=0)
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return problem, found, training, timing, footprint(problem), peak


# sphinx_gallery_end_ignore

# %%
# Estimators
# ----------
#
# Both estimators are fitted on the same T1 grid, with the proton density
# fixed to one. The matcher scores the two-block signal against every atom and
# returns the nearest. The table reduces both blocks to the unified value and
# interpolates along the curve; ``combine`` is the only information it is
# given about the sequence. Neither is given the invertible range.
grid = torch.linspace(50.0, 6000.0, 60)

lookup = LookupTable(simulator.bind(M0=1.0), combine=unified).fit(T1=grid, seed=0)
maps = lookup.map(measured)  # {"T1": ...}, one value per voxel

match = DictionaryMatcher(simulator.bind(M0=1.0)).fit(T1=grid, seed=0)

# %%
# Table size
# ----------
#
# Sweeping the number of grid points separates the method from the sampling.
# Times are the best of three passes over the slice.

# sphinx_gallery_start_ignore
POINTS = (30, 60, 120, 250, 500, 1000, 2000)
matched = {}
looked_up = {}
for points in POINTS:
    _, found, _, timing, _, _ = estimated(DictionaryMatcher, points)
    matched[points] = (error(found["T1"], truth), timing)
    problem, found, _, timing, _, _ = estimated(
        lambda acq: LookupTable(acq, combine=unified), points
    )
    looked_up[points] = (error(found["T1"], truth), timing)

print(f"\n{'points':>7}{'match':>10}{'table':>10}{'match':>10}{'table':>10}")
print(f"{'':>7}{'error':>10}{'error':>10}{'time':>10}{'time':>10}")
print("-" * 47)
for points in POINTS:
    print(
        f"{points:>7}{matched[points][0]:9.2f}%{looked_up[points][0]:9.2f}%"
        f"{1e3 * matched[points][1]:8.1f}ms{1e3 * looked_up[points][1]:8.1f}ms"
    )

fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))
axes[0].plot(
    POINTS, [matched[n][0] for n in POINTS], "-o", color=SERIES[0], label="match"
)
axes[0].plot(
    POINTS,
    [looked_up[n][0] for n in POINTS],
    "-*",
    color=SERIES[1],
    label="lookup table",
)
axes[0].set(
    xlabel="points on the curve",
    ylabel="median relative error (%)",
    xscale="log",
    yscale="log",
    title="error",
)
axes[1].plot(
    POINTS, [1e3 * matched[n][1] for n in POINTS], "-o", color=SERIES[0], label="match"
)
axes[1].plot(
    POINTS,
    [1e3 * looked_up[n][1] for n in POINTS],
    "-*",
    color=SERIES[1],
    label="lookup table",
)
axes[1].set(
    xlabel="points on the curve",
    ylabel="time to map the slice (ms)",
    xscale="log",
    yscale="log",
    title="time",
)
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The error floor is set by the noise. The table is within 0.2 percentage
# points of it from 30 points on. The match is about nine times worse at 30
# points and reaches the same floor at about 120, at a cost that grows with
# the number of points: it takes one comparison per atom per voxel, where the table takes a
# binary search that grows with the logarithm of the number of points. A
# sufficiently fine grid makes the match equal to the table; the table does not
# need to be told how fine the grid must be.
#
# Maps
# ----
#
# The maps use the number of points each method needs: 60 for the table and
# 2000 for the match, a grid fine enough not to limit it.

# sphinx_gallery_start_ignore
TABLE_POINTS = 60
MATCH_POINTS = 2000

problem, table_maps, table_training, table_time, table_model, table_peak = estimated(
    lambda acq: LookupTable(acq, combine=unified), TABLE_POINTS
)
_, match_maps, match_training, match_time, match_model, match_peak = estimated(
    DictionaryMatcher, MATCH_POINTS
)
print(
    f"the table keeps {problem.points} of {TABLE_POINTS} points -- the "
    f"monotonic run -- and spans unified "
    f"{problem.span[0]:.2f} to {problem.span[1]:.2f}"
)
# sphinx_gallery_end_ignore

# %%
# Neither estimator returns M0. The two blocks predicted for the estimated T1
# are a shape of which the measurement is a multiple, so the proton density is
# one inner product per voxel:


def proton_density(maps):
    """The scale of the measurement relative to the blocks the answer predicts."""
    predicted = simulator.simulate(T1=maps["T1"], M0=1.0)
    return (predicted * measured).sum(-1) / predicted.square().sum(-1).clamp_min(1e-12)


# sphinx_gallery_start_ignore
estimates = {
    "lookup": (table_maps, proton_density(table_maps)),
    "match": (match_maps, proton_density(match_maps)),
}

print(
    f"\n{'method':<24}{'train':>9}{'map':>9}{'model':>10}{'peak':>10}{'T1':>8}{'M0':>8}"
)
print("-" * 78)
for short, name, training, timing, model, peak in (
    (
        "lookup",
        f"lookup, {TABLE_POINTS} points",
        table_training,
        table_time,
        table_model,
        table_peak,
    ),
    (
        "match",
        f"match, {MATCH_POINTS} atoms",
        match_training,
        match_time,
        match_model,
        match_peak,
    ),
):
    found, m0 = estimates[short]
    print(
        f"{name:<24}{training:8.2f}s{1e3 * timing:7.1f}ms"
        f"{model:6.2f} MiB{held(peak)}"
        f"{error(found['T1'], truth):7.2f}%{error(m0, density):7.2f}%"
    )

panels = [
    ("T1", T1_true, {k: found["T1"] for k, (found, _) in estimates.items()}),
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

# Each parameter's errors on a scale of their own.
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
    scalebar(handle, axes[row], f"|error|, {label}")
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
#    With blochsim, map T1 and M0 in BrainWeb subject 0, slice 90, from a
#    two-block MP2RAGE (TI 800 and 2700 ms, flip angles 4 and 5 deg, TR 6.7 ms,
#    128 shots, TR_MP2RAGE 6 s, inversion efficiency 0.96) at 0.5% noise.
#    Fit a LookupTable on the unified image and a DictionaryMatcher over the
#    same T1 grid. Sweep the grid from 30 to 2000 points and report the median
#    relative T1 error and the mapping time of both; derive M0 from the
#    predicted blocks.
