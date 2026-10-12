"""
=====================================
T2 mapping by nonlinear least squares
=====================================

You map T2 from a multi-echo spin echo by nonlinear least squares and compare
the result with dictionary matching on the same slice.

The comparison turns on a nuisance parameter. A magnitude reconstruction sits
on a noise floor, so the decay does not reach zero. Unlike the proton density,
this offset does not divide out of a normalized match: ignoring it biases T2,
and adding it to the dictionary multiplies the number of atoms. A fit instead
adds one column to the Jacobian, at the cost of no guarantee that it finds the
global minimum.

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

from blochsim.estimators import DictionaryMatcher, NonlinearLeastSquares
from blochsim.simulators import MultiEchoSimulator

# %%
# Phantom
# -------
#
# The phantom is BrainWeb subject 0, slice 90: an axial 1 mm slice through the
# lateral ventricles. BrainWeb provides fuzzy tissue memberships, so each voxel
# holds a fraction of each tissue and its T2 is the fraction-weighted average.
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
    ("T2", T2_true),
    ("M0", M0_true),
)
figure, axes = canvas(1, 2, mask.shape, bars=2, extra=1.1)
for axis, (name, values) in zip(axes[0], MAPS, strict=True):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
plt.show()
# sphinx_gallery_end_ignore
truth = torch.as_tensor(T2_true[mask].copy())
density = torch.as_tensor(M0_true[mask].copy())

# %%
# Measurement and noise floor
# ---------------------------
#
# The sequence is a multi-echo spin echo with 16 echoes from 10 to 200 ms. The
# signal is the proton density times the T2 decay, plus a constant offset. A
# magnitude reconstruction rectifies the noise, so late echoes measure the
# floor and not zero, and a fit that does not model it absorbs it into T2.
ECHOES = 16
TE = torch.linspace(10.0, 200.0, ECHOES)
NOISE_STD = 0.005
FLOOR = 0.05

simulator = MultiEchoSimulator(TE=TE)

clean = simulator.simulate(T2=truth, M0=density, offset=FLOOR)
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)

# %%
# The 70 ms decay approaches the floor only in the last echoes, where the
# offset and T2 are hardest to separate; the long-T2 voxel never reaches it.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH))
for i, (value, name) in enumerate(
    ((70.0, "white matter"), (110.0, "grey matter"), (300.0, "CSF"))
):
    decay = simulator.simulate(T2=value, M0=1.0, offset=FLOOR)
    ax.semilogy(
        TE, decay, "-o", ms=3, color=SERIES[i], label=f"{name}, T2 {value:.0f} ms"
    )
ax.axhline(FLOOR, color=MUTED, ls="--", lw=1, label="noise floor")
ax.set(xlabel="TE (ms)", ylabel="signal (a.u.)")
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Nonlinear fit
# -------------
#
# ``NonlinearLeastSquares`` takes the range of every unknown, a starting value
# and the noise level. All voxels are updated in the same pass; each carries
# its own damping, accepts or rejects its own step, and stops when it
# converges.
#
# The bounds are enforced by fitting a transformed variable, not by clipping:
# no iterate leaves the interval and no bound is active at the solution. The
# transformation also puts all parameters on a common scale, which the damping
# assumes.
BOUNDS = {"T2": (10.0, 500.0), "M0": (0.1, 2.0), "offset": (0.0, 0.2)}
START = {"T2": 100.0, "M0": 1.0, "offset": 0.02}

fit = NonlinearLeastSquares(simulator, bounds=BOUNDS, initial=START).fit(
    BOUNDS, noise_std=NOISE_STD, seed=0
)

maps = fit.map(measured)  # {"T2": ..., "M0": ..., "offset": ...}


# sphinx_gallery_start_ignore
def fitted(unknown):
    """Fit these parameters, holding the rest of the model at the truth."""
    fixed = {name: START[name] for name in START if name not in unknown}
    if "offset" in fixed:
        fixed["offset"] = FLOOR
    problem = NonlinearLeastSquares(
        simulator.bind(**fixed),
        bounds={name: BOUNDS[name] for name in unknown},
        initial={name: START[name] for name in unknown},
    )
    start = time.perf_counter()
    problem.fit(
        **{name: BOUNDS[name] for name in unknown},
        noise_std=NOISE_STD,
        seed=0,
    )
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return found, training, timing, footprint(problem), peak


two_maps, two_training, two_time, two_model, two_peak = fitted(("T2", "M0"))
three_maps, three_training, three_time, three_model, three_peak = fitted(
    ("T2", "M0", "offset")
)

print(
    f"fitted floor, median {float(three_maps['offset'].median()):.4f} against {FLOOR}"
)
# sphinx_gallery_end_ignore

# %%
# Dictionary match
# ----------------
#
# The same problem posed to a dictionary. A match normalizes both sides, so
# any positive scale is free and M0 drops out of the grid. The offset survives
# normalization, so modelling it requires putting it on the grid. The first
# dictionary ignores the offset:
T2_GRID = torch.logspace(1.0, np.log10(500.0), 400)

match = DictionaryMatcher(simulator.bind(M0=1.0, offset=0.0)).fit(T2=T2_GRID, seed=0)

# %%
# The second has the offset on the grid, which is then the product of the T2
# and offset grids: 400 by 40 atoms.
offsets = torch.linspace(0.0, 0.15, 40)
grid_t2, grid_offset = torch.meshgrid(T2_GRID, offsets, indexing="ij")

wide = DictionaryMatcher(simulator.bind(M0=1.0)).fit(
    T2=grid_t2.reshape(-1), offset=grid_offset.reshape(-1), seed=0
)


# sphinx_gallery_start_ignore
def matched(floors):
    """Fit a matcher over T2, and over this many values of the offset."""
    if floors == 1:
        problem = DictionaryMatcher(simulator.bind(M0=1.0, offset=0.0))
        ranges = {"T2": T2_GRID}
        atoms = T2_GRID.numel()
    else:
        offsets = torch.linspace(0.0, 0.15, floors)
        grid_t2, grid_offset = torch.meshgrid(T2_GRID, offsets, indexing="ij")
        problem = DictionaryMatcher(simulator.bind(M0=1.0))
        ranges = {
            "T2": grid_t2.reshape(-1),
            "offset": grid_offset.reshape(-1),
        }
        atoms = grid_t2.numel()
    start = time.perf_counter()
    problem.fit(ranges, seed=0)
    training = time.perf_counter() - start
    maps, timing, peak = mapped(problem)
    return atoms, maps, training, timing, footprint(problem), peak


FLOOR_VALUES = (1, 10, 20, 40, 80)
matches = {floors: matched(floors) for floors in FLOOR_VALUES}
# sphinx_gallery_end_ignore

# %%
# Cost and accuracy
# -----------------
#
# Times are the best of three passes after a warm-up. *model* is the memory the
# fitted estimator holds between volumes. *peak* is the maximum GPU memory
# allocated while mapping the slice, blank on a machine without a GPU. The
# error is the median relative T2 error over the brain voxels.

# sphinx_gallery_start_ignore
print(
    f"\n{'method':<30}{'atoms':>8}{'train':>8}{'map':>8}{'model':>10}"
    f"{'peak':>10}{'T2':>8}"
)
print("-" * 82)
for floors in FLOOR_VALUES:
    atoms, found, training, timing, model, peak = matches[floors]
    name = "match, T2 only" if floors == 1 else f"match, T2 x {floors} offsets"
    print(
        f"{name:<30}{atoms:>8}{training:7.1f}s{timing:7.2f}s"
        f"{model:6.1f} MiB{held(peak)}{error(found['T2'], truth):7.1f}%"
    )
for name, training, timing, model, peak, found in (
    (
        "fit, T2 + M0, floor known",
        two_training,
        two_time,
        two_model,
        two_peak,
        two_maps,
    ),
    (
        "fit, T2 + M0 + offset",
        three_training,
        three_time,
        three_model,
        three_peak,
        three_maps,
    ),
):
    print(
        f"{name:<30}{'--':>8}{training:7.1f}s{timing:7.2f}s"
        f"{model:6.1f} MiB{held(peak)}{error(found['T2'], truth):7.1f}%"
    )
# sphinx_gallery_end_ignore

# %%
# Interpretation
# --------------
#
# The T2-only match is the fastest method here and the least accurate: the
# model it matches is not the model that produced the data. The residual it
# minimizes is small, at the wrong T2, and the estimator does not report the
# mismatch.
#
# With the offset on the grid the match recovers. The cost is the number of
# offset values: ten values multiply the number of atoms and the memory by ten,
# for a parameter that adds one column to the fit's Jacobian.
#
# The fit is slower in wall-clock time, because a Levenberg-Marquardt loop takes
# tens of passes where a match takes one. Each additional nuisance parameter
# multiplies the number of atoms but adds one column to the Jacobian, so the
# time curves cross at some number of nuisance values, and the memory curves
# have already crossed.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))
atoms = [matches[floors][0] for floors in FLOOR_VALUES]
axes[0].plot(
    atoms, [matches[f][3] for f in FLOOR_VALUES], "-o", color=SERIES[0], label="match"
)
axes[0].axhline(three_time, color=SERIES[1], ls="--", label="fit, 3 unknowns")
axes[0].set(
    xlabel="atoms in the dictionary",
    ylabel="time to map the slice (s)",
    xscale="log",
    yscale="log",
    title="time",
)
axes[1].plot(
    atoms, [matches[f][4] for f in FLOOR_VALUES], "-o", color=SERIES[0], label="match"
)
axes[1].axhline(three_model, color=SERIES[1], ls="--", label="fit, 3 unknowns")
axes[1].set(
    xlabel="atoms in the dictionary",
    ylabel="estimator memory (MiB)",
    xscale="log",
    yscale="log",
    title="memory",
)
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Maps
# ----
#
# T2 from the match without the offset, the match with it, and the fit of all
# three parameters, with the absolute error in the second row.

# sphinx_gallery_start_ignore
shown = {
    "match, T2": matches[1][1]["T2"],
    "match + offset": matches[40][1]["T2"],
    "fit, all three": three_maps["T2"],
}

cmap, limits, label = STYLE["T2"]
fig, axes = canvas(2, 1 + len(shown), mask.shape)
panel(axes[0, 0], T2_true, cmap, limits, title="truth")
axes[1, 0].set_visible(False)

residuals = {name: np.abs(painted(values) - T2_true) for name, values in shown.items()}
top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
for column, (name, values) in enumerate(shown.items(), start=1):
    found_map = panel(axes[0, column], painted(values), cmap, limits, title=name)
    error_map = panel(axes[1, column], residuals[name], "inferno", (0.0, top or 1.0))
scalebar(found_map, axes[0], label)
scalebar(error_map, axes[1, 1:], f"|error|, {label}")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Limits
# ------
#
# The fit has no guarantee of reaching the global minimum. Here it started every
# voxel at 100 ms and converged to the correct T2 everywhere, because the
# residual of a single exponential decay has one minimum. A model whose
# residual has several, such as a fingerprinting train or a fat-water fit at a
# wrong field map, can start in the wrong basin and stay there. A match cannot,
# because it scores every atom.
#
# Equality constraints belong in the model, not in the bounds. Two fractions
# that sum to one are written with one as the unknown and the other as ``1 - f``
# inside the model, so the constraint holds at every iterate.
#
# Specification
# -------------
#
# What this page did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, map T2 in BrainWeb subject 0, slice 90, from a 16-echo
#    multi-echo spin echo (TE 10 to 200 ms) with a constant noise-floor offset
#    of 0.05 and Gaussian noise of 0.005. Fit T2, M0 and the offset with
#    NonlinearLeastSquares (bounds T2 10-500 ms, M0 0.1-2, offset 0-0.2).
#    Compare with a DictionaryMatcher over 400 log-spaced T2 values, without
#    the offset and with 10 to 80 offset values on the grid. Report mapping
#    time, estimator memory and median relative T2 error, and plot the maps.
