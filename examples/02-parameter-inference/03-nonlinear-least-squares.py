"""
=====================================
T2 mapping by nonlinear least squares
=====================================

A dictionary is a sampling of the model, and its size is the product of the
grids it samples. A nonlinear fit walks downhill on the model itself, so a
parameter costs it one more column of the Jacobian rather than one more factor
in a product. What it gives up is the guarantee: it finds a local minimum of
the residual, and which one depends on where it started.

This example maps T2 from a multi-echo spin echo, and the parameter that
decides the comparison is a nuisance. A magnitude reconstruction sits on a
noise floor, so the decay does not go to zero -- and unlike the proton density,
which divides out of a normalized match for free, a constant added to the decay
does not. Ignore it and the T2 is biased; put it on the grid and the grid
multiplies.

Both routes map the same BrainWeb slice, and both report what they cost in time
and in peak memory.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim brainweb-dl cmap

# %%
#
# The problem is stated over a simulator carrying the sequence and filled in
# by an estimator. :func:`~torchsim.execution` decides where that work runs,
# and the timings below are taken inside it.
#

# sphinx_gallery_start_ignore
import csv
import warnings
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
from cmap import Colormap
from brainweb_dl import get_mri

warnings.filterwarnings("ignore")


# Fuderer et al. (Magn. Reson. Med. 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()

# Colormap, window and unit per parameter. Both relaxation windows stop well
# short of CSF, so that white and grey matter -- 500 against 833 ms in T1, 70
# against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "T1 [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "T2 [ms]"),
    "M0": ("gray", (0.0, 1.0), "M0"),
}


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
    """One colorbar for a group of panels, so none gives up width to its own."""
    axes = list(np.ravel(axes))
    axes[0].figure.colorbar(handle, ax=axes, label=label, shrink=0.92, aspect=20)


# Every panel on this page is drawn at the same size, so any two figures can be
# read against each other. The side is set by the widest grid, which fills the
# documentation column; a figure with fewer columns is narrower, not larger.
PAGE_WIDTH = 8.6  # inches, the width of the documentation column
BAR_WIDTH = 0.8  # what one colorbar takes out of it
PANEL = (PAGE_WIDTH - 1 * BAR_WIDTH) / 4  # one image panel


def canvas(rows, columns, shape, *, bars=1, extra=0.6):
    """A grid of image panels, in the proportion of the images.

    ``bars`` is how many colorbars a row carries and ``extra`` the height left
    over the panels, for titles and for a figure title where there is one.
    """
    return plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(
            columns * PANEL + bars * BAR_WIDTH,
            PANEL * shape[0] / shape[1] * rows + extra,
        ),
    )


# Figures are read at gallery scale, so the type sizes are set once here.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.6),
        "savefig.dpi": 110,
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 19,
        "figure.constrained_layout.use": True,
    }
)


def key(axes, ncols=1):
    """The legend above what it describes, clear of the curves and the titles.

    Takes a figure, where every panel is showing the same series, and puts one
    legend over the whole of it. Takes an axis, or several, where the panels
    differ, and puts a legend over each -- every titled panel in the figure
    then ends up with the same padding, so the titles line up whether or not
    that panel carries one, which is only known once it has been laid out.
    """
    if hasattr(axes, "add_subplot"):
        handles, labels = axes.axes[0].get_legend_handles_labels()
        return axes.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=ncols,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.4,
        )
    axes = [axes] if hasattr(axes, "get_legend_handles_labels") else list(axes)
    figure = axes[0].figure
    legends = [
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=ncols,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.6,
            columnspacing=1.4,
        )
        for axis in axes
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tallest = max(legend.get_window_extent(renderer).height for legend in legends)
    for axis in figure.axes:
        if axis.get_title():
            axis.set_title(axis.get_title(), pad=72.0 * tallest / figure.dpi + 4.0)
    return legends


# sphinx_gallery_end_ignore
import time

import numpy as np
import torch

import torchsim
from torchsim.estimators import DictionaryMatcher, NonlinearLeastSquares
from torchsim.simulators import MultiEchoSimulator

# %%
#
# A brain to map
# --------------
#
# The phantom is BrainWeb subject 0, slice 90 -- an axial slice at 1 mm
# through the lateral ventricles, carrying CSF, grey matter, white matter and
# the glial matter between them. BrainWeb publishes it as *fuzzy* memberships:
# every voxel holds a fraction of each tissue rather than a label, and
# weighting the relaxation times BrainWeb tabulates by those fractions gives
# the maps below.
#
# The fractions matter more than the anatomy. A third of these brain voxels
# are mixtures of two tissues or more, so the truth is a continuum rather than
# four values, and an estimator cannot do well merely by having seen the right
# four answers.
#

# sphinx_gallery_start_ignore
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
tissues = list(csv.DictReader(table.open()))
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

share = np.maximum(occupancy, 1e-6)
T2_true = np.where(mask, fractions @ tissue_T2 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

truth = torch.as_tensor(T2_true[mask].copy())
density = torch.as_tensor(M0_true[mask].copy())

figure, axes = canvas(1, 2, mask.shape, bars=2, extra=1.1)
for axis, values, name in ((axes[0, 0], T2_true, "T2"), (axes[0, 1], M0_true, "M0")):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
# sphinx_gallery_end_ignore

# %%
#
# The measurement, and the floor it sits on
# -----------------------------------------
#
# Sixteen echoes out to 200 ms. The decay is scaled by the proton density and
# offset by a constant: a magnitude reconstruction rectifies the noise, so what
# the late echoes measure is not zero but the noise floor, and a fit that does
# not say so will absorb it into T2.
#
ECHOES = 16
TE = torch.linspace(10.0, 200.0, ECHOES)
NOISE_STD = 0.005
FLOOR = 0.05

acquisition = MultiEchoSimulator(TE=TE)

clean = acquisition.simulate(T2=truth, M0=density, offset=FLOOR)
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)

# %%
#
# Three decays and what the floor does to them. The short-T2 voxel is on the
# floor by the fourth echo, so most of its train says nothing about T2 and
# everything about the offset; the long-T2 one never gets there.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.76))
for value, name in ((70.0, "white matter"), (110.0, "grey matter"), (300.0, "CSF")):
    decay = acquisition.simulate(T2=value, M0=1.0, offset=FLOOR)
    axis.semilogy(
        TE.numpy(), decay.numpy(), "-o", ms=3, label=f"{name}, T2 {value:.0f} ms"
    )
axis.axhline(FLOOR, color="k", ls="--", lw=1)
axis.text(TE[-1], FLOOR * 1.15, "noise floor", ha="right")
axis.set(xlabel="Echo time [ms]", ylabel="signal", title="what is measured")
axis.grid(alpha=0.3)
key(axis, ncols=3)
# sphinx_gallery_end_ignore


# sphinx_gallery_start_ignore
def footprint(problem):
    """MiB the fitted estimator itself holds."""
    held = sum(t.numel() * t.element_size() for t in problem.buffers())
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


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


# sphinx_gallery_end_ignore

# %%
#
# The fit
# -------
#
# What is unknown, over what range, from what acquisition, at what noise level
# -- and then :class:`~torchsim.NonlinearLeastSquares` to fill it in. Every
# voxel steps together in the same pass, carries its own damping, accepts or
# rejects on its own, and drops out when it has converged.
#
# The bounds are not clipping. A bound is kept by fitting a transformed
# variable, so no iterate ever leaves the interval and the bound cannot be
# sitting exactly on the answer -- which also puts every parameter on the same
# scale whatever its units, and that is what the damping term assumes.
#
BOUNDS = {"T2": (10.0, 500.0), "M0": (0.1, 2.0), "offset": (0.0, 0.2)}
START = {"T2": 100.0, "M0": 1.0, "offset": 0.02}

fit = NonlinearLeastSquares(acquisition, bounds=BOUNDS, initial=START).fit(
    BOUNDS, noise_std=NOISE_STD, seed=0
)

maps = fit.map(measured)  # {"T2": ..., "M0": ..., "offset": ...}


# sphinx_gallery_start_ignore
def fitted(unknown):
    """Fit these parameters, holding the rest of the model at the truth."""
    held = {name: START[name] for name in START if name not in unknown}
    if "offset" in held:
        held["offset"] = FLOOR
    problem = NonlinearLeastSquares(
        acquisition.bind(**held),
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
#
# The match
# ---------
#
# The same problem given to a dictionary. Its grid does not need a proton
# density: a match normalizes both sides, so any positive scale is matched for
# free, and one parameter of the three is gone before the grid is built.
#
# The offset is not so kind. It survives normalization, so a match that wants
# to model it has to put it on the grid -- and the grid is then the product.
#
T2_GRID = torch.logspace(1.0, np.log10(500.0), 400)

match = DictionaryMatcher(acquisition.bind(M0=1.0, offset=0.0)).fit(T2=T2_GRID, seed=0)

# %%
#
# To model the floor the match has to put it on the grid, and the grid is then
# the product of the two.
#
offsets = torch.linspace(0.0, 0.15, 40)
grid_t2, grid_offset = torch.meshgrid(T2_GRID, offsets, indexing="ij")

wide = DictionaryMatcher(acquisition.bind(M0=1.0)).fit(
    T2=grid_t2.reshape(-1), offset=grid_offset.reshape(-1), seed=0
)


# sphinx_gallery_start_ignore
def matched(floors):
    """Fit a matcher over T2, and over this many values of the offset."""
    if floors == 1:
        problem = DictionaryMatcher(acquisition.bind(M0=1.0, offset=0.0))
        ranges = {"T2": T2_GRID}
        atoms = T2_GRID.numel()
    else:
        offsets = torch.linspace(0.0, 0.15, floors)
        grid_t2, grid_offset = torch.meshgrid(T2_GRID, offsets, indexing="ij")
        problem = DictionaryMatcher(acquisition.bind(M0=1.0))
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
#
# What it cost, and what it got
# -----------------------------
#
# Best of three passes each, after one warm-up that leaves out the measurement
# :func:`~torchsim.execution` makes the first time it meets a workload. The
# **model** is what the fitted estimator carries between volumes; the **peak**
# is the high-water mark on the card while the slice was mapped.
#

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
        f"{model:6.1f} MiB{peak:6.0f} MiB{error(found['T2'], truth):7.1f}%"
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
        f"{model:6.1f} MiB{peak:6.0f} MiB{error(found['T2'], truth):7.1f}%"
    )
# sphinx_gallery_end_ignore

# %%
#
# Reading the table
# -----------------
#
# The first row is the trap. A T2-only match is the quickest thing here and it
# is wrong by an order of magnitude more than anything else, because the model
# it matched against was not the model that produced the data. Nothing about
# the estimator says so -- the residual it minimized is small, and it is small
# at the wrong T2.
#
# Once the offset is on the grid the match recovers, and the cost of recovering
# is the whole point: ten values of one nuisance is ten times the atoms, and
# the memory follows, for a parameter that took the fit one more column and no
# more storage at all. The peak stops climbing at the top of the sweep only
# because streaming has taken over and is sizing its chunk to a budget.
#
# The fit is the slower of the two in wall clock, and on this problem it stays
# that way: a Levenberg-Marquardt loop is tens of passes over the model where a
# match is one pass over the atoms. What does not happen to it is growth. Every
# nuisance the scale does not divide out multiplies the grid again and adds one
# column to the Jacobian, so where the two cross is arithmetic rather than
# opinion -- and the memory has crossed already.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3))
atoms = [matches[floors][0] for floors in FLOOR_VALUES]
axes[0].plot(atoms, [matches[f][3] for f in FLOOR_VALUES], "-o", label="match")
axes[0].axhline(three_time, color="crimson", ls="--", label="fit, 3 unknowns")
axes[0].set(
    xlabel="Atoms in the dictionary",
    ylabel="Time to map the slice [s]",
    xscale="log",
    yscale="log",
    title="time",
)
axes[1].plot(atoms, [matches[f][5] for f in FLOOR_VALUES], "-o", label="match")
axes[1].axhline(three_peak, color="crimson", ls="--", label="fit, 3 unknowns")
axes[1].set(
    xlabel="Atoms in the dictionary",
    ylabel="Peak device memory [MiB]",
    xscale="log",
    yscale="log",
    title="memory",
)
for axis in axes:
    axis.grid(alpha=0.3)
key(figure, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# The maps
# --------
#


# sphinx_gallery_start_ignore
def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


# Column headings sit over a two-inch panel; the table above carries the rest.
shown = {
    "match, T2": matches[1][1]["T2"],
    "match + offset": matches[40][1]["T2"],
    "fit, all three": three_maps["T2"],
}

cmap, limits, label = STYLE["T2"]
figure, axes = canvas(2, 1 + len(shown), mask.shape)
panel(axes[0, 0], T2_true, cmap, limits, title="truth")
axes[1, 0].set_visible(False)

residuals = {name: np.abs(painted(values) - T2_true) for name, values in shown.items()}
top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
for column, (name, values) in enumerate(shown.items(), start=1):
    found = panel(axes[0, column], painted(values), cmap, limits, title=name)
    error = panel(axes[1, column], residuals[name], "inferno", (0.0, top or 1.0))
scalebar(found, axes[0], label)
scalebar(error, axes[1, 1:], f"|error|, {label}")
# sphinx_gallery_end_ignore

# %%
#
# Where a fit is the wrong choice
# -------------------------------
#
# It has no guarantee. The fit above started every voxel at 100 ms and landed
# on the right answer everywhere, which is a property of an exponential and not
# of nonlinear least squares: the residual of a single decay has one minimum,
# so where it starts does not matter. A model whose residual has several -- a
# fingerprinting train, a fat-water fit at a wrong initial field map -- can be
# started in the wrong basin and stay there, and a match cannot, because it
# scores every atom.
#
# Equality constraints belong in the model rather than in the bounds. A fit
# where two fractions must sum to one is written with one of them as the
# unknown and the other as ``1 - f`` inside the model, so the constraint holds
# identically at every iterate rather than being restored after each.
#
