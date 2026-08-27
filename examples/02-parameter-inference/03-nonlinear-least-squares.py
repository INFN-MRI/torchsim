"""
==================================================
Fitting the model instead of a sampling of it
==================================================

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
#    !pip install torchsim brainweb-dl

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
from brainweb_dl import get_mri

warnings.filterwarnings("ignore")



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


# sphinx_gallery_end_ignore
import time

import numpy as np
import torch

import torchsim
from torchsim import ParameterMapping
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

figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
panel(axes[0], T2_true, "magma", (0, 350), label="T2 [ms]")
panel(axes[1], M0_true, "magma", (0, 1.1), label="M0")
figure.suptitle("BrainWeb subject 0, slice 90 -- what the map should come out as")
figure.tight_layout()
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
figure, axis = plt.subplots(figsize=(6.5, 3.6))
for value, name in ((70.0, "white matter"), (110.0, "grey matter"), (300.0, "CSF")):
    decay = acquisition.simulate(T2=value, M0=1.0, offset=FLOOR)
    axis.semilogy(TE.numpy(), decay.numpy(), "-o", ms=3, label=f"{name}, T2 {value:.0f} ms")
axis.axhline(FLOOR, color="k", ls="--", lw=1)
axis.text(TE[-1], FLOOR * 1.15, "noise floor", ha="right", fontsize=8)
axis.set(xlabel="Echo time [ms]", ylabel="signal", title="what is measured")
axis.legend(fontsize=8), axis.grid(alpha=0.3)
figure.tight_layout()
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
def footprint(problem):
    """MiB the fitted estimator itself holds."""
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

fit = ParameterMapping(
    acquisition, noise_std=NOISE_STD, seed=0, **BOUNDS
).train(NonlinearLeastSquares(bounds=BOUNDS, initial=START))

maps = fit(measured)  # {"T2": ..., "M0": ..., "offset": ...}

# sphinx_gallery_start_ignore
def fitted(unknown):
    """Fit these parameters, holding the rest of the model at the truth."""
    held = {name: START[name] for name in START if name not in unknown}
    if "offset" in held:
        held["offset"] = FLOOR
    problem = ParameterMapping(
        acquisition.bind(**held),
        noise_std=NOISE_STD,
        seed=0,
        **{name: BOUNDS[name] for name in unknown},
    )
    start = time.perf_counter()
    problem.train(
        NonlinearLeastSquares(
            bounds={name: BOUNDS[name] for name in unknown},
            initial={name: START[name] for name in unknown},
        )
    )
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return found, training, timing, footprint(problem), peak


two_maps, two_training, two_time, two_model, two_peak = fitted(("T2", "M0"))
three_maps, three_training, three_time, three_model, three_peak = fitted(
    ("T2", "M0", "offset")
)

print(f"fitted floor, median {float(three_maps['offset'].median()):.4f} "
      f"against {FLOOR}")
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

match = ParameterMapping(
    acquisition.bind(M0=1.0, offset=0.0), T2=T2_GRID, seed=0
).train(DictionaryMatcher())

# %%
#
# To model the floor the match has to put it on the grid, and the grid is then
# the product of the two.
#
offsets = torch.linspace(0.0, 0.15, 40)
grid_t2, grid_offset = torch.meshgrid(T2_GRID, offsets, indexing="ij")

wide = ParameterMapping(
    acquisition.bind(M0=1.0),
    T2=grid_t2.reshape(-1),
    offset=grid_offset.reshape(-1),
    seed=0,
).train(DictionaryMatcher())


# sphinx_gallery_start_ignore
def matched(floors):
    """Fit a matcher over T2, and over this many values of the offset."""
    if floors == 1:
        problem = ParameterMapping(
            acquisition.bind(M0=1.0, offset=0.0), T2=T2_GRID, seed=0
        )
        atoms = T2_GRID.numel()
    else:
        offsets = torch.linspace(0.0, 0.15, floors)
        grid_t2, grid_offset = torch.meshgrid(T2_GRID, offsets, indexing="ij")
        problem = ParameterMapping(
            acquisition.bind(M0=1.0),
            T2=grid_t2.reshape(-1),
            offset=grid_offset.reshape(-1),
            seed=0,
        )
        atoms = grid_t2.numel()
    start = time.perf_counter()
    problem.train(DictionaryMatcher())
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
print(f"\n{'method':<30}{'atoms':>8}{'train':>8}{'map':>8}{'model':>10}"
      f"{'peak':>10}{'T2':>8}")
print("-" * 82)
for floors in FLOOR_VALUES:
    atoms, found, training, timing, model, peak = matches[floors]
    name = "match, T2 only" if floors == 1 else f"match, T2 x {floors} offsets"
    print(f"{name:<30}{atoms:>8}{training:7.1f}s{timing:7.2f}s"
          f"{model:6.1f} MiB{peak:6.0f} MiB{error(found['T2'], truth):7.1f}%")
for name, training, timing, model, peak, found in (
    ("fit, T2 + M0, floor known", two_training, two_time, two_model, two_peak,
     two_maps),
    ("fit, T2 + M0 + offset", three_training, three_time, three_model, three_peak,
     three_maps),
):
    print(f"{name:<30}{'--':>8}{training:7.1f}s{timing:7.2f}s"
          f"{model:6.1f} MiB{peak:6.0f} MiB{error(found['T2'], truth):7.1f}%")
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
figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
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
    axis.grid(alpha=0.3), axis.legend(fontsize=8)
figure.tight_layout()
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


shown = {
    "match, T2 only": matches[1][1]["T2"],
    "match, T2 x 40 offsets": matches[40][1]["T2"],
    "fit, T2 + M0 + offset": three_maps["T2"],
}

columns = 1 + len(shown)
figure, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 6.8))
panel(axes[0, 0], T2_true, "viridis", (0, 350))
axes[0, 0].set_title("truth", fontsize=10)
axes[1, 0].set_visible(False)

residuals = {name: np.abs(painted(values) - T2_true) for name, values in shown.items()}
top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
for column, (name, values) in enumerate(shown.items(), start=1):
    panel(
        axes[0, column],
        painted(values),
        "viridis",
        (0, 350),
        label="T2 [ms]" if column == columns - 1 else None,
    )
    axes[0, column].set_title(name, fontsize=9)
    panel(
        axes[1, column],
        residuals[name],
        "inferno",
        (0.0, top or 1.0),
        label="|error| [ms]" if column == columns - 1 else None,
    )
figure.tight_layout()
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
