"""
================================================
One unknown: reading T1 off an MP2RAGE curve
================================================

A dictionary spans the parameters jointly, and its size is the product of the
grids. With a single unknown that product is one grid, and the dictionary
degenerates: the atoms lie on a *curve* rather than filling a space, and the
nearest one is found by looking along it.

Once the atoms are on a curve, interpolating between the two nearest costs
nothing and removes the grid spacing from the answer entirely -- which is the
only thing a matched estimate was limited by once the signal is fit to be
matched at all. That is what :class:`~torchsim.LookupTable` does, and it is how
an MP2RAGE T1 map is made.

This example maps a BrainWeb slice from a two-block MP2RAGE, by interpolation
and by matching the same curve, and sweeps the number of points to show which
of the two is limited by it.
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
from torchsim.estimators import DictionaryMatcher, LookupTable
from torchsim.simulators import MP2RAGESimulator

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
tissue_T1 = np.array([float(row["T1 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(row["PD (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior, and an image is
# drawn from its first row down. Flipping here puts anterior at the top of
# every figure below rather than in each one of them.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5

# A mixed voxel is given the relaxation time its tissues average to. That is
# the parameter a fit can actually return: no single T1 explains a voxel that
# is half one tissue and half another.
share = np.maximum(occupancy, 1e-6)
T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

truth = torch.as_tensor(T1_true[mask].copy())
density = torch.as_tensor(M0_true[mask].copy())

figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
panel(axes[0], T1_true, "magma", (0, 3000), label="T1 [ms]")
panel(axes[1], M0_true, "magma", (0, 1.1), label="M0")
figure.suptitle("BrainWeb subject 0, slice 90 -- what the map should come out as")
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# The protocol
# ------------
#
# One inversion, two spoiled gradient-echo blocks read at two inversion times.
# Both blocks sample the centre of k-space of their own shot train, so what a
# voxel contributes is two numbers -- and T1 is the only tissue property that
# moves them, because the train spoils after every readout and nothing
# transverse survives an interval.
#
PROTOCOL = dict(
    TI=(800.0, 2700.0),
    flip=(4.0, 5.0),
    TRspgr=6.7,
    TRmp2rage=6000.0,
    nshots=128,
)
INVERSION_EFFICIENCY = 0.96

acquisition = MP2RAGESimulator(**PROTOCOL, inv_efficiency=INVERSION_EFFICIENCY)

# %%
#
# The curve
# ---------
#
# Neither block on its own says T1: both are scaled by the proton density and
# by the receive gain. Their *unified* combination divides that scale out, and
# what is left is a number between -0.5 and 0.5 that depends on T1 alone.
#
# Which combination makes a curve monotonic is a property of the sequence
# rather than of the table, so it is handed over rather than assumed.
#


def unified(blocks):
    """The MP2RAGE unified image: scale-free, and a function of T1 alone."""
    return (blocks[..., 0] * blocks[..., 1]) / blocks.square().sum(-1).clamp_min(1e-12)


# %%
#
# The curve is not monotonic over every T1, and where it turns back on itself
# it has no inverse. :class:`~torchsim.LookupTable` keeps the longest
# monotonic run and reports what it spans, so the range the protocol can
# actually invert is a number rather than an assumption.
#
sweep = torch.arange(50.0, 6000.0, 10.0)
curve = unified(acquisition.simulate(T1=sweep, M0=1.0))

# sphinx_gallery_start_ignore
turning = int(curve.argmin()) if curve[0] > curve[-1] else int(curve.argmax())

figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
blocks = acquisition.simulate(T1=sweep, M0=1.0)
axes[0].plot(sweep.numpy(), blocks[:, 0].numpy(), label=f"TI = {PROTOCOL['TI'][0]:.0f} ms")
axes[0].plot(sweep.numpy(), blocks[:, 1].numpy(), label=f"TI = {PROTOCOL['TI'][1]:.0f} ms")
axes[0].set(xlabel="T1 [ms]", ylabel="magnetization", title="the two blocks")
axes[0].legend(fontsize=8)

axes[1].plot(sweep.numpy(), curve.numpy(), color="crimson")
axes[1].axvline(float(sweep[turning]), color="k", ls="--", lw=1)
axes[1].set(
    xlabel="T1 [ms]",
    ylabel="unified image",
    title="the curve a T1 is read off",
)
for axis in axes:
    axis.grid(alpha=0.3)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# The measurement
# ---------------
#
# Both blocks, at the true T1 and proton density of every brain voxel, with
# noise at half a percent of the peak magnetization.
#
clean = acquisition.simulate(T1=truth, M0=density)
NOISE_STD = float(0.005 * clean.abs().max())

generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)


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


def estimated(method, points):
    """Fit this method over a grid of this many points, then map the slice."""
    grid = torch.linspace(50.0, 6000.0, points)
    problem = ParameterMapping(acquisition.bind(M0=1.0), T1=grid, seed=0)
    start = time.perf_counter()
    problem.train(method)
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return problem, found, training, timing, footprint(problem), peak


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


# sphinx_gallery_end_ignore

# %%
#
# Two estimators over the same points
# -----------------------------------
#
# Both methods are given the same T1 grid. The match compares the two-block
# signal against every atom and takes the nearest; the table reduces both
# blocks to the unified number and interpolates along the curve, so ``combine``
# is the whole of what it needs to be told.
#
# Neither is told the range in advance -- what the table can invert falls out
# of the curve, and where the match saturates falls out of the grid.
#
grid = torch.linspace(50.0, 6000.0, 60)

table = ParameterMapping(acquisition.bind(M0=1.0), T1=grid, seed=0).train(
    LookupTable(combine=unified)
)

maps = table(measured)  # {"T1": ...}, one value per voxel

match = ParameterMapping(acquisition.bind(M0=1.0), T1=grid, seed=0).train(
    DictionaryMatcher()
)

# %%
#
# Sweeping the grid the two are given says how much of the difference is the
# method and how much is the sampling. Times are the best of three passes over
# the slice.
#

# sphinx_gallery_start_ignore
POINTS = (30, 60, 120, 250, 500, 1000, 2000)
matched = {}
looked_up = {}
for points in POINTS:
    _, found, _, timing, _, _ = estimated(DictionaryMatcher(), points)
    matched[points] = (error(found["T1"], truth), timing)
    problem, found, _, timing, _, _ = estimated(LookupTable(combine=unified), points)
    looked_up[points] = (error(found["T1"], truth), timing)

print(f"\n{'points':>7}{'match':>10}{'table':>10}{'match':>10}{'table':>10}")
print(f"{'':>7}{'error':>10}{'error':>10}{'time':>10}{'time':>10}")
print("-" * 47)
for points in POINTS:
    print(f"{points:>7}{matched[points][0]:9.2f}%{looked_up[points][0]:9.2f}%"
          f"{1e3 * matched[points][1]:8.1f}ms{1e3 * looked_up[points][1]:8.1f}ms")
# sphinx_gallery_end_ignore

# %%
#
# The table is at its floor from the coarsest grid tried and does not move
# again. The match starts an order of magnitude worse and climbs to the same
# place, and everything it spends getting there is spent on the grid: the
# search is one comparison per atom per voxel, so its time grows with the point
# count while the table's binary search grows with the logarithm of it.
#
# The floor both arrive at is the noise, which is what is left once the grid is
# gone. That is the claim, and it is narrower than "interpolation is more
# accurate": a fine enough grid matches a table exactly, and the table's
# advantage is that it never had to be told how fine.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(POINTS, [matched[n][0] for n in POINTS], "-o", label="match")
axes[0].plot(POINTS, [looked_up[n][0] for n in POINTS], "-*", label="lookup table")
axes[0].set(
    xlabel="Points on the curve",
    ylabel="Median relative error [%]",
    xscale="log",
    yscale="log",
    title="what the grid costs",
)
axes[1].plot(POINTS, [1e3 * matched[n][1] for n in POINTS], "-o", label="match")
axes[1].plot(POINTS, [1e3 * looked_up[n][1] for n in POINTS], "-*", label="lookup table")
axes[1].set(
    xlabel="Points on the curve",
    ylabel="Time to map the slice [ms]",
    xscale="log",
    yscale="log",
    title="what it costs to pay it",
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
# At the point count each method needs: the table at sixty, the match at a grid
# fine enough that the grid is no longer what limits it.
#

# sphinx_gallery_start_ignore
TABLE_POINTS = 60
MATCH_POINTS = 2000

problem, table_maps, table_training, table_time, table_model, table_peak = estimated(
    LookupTable(combine=unified), TABLE_POINTS
)
_, match_maps, match_training, match_time, match_model, match_peak = estimated(
    DictionaryMatcher(), MATCH_POINTS
)
print(f"the table keeps {problem.method.rank} of {TABLE_POINTS} points -- the "
      f"monotonic run -- and spans unified "
      f"{problem.method.span[0]:.2f} to {problem.method.span[1]:.2f}")
# sphinx_gallery_end_ignore

# %%
#
# Neither method estimates M0, and neither has to. Both answer with a T1, and
# the two blocks that T1 predicts are a shape the measurement is some multiple
# of -- so the multiple is a projection, one inner product per voxel.
#


def proton_density(maps):
    """The scale the measurement is, of the blocks the answer predicts."""
    predicted = acquisition.simulate(T1=maps["T1"], M0=1.0)
    return (predicted * measured).sum(-1) / predicted.square().sum(-1).clamp_min(1e-12)


M0_map = proton_density(maps)

# sphinx_gallery_start_ignore
estimates = {
    f"lookup, {TABLE_POINTS} points": (table_maps, proton_density(table_maps)),
    f"match, {MATCH_POINTS} atoms": (match_maps, proton_density(match_maps)),
}

print(f"\n{'method':<24}{'train':>9}{'map':>9}{'model':>10}{'peak':>10}"
      f"{'T1':>8}{'M0':>8}")
print("-" * 78)
for name, training, timing, model, peak in (
    (f"lookup, {TABLE_POINTS} points", table_training, table_time, table_model,
     table_peak),
    (f"match, {MATCH_POINTS} atoms", match_training, match_time, match_model,
     match_peak),
):
    found, m0 = estimates[name]
    print(f"{name:<24}{training:8.2f}s{1e3 * timing:7.1f}ms"
          f"{model:6.2f} MiB{peak:6.0f} MiB"
          f"{error(found['T1'], truth):7.2f}%{error(m0, density):7.2f}%")
# sphinx_gallery_end_ignore

# %%

# sphinx_gallery_start_ignore
def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


panels = [
    ("T1 [ms]", T1_true, {name: found["T1"] for name, (found, _) in estimates.items()},
     (0, 3000)),
    ("M0", M0_true, {name: m0 for name, (_, m0) in estimates.items()}, (0, 1.1)),
]

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
# What makes this work is the single unknown, and it is worth being clear about
# what bought it. The train spoils after every readout, so no T2 enters; the
# proton density and the receive gain divide out of the unified combination;
# and the inversion efficiency was *given* rather than estimated. Each of those
# is an assumption, and each is what turns a surface into a curve.
#
# The one that usually breaks first is the transmit field. A flip angle that is
# not what was prescribed moves the curve, and the T1 read off it moves with
# it -- which is why an MP2RAGE protocol is designed for a curve that is as
# flat in B1 as it can be made, and why a separately measured B1 map is
# sometimes handed in alongside. A property measured per voxel is what
# :class:`~torchsim.ParameterMapping` calls ``known``, and it is the one thing
# a table cannot take: a curve per voxel is not a curve.
#
# The other end of the range is the turning point. Beyond it the curve doubles
# back, the table keeps only the monotonic run, and every voxel past it is
# reported at the endpoint. For a brain at these inversion times that lies
# above CSF and nothing is lost; for a phantom of long-T1 solutions it would
# not, and the protocol rather than the estimator is what would have to change.
#
