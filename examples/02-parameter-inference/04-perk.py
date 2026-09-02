"""
=============================
PERK: kernel ridge regression
=============================

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
PANEL = (PAGE_WIDTH - 3 * BAR_WIDTH) / 3  # one image panel


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
from torchsim import Subspace
from torchsim.estimators import PERK, DictionaryMatcher
from torchsim.simulators import MRFSimulator

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

figure, axes = canvas(1, 3, mask.shape, bars=3, extra=1.1)
for axis, values, name in (
    (axes[0, 0], T1_true, "T1"),
    (axes[0, 1], T2_true, "T2"),
    (axes[0, 2], M0_true, "M0"),
):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
# sphinx_gallery_end_ignore

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
# returns transverse magnetization to the imaginary axis: the trajectory comes
# back real to within 3e-8, which halves both the dictionary and the
# arithmetic that searches it.
#
fingerprints = acquisition.simulate(
    T1=torch.tensor([500.0, 833.0, 2569.0]), T2=torch.tensor([70.0, 83.0, 329.0])
).real

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3))
axes[0].plot(repetition.numpy(), flip.numpy())
axes[0].set(xlabel="Repetition", ylabel="Flip angle [deg]", title="the schedule")
axes[0].grid(alpha=0.3)
for row, name in enumerate(("white matter", "grey matter", "CSF")):
    axes[1].plot(repetition.numpy(), fingerprints[row].numpy(force=True), label=name)
axes[1].set(xlabel="Repetition", ylabel="signal", title="fingerprints")
axes[1].grid(alpha=0.3)
key(axes[1], ncols=3)
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


# sphinx_gallery_start_ignore
def footprint(problem):
    """MiB the fitted estimator itself holds -- a dictionary, or a regression."""
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


def log_uniform(low, high, count):
    """``count`` draws spread evenly in the logarithm."""
    span = torch.rand(count, generator=prior)
    return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))


# sphinx_gallery_end_ignore

# %%
#
# The problem, stated once
# ------------------------
#
# What is unknown, over what range, from what acquisition, at what noise level.
# The method that fills it in is a separate choice, and the only thing that
# changes between the answers below.
#
# Both relaxation times span more than a decade, so the prior is drawn
# logarithmically: sampling uniformly would spend most of the budget on long
# T1, where the trajectories are nearly parallel and carry little information.
#
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)

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
training_signals, _, _ = (
    PERK(acquisition)
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

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.59))
axis.semilogy(ranks, outside, "o-", label="left outside the basis")
axis.axhline(noise_energy, color="crimson", linestyle="--", label="added by the noise")
axis.axvline(RANK, color="0.5", linewidth=1)
axis.set(
    xlabel="rank",
    ylabel="relative energy",
    title="what projecting through the basis loses",
)
axis.grid(alpha=0.3)
key(axis, ncols=2)
# sphinx_gallery_end_ignore

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


FEATURES = 1000

perk = PERK(acquisition, n_features=FEATURES, regularization=1e-6, normalize=True).fit(
    T1=log_uniform(*T1_RANGE, SAMPLES),
    T2=log_uniform(*T2_RANGE, SAMPLES),
    noise_std=NOISE_STD,
    seed=0,
    rank=RANK,
    samples=SAMPLES,
)

maps = perk(measured)  # {"T1": ..., "T2": ...}, one value per voxel


# sphinx_gallery_start_ignore
def regressed(features):
    """Train a regression of this size, then map the slice."""
    start = time.perf_counter()
    problem = PERK(
        acquisition, n_features=features, regularization=1e-6, normalize=True
    ).fit(
        T1=log_uniform(*T1_RANGE, SAMPLES),
        T2=log_uniform(*T2_RANGE, SAMPLES),
        noise_std=NOISE_STD,
        seed=0,
        rank=RANK,
        samples=SAMPLES,
    )
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return found, training, timing, footprint(problem), peak


regressions = {features: regressed(features) for features in (500, 1000, 4000)}
perk_maps, perk_training, perk_mapping, perk_model, perk_peak = regressions[FEATURES]
# sphinx_gallery_end_ignore

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

# sphinx_gallery_start_ignore
start = time.perf_counter()
# sphinx_gallery_end_ignore
matched = DictionaryMatcher(acquisition).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), rank=RANK, seed=0
)

# sphinx_gallery_start_ignore
match_training = time.perf_counter() - start
match_maps, match_mapping, match_peak = mapped(matched)
match_model = footprint(matched)
print(
    f"dictionary: {grid_t1.numel()} atoms at rank {RANK}; "
    f"regression: {FEATURES} features from {SAMPLES} training draws"
)
# sphinx_gallery_end_ignore

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


M0_map = proton_density(maps)

# sphinx_gallery_start_ignore
estimates = {
    "match": (match_maps, proton_density(match_maps)),
    "PERK": (perk_maps, proton_density(perk_maps)),
}
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
def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


print(
    f"\n{'method':<26}{'train':>8}{'map':>8}{'model':>10}{'peak':>10}{'T1':>8}{'T2':>8}"
)
print("-" * 78)
rows = [
    (
        f"match, rank {RANK}",
        match_training,
        match_mapping,
        match_model,
        match_peak,
        match_maps,
    )
]
for features, (found, training, timing, model, peak) in regressions.items():
    rows.append((f"PERK, {features} features", training, timing, model, peak, found))

for name, training, timing, model, peak, found in rows:
    print(
        f"{name:<26}{training:7.1f}s{timing:7.2f}s"
        f"{model:6.1f} MiB{peak:6.0f} MiB"
        f"{error(found['T1'], truth['T1']):7.1f}%"
        f"{error(found['T2'], truth['T2']):7.1f}%"
    )
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


panels = [
    ("T1", T1_true, {name: found["T1"] for name, (found, _) in estimates.items()}),
    ("T2", T2_true, {name: found["T2"] for name, (found, _) in estimates.items()}),
    ("M0", M0_true, {name: m0 for name, (_, m0) in estimates.items()}),
]

figure, axes = canvas(len(panels), 1 + len(estimates), mask.shape)
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

# Each parameter's errors on a scale of their own: an error map read at the
# scale of the map it came from is a black rectangle.
figure, axes = canvas(len(panels), len(estimates), mask.shape)
for row, (name, reference, found) in enumerate(panels):
    label = STYLE[name][2]
    residuals = {
        method: np.abs(painted(values) - reference) for method, values in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
    for column, (method, values) in enumerate(residuals.items()):
        error = panel(
            axes[row, column],
            values,
            "inferno",
            (0.0, top or 1.0),
            title=f"\u0394 {method}" if row == 0 else None,
        )
    unit = label[label.find(" [") :] if "[" in label else ""
    scalebar(error, axes[row], f"|error|{unit}")
# sphinx_gallery_end_ignore

# %%
#
# How sure is it?
# ---------------
#
# A map without an error bar is an assertion. Asking for one is a keyword:
# ``uncertainty=True`` returns the standard deviation the noise leaves on each
# map beside the maps themselves, under the noise the fit was told about.
#
# What PERK does to answer is run itself. Mapping the slice is a matrix
# multiply and takes hundredths of a second, so rather than linearize -- which
# at a realistic noise level understates the spread, because the features are
# cosines and the noise turns them far enough to matter -- it adds the declared
# noise a couple of dozen times over and spreads the answers.
#
maps, deviation = perk(measured, uncertainty=True)

# %%
#
# What that spread should be compared against is the Cramer-Rao bound: the
# lowest standard deviation any *unbiased* estimate could have from this train
# at this noise level. It is a property of the acquisition rather than of the
# method, so it says how much of the distance between the map and the truth the
# sequence is responsible for.
#
# The bound is read off the Jacobian at the true relaxation times, scaled by
# the proton density each voxel actually has -- a voxel with half the
# magnetization carries half the signal and so twice the standard deviation.
#
_, sensitivity = acquisition.jacobian("T1 T2".split(), **truth)
sensitivity = sensitivity.real * density[:, None, None]
floor = torchsim.crlb(
    sensitivity, noise_variance=NOISE_STD**2, singular="infinite"
).sqrt()
bound = {"T1": floor[:, 0], "T2": floor[:, 1]}

# %%
#
# And the third term, which neither of those contains: the bias. A regression
# trained on a prior answers with the prior where the data is weak, and that
# error is the same in every noise realization -- so repeating the measurement
# never reveals it and the spread does not contain it.
#
bias = {name: maps[name] - truth[name] for name in truth}

# sphinx_gallery_start_ignore
print(
    f"{'':<10}{'PERK spread':>14}{'CRLB':>10}{'|bias|':>10}   (median over the brain)"
)
for name in ("T1", "T2"):
    print(
        f"{name:<10}{float(deviation[name].median()):11.1f} ms"
        f"{float(bound[name].median()):7.1f} ms"
        f"{float(bias[name].abs().median()):7.1f} ms"
    )

bands = (
    ("white matter", truth["T1"] < 1000.0),
    ("grey matter", (truth["T1"] >= 1000.0) & (truth["T1"] < 2000.0)),
    ("CSF", truth["T1"] >= 2000.0),
)
print()
print(
    f"{'':<14}{'voxels':>8}{'T1 spread':>12}{'T1 CRLB':>10}"
    f"{'T2 spread':>12}{'T2 CRLB':>10}"
)
for label, where in bands:
    print(
        f"{label:<14}{int(where.sum()):8d}"
        f"{float(deviation['T1'][where].median()):9.1f} ms"
        f"{float(bound['T1'][where].median()):7.1f} ms"
        f"{float(deviation['T2'][where].median()):9.1f} ms"
        f"{float(bound['T2'][where].median()):7.1f} ms"
    )

figure, axes = canvas(2, 3, mask.shape)
for row, name in enumerate(("T1", "T2")):
    columns = (
        ("PERK spread", deviation[name]),
        ("CRLB", bound[name]),
        ("|bias|", bias[name].abs()),
    )
    top = max(
        float(np.percentile(values.numpy(force=True), 98)) for _, values in columns
    )
    for column, (title, values) in enumerate(columns):
        handle = panel(
            axes[row, column],
            painted(values),
            "inferno",
            (0.0, top or 1.0),
            title=title if row == 0 else None,
            ylabel=STYLE[name][2] if column == 0 else None,
        )
    scalebar(handle, axes[row], "ms")
# sphinx_gallery_end_ignore

# %%
#
# The three panels are on one scale per parameter, which is what makes them
# worth putting beside each other.
#
# The spread follows the sequence. It is smallest in white matter, larger in
# grey, and largest in CSF -- and the bound moves the same way, because a long
# T1 is what this train resolves least. A wide error bar in the ventricles is
# the acquisition saying so, not the regression failing.
#
# What the two parameters do not share is how far above the bound they sit. T1
# is estimated to within a small multiple of what the train allows; T2 is
# several times worse than allowed, and it is T2 whose error moves when the
# feature count is swept. The bound separates those two statements: one is a
# sequence to redesign, the other a regression to enlarge.
#
# The bias is the term to keep in view. It is the size of the spread here, and
# no amount of repeating the scan would show it -- which is why the map beside
# the truth, further up, is not replaced by the map beside its error bar.
#
# A method whose answer is a grid point has no such number to state. Asking a
# :class:`~torchsim.DictionaryMatcher` for one says so rather than inventing
# it: a match moves in steps, so the noise does not move it a little.
# :class:`~torchsim.NonlinearLeastSquares` does state one, and there it is the
# standard error the fit reports -- the inverse Fisher matrix at the solution,
# which is the bound above read at the answer rather than at a truth nobody
# has.
