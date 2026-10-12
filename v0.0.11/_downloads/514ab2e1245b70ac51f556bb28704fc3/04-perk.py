"""
=============================
PERK: kernel ridge regression
=============================

You map T1, T2 and proton density in a brain slice with PERK, vary the size of
the regression, and read the uncertainty it reports against the Cramér-Rao
bound.

PERK builds no dictionary. It is a kernel regression trained on signals drawn
from a prior instead of laid on a grid. At inference it projects a signal onto
a fixed set of random Fourier features and reads the estimate off a linear
combination of them. The cost per voxel does not depend on the number of
unknown parameters, and the training is paid once.

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

from blochsim import Subspace, crlb
from blochsim.estimators import PERK, DictionaryMatcher
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
# The readouts advance the configuration states and do not rewind them, so no
# transverse magnetization is returned to the imaginary axis and the signal
# evolution is real to float32 precision. The regression then needs only the
# real part. These are the evolutions of white matter, grey matter and CSF:
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
# The unknowns are T1 and T2. Both span more than a decade, so the prior is
# logarithmic: uniform sampling would spend most of the training draws on long
# T1, where the evolutions are nearly parallel.
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)

# %%
# Subspace basis
# --------------
#
# The 400 contrasts span far fewer than 400 independent directions.
# ``Subspace.fit`` computes a basis from simulated evolutions and reports the
# fraction of their energy that a given rank retains; one minus that fraction
# is the relative squared error of projecting onto the basis and back.
training_signals, _, _ = (
    PERK(simulator)
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
# adds, so rank 4 is used from here on.
#
# Training
# --------
#
# ``fit`` draws 20 000 (T1, T2) pairs from the prior, simulates them, adds the
# noise level of the scan, and solves the regularized linear system against
# 1000 random Fourier features. The fitted regression maps a measurement to
# the estimates directly:
FEATURES = 1000

perk = PERK(simulator, n_features=FEATURES, regularization=1e-6, normalize=True).fit(
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
        simulator, n_features=features, regularization=1e-6, normalize=True
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

# A compressed dictionary match over a grid fine enough not to limit it, kept as
# the reference row of the table below.
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 200)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 100)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")

start = time.perf_counter()
matched = DictionaryMatcher(simulator).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), rank=RANK, seed=0
)
match_training = time.perf_counter() - start
match_maps, match_mapping, match_peak = mapped(matched)
match_model = footprint(matched)
estimates = {
    "match": (match_maps, match_maps["M0"]),
    "PERK": (perk_maps, perk_maps["M0"]),
}
print(
    f"dictionary: {grid_t1.numel()} atoms at rank {RANK}; "
    f"regression: {FEATURES} features from {SAMPLES} training draws"
)
# sphinx_gallery_end_ignore

# %%
# Cost and accuracy
# -----------------
#
# The table compares PERK at three sizes with a dictionary match at rank 4 as
# the reference. Times are the best of three passes after a warm-up. *model* is
# the memory the fitted estimator holds between volumes. *peak* is the maximum
# GPU memory allocated while mapping the slice, blank on a machine without a
# GPU. Errors are medians over the brain voxels, relative to the truth.

# sphinx_gallery_start_ignore
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
        f"{model:6.1f} MiB{held(peak)}"
        f"{error(found['T1'], truth['T1']):7.1f}%"
        f"{error(found['T2'], truth['T2']):7.1f}%"
    )
# sphinx_gallery_end_ignore

# %%
# Maps
# ----
#
# The maps of both methods, and below them the absolute error on a scale of its
# own for each parameter.

# sphinx_gallery_start_ignore
panels = [
    ("T1", T1_true, {k: found["T1"] for k, (found, _) in estimates.items()}),
    ("T2", T2_true, {k: found["T2"] for k, (found, _) in estimates.items()}),
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
# Uncertainty
# -----------
#
# With ``uncertainty=True`` the regression returns a second set of maps, the
# expected distance of the estimate from the truth. PERK learns it during
# training from the residuals of its own fit, so reporting it costs a matrix
# multiplication and not a second pass over the volume:
maps, spread = perk(measured, uncertainty=True)

# %%
# The reference is the Cramér-Rao bound: the lowest standard deviation an
# unbiased estimator could reach from this train at this noise level. The bound
# is a property of the sequence, so the gap between it and the spread is what
# the method loses. The bound comes from the Jacobian of the signal with respect
# to T1 and T2, scaled by the proton density:
_signal, sensitivity = simulator.jacobian(["T1", "T2"], **truth)
sensitivity = sensitivity.real * density[:, None, None]
floor = crlb(sensitivity, noise_variance=NOISE_STD**2, singular="infinite")
bound = {"T1": floor[:, 0].sqrt(), "T2": floor[:, 1].sqrt()}

# sphinx_gallery_start_ignore
relative = {name: 100.0 * spread[name] / maps[name].clamp_min(1e-6) for name in bound}

print(f"{'':<6}{'PERK':>10}{'CRLB':>10}{'PERK':>9}   (median over the brain)")
for name in ("T1", "T2"):
    print(
        f"{name:<6}{float(spread[name].median()):7.1f} ms"
        f"{float(bound[name].median()):7.1f} ms"
        f"{float(relative[name].median()):8.1f}%"
    )

fig, axes = canvas(2, 3, mask.shape)
for row, name in enumerate(("T1", "T2")):
    cmap, _limits, label = STYLE[name]
    absolute = (("PERK", spread[name]), ("CRLB", bound[name]))
    top = max(
        float(np.percentile(values.numpy(force=True), 98)) for _, values in absolute
    )
    for column, (title, values) in enumerate(absolute):
        handle = panel(
            axes[row, column],
            painted(values),
            cmap,
            (0.0, top or 1.0),
            title=title if row == 0 else None,
            ylabel=label if column == 0 else None,
        )
    fig.colorbar(handle, ax=axes[row, :2], label="ms", shrink=0.92, aspect=20)
    relative_map = panel(
        axes[row, 2],
        painted(relative[name]),
        cmap,
        (0.0, float(np.percentile(relative[name].numpy(force=True), 98)) or 1.0),
        title="PERK, relative" if row == 0 else None,
    )
    fig.colorbar(relative_map, ax=axes[row, 2], label="%", shrink=0.92, aspect=20)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The two absolute columns share a scale in each row; the third column is the
# PERK spread as a percentage of the relaxation time, which shows whether a
# spread of ten milliseconds is small.
#
# Both quantities are largest in CSF, whose long T1 this train resolves least.
# The gap between the spread and the bound is a separate matter: the T1 spread
# is a few times the bound, the T2 spread is more than an order of magnitude
# above it, and the T2 error changed most when the number of features was
# varied. The T1 gap points to the sequence, the T2 gap to the size of the
# regression.
#
# The spread is not the noise alone. A regression trained on a prior returns
# the prior where the data are weak, and is wrong in the same way in every
# realization, so repeating the scan would not reveal that part.
#
# Specification
# -------------
#
# What this page did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, map T1, T2 and M0 in BrainWeb subject 0, slice 90, from an
#    inversion-prepared MRF train of 400 repetitions at 2% noise. Fit a PERK
#    regression (20 000 log-uniform training draws, subspace rank 4, 500, 1000
#    and 4000 random Fourier features) and compare it with a rank-4
#    DictionaryMatcher on the cost table and the maps. Return the PERK
#    uncertainty and compare it with the Cramér-Rao bound computed from the
#    simulator Jacobian.
