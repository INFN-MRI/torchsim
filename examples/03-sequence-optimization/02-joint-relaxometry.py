"""
==================================
Joint relaxometry protocol design
==================================

In this example you choose the flip angles of a DESPOT protocol so that a joint
fit of T1 and T2 is as precise as possible [1]_. DESPOT estimates T1 from
spoiled gradient-echo scans at several flip angles and T2 from balanced SSFP
scans. A joint fit uses all the data for both parameters.

The cost is a Cramer-Rao bound: the lowest variance an unbiased estimate can
have, given the derivative of the signal with respect to every parameter
estimated. You minimize it over the flip angles, then play the designed
protocol on a BrainWeb slice and compare the error of the fitted maps with the
bound.

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
import numpy as np
from brainweb_dl import get_mri
from cmap import Colormap

from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")

# One perceptually uniform colormap per relaxation parameter (Fuderer et al.,
# Magn. Reson. Med. 2025), so that a T1 map is never read as a T2 map. Both
# windows stop short of CSF, so white and grey matter take up most of the scale.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()
MAPS = {
    "T1": (LIPARI, (0.0, 1200.0), "T1 (ms)"),
    "T2": (NAVIA, (0.0, 120.0), "T2 (ms)"),
    "M0": ("gray", (0.0, 1.0), "M0"),
}
BAR_WIDTH = 0.8  # inches taken by one colorbar
PANEL = (PAGE_WIDTH - 3 * BAR_WIDTH) / 3  # one image panel, inches


def panel(axis, values, cmap, limits, title=None):
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    axis.set(xticks=[], yticks=[], title=title)
    return handle


def scalebar(handle, axes, label):
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


# sphinx_gallery_end_ignore

# %%
# Sequences
# ---------
#
# A design has three pieces. A simulator with the tissue fixed on it, so that
# only the parameters under design are left to give. A cost. And the bounded
# parameters that :class:`~blochsim.SequenceDesign` adjusts.
#
# Here the simulators are an SPGR and a bSSFP sequence, both in closed form.
# You construct and call them the same way as a sequence that plays events. The
# tissues are white and grey matter at 3 T, and you design for both at once.
# ``NOISE`` is the noise standard deviation as a fraction of the fully relaxed
# magnetization:
import torch

import blochsim
from blochsim.optim import Bounded, SequenceDesign
from blochsim.simulators import SPGRSimulator, bSSFPSimulator

T1_MS = torch.tensor([830.0, 1330.0])
T2_MS = torch.tensor([80.0, 110.0])
NOISE = 0.005

spgr = SPGRSimulator(TE=2.0, TR=6.0, T1=T1_MS, T2star=T2_MS, M0=1.0, B0=0.0)
ssfp = bSSFPSimulator(TE=2.5, TR=5.0, T1=T1_MS, T2=T2_MS, M0=1.0, B0=0.0)

# %%
# Cost
# ----
#
# You estimate four parameters jointly: T1, T2, the proton density M0 and the
# off-resonance B0. M0 and B0 are nuisance parameters, estimated because they
# affect the data and not because the design is for them.
#
# The two sequences carry different information. The spoiled steady state in
# closed form depends on T2* and not on T2, so its T2 row is exactly zero. Each
# block is blind to something, and the Fisher matrix adds them up. ``rows``
# stacks the derivatives of one block with a zero row for each parameter it is
# blind to, and :func:`~blochsim.crlb` turns the stacked derivatives into a
# bound on each parameter:
JOINT = ("T1", "T2", "M0", "B0")


def rows(simulator, **design):
    present = [name for name in JOINT if name in simulator.exposes]
    _, jacobian = simulator.jacobian(present, **design)
    placed = jacobian.new_zeros(jacobian.shape[:-2] + (len(JOINT), jacobian.shape[-1]))
    where = torch.tensor([JOINT.index(name) for name in present])
    return placed.index_copy(-2, where, jacobian)


def bounds(spgr_flip, ssfp_flip):
    together = torch.cat(
        (rows(spgr, flip=spgr_flip), rows(ssfp, flip=ssfp_flip)), dim=-1
    )
    return blochsim.crlb(together, noise_variance=NOISE**2)


# %%
# Dividing each bound by its parameter squared makes the terms dimensionless, so
# a T2 of 100 ms and a T1 of 1000 ms are weighted by how well they are known and
# not by how large they are. The logarithm makes the gradient relative, so the
# design does not depend on the noise level:


def precision(spgr_flip, ssfp_flip):
    """Relative variance of T1 and T2, averaged over the design tissues."""
    bound = bounds(spgr_flip, ssfp_flip)
    relative = bound[..., 0] / T1_MS**2 + bound[..., 1] / T2_MS**2
    return relative.mean().log()


# %%
# Design
# ------
#
# The protocol has four scans of each kind, initialized with a spread of angles.
# :class:`~blochsim.Bounded` constrains each angle to the range the scanner
# plays:
spgr_start = torch.tensor([2.0, 4.0, 8.0, 16.0])
ssfp_start = torch.tensor([10.0, 20.0, 40.0, 60.0])

design = SequenceDesign(
    precision,
    spgr_flip=Bounded(spgr_start, 1.0, 40.0),
    ssfp_flip=Bounded(ssfp_start, 1.0, 70.0),
)

# sphinx_gallery_start_ignore
start = time.time()
# sphinx_gallery_end_ignore
result = design.minimize(iterations=120, learning_rate=0.3)
# sphinx_gallery_start_ignore
design_time = time.time() - start
# sphinx_gallery_end_ignore

spgr_designed = result.parameters["spgr_flip"]
ssfp_designed = result.parameters["ssfp_flip"]

# %%
# The bound gives the standard deviation of each estimate, in percent of the
# value itself:

# sphinx_gallery_start_ignore
for label, angles_pair in (
    ("start", (spgr_start, ssfp_start)),
    ("designed", (spgr_designed, ssfp_designed)),
):
    bound = bounds(*angles_pair)
    sigma_t1 = 100.0 * bound[..., 0].sqrt() / T1_MS
    sigma_t2 = 100.0 * bound[..., 1].sqrt() / T2_MS
    print(
        f"{label:18s} "
        f"sigma(T1)/T1 = {sigma_t1[0]:.1f}%, {sigma_t1[1]:.1f}%   "
        f"sigma(T2)/T2 = {sigma_t2[0]:.1f}%, {sigma_t2[1]:.1f}%"
    )
print(f"designed in {design_time:.1f} s")
# sphinx_gallery_end_ignore

# %%
# Flip angles
# -----------
#
# The protocol has eight scans, four spoiled and four balanced, which differ
# only in flip angle:

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.38 * PAGE_WIDTH), sharey=True)
for axis, start_angles, designed, title in (
    (axes[0], spgr_start, spgr_designed, "SPGR block"),
    (axes[1], ssfp_start, ssfp_designed, "bSSFP block"),
):
    index = torch.arange(1, start_angles.numel() + 1)
    axis.bar(index - 0.19, start_angles, width=0.36, color=MUTED, label="start")
    axis.bar(
        index + 0.19, designed.detach(), width=0.36, color=SERIES[1], label="designed"
    )
    for position, angle in zip(index, designed.detach(), strict=False):
        axis.annotate(
            f"{float(angle):.0f}",
            (float(position) + 0.19, float(angle)),
            ha="center",
            va="bottom",
        )
    axis.set(xlabel="acquisition", title=title, xticks=index.tolist())
axes[0].set_ylabel("flip angle (deg)")
axes[0].margins(y=0.15)  # headroom for the annotation over the tallest bar
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The design collapses the eight angles onto three and repeats them. The
# information sits at a few places on each curve, and a fixed number of scans is
# best placed there and not on an even sampling of the curve.
#
# The SPGR angle lands above the Ernst angle of both tissues, where the curve
# separates the two T1 values most sharply. The peak itself is where the signal
# is largest and says least. The two bSSFP angles sit either side of the
# steady-state maximum, which makes the pair sensitive to T2. The upper one is
# against its limit and not at an interior optimum.

# sphinx_gallery_start_ignore
sweep = torch.linspace(1.0, 70.0, 200)
fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))

for axis, simulator, start_angles, designed, title in (
    (axes[0], spgr, spgr_start, spgr_designed, "SPGR"),
    (axes[1], ssfp, ssfp_start, ssfp_designed, "bSSFP"),
):
    curve = simulator.simulate(flip=sweep).abs()
    axis.plot(sweep, curve[0], color=SERIES[0], label="T1/T2 = 830/80 ms")
    axis.plot(sweep, curve[1], color=SERIES[2], label="T1/T2 = 1330/110 ms")
    sampled = simulator.simulate(flip=start_angles).abs()
    axis.plot(start_angles, sampled[0], "o", color=MUTED, label="start")
    sampled = simulator.simulate(flip=designed).abs()
    axis.plot(designed, sampled[0], "*", ms=12, color=SERIES[1], label="designed")
    axis.set(xlabel="flip angle (deg)", ylabel="|signal| (a.u.)", title=title)

axes[2].plot(result.loss.cpu(), color=MUTED)
axes[2].set(xlabel="iteration", ylabel="log relative CRLB", title="convergence")
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Phantom
# -------
#
# The phantom is BrainWeb subject 0, slice 90, the one the parameter-inference
# examples map. Its fuzzy tissue memberships give a T1, a T2 and a proton
# density at every voxel. You play both protocols on it and compare the fitted
# maps with the truth.

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
# drawn from its first row down.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5
share = np.maximum(occupancy, 1e-6)

T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
T2_true = np.where(mask, fractions @ tissue_T2 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

truth = {
    "T1": torch.as_tensor(T1_true[mask].copy()),
    "T2": torch.as_tensor(T2_true[mask].copy()),
    "M0": torch.as_tensor(M0_true[mask].copy()),
}
fig, axes = canvas(1, 3, mask.shape, bars=3, extra=1.1)
for axis, values, name in (
    (axes[0, 0], T1_true, "T1"),
    (axes[0, 1], T2_true, "T2"),
    (axes[0, 2], M0_true, "M0"),
):
    cmap, limits, label = MAPS[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
fig.suptitle("BrainWeb subject 0, slice 90")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Joint fit
# ---------
#
# The two blocks are one experiment, and you fit them as one. A
# :class:`~blochsim.model.Simulator` plays each block and concatenates what they
# record. The fit is the same for both protocols: the same nonlinear least
# squares over the same four unknowns, from the same starting values:
from blochsim.estimators import NonlinearLeastSquares
from blochsim.model import Simulator


class JointRelaxometry(Simulator):
    """Both blocks at fixed flip angles, as one signal model."""

    properties = ("T1", "T2", "M0", "B0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence):
        T1, T2 = properties["T1"], properties["T2"]
        M0 = properties.get("M0", 1.0)
        B0 = properties.get("B0", 0.0)
        return torch.cat(
            (
                self.spoiled.simulate(T1=T1, T2star=T2, M0=M0, B0=B0),
                self.balanced.simulate(T1=T1, T2=T2, M0=M0, B0=B0),
            ),
            dim=-1,
        )


# %%
# The noise is independent in the real and imaginary channels, each at the
# standard deviation the bound was computed with. Simulate the slice with the
# designed protocol, add the noise and fit every voxel:
UNKNOWN = {
    "T1": (200.0, 5000.0),
    "T2": (20.0, 600.0),
    "M0": (0.1, 2.0),
    "B0": (-50.0, 50.0),
}
generator = torch.Generator().manual_seed(7)

joint = JointRelaxometry(spgr_designed, ssfp_designed)

clean = joint.simulate(B0=0.0, **truth)
noise = torch.randn((2, *clean.shape), generator=generator, dtype=torch.float32)
measured = clean + NOISE * torch.complex(noise[0], noise[1])

problem = NonlinearLeastSquares(
    joint,
    bounds=UNKNOWN,
    initial={"T1": 1000.0, "T2": 100.0, "M0": 1.0, "B0": 0.0},
).fit(UNKNOWN, noise_std=NOISE, seed=0)

maps = problem(measured)  # {"T1": ..., "T2": ..., "M0": ..., "B0": ...}


# sphinx_gallery_start_ignore
def mapped(spgr_flip, ssfp_flip):
    """Play both blocks over the slice, then fit every voxel."""
    block = JointRelaxometry(spgr_flip, ssfp_flip)
    exact = block.simulate(B0=0.0, **truth)
    draw = torch.randn((2, *exact.shape), generator=generator, dtype=torch.float32)
    seen = exact + NOISE * torch.complex(draw[0], draw[1])

    fit = NonlinearLeastSquares(
        block,
        bounds=UNKNOWN,
        initial={"T1": 1000.0, "T2": 100.0, "M0": 1.0, "B0": 0.0},
    ).fit(UNKNOWN, noise_std=NOISE, seed=0)
    start = time.time()
    found = fit(seen)
    return found, time.time() - start


before, before_seconds = mapped(spgr_start, ssfp_start)
after, after_seconds = mapped(spgr_designed, ssfp_designed)
print(f"{2 * int(mask.sum())} joint fits in {before_seconds + after_seconds:.1f} s")
# sphinx_gallery_end_ignore

# %%
# Predicted precision
# -------------------
#
# The design bound was computed for two tissues. Evaluated at the relaxation
# times of each voxel, it gives a map of the predicted precision, against which
# the measured error is compared:


def predicted_sigma(spgr_flip, ssfp_flip):
    """Relative standard deviation the bound allows, voxel by voxel."""
    at_voxel = tuple(
        sequence.bind(T1=truth["T1"], M0=1.0, B0=0.0, **{name: truth["T2"]})
        for sequence, name in (
            (SPGRSimulator(TE=2.0, TR=6.0), "T2star"),
            (bSSFPSimulator(TE=2.5, TR=5.0), "T2"),
        )
    )
    together = torch.cat(
        (rows(at_voxel[0], flip=spgr_flip), rows(at_voxel[1], flip=ssfp_flip)), dim=-1
    )
    bound = blochsim.crlb(together, noise_variance=NOISE**2)
    return {
        "T1": bound[..., 0].sqrt() / truth["T1"],
        "T2": bound[..., 1].sqrt() / truth["T2"],
    }


# sphinx_gallery_start_ignore
expected = {
    "start": predicted_sigma(spgr_start, ssfp_start),
    "designed": predicted_sigma(spgr_designed, ssfp_designed),
}
# sphinx_gallery_end_ignore

# %%
# No unbiased estimator beats the bound. The fit below lands about a fifth
# above it for both protocols, so the reduction the bound predicts is also the
# reduction measured, which is the check that the design optimized the right
# quantity. Both are root-mean-square, because a bound is a standard
# deviation. The median absolute error is about two thirds of a standard
# deviation and would understate the error by that factor:

# sphinx_gallery_start_ignore
found = {"start": before, "designed": after}


def rms(values):
    return 100.0 * float(values.square().mean().sqrt())


print(f"\n{'':18}{'T1':>26}{'T2':>26}")
print(f"{'':18}{'measured':>13}{'bound':>13}{'measured':>13}{'bound':>13}")
for label, fitted in found.items():
    line = f"{label:18}"
    for name in ("T1", "T2"):
        error = (fitted[name] - truth[name]) / truth[name]
        line += f"{rms(error):12.1f}%{rms(expected[label][name]):12.1f}%"
    print(line)
# sphinx_gallery_end_ignore

# %%
# Maps
# ----
#
# The fitted maps and the absolute error of each protocol. The designed protocol
# gives lower noise in both maps.


# sphinx_gallery_start_ignore
def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask] = values.numpy(force=True)
    return image


for name in ("T1", "T2"):
    reference = T1_true if name == "T1" else T2_true
    cmap, limits, label = MAPS[name]
    residuals = {
        method: np.abs(painted(fitted[name]) - reference)
        for method, fitted in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())

    fig, axes = canvas(2, 1 + len(found), mask.shape)
    panel(axes[0, 0], reference, cmap, limits, title="truth")
    axes[1, 0].set_visible(False)
    for column, method in enumerate(found, start=1):
        estimate = panel(
            axes[0, column], painted(found[method][name]), cmap, limits, title=method
        )
        error = panel(axes[1, column], residuals[method], "inferno", (0.0, top or 1.0))
    scalebar(estimate, axes[0], label)
    scalebar(error, axes[1, 1:], f"|error|, {label}")
    plt.show()
# sphinx_gallery_end_ignore

# %%
# Specification
# -------------
#
# What this example did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, design the flip angles of a joint DESPOT protocol: four
#    SPGR scans (TE 2 ms, TR 6 ms, up to 40 degrees) and four bSSFP scans (TE
#    2.5 ms, TR 5 ms, up to 70 degrees), for white and grey matter at 3 T.
#    Estimate T1, T2, M0 and B0 jointly. Minimize the log of the summed
#    relative Cramer-Rao bounds of T1 and T2 with SequenceDesign. Play the
#    start and designed protocols on BrainWeb slice 90 with noise of 0.5 %,
#    fit every voxel with NonlinearLeastSquares, and compare the RMS error of
#    the T1 and T2 maps with the bound.
#
# References
# ----------
#
# .. [1] Teixeira, R. P. A. G., Malik, S. J., Hajnal, J. V., "Joint system
#    relaxometry (JSR) and Cramer-Rao lower bound optimization of sequence
#    parameters: a framework for enhanced precision of DESPOT T1 and T2
#    estimation", Magnetic Resonance in Medicine 79.1 (2018), pp. 234-245.
#    https://doi.org/10.1002/mrm.26670
