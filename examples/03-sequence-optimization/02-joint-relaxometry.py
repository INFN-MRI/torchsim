"""
=========================================
Designing a joint relaxometry protocol
=========================================

DESPOT estimates T1 from a set of spoiled gradient-echo acquisitions at
different flip angles and T2 from a set of balanced SSFP acquisitions. Fitting
them jointly rather than one after the other uses all the data for both
parameters, and then the flip angles themselves can be chosen to make the joint
estimate as precise as possible [1]_.

The cost here is a Cramer-Rao bound: the lowest variance an unbiased estimate
of T1 and T2 can have, given the derivative of each sequence's signal with
respect to every parameter being estimated. Minimizing it chooses where on each
signal curve the scan time is spent.

.. [1] Teixeira RPAG, Malik SJ, Hajnal JV. Joint system relaxometry (JSR) and
   Cramer-Rao lower bound optimization of sequence parameters: a framework for
   enhanced precision of DESPOT T1 and T2 estimation.
   Magn Reson Med. 2018;79:234-245.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim brainweb-dl

# %%
#
# The two closed-form sequences being designed, the three pieces a design is
# stated in, and :func:`~torchsim.crlb`, which is the cost.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# sphinx_gallery_end_ignore
import time

import torch

import torchsim
from torchsim.optim import Bounded, SequenceDesign
from torchsim.simulators import SPGRSimulator, bSSFPSimulator

# %%
#
# The two sequences
# -----------------
#
# A design problem is stated in three pieces. The acquisition is a simulator
# with the tissue it is being designed for already fixed on it, so only the
# parameters under design are left to give. Both sequences here are closed
# forms, and they are constructed and asked exactly as a state-machine
# sequence would be.
#

# White and grey matter at 3 T -- the design is for both at once.
T1_MS = torch.tensor([830.0, 1330.0])
T2_MS = torch.tensor([80.0, 110.0])
# Noise standard deviation, as a fraction of the fully relaxed magnetization.
NOISE = 0.005

spgr = SPGRSimulator(TE=2.0, TR=6.0, T1=T1_MS, T2star=T2_MS, M0=1.0, B0=0.0)
ssfp = bSSFPSimulator(TE=2.5, TR=5.0, T1=T1_MS, T2=T2_MS, M0=1.0, B0=0.0)

# %%
#
# The cost
# --------
#
# Four parameters are estimated jointly: T1, T2, the proton density and the
# off-resonance. The last two are nuisances -- they have to be estimated
# because they affect the data, but the design is not for them.
#
# The two sequences do not carry the same information, and neither does this
# implementation pretend they do: the spoiled steady state written in closed
# form depends on T2\* rather than T2, so its T2 row is exactly zero. That is
# the structure of joint relaxometry rather than a limitation -- each block is
# blind to something, and the Fisher matrix adds them up.
#
JOINT = ("T1", "T2", "M0", "B0")


def rows(acquisition, **design):
    """The Jacobian rows for every joint parameter, zero where the block is blind."""
    present = [name for name in JOINT if name in acquisition.exposes]
    _, jacobian = acquisition.jacobian(present, **design)
    placed = jacobian.new_zeros(
        jacobian.shape[:-2] + (len(JOINT), jacobian.shape[-1])
    )
    where = torch.tensor([JOINT.index(name) for name in present])
    return placed.index_copy(-2, where, jacobian)


def bounds(spgr_flip, ssfp_flip):
    """The Cramer-Rao bound on each joint parameter, for each design tissue."""
    together = torch.cat(
        (rows(spgr, flip=spgr_flip), rows(ssfp, flip=ssfp_flip)), dim=-1
    )
    return torchsim.crlb(together, noise_variance=NOISE**2)


# %%
#
# The cost is the whole of what makes this problem this problem, and it is
# four lines. Dividing each bound by its own parameter squared makes the two
# terms dimensionless, so a 100 ms T2 and a 1000 ms T1 are weighted by how
# well they are known rather than by how large they are; the logarithm makes
# the gradient relative, so the design does not depend on the noise level.
#


def precision(spgr_flip, ssfp_flip):
    """Relative variance of T1 and T2, averaged over the design tissues."""
    bound = bounds(spgr_flip, ssfp_flip)
    relative = bound[..., 0] / T1_MS**2 + bound[..., 1] / T2_MS**2
    return relative.mean().log()


# %%
#
# The design
# ----------
#
# Four acquisitions of each kind, starting from a spread of angles. The limits
# are what the scanner will play, and they are enforced exactly -- no iterate
# is ever outside them.
#
spgr_start = torch.tensor([2.0, 4.0, 8.0, 16.0])
ssfp_start = torch.tensor([10.0, 20.0, 40.0, 60.0])

design = SequenceDesign(
    precision,
    spgr_flip=Bounded(spgr_start, 1.0, 40.0),
    ssfp_flip=Bounded(ssfp_start, 1.0, 70.0),
)

start = time.time()
result = design.minimize(iterations=120, learning_rate=0.3)
design_time = time.time() - start

spgr_designed = result.parameters["spgr_flip"]
ssfp_designed = result.parameters["ssfp_flip"]

# %%
#
# What it buys, as the number a spectroscopist would quote: the standard
# deviation of each estimate as a percentage of the value itself.
#
for label, angles_pair in (
    ("published spread", (spgr_start, ssfp_start)),
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

# %%
#
# The schedule
# ------------
#
# What a scanner is actually handed: eight acquisitions, four spoiled and four
# balanced, each with a flip angle and nothing else changing between them. This
# is the protocol table a radiographer would read, before and after.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for axis, start_angles, designed, title in (
    (axes[0], spgr_start, spgr_designed, "SPGR block"),
    (axes[1], ssfp_start, ssfp_designed, "bSSFP block"),
):
    index = torch.arange(1, start_angles.numel() + 1)
    axis.bar(index - 0.19, start_angles, width=0.36, color="grey",
             label="published spread")
    axis.bar(index + 0.19, designed.detach(), width=0.36, color="crimson",
             label="designed")
    for position, angle in zip(index, designed.detach()):
        axis.annotate(f"{float(angle):.0f}", (float(position) + 0.19, float(angle)),
                      ha="center", va="bottom", fontsize=7)
    axis.set(xlabel="Acquisition", title=title, xticks=index.tolist())
    axis.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Flip angle [deg]")
axes[0].legend(fontsize=8)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# Where the angles went
# ---------------------
#
# The design collapses eight distinct angles onto three, and repeats them.
# That is what an optimal design does: the information sits at a few places on
# each curve, and the best use of a fixed number of acquisitions is to spend
# them there rather than to sample the curve evenly.
#
# Where those places are is worth reading off the figure. The SPGR angle lands
# above the Ernst angle of both tissues, on the side where the curve separates
# the two T1 values most sharply -- the peak itself is where the signal is
# largest and where it says least. The two bSSFP angles sit either side of the
# steady-state maximum, which is what makes the pair sensitive to T2. The
# upper one is against its limit rather than at an interior optimum, so
# raising the limit would move it; that limit is a real one, being what the
# deposited RF power allows.
#
sweep = torch.linspace(1.0, 70.0, 200)
# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))

for axis, acquisition, start_angles, designed, title in (
    (axes[0], spgr, spgr_start, spgr_designed, "SPGR"),
    (axes[1], ssfp, ssfp_start, ssfp_designed, "bSSFP"),
):
    curve = acquisition.simulate(flip=sweep).abs()
    axis.plot(sweep, curve[0], label="T1/T2 = 830/80 ms")
    axis.plot(sweep, curve[1], label="T1/T2 = 1330/110 ms")
    sampled = acquisition.simulate(flip=start_angles).abs()
    axis.plot(start_angles, sampled[0], "o", color="grey", label="start")
    sampled = acquisition.simulate(flip=designed).abs()
    axis.plot(designed, sampled[0], "*", ms=14, color="crimson", label="designed")
    axis.set(xlabel="Flip angle [deg]", ylabel="|signal|", title=title)
    axis.grid(alpha=0.3)
axes[0].legend(fontsize=7)

axes[2].plot(result.loss.cpu())
axes[2].set(
    xlabel="Iteration", ylabel="log relative CRLB", title="convergence"
)
axes[2].grid(alpha=0.3)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# What that means for the maps
# ----------------------------
#
# A bound is a promise about variance, and the place to cash it is a brain. The
# same BrainWeb phantom the parameter-inference examples map gives a T1, a T2
# and a proton density that are known at every voxel, so both protocols can be
# played on it and the answers compared against something rather than against
# each other.
#
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

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
print(f"\n{int(mask.sum())} brain voxels, "
      f"T1 {float(truth['T1'].min()):.0f}-{float(truth['T1'].max()):.0f} ms, "
      f"T2 {float(truth['T2'].min()):.0f}-{float(truth['T2'].max()):.0f} ms")

# %%
#
# The two blocks are one experiment, so they are fitted as one: a
# :class:`~torchsim.model.SignalModel` that plays each and concatenates what
# they record. The fit has to be the one thing that does not differ between the
# protocols, so it is the same nonlinear least squares over the same four
# unknowns, started from the same guess.
#
from torchsim import ParameterMapping
from torchsim.estimators import NonlinearLeastSquares
from torchsim.model import SignalModel


class JointRelaxometry(SignalModel):
    """Both blocks at fixed flip angles, as one signal model."""

    properties = ("T1", "T2", "M0", "B0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence):
        """The two blocks, end to end along the contrast axis."""
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
#
# The noise is independent on the real and the imaginary channel, each at the
# standard deviation the bound was computed with -- which is what makes the two
# comparable at all.
#
UNKNOWN = {
    "T1": (200.0, 5000.0),
    "T2": (20.0, 600.0),
    "M0": (0.1, 2.0),
    "B0": (-50.0, 50.0),
}
generator = torch.Generator().manual_seed(7)


def mapped(spgr_flip, ssfp_flip):
    """Play both blocks over the slice, then fit every voxel."""
    joint = JointRelaxometry(spgr_flip, ssfp_flip)
    clean = joint.simulate(B0=0.0, **truth)
    noise = torch.randn((2, *clean.shape), generator=generator, dtype=torch.float32)
    measured = clean + NOISE * torch.complex(noise[0], noise[1])

    problem = ParameterMapping(joint, noise_std=NOISE, seed=0, **UNKNOWN).train(
        NonlinearLeastSquares(
            bounds=UNKNOWN,
            initial={"T1": 1000.0, "T2": 100.0, "M0": 1.0, "B0": 0.0},
        )
    )
    start = time.time()
    maps = problem(measured)
    return maps, time.time() - start


before, before_seconds = mapped(spgr_start, ssfp_start)
after, after_seconds = mapped(spgr_designed, ssfp_designed)
print(f"{2 * int(mask.sum())} joint fits in "
      f"{before_seconds + after_seconds:.1f} s")

# %%
#
# The bound was computed for two tissues; the slice has thousands. Evaluating
# it at every voxel's own relaxation times turns it from a number about white
# and grey matter into a *predicted* precision map, which is what the measured
# error is then read against.
#


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
    bound = torchsim.crlb(together, noise_variance=NOISE**2)
    return {
        "T1": bound[..., 0].sqrt() / truth["T1"],
        "T2": bound[..., 1].sqrt() / truth["T2"],
    }


expected = {
    "published spread": predicted_sigma(spgr_start, ssfp_start),
    "designed": predicted_sigma(spgr_designed, ssfp_designed),
}

# %%
#
# What the design bought, over the brain rather than over two tissues. No
# unbiased estimator can beat the bound and a good one comes close to it, so
# the two columns agreeing is the check that the design optimized the right
# thing.
#
# Both are root-mean-square over the brain, because a bound is a standard
# deviation: the median of an absolute error is about two thirds of one, and
# comparing the two would flatter the estimator by exactly that factor.
#
found = {"published spread": before, "designed": after}


def rms(values):
    """The second moment, which is what a standard deviation is."""
    return 100.0 * float(values.square().mean().sqrt())


print(f"\n{'':18}{'T1':>26}{'T2':>26}")
print(f"{'':18}{'measured':>13}{'bound':>13}{'measured':>13}{'bound':>13}")
for label, maps in found.items():
    line = f"{label:18}"
    for name in ("T1", "T2"):
        error = (maps[name] - truth[name]) / truth[name]
        line += f"{rms(error):12.1f}%{rms(expected[label][name]):12.1f}%"
    print(line)

# %%
#
# The maps, and the error each protocol leaves. The designed protocol is not a
# different picture -- it is the same picture with less noise in it, which is
# what a precision design buys and all it buys.
#


def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


# sphinx_gallery_start_ignore
def panel(axis, values, cmap, limits, label=None):
    """One map, with a colorbar every panel loses the same width to."""
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    bar = axis.figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
    if label is None:
        bar.ax.set_visible(False)
    else:
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


for name, reference, limits in (("T1", T1_true, (0, 3000)), ("T2", T2_true, (0, 350))):
    residuals = {
        label: np.abs(painted(maps[name]) - reference) for label, maps in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())

    figure, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    panel(axes[0, 0], reference, "magma", limits)
    axes[0, 0].set_title("truth", fontsize=10)
    axes[0, 0].set_ylabel(f"{name} [ms]", fontsize=11)
    axes[1, 0].set_visible(False)
    for column, label in enumerate(found, start=1):
        panel(axes[0, column], painted(found[label][name]), "magma", limits,
              label=f"{name} [ms]" if column == 2 else None)
        axes[0, column].set_title(label, fontsize=10)
        panel(axes[1, column], residuals[label], "inferno", (0.0, top or 1.0),
              label=f"|error| {name} [ms]" if column == 2 else None)
    axes[1, 1].set_ylabel("|error|", fontsize=11)
    figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# Reading the result honestly
# ---------------------------
#
# The bound is a bound, not a prediction: an estimator can be worse than it and
# none can be better. The measured error sits about a fifth above it, and the
# gap is the fit rather than the design -- a Cramer-Rao bound describes a
# linearization, and over a slice that runs from white matter to CSF the fit
# is not linear everywhere. What carries over is the *ratio*: the design lowers
# the bound by about a third and lowers the measured error by about a third,
# which is the claim being made.
#
# Raise the noise and that stops holding. The fit becomes biased near the ends
# of its bounds, the long-T1 voxels lose their conditioning first, and the
# design's advantage shrinks -- which is the honest limit of designing against
# a bound rather than against an estimator.
#
# Nothing above is specific to DESPOT except the cost. The acquisition, the
# bounded parameters and the loop are the same three pieces that design a
# sequence for image quality rather than for precision.
#
