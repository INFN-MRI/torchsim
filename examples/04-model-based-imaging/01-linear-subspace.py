"""
==============================================
Reconstructing in a linear subspace
==============================================

A quantitative scan is usually reconstructed twice: once to make one image per
contrast, and again -- voxel by voxel -- to turn those images into parameter
maps. The first step has no idea what the second one is for, so it spends its
effort recovering eight images when the answer is two numbers per voxel, and it
recovers each of them from its own undersampled data with no help from the
others.

A **linear subspace** removes both problems without leaving linear algebra. The
signals a train can produce span far fewer directions than it has contrasts, so
writing the series in that basis and reconstructing the *coefficients* both
shortens the unknown and ties the echoes together. There are no local minima and
no starting guess: it is a least-squares problem like any other.

This example reconstructs one undersampled radial multi-echo spin echo three
ways -- gridding, conjugate gradients per echo, and a subspace -- and reports
what each costs and what each gets wrong.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install torchsim brainweb-dl cmap mri-nufft[finufft,cufinufft] deepinv

# %%
#
# The phantom is BrainWeb's, reached through ``brainweb-dl``: ``get_mri``
# fetches the fuzzy tissue memberships, and the package ships the table of
# relaxation times that goes with them -- which is what the two standard
# library imports read.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from cmap import Colormap


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

# sphinx_gallery_end_ignore
import csv
from pathlib import Path

import brainweb_dl
from brainweb_dl import get_mri

# %%
#
# The Fourier encoding is not TorchSim's and never will be. ``mri-nufft``
# supplies the radial trajectory and the non-uniform transform that plays it;
# ``deepinv`` supplies the linear solver a Gauss-Newton step hands its
# linearized problem to, and the :class:`~deepinv.physics.LinearPhysics` base
# class the encoding operator below is written against.
#
# That base class is the whole of the adapter: anything exposing ``A`` and
# ``A_adjoint`` composes with what TorchSim supplies, so the operator built a
# few cells down is the only glue this integration needs.
#
import mrinufft
from deepinv.optim.linear import least_squares
from deepinv.physics import LinearPhysics
from mrinufft.operators.subspace import MRISubspace
from mrinufft.trajectories import initialize_2D_radial

# %%
#
# From TorchSim: the sequence, the estimator the contrast-then-fit routes
# need, and :attr:`~torchsim.Subspace.modes`, which hands the temporal
# basis to mri-nufft in the layout its subspace operator reads.
#
import time

import numpy as np
import torch

from torchsim.estimators import DictionaryMatcher
from torchsim.simulators import MultiEchoSimulator


# %%
#
# What the experiment is: a 96 matrix read as 16 radial spokes per echo, eight
# echoes, and the rank the temporal basis is fitted at.
#
SIZE = 96
ECHOES = 8
SPOKES = 16
SAMPLES = 192
RANK = 3

# deepinv's ``gamma`` is the *inverse* regularization weight: it minimizes
# ``||Ax - y||^2 + (1/gamma)||x||^2``, so a smaller number regularizes harder.
# Each route below was given the best of a short sweep -- a few lines, not
# shown -- so what the table compares is routes rather than tuning effort.
CONTRAST_GAMMA = 0.01
SUBSPACE_GAMMA = 10.0

# The GPU transform is used when it is both installed and usable; the
# simulation follows it, so the images and the operator meet on one device.
on_gpu = torch.cuda.is_available() and mrinufft.check_backend("cufinufft")
device = "cuda" if on_gpu else "cpu"
backend = "cufinufft" if on_gpu else "finufft"

# %%
#
# A brain with a known answer
# ---------------------------
#
# The phantom is BrainWeb subject 0, slice 90 -- an axial slice at 1 mm
# through the lateral ventricles, resampled to the matrix reconstructed here.
# BrainWeb publishes it as *fuzzy* memberships, so a voxel carries a fraction
# of CSF, grey matter, white matter and glial matter rather than a label, and
# weighting the tabulated relaxation times by those fractions gives a T2 map
# whose mixed voxels sit between the pure ones -- an answer that is known
# everywhere, mixtures included.
#

# sphinx_gallery_start_ignore
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
tissue_T2 = np.array([float(r["T2 (ms)"]) for r in rows])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(r["PD (ms)"]) for r in rows])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior, and an image is
# drawn from its first row down. Flipping here puts anterior at the top of
# every figure below rather than in each one of them.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
share = np.maximum(occupancy, 1e-6)


def resampled(values):
    """The slice at the matrix size this example reconstructs."""
    grid = torch.as_tensor(np.asarray(values, np.float32))[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0].to(device)


T2_true = resampled(np.where(occupancy > 0.5, fractions @ tissue_T2 / share, 0.0))
M0_true = resampled(np.where(occupancy > 0.5, fractions @ tissue_PD, 0.0))
brain = resampled((occupancy > 0.5).astype(np.float32)) > 0.5
T2_true = torch.where(
    brain, T2_true.clamp(20.0, 400.0), torch.tensor(20.0, device=device)
)

# sphinx_gallery_end_ignore

# %%
#
# What the scanner played
# -----------------------
#
# A multi-echo spin echo, read out on a golden-angle radial trajectory that
# rotates between echoes. Sixteen spokes per echo across a 96-sample matrix is
# roughly ninefold undersampled, which is where the three routes start to
# disagree.
#
TE = torch.linspace(10.0, 150.0, ECHOES)
simulator = MultiEchoSimulator(TE=TE)

images = (
    torch.as_tensor(simulator.to(device).simulate(T2=T2_true)).to(torch.complex64)
    * M0_true.to(torch.complex64)[..., None]
)

trajectory = (
    initialize_2D_radial(SPOKES * ECHOES, SAMPLES, tilt="golden")
    .astype(np.float32)
    .reshape(ECHOES, SPOKES * SAMPLES, 2)
)

build = mrinufft.get_operator(backend)
per_echo = [
    build(trajectory[echo], (SIZE, SIZE), n_coils=1, squeeze_dims=False, density=True)
    for echo in range(ECHOES)
]


class RadialEncoding(LinearPhysics):
    """``(batch, echoes, x, y)`` images to k-space, one trajectory per echo.

    This is the whole of ``P F C`` for this experiment, and none of it is
    TorchSim's: it wraps mri-nufft, which is what a real pipeline would do
    with its own trajectory, its own density compensation and its own coils.
    """

    def A(self, x, **kwargs):
        return torch.stack(
            [per_echo[e].op(x[:, e][:, None])[:, 0] for e in range(ECHOES)], 1
        )

    def A_adjoint(self, y, **kwargs):
        return torch.stack(
            [per_echo[e].adj_op(y[:, e][:, None])[:, 0] for e in range(ECHOES)], 1
        )


encoding = RadialEncoding()
kspace = encoding.A(images.movedim(-1, 0)[None])

# The k-space is scaled so the adjoint image peaks at one. Every damping
# weight below is then a number about the model rather than about the
# receiver gain, which is what makes one choice of it transferable.
gridded = encoding.A_adjoint(kspace)[0].movedim(0, -1)
scale = float(gridded.abs().max())
kspace, gridded = kspace / scale, gridded / scale

# sphinx_gallery_start_ignore
undersampling = (0.5 * np.pi * SIZE) / SPOKES
print(f"{SPOKES} spokes per echo: {undersampling:.0f}x undersampled")
# sphinx_gallery_end_ignore

# %%
#
# The object, and how it is sampled. The maps on the left are what every route
# below is trying to recover; the spokes on the right are all that is measured
# of them -- one echo's worth, rotated by the golden angle from the echo before
# it, so the echoes together cover k-space more evenly than any one of them
# does.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.58))
for axis, values, name, title in (
    (axes[0], T2_true, "T2", "ground truth"),
    (axes[1], M0_true, "M0", "proton density"),
):
    cmap, limits, label = STYLE[name]
    handle = panel(axis, values.cpu().numpy(), cmap, limits, title=title)
    axis.figure.colorbar(handle, ax=axis, label=label, fraction=0.046, pad=0.03)
    axis.set_box_aspect(1)

for echo in (0, ECHOES // 2, ECHOES - 1):
    arm = trajectory[echo].reshape(SPOKES, SAMPLES, 2)
    for spoke in range(SPOKES):
        axes[2].plot(
            arm[spoke, :, 0],
            arm[spoke, :, 1],
            lw=0.4,
            color=plt.cm.plasma(echo / (ECHOES - 1)),
        )
axes[2].set(
    xlabel="$k_x$",
    ylabel="$k_y$",
    title=f"{SPOKES} spokes per echo, 3 of {ECHOES} shown",
)
# An image keeps its own aspect and a line plot fills whatever it is given, so
# the box each panel is drawn into has to be fixed for the row to line up.
axes[2].set_box_aspect(1)
# A colorbar on the trajectory would have nothing to scale, but the panel has
# to lose the same width as the two beside it or the images shrink.
bar = figure.colorbar(handle, ax=axes[2], fraction=0.046, pad=0.03)
bar.ax.set_visible(False)
# sphinx_gallery_end_ignore

# %%
#
# The estimator, and the basis
# ----------------------------
#
# One :class:`~torchsim.DictionaryMatcher` states the problem, and it serves
# every route below. Asking it for a rank fits a temporal basis to the training
# signals: that basis is what the subspace reconstruction is given, and the
# coefficients it returns come straight back to the same mapping.
#
# Three directions hold essentially all of an eight-echo exponential, which is
# read off the basis rather than assumed.
#
grid = torch.linspace(20.0, 400.0, 500)
mapping = DictionaryMatcher(simulator).fit(T2=grid, M0=1.0, rank=RANK, seed=0)

# sphinx_gallery_start_ignore
print(f"rank {RANK} of {ECHOES} contrasts keeps {mapping.subspace.retained:.6f}")


def clock():
    """Wall clock, with the card caught up first."""
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def report(name, seconds, found):
    """One route's cost and its error over the brain.

    Every route is timed the same way -- reconstruction *and* fit -- because
    that is what a pipeline costs. A route that reconstructs quickly and then
    fits eight images is not a quick route.
    """
    error = (found[brain] - T2_true[brain]).abs()
    print(
        f"{name:<26} {seconds:5.1f}s   "
        f"T2 error {float(error.mean()):5.1f} ms "
        f"({100 * float((error / T2_true[brain]).mean()):4.1f}%)"
    )
    return found


# sphinx_gallery_end_ignore

# %%
#
# Reconstruct each contrast, then fit
# -----------------------------------
#
# The conventional pipeline, in its two usual forms. Gridding is the adjoint
# operator with a density weighting -- one pass, smooth, and biased. Iterating
# instead solves each echo's own least-squares problem, which is what a
# CG-SENSE reconstruction does.
#
# Both are given the same estimator afterwards, so what is being compared is
# the reconstruction and not the fit.
#

# sphinx_gallery_start_ignore
started = clock()
# sphinx_gallery_end_ignore
adjoint = mapping(gridded)["T2"]

# sphinx_gallery_start_ignore
report("adjoint per echo", clock() - started, adjoint)
started = clock()
# sphinx_gallery_end_ignore
images = least_squares(
    A=encoding.A,
    AT=encoding.A_adjoint,
    y=kspace,
    gamma=CONTRAST_GAMMA,
    solver="CG",
    max_iter=40,
)
separate = mapping(images[0].movedim(0, -1))["T2"]

# sphinx_gallery_start_ignore
report("iterative per echo", clock() - started, separate)
# sphinx_gallery_end_ignore

# %%
#
# Iterating gains nothing here, and the reason is worth stating: sixteen
# spokes of 192 samples is 3072 measurements against 9216 unknowns, so each
# echo on its own is an underdetermined problem and there is nothing to
# converge *to* that the density-weighted adjoint has not already found. What
# buys accuracy is a constraint that reaches across the echoes -- which is
# what both remaining routes are.
#
# %%
#
# Reconstruct the coefficients instead
# ------------------------------------
#
# The signal is written in the basis fitted above and the *coefficients* are
# reconstructed, three of them instead of eight images. Now the echoes
# constrain one another, and the problem is 3072 measurements against 3456
# unknowns rather than eight separate underdetermined ones. It stays linear,
# so it has no local minima and no starting guess.
#
# ``mapping.subspace.modes`` hands the basis over in the layout mri-nufft's
# subspace operator reads -- rank first, plain transpose -- and
# ``from_coefficients`` takes what comes back without projecting it a second
# time. The solver is the same one route one used, on a different operator.
#
# One operator serves every echo here, which the subspace structure is what
# allows: a single coefficient image is transformed once against all the
# echoes' samples together.
#
flat = build(
    trajectory.reshape(-1, 2), (SIZE, SIZE), n_coils=1, squeeze_dims=False, density=True
)
projected = MRISubspace(flat, mapping.subspace.modes.to(device))
projected.n_batchs, projected.n_coils = 1, 1

# sphinx_gallery_start_ignore
started = clock()
# sphinx_gallery_end_ignore
coefficients = least_squares(
    A=projected.op,
    AT=projected.adj_op,
    y=kspace[:, :, None, :],
    gamma=SUBSPACE_GAMMA,
    solver="CG",
    max_iter=40,
)
linear = mapping.from_coefficients(coefficients[0][:, 0].movedim(0, -1))["T2"]

# sphinx_gallery_start_ignore
report("iterative subspace", clock() - started, linear)
# sphinx_gallery_end_ignore

# %%
#
# The maps
# --------
#
# The subspace is the only one of the three that constrains the echoes against
# one another, and it lands at about half the error of either route that
# reconstructs each contrast on its own -- in a fraction of the time, because
# one operator serves every echo.
#

# sphinx_gallery_start_ignore
shown = (
    ("truth", T2_true),
    ("adjoint per echo", adjoint),
    ("iterative per echo", separate),
    ("iterative subspace", linear),
)

cmap, limits, label = STYLE["T2"]
figure, axes = canvas(2, len(shown), T2_true.shape)
axes[1, 0].set_visible(False)
for column, (title, values) in enumerate(shown):
    picture = torch.where(brain, values, torch.tensor(0.0, device=device))
    estimate = panel(axes[0, column], picture.cpu().numpy(), cmap, limits, title=title)
    if column == 0:
        continue
    difference = torch.where(
        brain, (values - T2_true).abs(), torch.tensor(0.0, device=device)
    )
    error = panel(axes[1, column], difference.cpu().numpy(), "inferno", (0, 80))
scalebar(estimate, axes[0], label)
scalebar(error, axes[1, 1:], f"|error|, {label}")
# sphinx_gallery_end_ignore

# %%
#
# Where a subspace stops being cheap
# ----------------------------------
#
# Eight echoes of a single exponential are the case a subspace is best at:
# three directions hold essentially all of the signal, and what is left for
# anything more elaborate to recover is close to nothing.
#
# Two things break that. A phase-modulated signal -- a balanced steady state
# through a field map, a fingerprinting train with a varying RF phase -- needs
# tens of components rather than three, and the coefficient problem stops being
# smaller than the image problem. And a model with several parameters has no
# small basis at all, because the basis has to span the *product* of the
# ranges. In both cases the model itself has to go inside the operator, which
# is the nonlinear route.
#
# The rank is not a guess either way. :attr:`~torchsim.Subspace.retained` says
# what a basis keeps before anything is projected through it, so how well a
# subspace can possibly do is known before the reconstruction is run.
#
