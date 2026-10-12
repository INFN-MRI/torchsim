"""
=========================
Subspace reconstruction
=========================

In this Application you reconstruct one undersampled radial multi-echo spin
echo three ways: by gridding, by conjugate gradients on each echo separately,
and in a linear subspace. You compare what each costs and how far its T2 map is
from the truth.

Quantitative maps are usually obtained in two steps: reconstruct one image per
contrast, then fit the maps voxel by voxel. The first step recovers eight images
from undersampled data although each voxel has two unknowns. The multi-echo
signals span far fewer directions than there are echoes, so you can
reconstruct the coefficients along those directions instead. This reduces the
number of unknowns and couples the echoes.

Prerequisites: Course lessons :doc:`../01-framework/01-first-simulation`.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install blochsim matplotlib brainweb-dl cmap mri-nufft[finufft,cufinufft] deepinv
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# %%
# Setup
# -----
#
# blochsim simulates the signal. The radial trajectory and the non-uniform
# Fourier transform come from mri-nufft, and the linear solver and the
# :class:`~deepinv.physics.LinearPhysics` base class come from deepinv. Anything
# that provides ``A`` and ``A_adjoint`` works with the solver.
import time

import mrinufft
import numpy as np
import torch

from deepinv.optim.linear import least_squares
from deepinv.physics import LinearPhysics
from mrinufft.operators.subspace import MRISubspace
from mrinufft.trajectories import initialize_2D_radial

from blochsim.estimators import DictionaryMatcher
from blochsim.simulators import MultiEchoSimulator

# sphinx_gallery_start_ignore
import csv
import warnings
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
from brainweb_dl import get_mri
from cmap import Colormap

from figure_style import PAGE_WIDTH, SERIES

warnings.filterwarnings("ignore")

# The window stops short of CSF, so white and grey matter fill the scale.
NAVIA = Colormap("crameri:navia").to_matplotlib()
T2_WINDOW = (0.0, 120.0)
BAR_WIDTH = 0.8
PANEL = (PAGE_WIDTH - BAR_WIDTH) / 4


def panel(axis, values, cmap, limits, title=None):
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title)
    return handle


def scalebar(handle, axes, label):
    axes = list(np.ravel(axes))
    axes[0].figure.colorbar(handle, ax=axes, label=label, shrink=0.92, aspect=20)


def canvas(rows, columns, shape):
    return plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(
            columns * PANEL + BAR_WIDTH,
            PANEL * shape[0] / shape[1] * rows + 0.6,
        ),
    )


# sphinx_gallery_end_ignore

# %%
# The experiment is a 96 x 96 matrix read as 16 radial spokes of 192 samples per
# echo, with 8 echoes. The temporal basis has rank 3.
SIZE = 96
ECHOES = 8
SPOKES = 16
SAMPLES = 192
RANK = 3

# The GPU transform is used when it is installed and usable, and the simulation
# follows it so that the images and the operator are on one device.
on_gpu = torch.cuda.is_available() and mrinufft.check_backend("cufinufft")
device = "cuda" if on_gpu else "cpu"
backend = "cufinufft" if on_gpu else "finufft"

# %%
# Phantom
# -------
#
# The phantom is BrainWeb subject 0, slice 90, resampled to 96 x 96. BrainWeb
# gives fuzzy tissue memberships, not labels. Weighting the tabulated T2 and
# proton density of each tissue by its membership gives maps in which mixed
# voxels lie between the pure tissues, known everywhere.

# sphinx_gallery_start_ignore
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
tissue_T2 = np.array([float(r["T2 (ms)"]) for r in rows])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(r["PD (ms)"]) for r in rows])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# Anterior at the top of every figure.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
share = np.maximum(occupancy, 1e-6)


def resampled(values):
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
# Sequence and sampling
# ---------------------
#
# The sequence is a multi-echo spin echo with eight echoes from 10 to 150 ms.
# Simulating it over the T2 map gives one image per echo, scaled by the proton
# density:
TE = torch.linspace(10.0, 150.0, ECHOES)
simulator = MultiEchoSimulator(TE=TE)

images = (
    torch.as_tensor(simulator.to(device).simulate(T2=T2_true)).to(torch.complex64)
    * M0_true.to(torch.complex64)[..., None]
)

# %%
# Each echo is sampled by 16 radial spokes, rotated by the golden angle from the
# echo before, so the echoes together cover k-space more evenly than any one
# does. 16 spokes across a 96-sample matrix is about ninefold undersampling,
# which is where the reconstructions differ.
#
# The encoding operator maps the images of all echoes to k-space, with one
# trajectory per echo. It wraps mri-nufft, as a real pipeline would wrap its own
# trajectory, density compensation and coils:
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
    """``(batch, echoes, x, y)`` images to k-space, one trajectory per echo."""

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

# %%
# Scale the k-space so that the gridded image peaks at one. The damping weights
# below are then numbers about the model, not about the receiver gain:
gridded = encoding.A_adjoint(kspace)[0].movedim(0, -1)
scale = float(gridded.abs().max())
kspace, gridded = kspace / scale, gridded / scale

# sphinx_gallery_start_ignore
print(f"{SPOKES} spokes per echo: {0.5 * np.pi * SIZE / SPOKES:.0f}x undersampled")

figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 0.3 * PAGE_WIDTH))
for axis, values, cmap, limits, label, title in (
    (axes[0], T2_true, NAVIA, T2_WINDOW, "T2 [ms]", "ground truth"),
    (axes[1], M0_true, "gray", (0.0, 1.0), "M0", "proton density"),
):
    handle = panel(axis, values.cpu().numpy(), cmap, limits, title=title)
    figure.colorbar(handle, ax=axis, label=label, fraction=0.046, pad=0.03)
    axis.set_box_aspect(1)
for shade, echo in enumerate((0, ECHOES // 2, ECHOES - 1)):
    arm = trajectory[echo].reshape(SPOKES, SAMPLES, 2)
    for spoke in range(SPOKES):
        axes[2].plot(
            arm[spoke, :, 0],
            arm[spoke, :, 1],
            lw=0.4,
            color=SERIES[shade],
        )
axes[2].set(xlabel="$k_x$", ylabel="$k_y$", title="3 of 8 echoes")
axes[2].set_box_aspect(1)
bar = figure.colorbar(handle, ax=axes[2], fraction=0.046, pad=0.03)
bar.ax.set_visible(False)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The maps are what every reconstruction below recovers, and the spokes are all
# that is measured of them.
#
# Temporal basis
# --------------
#
# A :class:`~blochsim.DictionaryMatcher` serves every reconstruction below as
# the estimator that turns images into T2. Giving ``fit`` a rank fits a temporal
# basis to the simulated signals of a grid of T2 values. Three directions hold
# essentially all of an eight-echo exponential decay:
grid = torch.linspace(20.0, 400.0, 500)
mapping = DictionaryMatcher(simulator).fit(T2=grid, M0=1.0, rank=RANK, seed=0)
print(f"rank {RANK} of {ECHOES} keeps {mapping.subspace.retained:.6f} of the signal")

# sphinx_gallery_start_ignore


def clock():
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def report(name, seconds, found):
    error = (found[brain] - T2_true[brain]).abs()
    print(
        f"{name:<20} {seconds:5.1f}s   "
        f"T2 error {float(error.mean()):5.1f} ms "
        f"({100 * float((error / T2_true[brain]).mean()):4.1f}%)"
    )
    return found


# sphinx_gallery_end_ignore

# %%
# :attr:`~blochsim.Subspace.retained` is the fraction of the signal the basis
# keeps, so you choose the rank from it before projecting anything.
#
# Reconstruction contrast by contrast
# -----------------------------------
#
# The conventional route has two forms. Gridding is the adjoint with density
# compensation: one pass, smooth, biased. The iterative form solves a
# least-squares problem for each echo, as CG-SENSE does. Both are followed by
# the same estimator, so the comparison is between reconstructions.
#
# deepinv's ``gamma`` is the inverse regularization weight, so a smaller value
# regularizes harder. Each route was given the best of a short sweep, not shown.

# sphinx_gallery_start_ignore
started = clock()
# sphinx_gallery_end_ignore
adjoint = mapping(gridded)["T2"]

# sphinx_gallery_start_ignore
report("adjoint per echo", clock() - started, adjoint)
started = clock()
# sphinx_gallery_end_ignore
images_cg = least_squares(
    A=encoding.A,
    AT=encoding.A_adjoint,
    y=kspace,
    gamma=0.01,
    solver="CG",
    max_iter=40,
)
separate = mapping(images_cg[0].movedim(0, -1))["T2"]

# sphinx_gallery_start_ignore
report("iterative per echo", clock() - started, separate)
# sphinx_gallery_end_ignore

# %%
# Iterating does not reduce the error. 16 spokes of 192 samples are 3072 measurements per
# echo against 9216 unknowns, so each echo alone is underdetermined, and the
# density-weighted adjoint is already the solution the iterations converge to.
# The gain has to come from a constraint across the echoes.
#
# Reconstruction in the subspace
# ------------------------------
#
# Expressing the signal in the temporal basis reduces the unknowns from 8 echo
# images to 3 coefficient images and couples the echoes. The problem has 24576
# measurements for 27648 unknowns. It remains linear, so it has a single minimum
# and needs no starting guess.
#
# ``mapping.subspace.modes`` gives the basis in the layout mri-nufft's subspace
# operator reads. One operator over the trajectories of all echoes replaces the
# per-echo operators, and the solver is the one the gridded route used.
# ``from_coefficients`` turns the coefficients into maps without projecting a
# second time.
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
    gamma=10.0,
    solver="CG",
    max_iter=40,
)
linear = mapping.from_coefficients(coefficients[0][:, 0].movedim(0, -1))["T2"]

# sphinx_gallery_start_ignore
report("iterative subspace", clock() - started, linear)
# sphinx_gallery_end_ignore

# %%
# Maps
# ----
#
# The T2 maps use a perceptually uniform colormap reserved for T2, so that a T2
# map is never read as a T1 map [1]_. The subspace is the only route that
# constrains the echoes against one another. On the CPU run of this page its mean
# error is about 55% of either per-echo route's (13% against 21%), at a longer
# reconstruction time than the iterative per-echo route; the timings printed above
# depend on the hardware.

# sphinx_gallery_start_ignore
shown = (
    ("truth", T2_true),
    ("adjoint", adjoint),
    ("iterative", separate),
    ("subspace", linear),
)

figure, axes = canvas(2, len(shown), T2_true.shape)
axes[1, 0].set_visible(False)
for column, (title, values) in enumerate(shown):
    picture = torch.where(brain, values, torch.tensor(0.0, device=device))
    estimate = panel(axes[0, column], picture.cpu().numpy(), NAVIA, T2_WINDOW, title)
    if column == 0:
        continue
    difference = torch.where(
        brain, (values - T2_true).abs(), torch.tensor(0.0, device=device)
    )
    error = panel(axes[1, column], difference.cpu().numpy(), "inferno", (0, 80))
scalebar(estimate, axes[0], "T2 [ms]")
scalebar(error, axes[1, 1:], "|error| [ms]")
plt.show()
# sphinx_gallery_end_ignore

# %%
# Limits
# ------
#
# Eight echoes of one exponential is the case a subspace handles best: three
# directions hold essentially all of the signal. Two things break that.
#
# - A phase-modulated signal, such as a balanced steady state through a field
#   map or a fingerprinting train with varying RF phase, needs tens of
#   components. The coefficient problem then is no smaller than the image
#   problem.
# - A model with several parameters has no small basis, because the basis must
#   span the product of their ranges.
#
# Both cases require the model inside the operator. That is the nonlinear route of
# :doc:`02-nonlinear-inversion`.
#
# References
# ----------
#
# .. [1] Fuderer M, et al. Color-map recommendation for MR relaxometry maps.
#        Magn Reson Med 93(2):490-506 (2025).
#
# Specification
# -------------
#
# What this Application did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, mri-nufft and deepinv, simulate an 8-echo multi-echo spin
#    echo (TE 10 to 150 ms) on a BrainWeb T2 phantom of 96 x 96 voxels. Sample
#    each echo with 16 golden-angle radial spokes of 192 samples. Fit a rank-3
#    temporal basis with a DictionaryMatcher and check how much signal it keeps.
#    Reconstruct T2 by gridding, by CG per echo, and by CG on the subspace
#    coefficients. Report the time and the mean T2 error over the brain of each.
