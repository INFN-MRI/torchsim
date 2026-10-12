"""
================================
Nonlinear inversion from k-space
================================

In this Application you reconstruct T2 maps directly from k-space, with the
signal model inside the forward operator, and compare the result with gridding
followed by a fit.

The forward operator is the chain

.. math::

   F = P \\, \\mathcal{F} \\, C \\, M

of sampling :math:`P`, Fourier encoding :math:`\\mathcal{F}`, coil sensitivities
:math:`C` and the signal model :math:`M` [1]_. Only :math:`M` changes with the
sequence, and it is the only factor blochsim supplies: a
:class:`~blochsim.ModelOperator` wraps any simulator as :math:`M`. The encoding
and the Gauss-Newton solver come from
`bartorch <https://github.com/pulserver/bartorch>`_. The problem is nonlinear
in the maps, so it needs a starting guess and an outer Gauss-Newton loop. In
exchange the model can have any number of parameters, where a linear subspace
would have to span the product of their ranges.

Prerequisites: Course lessons :doc:`../01-framework/01-first-simulation` and
:doc:`../01-framework/03-custom-signal-model`.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install blochsim bartorch matplotlib brainweb-dl cmap
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# %%
# Setup
# -----
#
# blochsim simulates the signal. bartorch provides the non-uniform Fourier
# transform, the nonlinear operator that carries the signal model into BART, and
# the iteratively regularized Gauss-Newton solver.
import time

import numpy as np
import torch

from bartorch import linop, nlop

from blochsim import ModelOperator
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
# echo, with 8 echoes. The baseline estimator uses a rank-3 temporal basis.
SIZE = 96
ECHOES = 8
SPOKES = 16
SAMPLES = 192
RANK = 3
STEPS = 8  # Gauss-Newton iterations
CG_ITERATIONS = 20  # conjugate-gradient iterations per linearized problem
ALPHA = 1e-3  # initial damping, halved at every iteration

# The simulation and the transform run on a GPU when one is available.
device = "cuda" if torch.cuda.is_available() else "cpu"

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
# The trajectory is ``(echoes, spokes, samples, 2)`` in grid units. Each echo
# has its own :class:`bartorch.linop.NUFFT` along its own spokes, and
# :func:`bartorch.linop.block_diag` stacks them into one operator from the
# ``(echoes, y, x)`` images to the ``(echoes, spokes, samples)`` k-space:
golden = torch.pi * (3.0 - 5.0**0.5)
angle = golden * torch.arange(ECHOES * SPOKES).reshape(ECHOES, SPOKES, 1)
radius = torch.linspace(-SIZE / 2, SIZE / 2, SAMPLES + 1)[:-1]
trajectory = torch.stack(
    (radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1
).to(device)

encoding = linop.block_diag(
    [
        linop.NUFFT(trajectory[echo : echo + 1], image_shape=(1, SIZE, SIZE))
        for echo in range(ECHOES)
    ]
)
kspace = encoding(images.movedim(-1, 0).contiguous())

# %%
# Gridding is the adjoint after a ramp density compensation. Scale the k-space
# so that the gridded image peaks at one; the damping weights below are then
# numbers about the model, not about the receiver gain:
ramp = (radius.abs() + 0.5).to(device) / (SIZE / 2)
gridded = encoding.H(kspace * ramp).movedim(0, -1)
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
    arm = trajectory[echo].cpu().numpy()
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
# The maps are what both reconstructions below recover, and the spokes are all
# that is measured of them.
#
# Gridding baseline
# -----------------
#
# The baseline reconstructs the eight images and then fits them. The fit needs
# an estimator, here a dictionary match over a rank-3 temporal basis, which
# holds essentially all of an eight-echo exponential decay. The nonlinear
# reconstruction has no such step, because its result is the maps.
grid = torch.linspace(20.0, 400.0, 500)
mapping = DictionaryMatcher(simulator).fit(T2=grid, M0=1.0, rank=RANK, seed=0)

# sphinx_gallery_start_ignore
print(f"rank {RANK} of {ECHOES} keeps {mapping.subspace.retained:.6f} of the signal")


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


started = clock()
# sphinx_gallery_end_ignore
adjoint = mapping(gridded)["T2"]
# sphinx_gallery_start_ignore
report("adjoint per echo", clock() - started, adjoint)
# sphinx_gallery_end_ignore

# %%
# Gridding is the adjoint with density compensation. 16 spokes of 192 samples are
# 3072 measurements per echo against 9216 unknowns, so each echo alone is
# underdetermined and iterating per echo does not reduce the error. The
# improvement has to come from a constraint across the echoes, which is the
# signal model.
#
# Signal model
# ------------
#
# The model stays in the forward operator and the maps are solved for against
# k-space. You declare two things:
#
# - the unknowns: ``T2`` and the complex amplitude the operator carries with it,
#   which combines proton density and receive phase;
# - the range of T2. The bound is enforced by solving for a transformed
#   variable, so no iterate leaves it. This matters more here than in a voxelwise
#   fit, because the model is evaluated in every voxel to predict every k-space
#   sample, and one unphysical voxel corrupts the whole residual.
operator = ModelOperator(simulator.to(device), "T2", bounds={"T2": (20.0, 400.0)})

# %%
# :class:`bartorch.nlop.ModelOperator` carries it into BART as a nonlinear
# operator from ``(channels, y, x)`` maps to ``(echoes, y, x)`` images. The
# amplitude starts from the first gridded echo, which costs nothing and makes the
# first Gauss-Newton step well conditioned:
M = nlop.ModelOperator(operator, (SIZE, SIZE))
initial = M.initial(T2=100.0, amplitude=gridded[..., 0]).to(device)

# %%
# Gauss-Newton solve
# ------------------
#
# The forward operator is the composition ``encoding @ M``. Each iteration of
# :class:`bartorch.nlop.IRGNM` linearizes it at the current maps, solves the
# linearized least-squares problem by conjugate gradients and steps, then halves
# the damping. blochsim provides the model and its derivative; the loop and the
# encoding are BART's.

# sphinx_gallery_start_ignore
started = clock()
# sphinx_gallery_end_ignore
solver = nlop.IRGNM(iterations=STEPS, alpha=ALPHA, cg_maxiter=CG_ITERATIONS)
found = solver(kspace, encoding @ M, x0=initial)

modelled = M.split(found)["T2"]

# sphinx_gallery_start_ignore
report("model-based", clock() - started, modelled)
# sphinx_gallery_end_ignore

# %%
# ``found`` holds the solved variables and ``M.split`` separates them into named
# maps. No fit follows.
#
# Time and memory
# ---------------
#
# Each conjugate-gradient step applies the Jacobian and its adjoint once. Each
# product is one application of the encoding operator and one of the model.
# Timing the four shows which part a faster reconstruction has to speed up.
point = operator.initial((SIZE, SIZE), T2=100.0).to(device)
tangent = torch.randn_like(point)
predicted = operator.A_jvp(point, tangent)
adjoint_image = encoding.H(kspace).movedim(0, -1)

# sphinx_gallery_start_ignore


def timed(call, repeats=5):
    call()  # the first call plans the transforms
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    if device == "cuda":
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / repeats


print(f"per conjugate-gradient step, {operator.channels} channels solved for:")
print(f"  model    J  v   {timed(lambda: operator.A_jvp(point, tangent)):6.1f} ms")
print(
    f"  model    J^H v  {timed(lambda: operator.A_vjp(point, adjoint_image)):6.1f} ms"
)
image_stack = predicted.movedim(-1, 0).contiguous()
print(f"  encoding A      {timed(lambda: encoding(image_stack)):6.1f} ms")
print(f"  encoding A^H    {timed(lambda: encoding.H(kspace)):6.1f} ms")

blocks = operator.jacobian(point)
print(
    f"the Jacobian this avoids holding: "
    f"{blocks.numel() * 8 / 2**20:.1f} MiB, against "
    f"{predicted.numel() * 8 / 2**20:.1f} MiB for a signal"
)
# sphinx_gallery_end_ignore

# %%
# The products do not build the Jacobian. Its blocks are voxels x channels x
# contrasts, where a signal is voxels x contrasts, so avoiding it saves the
# channel count times the signal size at every iteration.
#
# Maps
# ----
#
# The T2 maps use a perceptually uniform colormap reserved for T2 [2]_.

# sphinx_gallery_start_ignore
shown = (
    ("truth", T2_true),
    ("adjoint", adjoint),
    ("model-based", modelled),
)

figure, axes = plt.subplots(
    2,
    len(shown),
    squeeze=False,
    figsize=(len(shown) * PANEL + BAR_WIDTH, PANEL * 2 + 0.6),
)
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
# Alternative signal model
# ------------------------
#
# The model is the only part that names a relaxation time, and it is an ordinary
# :class:`~blochsim.Simulator`, the object used in the fitting and design
# Applications. Water-fat separation, T2* with a field map or a Look-Locker
# inversion recovery each require a different simulator. The operator, the loop
# and the encoding are unchanged.
#
# References
# ----------
#
# .. [1] Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. Physics-based
#        reconstruction methods for magnetic resonance imaging. Phil Trans R Soc
#        A 379:20200196 (2021).
# .. [2] Fuderer M, et al. Recommended colour maps for relaxation parameter
#        maps. Magn Reson Med (2025).
#
# Specification
# -------------
#
# What this Application did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim and bartorch, simulate an 8-echo multi-echo spin echo (TE 10
#    to 150 ms) on a BrainWeb T2 phantom of 96 x 96 voxels, sampled by 16
#    golden-angle radial spokes per echo. Reconstruct T2 two ways: gridding
#    followed by a dictionary match, and a blochsim ModelOperator over the
#    simulator with T2 bounded to 20-400 ms, wrapped as bartorch's
#    nlop.ModelOperator under linop.NUFFT and solved by nlop.IRGNM from a T2 of
#    100 ms.
#    Report time and mean T2 error over the brain, the time of each Jacobian
#    product, and the size of the Jacobian that the operator avoids storing.
