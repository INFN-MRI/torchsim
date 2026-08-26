"""
=======================
Model-based imaging
=======================

A quantitative scan is usually reconstructed twice: once to make one image per
contrast, and again -- voxel by voxel -- to turn those images into parameter
maps. The first step has no idea what the second one is for, so it spends its
effort recovering eight images when the answer is two numbers per voxel.

Physics-based reconstruction removes the intermediate step. The forward
operator is written as a chain

.. math::

   F = P \\, \\mathcal{F} \\, C \\, M

-- sampling, Fourier encoding, coil sensitivities, and the **signal model** --
and the parameter maps are solved for directly against the k-space that was
measured. Only the last factor changes with the sequence, and it is the only
one TorchSim supplies: :class:`~torchsim.recon.ModelOperator` turns any
:class:`~torchsim.Acquisition` into it, and the encoding comes from mri-nufft.

This example reconstructs one undersampled radial multi-echo spin echo three
ways -- gridding, a linear subspace, and the nonlinear model -- and reports
what each costs and what each gets wrong.

Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. *Physics-based reconstruction
methods for magnetic resonance imaging.* Phil Trans R Soc A 379:20200196
(2021).
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install torchsim brainweb-dl mri-nufft[finufft] deepinv

# %%
#
# The imports:
#
import warnings

warnings.filterwarnings("ignore")

import csv
import time
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
import mrinufft
import numpy as np
import torch
from brainweb_dl import get_mri
from deepinv.optim.linear import least_squares
from deepinv.physics import LinearPhysics
from mrinufft.operators.subspace import MRISubspace
from mrinufft.trajectories import initialize_2D_radial

from torchsim import Acquisition, ParameterMapping
from torchsim.estimators import DictionaryMatcher
from torchsim.recon import GaussNewton, ModelOperator, Schedule, iterative
from torchsim.simulators import MultiEchoSimulator

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

device = "cuda" if torch.cuda.is_available() else "cpu"
backend = "cufinufft" if device == "cuda" else "finufft"

# %%
#
# A brain with a known answer
# ---------------------------
#
# BrainWeb publishes its phantom as *fuzzy* tissue memberships, so a voxel
# carries a fraction of each tissue rather than a label. Weighting the
# tabulated relaxation times by those fractions gives a T2 map with the
# structure of a brain and an answer that is known everywhere, mixtures
# included.
#
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
tissue_T2 = np.array([float(r["T2 (ms)"]) for r in rows])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(r["PD (ms)"]) for r in rows])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
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
T2_true = torch.where(brain, T2_true.clamp(20.0, 400.0), torch.tensor(20.0, device=device))

print(f"{SIZE}x{SIZE} slice, {int(brain.sum())} brain voxels")
print(
    f"T2 {T2_true[brain].min():.0f}-{T2_true[brain].max():.0f} ms, "
    f"M0 {M0_true[brain].min():.2f}-{M0_true[brain].max():.2f}"
)

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
# The protocol stays on the host. :class:`~torchsim.recon.ModelOperator` takes
# it wherever the maps are, so nothing here has to be moved by hand.
#
TE = torch.linspace(10.0, 150.0, ECHOES)
acquisition = Acquisition(MultiEchoSimulator(TE=TE))

images = torch.as_tensor(acquisition.to(device).simulate(T2=T2_true)).to(
    torch.complex64
) * M0_true.to(torch.complex64)[..., None]

trajectory = initialize_2D_radial(
    SPOKES * ECHOES, SAMPLES, tilt="golden"
).astype(np.float32).reshape(ECHOES, SPOKES * SAMPLES, 2)

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
undersampling = (0.5 * np.pi * SIZE) / SPOKES
print(f"{SPOKES} spokes per echo: {undersampling:.0f}x undersampled")

# %%
#
# The estimator, and the basis
# ----------------------------
#
# One :class:`~torchsim.ParameterMapping` states the problem, and it serves
# both of the first two routes. Asking it for a rank fits a temporal basis to
# the training signals: that basis is what the subspace reconstruction is
# given, and the coefficients it returns come straight back to the same
# mapping.
#
# Three directions hold essentially all of an eight-echo exponential, which is
# read off the basis rather than assumed.
#
grid = torch.linspace(20.0, 400.0, 500)
mapping = ParameterMapping(
    acquisition, T2=grid, M0=1.0, rank=RANK, seed=0
).train(DictionaryMatcher())
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


# %%
#
# Route one: reconstruct each contrast, then fit
# ----------------------------------------------
#
# The conventional pipeline, in its two usual forms. Gridding is the adjoint
# operator with a density weighting -- one pass, smooth, and biased. Iterating
# instead solves each echo's own least-squares problem, which is what a
# CG-SENSE reconstruction does.
#
# Both are given the same estimator afterwards, so what is being compared is
# the reconstruction and not the fit.
#
started = clock()
adjoint = report("adjoint per echo", clock() - started, mapping(gridded)["T2"])

started = clock()
images = least_squares(
    A=encoding.A,
    AT=encoding.A_adjoint,
    y=kspace,
    gamma=CONTRAST_GAMMA,
    solver="CG",
    max_iter=40,
)
separate = report(
    "iterative per echo",
    clock() - started,
    mapping(images[0].movedim(0, -1))["T2"],
)

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
# Route two: a linear subspace
# ----------------------------
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

started = clock()
coefficients = least_squares(
    A=projected.op,
    AT=projected.adj_op,
    y=kspace[:, :, None, :],
    gamma=SUBSPACE_GAMMA,
    solver="CG",
    max_iter=40,
)
linear = report(
    "iterative subspace",
    clock() - started,
    mapping.from_coefficients(coefficients[0][:, 0].movedim(0, -1))["T2"],
)

# %%
#
# Route three: the nonlinear model
# --------------------------------
#
# The signal model stays inside the forward operator and the maps are solved
# for against k-space directly. Two things are declared and nothing else:
#
# * **what is unknown** -- ``T2``, plus the complex amplitude the operator
#   carries for it, which is proton density and receive phase together;
# * **what T2 may be** -- a box bound, kept by solving for a transformed
#   variable so no iterate is ever outside it. Under an encoding operator that
#   matters more than in a fit: the model is evaluated at every voxel to
#   predict every k-space sample, so one unphysical voxel corrupts the whole
#   residual.
#
# An equality constraint, were there one, would be written into the model
# instead -- see :class:`~torchsim.recon.ModelOperator`.
#
operator = ModelOperator(acquisition, "T2", bounds={"T2": (20.0, 400.0)})

# The amplitude starts from the first gridded echo, which is nearly free and
# is most of what makes the first Newton step sensible.
initial = operator.initial((1, SIZE, SIZE), T2=100.0).to(device)
initial[0, ..., 1] = gridded[..., 0].real
initial[0, ..., 2] = gridded[..., 0].imag

# %%
#
# The loop is an iteratively regularized Gauss-Newton: linearize, solve the
# linear problem that leaves, step, and lower the damping. TorchSim supplies
# the loop and the derivative, and **not the linear solver** --
# :func:`~torchsim.recon.iterative` hands the linearized problem to the same
# deepinv routine the two routes above called directly. Swapping in a proximal
# solver under a wavelet prior is a change to that one argument.
#
started = clock()
found = GaussNewton(
    Schedule(initial=1e-3, factor=0.5, minimum=1e-7),
    solve=iterative("CG", max_iter=20),
    max_iterations=8,
).minimize(operator, kspace, initial, encoding=encoding)
# No fit afterwards: the maps are what was solved for.
modelled = report(
    "model-based", clock() - started, operator.split(found.x)["T2"][0]
)

print(f"\nresidual {float(found.cost[0]):.3e} -> {float(found.cost[-1]):.3e}")
print(f"damping  {float(found.damping[0]):.0e} -> {float(found.damping[-1]):.0e}")

# %%
#
# Where the time goes
# -------------------
#
# Each conjugate-gradient step costs one product with the Jacobian and one
# with its adjoint, and each of those is the encoding operator once and the
# model once. Timing the four separately says which half a faster
# reconstruction would have to come from -- and on this problem they are
# comparable, so the model is not something to optimize around.
#
# Neither product builds the Jacobian. That is a memory argument rather than a
# speed one: the blocks are ``voxels x channels x contrasts`` where a signal
# is ``voxels x contrasts``, so what is not held is the channel count times
# the signal, every iteration.
#
tangent = torch.randn_like(initial)
predicted = operator.A_jvp(initial, tangent)
adjoint_image = encoding.A_adjoint(kspace).movedim(1, -1)


def timed(call, repeats=5):
    """Wall clock, after a warm-up, because the first call plans transforms."""
    call()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    if device == "cuda":
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / repeats


print(f"per conjugate-gradient step, {operator.channels} channels solved for:")
print(f"  model    J  v   {timed(lambda: operator.A_jvp(initial, tangent)):6.1f} ms")
print(f"  model    J^H v  {timed(lambda: operator.A_vjp(initial, adjoint_image)):6.1f} ms")
print(f"  encoding A      {timed(lambda: encoding.A(predicted.movedim(-1, 1))):6.1f} ms")
print(f"  encoding A^H    {timed(lambda: encoding.A_adjoint(kspace)):6.1f} ms")

blocks = operator.jacobian(initial)
print(
    f"\nthe Jacobian this avoids holding: "
    f"{blocks.numel() * 8 / 2**20:.1f} MiB, against "
    f"{predicted.numel() * 8 / 2**20:.1f} MiB for a signal"
)

# %%
#
# The maps
# --------
#
# The two routes that constrain the echoes against one another land in the
# same place, at about half the error of either route that reconstructs each
# contrast on its own -- and the subspace gets there in a fraction of the
# time. That is not the ordering the review reports, which had the nonlinear
# route slightly ahead; on eight echoes of a single exponential a rank-three
# basis leaves almost nothing for a nonlinear model to add, and the numbers
# here say so.
#
# Where the nonlinear route earns its cost is where a subspace stops being
# cheap: a phase-modulated signal needs tens of components rather than three,
# and a model with several parameters has no small basis at all.
#
shown = (
    ("truth", T2_true),
    ("adjoint per echo", adjoint),
    ("iterative per echo", separate),
    ("iterative subspace", linear),
    ("model-based", modelled),
)
figure, axes = plt.subplots(2, 5, figsize=(16, 6.5))
for column, (title, values) in enumerate(shown):
    picture = torch.where(brain, values, torch.tensor(0.0, device=device))
    handle = axes[0, column].imshow(
        picture.cpu(), cmap="viridis", vmin=0, vmax=250
    )
    axes[0, column].set_title(title)
    axes[0, column].axis("off")
    difference = torch.where(
        brain, (values - T2_true).abs(), torch.tensor(0.0, device=device)
    )
    axes[1, column].imshow(difference.cpu(), cmap="magma", vmin=0, vmax=80)
    axes[1, column].axis("off")
    axes[1, column].set_title("|error|" if column else "")
axes[1, 0].set_visible(False)
figure.colorbar(handle, ax=axes[0, :], shrink=0.8, label="T2 (ms)")
plt.show()

# %%
#
# Writing a different one
# -----------------------
#
# Nothing above is about T2. The model is the only thing that names a
# relaxation time, and it is an ordinary
# :class:`~torchsim.model.SignalModel` -- the same object the fitting and
# sequence-design examples use. Water-fat separation, T2* with a field map,
# a Look-Locker inversion recovery: each is a different ``evaluate``, and the
# operator, the loop and the encoding are unchanged.
#
# That is the contribution. A reconstruction library has to be told the
# physics; here the physics is the part you write, and everything that
# surrounds it already exists.
