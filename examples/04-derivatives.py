"""
==========================================
Automatic differentiation and precision
==========================================

Two things a derivative is for.

The first is a **derivative with respect to tissue**: how much the signal moves
when T1 or T2 moves. That is what a Cramer-Rao bound is built from, and we
check it against finite differences before trusting it.

The second is a **derivative with respect to the sequence**: how much a cost
moves when a flip angle moves. That is what designs a protocol, and here we
use it to choose the flip angles of a joint SPGR/bSSFP relaxometry experiment
so that T1 and T2 come out as precisely as the scan time allows [1]_.

.. [1] Teixeira RPAG, Malik SJ, Hajnal JV. Joint system relaxometry (JSR) and
   Cramer-Rao lower bound optimization of sequence parameters: a framework for
   enhanced precision of DESPOT T1 and T2 estimation.
   Magn Reson Med. 2018;79:234-245.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# First, the imports:
#
import warnings

warnings.filterwarnings("ignore")

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchsim

# %%
#
# The Cramer-Rao bound
# --------------------
#
# The bound is the lowest variance any unbiased estimate of a parameter can
# have. It is read off the Fisher information matrix, which is built from the
# derivative of the signal with respect to every parameter being estimated, so
# it needs exactly the derivative the simulator produces:
# :func:`torchsim.crlb` takes that derivative and returns one variance per
# parameter.
#
# We start with a single parameter -- T2 from a fast spin echo train -- so the
# derivative is one row and the bound is one number.
#


def t2_bound(flip, ESP, T1, T2):
    """The variance an unbiased T2 estimate cannot beat, per unit noise."""
    _, derivative = torchsim.fse_sim(
        flip=flip, ESP=ESP, T1=T1, T2=T2, diff="T2"
    )
    return torchsim.crlb(torch.as_tensor(derivative)[None, :]).sum()


# %%
#
# The cost is one number and the schedule is many, which is what reverse mode
# is for: build the cost on the signal and ask autograd for its gradient. The
# simulator reads which of its inputs carry one and picks its kernel from
# that.
#


def analytic_gradient(flip, ESP, T1, T2):
    flip = torch.as_tensor(flip, dtype=torch.float32).clone()
    flip.requires_grad = True

    cost = t2_bound(flip, ESP, T1, T2)
    (gradient,) = torch.autograd.grad(cost, flip)

    return cost.detach().cpu().numpy(), gradient.detach().cpu().numpy()


# %%
#
# As a reference, the same derivatives by finite differences. Inaccurate, but
# as easy to write -- and the point of the comparison is what each costs.
#


def finite_difference_signal(flip, ESP, T1, T2):
    signal = torchsim.fse_sim(flip=flip, ESP=ESP, T1=T1, T2=T2)
    step = 1.0
    moved = torchsim.fse_sim(flip=flip, ESP=ESP, T1=T1, T2=T2 + step)
    return signal, (moved - signal) / step


def finite_difference_bound(flip, ESP, T1, T2):
    _, derivative = finite_difference_signal(flip, ESP, T1, T2)
    return float(torchsim.crlb(torch.as_tensor(derivative)[None, :]).sum())


def finite_difference_gradient(flip, ESP, T1, T2):
    reference = finite_difference_bound(flip, ESP, T1, T2)
    moved = []
    for index in range(len(flip)):
        angles = flip.copy()
        angles[index] += 1.0
        moved.append(finite_difference_bound(angles, ESP, T1, T2))
    return reference, np.asarray(moved) - reference


# %%
#
# For one tissue -- T1 = 1000 ms, T2 = 100 ms -- and a constant refocusing
# train:
#
t1 = 1000.0
t2 = 100.0
angles = np.ones(48) * 60.0
esp = 5.0  # ms

# %%
#
# Run both, and time them:
#
start = time.time()
_, reference_derivative = finite_difference_signal(angles, esp, t1, t2)
finite_derivative_time = time.time() - start

start = time.time()
_, derivative = torchsim.fse_sim(flip=angles, ESP=esp, T1=t1, T2=t2, diff="T2")
analytic_derivative_time = time.time() - start

start = time.time()
_, finite_gradient = finite_difference_gradient(angles, esp, t1, t2)
finite_gradient_time = time.time() - start

start = time.time()
_, analytic = analytic_gradient(angles, esp, t1, t2)
analytic_gradient_time = time.time() - start

echoes = len(angles)
fsz = 10
plt.figure(figsize=(6, 9))
plt.subplot(4, 1, 1)
plt.rcParams.update({"font.size": 0.5 * fsz})
plt.plot(angles, ".")
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, echoes + 1])
plt.ylabel("Flip Angle [deg]", fontsize=fsz)

plt.subplot(4, 1, 2)
plt.rcParams.update({"font.size": 0.5 * fsz})
plt.plot(abs(derivative), "-k"), plt.plot(abs(reference_derivative), "*r")
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, echoes + 1])
plt.ylabel(r"$\frac{\partial signal}{\partial T2}$ [a.u.]", fontsize=fsz)
plt.legend(["Auto Diff", "Finite Diff"])

plt.subplot(4, 1, 3)
plt.rcParams.update({"font.size": 0.5 * fsz})
plt.plot(abs(analytic), "-k"), plt.plot(abs(finite_gradient), "*r")
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, echoes + 1])
plt.ylabel(r"$\frac{\partial CRLB}{\partial FA}$ [a.u.]", fontsize=fsz)
plt.legend(["Auto Diff", "Finite Diff"])

plt.subplot(4, 1, 4)
labels = ["derivative of signal", "CRLB objective gradient"]
finite_times = [round(finite_derivative_time, 2), round(finite_gradient_time, 2)]
analytic_times = [
    round(analytic_derivative_time, 2),
    round(analytic_gradient_time, 2),
]

position = np.arange(len(labels))
width = 0.35
finite_bars = plt.bar(
    position + width / 2, finite_times, width, label="Finite Diff"
)
analytic_bars = plt.bar(
    position - width / 2, analytic_times, width, label="Auto Diff"
)

plt.ylabel("Execution Time [s]", fontsize=fsz)
plt.xticks(position, labels, fontsize=fsz)
plt.legend()
plt.bar_label(finite_bars, padding=3, fontsize=fsz)
plt.bar_label(analytic_bars, padding=3, fontsize=fsz)
plt.tight_layout()

# %%
#
# The two derivatives agree, but the finite-difference gradient of the
# objective needs one extra simulation *per flip angle*, so its cost grows
# with the length of the echo train while reverse-mode AD does not.
#
# Designing a joint relaxometry protocol
# --------------------------------------
#
# DESPOT estimates T1 from a set of spoiled gradient-echo acquisitions at
# different flip angles and T2 from a set of balanced SSFP acquisitions.
# Fitting them jointly rather than one after the other uses all the data for
# both parameters, and then the flip angles themselves can be chosen to make
# the joint estimate as precise as possible.
#
# A design problem is stated in three pieces. An
# :class:`~torchsim.Acquisition` is a simulator with the tissue it is being
# designed for already in place, so only the parameters under design are left
# to give. Both sequences here are closed forms, and they are constructed and
# asked exactly as a state-machine sequence would be.
#
from torchsim.optim import Acquisition, Bounded, SequenceDesign
from torchsim.simulators import SPGRSimulator, bSSFPSimulator

# White and grey matter at 3 T -- the design is for both at once.
T1_MS = torch.tensor([830.0, 1330.0])
T2_MS = torch.tensor([80.0, 110.0])
# Noise standard deviation, as a fraction of the fully relaxed magnetization.
NOISE = 0.005

spgr = Acquisition(
    SPGRSimulator(TE=2.0, TR=6.0), T1=T1_MS, T2star=T2_MS, M0=1.0, B0=0.0
)
ssfp = Acquisition(
    bSSFPSimulator(TE=2.5, TR=5.0), T1=T1_MS, T2=T2_MS, M0=1.0, B0=0.0
)

# %%
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

# %%
#
# Nothing here was specific to DESPOT except the cost. Designing for image
# quality rather than for precision is the same three pieces with a different
# function in the middle, which is the next example.
#
