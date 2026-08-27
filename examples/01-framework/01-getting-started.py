"""
==========================================
Running a sequence, and differentiating it
==========================================

The shortest thing you can do with TorchSim is name a protocol, hand it tissue,
and read the signal. The second shortest is to ask for the derivative of that
signal, which costs one extra pass and is what everything else in this gallery
is built on -- a Cramer-Rao bound, a nonlinear fit, a model-based
reconstruction and a sequence design all begin with it.

This example uses a fast spin echo that ships with TorchSim. Nothing here is
written by hand: the simulator, the derivative and the bound are all calls.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# Everything below is torch and TorchSim: a simulator, which carries both the
# sequence and the tissue it is being asked about, and
# :func:`~torchsim.crlb`, which turns a derivative into the precision it
# allows.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# sphinx_gallery_end_ignore
import time

import torch

import torchsim
from torchsim.simulators import FSESimulator

# %%
#
# A protocol, and the tissue it is played on
# ------------------------------------------
#
# A simulator names what the scanner does: here a 48-echo refocused train at a
# 5 ms echo spacing. Its constructor takes whatever
# :meth:`~torchsim.model.SignalModel.simulate` takes and fixes it, so the
# tissue is written down once with the sequence and what is
# left to give at the call is the part still under discussion -- in this case
# the refocusing angles.
#
# Three tissues at 3 T, given as arrays, are simulated together. Every property
# broadcasts, so a whole slice is the same call with longer arrays.
#
ECHOES = 48
ESP_MS = 5.0

NAMES = ("white matter", "grey matter", "CSF")
T1_MS = torch.tensor([830.0, 1330.0, 4000.0])
T2_MS = torch.tensor([80.0, 110.0, 2000.0])

acquisition = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, M0=1.0)

flip = torch.full((ECHOES,), 60.0)
signal = acquisition.simulate(flip=flip)

print(f"{tuple(signal.shape)} -- one row per tissue, one column per echo")
print(f"first echo: {signal[:, 0].abs().numpy().round(3)}")

# %%
#
# A constant 60 degree refocusing train is not a train of spin echoes. Most of
# the magnetization is stored along the longitudinal axis and brought back
# later, so what is sampled at each echo is a sum of coherence pathways rather
# than a single exponential -- which is exactly why an extended phase graph is
# needed to predict it and a mono-exponential fit of the train would be wrong.
#
refocused = acquisition.simulate(flip=torch.full((ECHOES,), 180.0))

# sphinx_gallery_start_ignore

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for axis, values, title in (
    (axes[0], signal, "a 60 degree train"),
    (axes[1], refocused, "a 180 degree train"),
):
    for row, name in enumerate(NAMES):
        axis.plot(values[row].abs().numpy(), label=name)
    axis.set(xlabel="Echo", ylabel="|signal|", title=title)
    axis.grid(alpha=0.3)
axes[0].legend(fontsize=8)
figure.tight_layout()
# sphinx_gallery_end_ignore
# sphinx_gallery_end_ignore

# %%
#
# Derivative with respect to tissue
# ---------------------------------
#
# :meth:`~torchsim.model.SignalModel.jacobian` returns the signal and its derivative
# with respect to the properties named. It is forward mode, one directional
# derivative per property, so a Fisher matrix over four parameters costs four
# passes however many voxels are being simulated.
#
signal, dT2 = acquisition.jacobian("T2", flip=flip)
print(f"derivative shape {tuple(dT2.shape)}")

# %%
#
# Before trusting it, check it against a finite difference. The two should
# agree to the step size, and the comparison is worth making once for any new
# model rather than assumed.
#
STEP_MS = 1.0
moved = acquisition.simulate(flip=flip, T2=T2_MS + STEP_MS)
finite = (moved - signal) / STEP_MS

discrepancy = (dT2 - finite).abs().max() / dT2.abs().max()
print(f"largest disagreement with a {STEP_MS} ms step: {float(discrepancy):.2e}")

# %%
#
# What the derivative is for
# --------------------------
#
# The Cramer-Rao bound is the lowest variance any unbiased estimate of a
# parameter can have. It is read off the Fisher information matrix, which is
# built from exactly the derivative above, so :func:`torchsim.crlb` takes that
# derivative and returns one variance per parameter.
#
# With a single unknown the Jacobian is one row and the bound is one number.
# Reported as a percentage of T2 itself, it says how tightly this train could
# ever pin down each tissue.
#
NOISE = 0.005  # standard deviation, relative to the fully relaxed magnetization

bound = torchsim.crlb(dT2[:, None, :], noise_variance=NOISE**2)
for row, name in enumerate(NAMES):
    sigma = 100.0 * float(bound[row, 0].sqrt()) / float(T2_MS[row])
    print(f"{name:<14} sigma(T2)/T2 >= {sigma:5.2f}%")

# %%
#
# Derivative with respect to the sequence
# ---------------------------------------
#
# The bound above is a number, and the flip angles are forty-eight of them.
# That is what reverse mode is for: build a cost on the signal and ask autograd
# for its gradient with respect to the schedule. Which kernel runs is decided
# by which inputs carry a gradient, so nothing has to be declared.
#


def precision(shots, angles):
    """The T2 variance this train allows, averaged over the three tissues."""
    _, derivative = shots.jacobian("T2", flip=angles)
    return torchsim.crlb(derivative[:, None, :], noise_variance=NOISE**2).mean()


def analytic_gradient(shots, angles):
    """The cost and its gradient, in one reverse pass."""
    angles = angles.clone().requires_grad_(True)
    cost = precision(shots, angles)
    (gradient,) = torch.autograd.grad(cost, angles)
    return float(cost), gradient.detach()


# %%
#
# As a reference, the same gradient by finite differences: one extra simulation
# per flip angle, which is the cost that grows with the length of the train.
#


def finite_gradient(shots, angles):
    """The cost and its gradient, one perturbed simulation at a time."""
    with torch.no_grad():
        cost = float(precision(shots, angles))
        moved = torch.empty_like(angles)
        for index in range(angles.numel()):
            nudged = angles.clone()
            nudged[index] += 1.0
            moved[index] = precision(shots, nudged)
    return cost, moved - cost


# %%
#
# Run both, over a range of train lengths, and time them. One reverse pass
# answers for the whole schedule whatever its length; the finite difference
# needs one simulation per angle, and each of those simulations is itself
# longer.
#
LENGTHS = (24, 48, 96, 192)


def timed(call):
    """Wall clock, after a warm-up."""
    call()
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


reverse_seconds, difference_seconds = [], []
for echoes in LENGTHS:
    shots = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, M0=1.0)
    angles = torch.full((echoes,), 60.0)
    (cost, analytic), seconds = timed(lambda: analytic_gradient(shots, angles))
    reverse_seconds.append(seconds)
    (_, difference), seconds = timed(lambda: finite_gradient(shots, angles))
    difference_seconds.append(seconds)
    print(f"{echoes:>4} echoes   reverse {reverse_seconds[-1]:6.3f} s   "
          f"finite {difference_seconds[-1]:6.3f} s   "
          f"({difference_seconds[-1] / reverse_seconds[-1]:5.1f}x)")

# %%
#
# The two gradients agree in shape and in sign, which is what a design loop
# follows. They do not agree exactly, and should not: a one-degree step is a
# large one on a curve this bent, and the discrepancy is the finite
# difference's rather than the derivative's.
#

# sphinx_gallery_start_ignore

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].plot(dT2[0].abs().numpy(), "-k", label="forward mode")
axes[0].plot(finite[0].abs().numpy(), "*r", ms=4, label="finite difference")
axes[0].set(
    xlabel="Echo",
    ylabel=r"$|\partial\,\mathrm{signal}/\partial T_2|$",
    title=f"tissue derivative, {ECHOES} echoes",
)

axes[1].plot(analytic.numpy(), "-k", label="reverse mode")
axes[1].plot(difference.numpy(), "*r", ms=4, label="finite difference")
axes[1].set(
    xlabel="Echo",
    ylabel=r"$\partial\,\mathrm{CRLB}/\partial\alpha$",
    title=f"schedule gradient, {LENGTHS[-1]} echoes",
)

axes[2].plot(LENGTHS, reverse_seconds, "-ok", label="reverse mode")
axes[2].plot(LENGTHS, difference_seconds, "-*r", label="finite difference")
axes[2].set(
    xlabel="Echoes in the train",
    ylabel="Time for one gradient [s]",
    title="what each costs",
    xscale="log",
    yscale="log",
)
for axis in axes:
    axis.grid(alpha=0.3), axis.legend(fontsize=8)
figure.tight_layout()
# sphinx_gallery_end_ignore
# sphinx_gallery_end_ignore

# %%
#
# Where this goes
# ---------------
#
# The gradient with respect to tissue is what a fit descends and what a
# model-based reconstruction pushes through an encoding operator. The gradient
# with respect to the schedule is what designs a protocol: replace the cost
# above with one about image quality and the loop is unchanged, which is what
# the sequence-optimization examples do.
#
# Neither needed a kernel to be written. When the sequence you want is not one
# of the ones that ship, the next examples say what to write instead -- and it
# is a Python function either way.
#