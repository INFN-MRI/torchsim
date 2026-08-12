"""
=========================
Automatic differentiation
=========================

This example showcases the automatic differentiation capabilities
of the framework.

We first verify the analytical derivatives against finite differences and
compare their cost, then use them to design a refocusing train by minimizing
the Cramer-Rao lower bound on T2.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# First, we will import the required packages:
#
import warnings

warnings.filterwarnings("ignore")

from functools import partial

import numpy as np
import torch

from torch.func import jacrev

import matplotlib.pyplot as plt
import time

# %%
#
# We will show how to use automatic differentiation
# to automatically compute Cramer Rao Lower Bound.
#
# This can be used as a cost function to optimize acquisition schedules,
# for example for quantitative MRI
#
# We'll focus on a simple Fast Spin Echo acquisition:
#

import torchsim

# %%
#
# Cramer Rao Lower Bound is defined as the diagonal of the inverse
# of Fisher information matrix. This can be computed as
#


def calculate_crlb(grad, W=None, weight=1.0):
    if len(grad.shape) == 1:
        grad = grad[None, :]

    if W is None:
        W = torch.eye(grad.shape[0], dtype=grad.dtype, device=grad.device)

    J = torch.stack((grad.real, grad.imag), axis=0)  # (nparams, nechoes)
    J = J.permute(2, 1, 0)

    # calculate Fischer information matrix
    In = torch.einsum("bij,bjk->bik", J, J.permute(0, 2, 1))
    I = In.sum(axis=0)  # (nparams, nparams)

    # Invert
    return torch.trace(torch.linalg.inv(I) * W).real * weight


# %%
#
# notice that we used the trace as a cost function.
# For optimization, we need the gradient of this cost
# wrt sequence parameters.
#
# This can be obtained as:
#


def _crlb_cost(ESP, T1, T2, flip):

    # calculate signal and derivative
    _, grad = torchsim.fse_sim(flip=flip, ESP=ESP, T1=T1, T2=T2, diff="T2")

    # calculate cost
    return calculate_crlb(grad)


def crlb_cost(flip, ESP, T1, T2):
    flip = torch.as_tensor(flip, dtype=torch.float32)
    flip.requires_grad = True

    # get partial function
    _cost = partial(_crlb_cost, ESP, T1, T2)
    _dcost = jacrev(_cost)

    return _cost(flip).detach().cpu().numpy(), _dcost(flip).detach().cpu().numpy()


# %%
#
# As reference, we compute derivatives via finite differences
# approximation. This is inaccurate, but as easy to implement
# as automatic differentiation:
#


def fse_finitediff_grad(flip, ESP, T1, T2):
    sig = torchsim.fse_sim(flip=flip, ESP=ESP, T1=T1, T2=T2)

    # numerical derivative
    dt = 1.0
    dsig = torchsim.fse_sim(flip=flip, ESP=ESP, T1=T1, T2=T2 + dt)

    return sig, (dsig - sig) / dt


def _crlb_finitediff_cost(ESP, T1, T2, flip):

    # calculate signal and derivative
    _, grad = fse_finitediff_grad(flip, ESP, T1, T2)

    # calculate cost
    return calculate_crlb(grad).cpu().detach().numpy()


def crlb_finitediff_cost(flip, ESP, T1, T2):

    # initial cost
    cost0 = _crlb_finitediff_cost(ESP, T1, T2, flip)
    dcost = []

    for n in range(len(flip)):
        # get angles
        angles = flip.copy()
        angles[n] += 1.0
        dcost.append(_crlb_finitediff_cost(ESP, T1, T2, angles))

    return cost0, (np.asarray(dcost) - cost0)


# %%
#
# Now, we can compute optimization for a specific tissue.
#
# We assume T1 = 1000.0 ms and T2 = 100.0 ms:
#
t1 = 1000.0
t2 = 100.0

# %%
#
# Let's compute CRLB for a constant refocusing schedule:
#
angles = np.ones(48) * 60.0
esp = 5.0  # ms

# %%
#
# Run and plot timings:
#
tstart = time.time()
sig0, grad0 = fse_finitediff_grad(angles, esp, t1, t2)
tstop = time.time()
tgrad0 = tstop - tstart

tstart = time.time()
sig, grad = torchsim.fse_sim(flip=angles, ESP=esp, T1=t1, T2=t2, diff="T2")
tstop = time.time()
tgrad = tstop - tstart

# cost and derivative
tstart = time.time()
cost0, dcost0 = crlb_finitediff_cost(angles, esp, t1, t2)
tstop = time.time()
tcost0 = tstop - tstart

tstart = time.time()
cost, dcost = crlb_cost(angles, esp, t1, t2)
tstop = time.time()
tcost = tstop - tstart

nechoes = len(angles)
fsz = 10
plt.figure(figsize=(6, 9))
plt.subplot(4, 1, 1)
plt.rcParams.update({"font.size": 0.5 * fsz})
plt.plot(angles, ".")
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, nechoes + 1])
plt.ylabel("Flip Angle [deg]", fontsize=fsz)

plt.subplot(4, 1, 2)
plt.rcParams.update({"font.size": 0.5 * fsz})
plt.plot(abs(grad), "-k"), plt.plot(abs(grad0), "*r")
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, nechoes + 1])
plt.ylabel(r"$\frac{\partial signal}{\partial T2}$ [a.u.]", fontsize=fsz)
plt.legend(
    [
        "Auto Diff",
        "Finite Diff",
    ]
)

plt.subplot(4, 1, 3)
plt.rcParams.update({"font.size": 0.5 * fsz})
plt.plot(abs(dcost), "-k"), plt.plot(abs(dcost0), "*r")
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, nechoes + 1])
plt.ylabel(r"$\frac{\partial CRLB}{\partial FA}$ [a.u.]", fontsize=fsz)
plt.legend(["Auto Diff", "Finite Diff"])

plt.subplot(4, 1, 4)
labels = ["derivative of signal", "CRLB objective gradient"]
time_finite = [round(tgrad0, 2), round(tcost0, 2)]
time_auto = [round(tgrad, 2), round(tcost, 2)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
rects1 = plt.bar(x + width / 2, time_finite, width, label="Finite Diff")
rects2 = plt.bar(x - width / 2, time_auto, width, label="Auto Diff")

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel("Execution Time [s]", fontsize=fsz)
plt.xticks(x, labels, fontsize=fsz)
plt.legend()

plt.bar_label(rects1, padding=3, fontsize=fsz)
plt.bar_label(rects2, padding=3, fontsize=fsz)
plt.tight_layout()

# %%
#
# The two derivatives agree, but the finite-difference gradient of the
# objective needs one extra simulation *per flip angle*, so its cost grows
# with the length of the echo train while reverse-mode AD does not.
#
# Designing a refocusing train
# ----------------------------
#
# We can now feed that gradient to an optimizer. TorchSim ships the objective
# and the optimization loop, so the design problem is stated declaratively.
#
# Two things matter for a usable result. First, the data term is the
# *relative* Cramer-Rao bound, so short- and long-T2 species are weighted
# comparably. Second, raw T2 information is maximized by trains that alternate
# between extreme flip angles; such a train is not playable, so curvature and
# RF-power penalties select the smooth solution among the near-optimal ones.
#
from torchsim import FSET2Precision, SequenceOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
echo_train_length = 20

# a pair of design tissues bracketing the clinical range
objective = FSET2Precision(
    torch.tensor([800.0, 1400.0], device=device),
    torch.tensor([45.0, 120.0], device=device),
    esp,
)
optimizer = SequenceOptimizer(
    objective,
    bounds=(20.0, 180.0),
    iterations=80,
    learning_rate=0.08,
)

initial = torch.full((echo_train_length,), 120.0, device=device)
result = optimizer.optimize(initial)
optimized_flip = result.parameters.cpu().numpy()

# %%
#
# The optimizer converges to the shape used by clinical variable-flip-angle
# TSE: a high first refocusing pulse, a rapid drop to a pseudo-steady state
# that preserves magnetization, and a smooth ramp back up which recovers
# signal at the late echoes. The right panel shows what that buys, a slower
# and more informative decay than the constant train it started from.
#
echoes = np.arange(1, echo_train_length + 1)
figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))

axes[0].plot(echoes, optimized_flip, ".-")
axes[0].set(xlabel="Echo #", ylabel="Flip Angle [deg]", title="optimized train")
axes[0].set_ylim([0, 190])
axes[0].grid(alpha=0.3)

axes[1].semilogy(result.loss.cpu())
axes[1].set(xlabel="Iteration", ylabel="Objective", title="convergence")
axes[1].grid(alpha=0.3)

for label, train in [
    ("constant 120 deg", initial.cpu().numpy()),
    ("optimized", optimized_flip),
]:
    signal = torchsim.fse_sim(flip=train, ESP=esp, T1=1000.0, T2=100.0)
    axes[2].plot(echoes, abs(signal.numpy(force=True)), ".-", label=label)
axes[2].set(xlabel="Echo #", ylabel="Signal [a.u.]", title="T2 = 100 ms")
axes[2].legend(), axes[2].grid(alpha=0.3)
figure.tight_layout()
