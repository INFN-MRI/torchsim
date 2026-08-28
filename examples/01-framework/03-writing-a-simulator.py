"""
======================
Writing a signal model
======================

A signal model is written in two pieces. The **physics** says what a
voxel holds -- which tissue properties are exposed, and so which physics the
kernels carry -- and what each kind of event does to it. A **simulator** says
what order the events are played in. Everything else -- which kernel runs, how
the work is cut across memory and devices, how derivatives are taken --
follows from those two and is not yours to write.

This example builds an inversion-prepared SSFP fingerprinting sequence from
scratch, differentiates it with respect to tissue, and then differentiates a
cost with respect to the sequence.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# The two pieces a signal model is written from, and the trigger set that
# says what an unbalanced readout plays.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


# Every figure is drawn at the width of the documentation column, so none of
# them is scaled on the way in and type is the same size throughout.
PAGE_WIDTH = 8.6  # inches

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
from dataclasses import replace

import numpy as np
import torch

from torchsim.model import (
    UNBALANCED,
    Simulator,
    SpinPhysics,
)

# %%
# Saying what the events do
# -------------------------
# A :class:`~torchsim.model.SpinPhysics` is the physics. ``properties``
# maps the name a caller uses to the tissue field it fills, so the model keeps
# the vocabulary your protocol is written in while the engine keeps its own.
#
# It is also the whole of how you ask for physics. A field you do not name is
# never handed to the tissue, and the kernels leave its term out -- so a T1/T2
# model pays for no off-resonance turn, no diffusion attenuation and no flow
# winding. Name ``b0_hz`` and the off-resonance term comes back.
#
# ``operators`` is what each kind of event is realized as. ``UNBALANCED`` says
# a Readout is followed by one unbalanced gradient, which is what makes this
# an SSFP-FID rather than a balanced or a spoiled train. Swapping it is how
# you change that, and it is the only thing you change.

physics = SpinPhysics(
    properties={"T1": "t1_ms", "T2": "t2_ms"},
    operators=UNBALANCED,
)

# %%
# Saying what order they play in
# ------------------------------
# An :class:`~torchsim.model.Simulator` is the protocol. You do not
# write timestamps: ``layout`` returns the *operators* of one repetition in
# order, and the simulator turns the span each one holds into the timestamps a
# description carries.
#
# The operators are bound when the simulator is constructed. What ``layout``
# then produces is an ordinary description whose events carry their own action
# word, and from there the path is the fused one -- packing, the feature mask,
# offload and sharding. Nothing consults a trigger during a run.


class SSFPMRF(Simulator):
    """An Inversion, then one Excitation and one sample per repetition."""

    model = physics
    states = 10

    def layout(self, *, flip, TR, TI=0.0):
        """Return one repetition's operators, in the order they are played."""
        angles = torch.deg2rad(torch.as_tensor(flip))
        parts = [self.operators.inversion(duration_s=TI * 1e-3)]
        for index in range(angles.numel()):
            parts.append(self.operators.excitation(angles[index]))
            parts.append(self.operators.readout(duration_s=TR * 1e-3))
        return parts


# %%
# Running it
# ----------
# Property and sequence arguments are given together and told apart by
# ``properties``. A scalar property is one voxel; an array is a map, and every
# voxel runs at once.
#
flip = np.concatenate(
    (np.linspace(5.0, 60.0, 350), np.linspace(60.0, 1.0, 350), np.ones(180))
)

sequence = SSFPMRF(flip=flip, TR=10.0, TI=20.0)
signal = sequence.simulate(T1=1000.0, T2=100.0)

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(signal))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")
# sphinx_gallery_end_ignore

# %%
# The same call over a parameter map returns one row per voxel:
#
signals = sequence.simulate(
    T1=torch.tensor([500.0, 1000.0, 1500.0]),
    T2=torch.tensor([50.0, 100.0, 150.0]),
)

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(signals.T))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")
# sphinx_gallery_end_ignore

# %%
# Derivatives with respect to tissue: forward mode
# ------------------------------------------------
# A Bloch simulation records far more samples than it takes parameters, so a
# derivative with respect to tissue is cheapest taken forwards: one directional
# derivative per property yields every voxel's derivative at once, and the cost
# is one pass per property rather than per voxel.
#
# That is what :meth:`~torchsim.model.SignalModel.jacobian` does. A single name
# collapses the parameter axis; a sequence of names keeps it.
#
signal, jacobian = sequence.jacobian(("T1", "T2"), T1=1000.0, T2=100.0)

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(jacobian.T))
plt.xlabel("TR index")
plt.ylabel("signal jacobian [a.u.]")
# sphinx_gallery_end_ignore

# %%
# Derivatives with respect to the sequence: reverse mode
# ------------------------------------------------------
# The acquisition optimization problem runs the other way: one scalar cost,
# many sequence parameters. That is reverse mode, and it is deliberately not
# wrapped -- build a cost on the signal and call ``backward()``. The engine
# reads which of its inputs carry a gradient and picks its kernel from that, so
# a layer here would only hide the choice.
#
schedule = torch.tensor(flip, dtype=torch.float32, requires_grad=True)
recorded = sequence.simulate(T1=1000.0, T2=100.0, flip=schedule)
loss = -recorded.abs().square().sum()
loss.backward()

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(schedule.grad)
plt.xlabel("TR index")
plt.ylabel("d(loss) / d(flip) [1/deg]")
# sphinx_gallery_end_ignore

# %%
# Asking for more physics
# -----------------------
# Off-resonance is one line: name the tissue field, and the kernels start
# carrying the turn it puts on the states.
#


detuned = SSFPMRF(
    model=replace(physics, properties={"T1": "t1_ms", "T2": "t2_ms", "B0": "b0_hz"}),
    flip=flip,
    TR=10.0,
    TI=20.0,
).simulate(T1=1000.0, T2=100.0, B0=torch.tensor([0.0, 30.0, 60.0]))

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(detuned.T))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")
# sphinx_gallery_end_ignore

# %%
# Note that ``B0`` is a *map* here. Had it been left at its scalar default the
# model would have declared it and still paid nothing: a property at the value
# where it has no effect is reported as absent, and its term stays out of the
# kernel.
#
# A functional wrapper
# --------------------
# The shipped models come with one, and yours can too:


def ssfp_mrf_sim(flip, TR, T1, T2, TI=0.0, diff=None):
    """Simulate an inversion-prepared SSFP train, and differentiate it."""
    sequence = SSFPMRF(flip=flip, TR=TR, TI=TI)
    if diff is None:
        return sequence.simulate(T1=T1, T2=T2)
    return sequence.jacobian(diff, T1=T1, T2=T2)


signal, jacobian = ssfp_mrf_sim(flip, 10.0, 1000.0, 100.0, diff=("T1", "T2"))
# signal is (repetitions,); jacobian is (2, repetitions), one row per property
