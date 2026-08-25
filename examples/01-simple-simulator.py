"""
======================
Writing a signal model
======================

A signal model is written in two pieces. A **state-machine model** says what a
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
# We begin with the necessary imports:
#
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsim.model import (
    SPOILED,
    UNBALANCED,
    AbstractSimulator,
    StateMachineModel,
)

# %%
# Saying what the events do
# -------------------------
# A :class:`~torchsim.model.StateMachineModel` is the physics. ``properties``
# maps the name a caller uses to the tissue field it fills, so the model keeps
# the vocabulary your protocol is written in while the engine keeps its own.
#
# It is also the whole of how you ask for physics. A field you do not name is
# never handed to the tissue, and the kernels leave its term out -- so a T1/T2
# model pays for no off-resonance turn, no diffusion attenuation and no flow
# winding. Name ``b0_hz`` and the off-resonance term comes back.
#
# ``triggers`` is what each kind of event is realized as. ``UNBALANCED`` says
# a Readout is followed by one unbalanced gradient, which is what makes this
# an SSFP-FID rather than a balanced or a spoiled train. Swapping it is how
# you change that, and it is the only thing you change.

physics = StateMachineModel(
    properties={"T1": "t1_ms", "T2": "t2_ms"},
    triggers=UNBALANCED,
)

# %%
# Saying what order they play in
# ------------------------------
# An :class:`~torchsim.model.AbstractSimulator` is the protocol. You do not
# write timestamps: ``layout`` returns the *operators* of one repetition in
# order, and the simulator turns the span each one holds into the timestamps a
# description carries.
#
# The triggers are bound when the simulator is constructed. What ``layout``
# then produces is an ordinary description whose events carry their own action
# word, and from there the path is the fused one -- packing, the feature mask,
# offload and sharding. Nothing consults a trigger during a run.


class SSFPMRF(AbstractSimulator):
    """An Inversion, then one Excitation and one sample per repetition."""

    model = physics
    states = 10

    def layout(self, *, flip, TR, TI=0.0):
        """Return one repetition's operators, in the order they are played."""
        angles = torch.deg2rad(torch.as_tensor(flip))
        parts = [self.triggers.inversion(duration_s=TI * 1e-3)]
        for index in range(angles.numel()):
            parts.append(self.triggers.excitation(angles[index]))
            parts.append(self.triggers.readout(duration_s=TR * 1e-3))
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

plt.figure()
plt.plot(abs(signal))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")

# %%
# The same call over a parameter map returns one row per voxel:
#
signals = sequence.simulate(
    T1=torch.tensor([500.0, 1000.0, 1500.0]),
    T2=torch.tensor([50.0, 100.0, 150.0]),
)

plt.figure()
plt.plot(abs(signals.T))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")

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

plt.figure()
plt.plot(abs(jacobian.T))
plt.xlabel("TR index")
plt.ylabel("signal jacobian [a.u.]")

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

plt.figure()
plt.plot(schedule.grad)
plt.xlabel("TR index")
plt.ylabel("d(loss) / d(flip) [1/deg]")

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

plt.figure()
plt.plot(abs(detuned.T))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")

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
print(signal.shape, jacobian.shape)
