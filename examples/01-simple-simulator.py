"""
======================
Writing a signal model
======================

A signal model says two things: which tissue properties it exposes, and what
sequence it plays. Everything else -- which kernel runs, how the work is cut
across memory and devices, how derivatives are taken -- follows from those two
and is not yours to write.

This example builds an inversion-prepared SSFP fingerprinting model from
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
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsim.model import EpgModel
from torchsim.sequence import (
    SSFPFID,
    EventAction,
    SequenceDescription,
    compose,
    excitation,
    ideal_rf_definition,
    inversion,
    readout,
)

# %%
# Describing the sequence
# -----------------------
# A description is a stream of events at absolute times. You do not write those
# times: you lay out *operators* -- a pulse, a readout, a delay, a whole
# preparation -- and :func:`~torchsim.sequence.compose` turns the span each one
# holds into the timestamps the stream carries.
#
# Our sequence is an inversion, then one excitation and one sample per
# repetition, with the states wound on by an unbalanced gradient afterwards.
# ``EventAction.SHIFT_AFTER`` on the readout is what says so.


def ssfp_train(flip_rad, repetition_s, inversion_s):
    """Lay out one inversion-prepared unbalanced SSFP train."""
    modules = [inversion(duration_s=inversion_s)]
    for index in range(flip_rad.numel()):
        modules.append(excitation(flip_rad[index]))
        modules.append(
            readout(
                action=EventAction.SHIFT_AFTER,
                duration_s=repetition_s,
            )
        )
    return compose(*modules)


# %%
# Declaring the model
# -------------------
# ``properties`` maps the name a caller uses to the tissue field it fills, so
# the model keeps the vocabulary your protocol is written in while the engine
# keeps its own.
#
# It is also the whole of how you ask for physics. A field you do not name is
# never handed to the tissue, and the kernels leave its term out -- so a T1/T2
# model pays for no off-resonance turn, no diffusion attenuation and no flow
# winding. Name ``b0_hz`` and the off-resonance term comes back.
#
# ``describe`` receives the sequence arguments in the units a user quotes them
# in -- milliseconds, degrees -- and converts them, because that conversion is
# part of the sequence rather than of the engine.


class SSFPMRFModel(EpgModel):
    """Inversion-prepared unbalanced SSFP with a variable flip-angle schedule."""

    properties = {"T1": "t1_ms", "T2": "t2_ms"}
    simulator = SSFPFID()
    states = 10

    def describe(self, *, flip, TR, TI=0.0):
        """Return the train, in the seconds and radians a description carries."""
        events, duration_s = ssfp_train(
            torch.pi / 180.0 * torch.as_tensor(flip),
            torch.as_tensor(TR) * 1e-3,
            torch.as_tensor(TI) * 1e-3,
        )
        return SequenceDescription(
            subsequence_index=0,
            tr_duration_us=1e6 * duration_s,
            events=events,
            rf_definitions={0: ideal_rf_definition()},
        )


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

model = SSFPMRFModel()
signal = model.simulate(T1=1000.0, T2=100.0, flip=flip, TR=10.0, TI=20.0)

plt.figure()
plt.plot(abs(signal))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")

# %%
# The same call over a parameter map returns one row per voxel:
#
signals = model.simulate(
    T1=torch.tensor([500.0, 1000.0, 1500.0]),
    T2=torch.tensor([50.0, 100.0, 150.0]),
    flip=flip,
    TR=10.0,
    TI=20.0,
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
signal, jacobian = model.jacobian(
    ("T1", "T2"), T1=1000.0, T2=100.0, flip=flip, TR=10.0, TI=20.0
)

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
recorded = model.simulate(T1=1000.0, T2=100.0, flip=schedule, TR=10.0, TI=20.0)
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


class OffResonantMRFModel(SSFPMRFModel):
    """The same train, with an off-resonance map."""

    properties = {"T1": "t1_ms", "T2": "t2_ms", "B0": "b0_hz"}


detuned = OffResonantMRFModel().simulate(
    T1=1000.0, T2=100.0, B0=torch.tensor([0.0, 30.0, 60.0]), flip=flip, TR=10.0, TI=20.0
)

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
    model = SSFPMRFModel()
    values = {"flip": flip, "TR": TR, "T1": T1, "T2": T2, "TI": TI}
    if diff is None:
        return model.simulate(**values)
    return model.jacobian(diff, **values)


signal, jacobian = ssfp_mrf_sim(flip, 10.0, 1000.0, 100.0, diff=("T1", "T2"))
print(signal.shape, jacobian.shape)
