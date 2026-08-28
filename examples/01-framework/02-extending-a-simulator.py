"""
==========================================
Extending a simulator, and assembling one
==========================================

Between calling a sequence that ships with TorchSim and writing a signal model
from scratch there are two smaller steps, and most of what a user actually
wants is one of them.

The first is **giving an existing simulator physics it does not carry**. A
simulator declares which tissue properties it exposes, and that declaration is
what decides which terms the kernels evaluate. Adding a property to the
declaration is the whole of adding the physics: the timing, the operators and
the layout are untouched.

The second is **assembling a new sequence out of operators that already
exist**. A layout is a list of modules -- a pulse, a spoiler, a delay, a
readout -- and writing a sequence TorchSim does not ship is writing that list.

This example does both, and checks each against something that was already
true.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# A model is a frozen dataclass, so extending one is
# :func:`~dataclasses.replace`. The rest is the operator vocabulary a layout is
# written in, and the shipped fingerprinting sequence being extended.
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


def key(axes, ncols=1):
    """The legend above what it describes, clear of the curves and the titles.

    Takes a figure, where every panel is showing the same series, and puts one
    legend over the whole of it. Takes an axis, or several, where the panels
    differ, and puts a legend over each -- every titled panel in the figure
    then ends up with the same padding, so the titles line up whether or not
    that panel carries one, which is only known once it has been laid out.
    """
    if hasattr(axes, "add_subplot"):
        handles, labels = axes.axes[0].get_legend_handles_labels()
        return axes.legend(handles, labels, loc="outside upper center",
                           ncols=ncols, frameon=False, handlelength=1.6,
                           columnspacing=1.4)
    axes = [axes] if hasattr(axes, "get_legend_handles_labels") else list(axes)
    figure = axes[0].figure
    legends = [
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=ncols,
                    frameon=False, borderaxespad=0.0, handlelength=1.6,
                    columnspacing=1.4)
        for axis in axes
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tallest = max(legend.get_window_extent(renderer).height for legend in legends)
    for axis in figure.axes:
        if axis.get_title():
            axis.set_title(axis.get_title(), pad=72.0 * tallest / figure.dpi + 4.0)
    return legends


# sphinx_gallery_end_ignore
from dataclasses import replace

import torch

from torchsim.model import SPOILED, Simulator, SpinPhysics
from torchsim.sequence import Delay, Excitation, Spoil, module
from torchsim.simulators import MRFSimulator
# %%
#
# What a simulator declares
# -------------------------
#
# The physics is two things: a map from the names a caller uses to
# the tissue fields they fill, and a set of operators saying what each kind of
# event plays. The shipped fingerprinting sequence declares five properties.
#
MRFSimulator.model.properties

# %%
#
# The right-hand column is the point. Those are fields of
# :class:`~torchsim.TissueProperties`, and the kernels carry a term for each
# one they are given -- but only for the ones they are given. A property left
# out of the declaration is not a property set to a default: its term is never
# evaluated, and the run does not pay for it.
#
# The fields no simulator here claims are the interesting ones. Off-resonance,
# diffusion, flow, a chemically exchanging second pool, and the macromolecular
# pool that RF saturates are all terms the state machine already knows how to
# evaluate, waiting for a model to ask.
#

# sphinx_gallery_start_ignore
declared = set(MRFSimulator.model.properties.values())
available = [
    "b0_hz",
    "diffusion_um2_per_ms",
    "velocity_m_per_s",
    "bound_fraction",
    "bound_exchange_hz",
    "t1_bound_ms",
    "pool_b_fraction",
    "pool_b_shift_hz",
]
print("carried by the kernels, not asked for here:")
print("  " + ", ".join(name for name in available if name not in declared))
# sphinx_gallery_end_ignore

# %%
#
# Adding a bound pool
# -------------------
#
# Magnetization transfer is three fields: how much of the magnetization sits in
# the macromolecular pool, how fast that pool exchanges with the free water,
# and how fast it relaxes. Asking for them is one
# :func:`~dataclasses.replace` on the model.
#
# Nothing else changes. The train, the inversion, the readouts that wind the
# states on, and the flip angle schedule are all inherited, because none of
# them had anything to do with how many pools a voxel has.
#


class MTFingerprinting(MRFSimulator):
    """The shipped fingerprinting train, on a tissue with a bound pool."""

    model = replace(
        MRFSimulator.model,
        properties={
            **MRFSimulator.model.properties,
            "bound_fraction": "bound_fraction",
            "bound_exchange": "bound_exchange_hz",
            "T1_bound": "t1_bound_ms",
        },
    )


# %%
#
# The same protocol, built from both models:
#
CONTRASTS = 400
TR_MS = 10.0
TI_MS = 20.0

repetition = torch.arange(CONTRASTS, dtype=torch.float32)
flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / CONTRASTS) ** 2

single = MRFSimulator(flip=flip, TR=TR_MS, TI=TI_MS, states=20)
two_pool = MTFingerprinting(flip=flip, TR=TR_MS, TI=TI_MS, states=20)

WHITE_MATTER = {"T1": 830.0, "T2": 80.0}
POOL = {"bound_fraction": 0.12, "bound_exchange": 30.0, "T1_bound": 1000.0}

# %%
#
# Before looking at what the pool does, check what it does when there isn't
# one. ``bound_fraction`` is the *gate* on the whole second pool: at zero the
# exchange conserves nothing across pools and the bound pool starts empty, so
# the kernels drop the term entirely. The two models must then agree exactly,
# not nearly -- and if they do not, the extension changed something it had no
# business changing.
#
without = single.simulate(**WHITE_MATTER)
gated = two_pool.simulate(**WHITE_MATTER, bound_fraction=0.0)

with_pool = two_pool.simulate(**WHITE_MATTER, **POOL)
shift = (without - with_pool).abs().max() / without.abs().max()

# sphinx_gallery_start_ignore
print(f"\ngate closed, identical: {torch.equal(without, gated)}")
print(f"gate open, largest change: {100 * float(shift):.0f}% of the peak signal")
# sphinx_gallery_end_ignore

# %%
#
# A pool the readouts never sample, holding a tenth or so of the magnetization,
# moves the trajectory by a sizeable fraction of its peak. That is the case for
# asking: the fingerprint of white matter is not the fingerprint of a single
# pool with white matter's relaxation times, and a dictionary built from the
# smaller model is wrong in a way no amount of matching will recover.
#
# The derivative is what says whether it is *estimable* rather than merely
# present, and it comes from the extended model for the same one extra pass
# any other property costs.
#
_, sensitivity = two_pool.jacobian("bound_fraction", **WHITE_MATTER, **POOL)

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.1))
axes[0].plot(repetition.numpy(), flip.numpy())
axes[0].set(xlabel="Repetition", ylabel="Flip angle [deg]", title="the schedule")

axes[1].plot(without.abs().numpy(), label="one pool")
axes[1].plot(with_pool.abs().numpy(), label="bound fraction 0.12")
axes[1].set(xlabel="Repetition", ylabel="|signal|", title="what the pool does")

axes[2].plot(sensitivity.abs().numpy(), color="crimson")
axes[2].set(
    xlabel="Repetition",
    ylabel=r"$|\partial\,\mathrm{signal}/\partial f_\mathrm{b}|$",
    title="where it is visible",
)
for axis in axes:
    axis.grid(alpha=0.3)
key(axes[1])
# sphinx_gallery_end_ignore

# %%
#
# Assembling a sequence from operators
# ------------------------------------
#
# The other direction. A layout is a list of **operators** -- one module of a
# sequence each, knowing what it plays and how long it holds the timeline --
# and TorchSim ships the ordinary ones: an excitation, a refocusing pulse, an
# inversion, a delay, ideal spoiling, and a readout in each of the flavours a
# repetition can end in.
#
# Saturation recovery is a T1 measurement that is not among the shipped
# sequences and is four of those operators. Every measurement starts by
# destroying whatever magnetization was there -- a 90 degree pulse and a
# spoiler -- waits, and reads what has recovered.
#
# :func:`~torchsim.module` packages the pulse and the spoiler as one operator,
# so the layout reads as the three things a physicist would name.
#


class SaturationRecovery(Simulator):
    """Saturate, wait, read what recovered -- once per saturation time.

    Parameters are given at construction or at the call, as for any simulator:
    ``TS`` is the saturation times in milliseconds and ``flip`` the readout
    flip angle in degrees.
    """

    model = SpinPhysics(
        properties={"T1": "t1_ms", "T2": "t2_ms", "M0": "m0", "B1": "b1"},
        operators=SPOILED,
    )
    # Every block begins by destroying the transverse magnetization, so nothing
    # is carried in a dephased configuration and one order is the whole state.
    states = 1

    def layout(self, *, TS, flip, phases=0.0):
        """Return one saturate-wait-read block per saturation time."""
        waits = torch.atleast_1d(torch.as_tensor(TS)) * 1e-3
        angle = torch.deg2rad(torch.as_tensor(flip)).broadcast_to(waits.shape)
        turn = torch.deg2rad(torch.as_tensor(phases)).broadcast_to(waits.shape)

        saturate = module(Excitation(torch.pi / 2), Spoil(), duration_s=0.0)
        parts = []
        for index in range(waits.numel()):
            parts.append(saturate)
            parts.append(Delay(waits[index]))
            parts.append(self.operators.excitation(angle[index], turn[index]))
            parts.append(self.operators.readout(turn[index]))
        return parts


# %%
#
# ``self.operators`` is why the excitation and the readout are not named
# directly. ``SPOILED`` says a readout is followed by ideal transverse
# spoiling; swapping it for ``UNBALANCED`` or ``BALANCED`` changes what this
# sequence is without touching a line of the layout above.
#
# What it should say
# ------------------
#
# Saturation recovery is the one T1 experiment with an answer written down:
# what is read at a saturation time is
# :math:`M_0 \\sin\\alpha \\, (1 - e^{-T_S/T_1})`, because the saturation leaves
# nothing behind and the recovery is undisturbed until the readout.
#
# The state machine does not know that. It plays the events and it has never
# been told what sequence they add up to, so agreeing with the closed form is a
# real check rather than a tautology.
#
SATURATION_TIMES = torch.tensor([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0])
FLIP_DEG = 10.0

sequence = SaturationRecovery(TS=SATURATION_TIMES, flip=FLIP_DEG)
simulated = sequence.simulate(T1=830.0, T2=80.0, M0=1.0).abs()

expected = torch.sin(torch.deg2rad(torch.tensor(FLIP_DEG))) * (
    1 - torch.exp(-SATURATION_TIMES / 830.0)
)
# sphinx_gallery_start_ignore
print(f"\nlargest disagreement with the closed form: "
      f"{float((simulated - expected).abs().max()):.2e}")
# sphinx_gallery_end_ignore

# %%
#
# What the layout actually produced
# ---------------------------------
#
# :meth:`~torchsim.model.Simulator.describe` returns the event stream
# the layout lays down -- the same object a sequence arriving from a scanner
# would be read into, with a timestamp and an action word on every event. It is
# worth looking at once: a sequence that plays the wrong thing is far easier to
# see here than in the signal it produces.
#
description = sequence.describe(TS=SATURATION_TIMES, flip=FLIP_DEG)

# sphinx_gallery_start_ignore
print(f"\n{len(description.events)} events, "
      f"{description.tr_duration_us * 1e-3:.0f} ms long")
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
from torchsim.sequence import EventType

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
for event in description.events:
    when = float(event.timestamp_us) * 1e-3
    if event.type is EventType.RF:
        flip_deg = float(event.rf_amplitude_hz) * 180.0 / torch.pi
        axis.vlines(when, 0.0, flip_deg, color="crimson", lw=2)
    elif event.type is EventType.ADC:
        axis.plot(when, 0.0, "v", color="tab:blue", ms=7)
axis.plot([], [], color="crimson", lw=2, label="RF, height is the flip angle")
axis.plot([], [], "v", color="tab:blue", ms=7, label="ADC")
axis.set(
    xlabel="Time [ms]",
    ylabel="Flip angle [deg]",
    title="one saturate-wait-read block per saturation time",
    ylim=(-8, 100),
)
axis.grid(alpha=0.3)
key(axis, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# It differentiates, because everything does. Nothing was written to make that
# happen: the layout produced ordinary events and the engine took the
# derivative of the state machine that played them.
#
STEP_MS = 1.0
signal, dT1 = sequence.jacobian("T1", T1=830.0, T2=80.0, M0=1.0)
moved = sequence.simulate(T1=830.0 + STEP_MS, T2=80.0, M0=1.0)
difference = (moved - signal) / STEP_MS

# sphinx_gallery_start_ignore
print(f"agrees with a finite difference to "
      f"{float((dT1 - difference).abs().max()):.1e}")
# sphinx_gallery_end_ignore

# %%
#
# Three tissues over a denser set of saturation times, with the curves the
# closed form draws underneath:
#
times = torch.logspace(1.3, 3.7, 60)
NAMES = ("white matter", "grey matter", "CSF")
T1_MS = torch.tensor([830.0, 1330.0, 4000.0])

dense = SaturationRecovery(TS=times, flip=FLIP_DEG)
curves = dense.simulate(T1=T1_MS, T2=torch.tensor([80.0, 110.0, 2000.0]), M0=1.0)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.76))
for row, name in enumerate(NAMES):
    line, = axis.plot(times.numpy(), curves[row].abs().numpy(), label=name)
    analytic = torch.sin(torch.deg2rad(torch.tensor(FLIP_DEG))) * (
        1 - torch.exp(-times / T1_MS[row])
    )
    axis.plot(times.numpy(), analytic.numpy(), "--k", lw=0.8)
axis.set(
    xlabel="Saturation time [ms]",
    ylabel="|signal|",
    xscale="log",
    title="simulated, against the closed form (dashed)",
)
axis.grid(alpha=0.3)
key(axis, ncols=3)
# sphinx_gallery_end_ignore

# %%
#
# What is left to write
# ---------------------
#
# Neither half of this example wrote a kernel, an event, or a derivative.
# Extending a model was a declaration; assembling a sequence was a list.
#
# What neither can do is play something the operator vocabulary has no word
# for -- a preparation with its own internal timing, a readout that ends in a
# gradient nobody has spelled. That is one level further down, and it is still
# a Python function.
#
