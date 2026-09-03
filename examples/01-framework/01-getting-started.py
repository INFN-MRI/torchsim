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
        return axes.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=ncols,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.4,
        )
    axes = [axes] if hasattr(axes, "get_legend_handles_labels") else list(axes)
    figure = axes[0].figure
    legends = [
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=ncols,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.6,
            columnspacing=1.4,
        )
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
import math
import time
from functools import partial

import numpy as np
import torch

import torchsim
from torchsim.model import Simulator
from torchsim.sequence import EventType
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
signal = acquisition.simulate(flip=flip)  # (3, 48): one row per tissue

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
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
for axis, values, title in (
    (axes[0], signal, "a 60 degree train"),
    (axes[1], refocused, "a 180 degree train"),
):
    for row, name in enumerate(NAMES):
        axis.plot(values[row].abs().numpy(), label=name)
    axis.set(xlabel="Echo", ylabel="|signal|", title=title)
    axis.grid(alpha=0.3)
key(figure, ncols=3)
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
signal, dT2 = acquisition.jacobian("T2", flip=flip)  # dT2 is (3, 48) too

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

# sphinx_gallery_start_ignore
print(f"largest disagreement with a {STEP_MS} ms step: {float(discrepancy):.2e}")
# sphinx_gallery_end_ignore

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

# sphinx_gallery_start_ignore
for row, name in enumerate(NAMES):
    sigma = 100.0 * float(bound[row, 0].sqrt()) / float(T2_MS[row])
    print(f"{name:<14} sigma(T2)/T2 >= {sigma:5.2f}%")
# sphinx_gallery_end_ignore

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
# The reference it is measured against is the same gradient by finite
# differences -- one extra simulation per flip angle -- over a range of train
# lengths. One reverse pass answers for the whole schedule whatever its
# length; the difference needs one simulation per angle, and each of those
# simulations is itself longer.
#

# sphinx_gallery_start_ignore
LENGTHS = (24, 48, 96, 192)


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
    (cost, analytic), seconds = timed(partial(analytic_gradient, shots, angles))
    reverse_seconds.append(seconds)
    (_, difference), seconds = timed(partial(finite_gradient, shots, angles))
    difference_seconds.append(seconds)
    print(
        f"{echoes:>4} echoes   reverse {reverse_seconds[-1]:6.3f} s   "
        f"finite {difference_seconds[-1]:6.3f} s   "
        f"({difference_seconds[-1] / reverse_seconds[-1]:5.1f}x)"
    )
# sphinx_gallery_end_ignore

# %%
#
# The two gradients agree in shape and in sign, which is what a design loop
# follows. They do not agree exactly, and should not: a one-degree step is a
# large one on a curve this bent, and the discrepancy is the finite
# difference's rather than the derivative's.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.5))
axes[0].plot(dT2[0].abs().numpy(), "-k", label="forward mode")
axes[0].plot(finite[0].abs().numpy(), "*r", ms=4, label="finite difference")
axes[0].set(
    xlabel="Echo",
    ylabel=r"$|\partial\,\mathrm{signal}/\partial T_2|$",
    title="tissue derivative",
)

axes[1].plot(analytic.numpy(), "-k", label="reverse mode")
axes[1].plot(difference.numpy(), "*r", ms=4, label="finite difference")
axes[1].set(
    xlabel="Echo",
    ylabel=r"$\partial\,\mathrm{CRLB}/\partial\alpha$",
    title="schedule gradient",
)

axes[2].plot(LENGTHS, reverse_seconds, "-ok", label="reverse mode")
axes[2].plot(LENGTHS, difference_seconds, "-*r", label="finite difference")
axes[2].set(
    xlabel="Echoes",
    ylabel="Time [s]",
    title="what each costs",
    xscale="log",
    yscale="log",
    xticks=list(LENGTHS),
    xticklabels=[str(length) for length in LENGTHS],
)
axes[2].minorticks_off()
for axis in axes:
    axis.grid(alpha=0.3)
key(axes)
# sphinx_gallery_end_ignore

# %%
#
# The same thing in one call
# --------------------------
#
# Every sequence that ships also has a function, for the case where there is
# nothing to reuse: it takes the protocol and the tissue together, returns the
# signal, and returns the derivative too if ``diff`` names a property. It is
# the object above with the construction folded in, so the answer is the same
# to the bit.
#
signal, dT2 = torchsim.fse_sim(
    flip=flip, ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, diff="T2"
)

# sphinx_gallery_start_ignore
reference, reference_dT2 = acquisition.jacobian("T2", flip=flip)
print(
    f"agrees with the simulator to "
    f"{float((torch.as_tensor(dT2) - reference_dT2).abs().max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# Reach for it when a sequence is simulated once and nothing about it is being
# varied. The object is what a loop wants: it resolves the event stream on its
# first call and rebinds only the numbers that change afterwards, which is
# worth about eight times the whole call to a design or a dictionary sweep.
#

# %%
#
# Saying how the run is made
# --------------------------
#
# Everything so far took the defaults. Four settings decide what the run costs
# and how exact it is, and each is given to the constructor or to the call.
#
# ``states`` is how many configuration orders are carried. A refocused train
# winds one order per interval, and a pulse that is not a perfect 180 degrees
# splits the magnetization down every pathway those orders describe, so the
# answer is only as good as the number kept. Too few is a wrong answer rather
# than a slow one. Held against a train carrying far more than it needs:
#
signal_ref = acquisition.simulate(flip=flip)
converged = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, states=64).simulate(
    flip=flip
)
for orders in (4, 10, 16, 32, 48):
    truncated = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, states=orders)
    drift = float((truncated.simulate(flip=flip) - converged).abs().max())
    print(
        f"  {orders:2d} orders   {drift / float(converged.abs().max()):.1e} from converged"
    )

# %%
#
# It lands exactly at forty-eight, which is the number of echoes: a train that
# winds one order per interval can populate one more pathway per echo and no
# more, so carrying more orders than the train has intervals changes nothing.
# A spoiled sequence is the other case -- it discards the transverse orders
# every repetition, so a handful is enough however long it runs.
#
# The shipped default is chosen for the refocused trains these simulators are
# written for, and a 60 degree train is not one of them. It is the first
# setting to raise when a signal looks wrong late in an echo train, and the
# check above -- run once against a larger number -- is how you find out rather
# than assume.

# %%
#
# A simulator is worth holding on to. The structure of a sequence -- the order
# of its events, which of them record, how far it winds -- is settled the first
# time it runs and the numbers are rebound onto it afterwards, so a sweep that
# changes only flip angles never walks the event stream again. That happens by
# itself; what it is worth grows with the sequence, and a 500-repetition
# fingerprinting schedule is where it decides whether a dictionary sweep takes
# minutes or hours.
#
held = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)
rebuilt = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)

for name, simulator in (("held", held), ("rebuilt anew", rebuilt)):
    simulator.simulate(flip=flip)  # the first call is where the structure is read
    start = time.perf_counter()
    for _ in range(20):
        if name != "held":
            simulator = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)
        simulator.simulate(flip=flip)
    print(f"  {name:14s} {1e3 * (time.perf_counter() - start) / 20:6.2f} ms a call")

# %%
#
# ``execution`` says where the work goes. ``"auto"`` weighs the problem against
# what the cards have free -- work too small to repay a launch stays on the
# host, work that fits crosses in one piece, work that does not is streamed
# through in chunks. Naming a device insists on it, and a block settles it for
# everything inside:
#
with torchsim.execution("cpu"):
    on_the_host = acquisition.simulate(flip=flip)

print(
    f"  forced onto the host, agrees to {float((on_the_host - signal_ref).abs().max()):.1e}"
)

# %%
#
# ``stream`` and ``budget_bytes`` are for a volume rather than a dictionary:
# the first insists the run be cut into chunks even where it would have fit,
# the second caps what a chunk may hold. A whole-brain map at a hundred
# thousand voxels is the case they exist for, and it is written like this:
#
if torch.cuda.is_available():
    with torchsim.execution("cuda", stream=True, budget_bytes=1 << 28):
        streamed = acquisition.simulate(flip=flip)
    agreement = float((streamed.cpu() - signal_ref).abs().max())
    print(f"  streamed through a card in 256 MiB chunks, agrees to {agreement:.1e}")
else:
    print("  no card here, so the streamed run is skipped")

# The state a scanner is really in
# --------------------------------
#
# Everything so far started from equilibrium. A scanner does not: it plays the
# train over and over, and what it records is the state the train has settled
# into. Reaching that by playing it out is hundreds of repetitions, every one
# of them the full cost of the sequence.
#
# ``repetitions="auto"`` reads the limit off a handful of playings instead. A
# settled signal is a constant plus decaying modes, so finitely many terms fix
# where it is going, and the answer is the one that running there arrives at.
#
settling = FSESimulator(ESP=ESP_MS, TR=500.0, T1=T1_MS, T2=T2_MS, states=20)
first_pass = settling.simulate(flip=flip)
settled = settling.simulate(flip=flip, repetitions="auto")
played_out = settling.simulate(flip=flip, repetitions=200)

print(f"  one playing          {float(first_pass[0].abs().max()):.5f}")
print(f'  repetitions="auto"   {float(settled[0].abs().max()):.5f}')
print(f"  200 playings         {float(played_out[0].abs().max()):.5f}")
print(
    "  the last two agree to "
    f"{float((settled - played_out).abs().max() / played_out.abs().max()):.1e}"
)

# %%
#
# The first playing is wrong by a fraction that a short TR makes large, which
# is what a simulation of a steady-state sequence gets wrong if it starts from
# equilibrium and stops. Every shipped simulator takes the setting, and it
# costs a fraction of what running there costs.
#


# %%
#
# A sequence someone else assembled
# ---------------------------------
#
# Nothing above required TorchSim to have written the sequence. What a
# simulator lays down, what a Pulseq design exports and what the scanner
# streams back are one object: a **sequence description**, and it is small
# enough to read.
#
# It is a repetition's worth of events, the pulses those events drive, and the
# transmit shims they are driven on. An event is one of three things -- time
# passing, an RF pulse, an ADC window -- a timestamp in microseconds, and the
# handful of numbers that kind of event carries.
#
described = acquisition.describe(flip=flip, ESP=ESP_MS, TR=3000.0)

# sphinx_gallery_start_ignore
print(
    f"  {len(described.events)} events over "
    f"{described.tr_duration_us * 1e-3:.0f} ms, "
    f"{len(described.rf_definitions)} pulse shape, "
    f"{len(described.shim_definitions)} shims"
)
for event in described.events[:4]:
    print(
        f"    {event.type.name:<4s} at {event.timestamp_us / 1000:7.2f} ms   "
        + (
            f"{event.rf_use.name.lower()}, "
            f"{np.degrees(float(event.rf_amplitude_hz)):.0f} deg"
            if event.type is EventType.RF
            else f"{event.adc_role.name.lower()}"
            if event.type is EventType.ADC
            else f"{event.action.name.lower()}"
        )
    )
# sphinx_gallery_end_ignore

# %%
#
# On the wire those numbers are positional -- the scanner sends a flat row per
# event -- and in Python they are named. ``event.rf_use`` is the Pulseq tag the
# designer wrote, ``event.rf_amplitude_hz`` the flip in radians, and
# ``event.adc_role`` says whether a window is the centre of an echo. Nothing
# has to be inferred from timing.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
axis.axis("off")
axis.set(xlim=(0, 1), ylim=(0, 1))


def _box(x, y, text, colour, size=13.0):
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colour, edgecolor="none"),
    )


def _arrow(start_at, end_at, label=None, side="right"):
    axis.annotate(
        "",
        xy=end_at,
        xytext=start_at,
        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.8, shrinkA=2, shrinkB=2),
    )
    if label:
        middle = ((start_at[0] + end_at[0]) / 2, (start_at[1] + end_at[1]) / 2)
        axis.text(
            middle[0] + (0.015 if side == "right" else -0.015),
            middle[1],
            label,
            ha="left" if side == "right" else "right",
            va="center",
            fontsize=11.5,
            color="0.35",
            style="italic",
        )


_box(0.22, 0.92, "a Pulseq design\n.seq, or a script", "tab:blue")
_box(
    0.78,
    0.92,
    "the scanner's SEQDESC stream\nwaveforms 999 / 1000 / 1002 / 1005",
    "tab:orange",
    11.5,
)
_box(0.50, 0.46, "SequenceDescription", "0.25", 15.0)
axis.text(
    0.50,
    0.32,
    "events  ·  rf_definitions  ·  shim_definitions",
    ha="center",
    fontsize=11.5,
    color="0.35",
)
_box(0.50, 0.12, "Simulator.from_description", "tab:green", 14.0)
_arrow((0.22, 0.80), (0.44, 0.56), "sequence_descriptor()", side="left")
_arrow((0.78, 0.80), (0.56, 0.56), "decode_sequence_description()", side="right")
_arrow((0.50, 0.27), (0.50, 0.19))

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
FIRST = 6
shown = [e for e in described.events if e.timestamp_us <= FIRST * ESP_MS * 1000.0]
for event in shown:
    when = float(event.timestamp_us) / 1000.0
    if event.type is EventType.RF:
        axis.vlines(
            when, 0.0, np.degrees(float(event.rf_amplitude_hz)), color="crimson", lw=2.5
        )
    elif event.type is EventType.ADC:
        axis.plot(when, 0.0, "v", color="tab:blue", ms=9)
axis.plot([], [], color="crimson", lw=2.5, label="RF, height is the flip")
axis.plot([], [], "v", color="tab:blue", ms=9, label="ADC")
axis.set(
    xlabel="time [ms]",
    ylabel="flip angle [deg]",
    title=f"the first {FIRST} echoes of the description above",
    ylim=(-8, 105),
)
axis.grid(alpha=0.3)
key(axis, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# In practice a description is never typed out -- the scanner computes it from
# the sequence file, and a Pulseq design exports it. But it is worth writing
# one by hand once, because that is what shows there is nothing else in it:
# :func:`~torchsim.description` lays operators out and fills in the rest.
#
by_hand = torchsim.description(
    torchsim.Excitation(math.pi / 2, math.pi / 2),
    *[
        part
        for _ in range(ECHOES)
        for part in (
            torchsim.Delay(0.5 * ESP_MS * 1e-3),
            torchsim.Refocusing(math.radians(60.0), 0.0),
            torchsim.Delay(0.5 * ESP_MS * 1e-3),
            torchsim.Readout(0.0),
        )
    ],
)

print(
    f"  written by hand: {len(by_hand.events)} events over "
    f"{by_hand.tr_duration_us * 1e-3:.0f} ms"
)

# %%
#
# :meth:`~torchsim.model.Simulator.from_description` runs one. The events are
# already concrete -- each carries the action word that says whether it winds,
# spoils or records -- so no layout is walked and nothing is inferred. What the
# model supplies is only the vocabulary the tissue is written in.
#
from_stream = Simulator.from_description(
    described, acquisition.model, states=10, T1=T1_MS, T2=T2_MS
)
streamed_signal = from_stream.simulate()

# %%
#
# It differentiates like anything else, because the derivative follows from the
# events and not from who wrote them:
#
_, streamed_dT2 = from_stream.jacobian("T2")

# sphinx_gallery_start_ignore
print(f"  signal {tuple(streamed_signal.shape)}, dT2 {tuple(streamed_dT2.shape)}")
# sphinx_gallery_end_ignore

# %%
#
# One thing to know when comparing it against the shipped simulator: a
# description carries the events, and a simulator may carry physics *around*
# them. :class:`~torchsim.simulators.FSESimulator` folds in the recovery
# between one train and the next in closed form, which is not an event and so
# is not in the stream. The shape of the train is the same; the driven
# equilibrium the shipped object adds does not come along.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(1, 1, figsize=(PAGE_WIDTH, 3.2))
axis.plot(signal_ref[0].abs().numpy(), "-k", lw=2, label="FSESimulator")
axis.plot(
    streamed_signal[0].abs().numpy(), "--r", lw=2, label="from the description alone"
)
axis.set(
    xlabel="Echo", ylabel="|signal|", title="white matter, the same train both ways"
)
axis.grid(alpha=0.3)
key(axis, ncols=2)
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
