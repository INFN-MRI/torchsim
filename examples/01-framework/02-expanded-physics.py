"""
===========================
The physics a model carries
===========================

A signal model declares which tissue properties it exposes, and that
declaration is the whole of what decides which terms the kernels evaluate. A
model of T1 and T2 pays for no exchange pool, no diffusion attenuation and no
flow winding; naming a property is what brings its term back, and naming one
costs nothing until a voxel is given a value at which it does something.

So the physics available here is a vocabulary rather than a set of modes. This
example walks the whole of it -- the transmit field and the array that produces
it, off resonance, an imperfect inversion, a second pool that exchanges, a pool
too broad to image, both at once, diffusion and flow -- and then the steady
state a repeated train settles into, which is a property of the sequence rather
than of the tissue.

Each is checked against something that was already true of it.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# A model is a frozen dataclass, so asking for more physics is
# :func:`dataclasses.replace` over its property map.
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
from dataclasses import replace

import numpy as np
import torch

from torchsim import EpgEngine, ShimDefinition, TissueProperties, fse_description
from torchsim.model import BALANCED
from torchsim.simulators import MRFSimulator

# %%
# What a model declares
# ---------------------
#
# The physics of a shipped simulator is a mapping from the names a protocol is
# written in to the fields of a voxel. The left column is what a caller passes;
# the right column is what the kernels carry.
#
MODEL = MRFSimulator.model

# sphinx_gallery_start_ignore
print("  what a caller writes -> what a voxel holds")
for name, field in MODEL.properties.items():
    print(f"  {name:16s} -> {field}")
print()
print("  fields no shipped fingerprinting model claims:")
for field in TissueProperties.__dataclass_fields__:
    if field not in set(MODEL.properties.values()):
        print(f"    {field}")
# sphinx_gallery_end_ignore

# %%
#
# The unclaimed fields are the subject of this page. Every one of them is a
# term the kernels already contain and skip, so reaching it is naming it:
#


def carrying(**extra):
    """The fingerprinting model, told about more of what a voxel holds."""
    return replace(MODEL, properties=dict(MODEL.properties, **extra))


# %%
#
# One schedule runs throughout -- four hundred repetitions of a ramped flip
# angle after an inversion, which is a fingerprinting train and drives every
# pathway hard enough to see what a term does to it.
#
FLIP_DEG = np.concatenate((np.linspace(5.0, 55.0, 200), np.linspace(55.0, 5.0, 200)))
SCHEDULE = dict(flip=FLIP_DEG, TR=10.0, TI=20.0, states=20)
T1_MS, T2_MS = 1000.0, 80.0

baseline = MRFSimulator(**SCHEDULE).simulate(T1=T1_MS, T2=T2_MS)

# %%
# The transmit field a voxel sits in
# ----------------------------------
#
# ``B1`` scales the flip angle a voxel actually turns, and a fingerprinting
# train is sensitive to it in a way a single contrast is not: the flip changes
# every repetition, so a transmit error is a distortion of the trajectory
# rather than a scaling of it. That is what makes it estimable alongside T1 and
# T2, and what makes ignoring it a bias in both.
#
transmit = torch.tensor([0.7, 0.85, 1.0, 1.15])
scaled = MRFSimulator(**SCHEDULE).simulate(T1=T1_MS, T2=T2_MS, B1=transmit)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6))
for row, value in enumerate(transmit):
    axis.plot(abs(scaled[row]), label=f"B1 = {float(value):.2f}")
axis.set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="a transmit error reshapes the trajectory",
)
axis.grid(alpha=0.3)
key(axis, ncols=4)
# sphinx_gallery_end_ignore

# %%
# The array that produces it
# --------------------------
#
# On a parallel transmit system the field is what several channels put there
# together, so a voxel's transmit is a complex sum rather than a number.
# ``b1`` and ``b1_phase_rad`` then carry one row per channel, and a
# :class:`~torchsim.ShimDefinition` says how hard each channel is driven and at
# what phase -- the shim.
#
# The check that matters is cancellation. Four channels whose sensitivities sit
# a quarter turn apart, all driven alike, put nothing on the voxel at all;
# a shim that counter-rotates them puts back the whole of it. Summing
# magnitudes and phases separately cannot do that, which is why the array is
# resolved into the field before the state machine sees it.
#
CHANNELS, VOXELS = 4, 3
sensitivity = torch.full((CHANNELS, VOXELS), 1.0 / CHANNELS)
sensitivity_phase = (
    (torch.arange(CHANNELS)[:, None] * 2.0 * math.pi / CHANNELS)
    .expand(CHANNELS, VOXELS)
    .contiguous()
    .float()
)

echo_train = fse_description(
    torch.deg2rad(torch.full((8,), 150.0)),
    echo_spacing_s=5e-3,
    phases_rad=0.5 * math.pi,
    excitation_phase_rad=0.5 * math.pi,
)
array_tissue = TissueProperties(
    t1_ms=torch.linspace(600.0, 1400.0, VOXELS),
    t2_ms=torch.linspace(40.0, 120.0, VOXELS),
    b1=sensitivity,
    b1_phase_rad=sensitivity_phase,
)

shims = {
    "driven alike": ShimDefinition(0, (1.0,) * CHANNELS, (0.0,) * CHANNELS),
    "counter-rotated": ShimDefinition(
        0,
        (1.0,) * CHANNELS,
        tuple(-c * 2.0 * math.pi / CHANNELS for c in range(CHANNELS)),
    ),
}
shimmed = {
    label: EpgEngine()
    .simulate(
        replace(echo_train, shim_definitions={shim.id: shim}), array_tissue, nstates=12
    )
    .signal
    for label, shim in shims.items()
}

# sphinx_gallery_start_ignore
for label, signal in shimmed.items():
    print(
        f"  {label:16s} first echo {[round(float(v), 4) for v in signal[:, 0].abs()]}"
    )
# sphinx_gallery_end_ignore

# %%
#
# A shim is a property of the *pulse*, not of the sequence: an event names the
# shim it is driven on, so an excitation and a refocusing pulse can sit on
# different ones. What every backend and every derivative sees is still the two
# per-voxel buffers, so nothing downstream knows that channels exist.
#
# The shaped pulse a channel plays -- the envelope itself, and what it does
# across a slice -- is in :ref:`the getting-started example
# <sphx_glr_generated_autoexamples_01-framework_01-getting-started.py>`.
#
# Off resonance
# -------------
#
# ``B0`` turns the transverse states between one event and the next. Whether
# that reaches the signal is a question about the sequence rather than about
# the field: a train that dephases by a whole configuration order every
# repetition separates the orders completely and comes back insensitive to it,
# while a balanced train keeps them together and bands.
#
# So the demonstration is balanced -- the readout is the only thing that
# changes -- and the banding period is the one a balanced train has, ``1 / TR``.
#
TR_MS = 10.0
offsets_hz = torch.linspace(-150.0, 150.0, 121)
balanced = MRFSimulator(
    model=replace(carrying(B0="b0_hz"), operators=BALANCED),
    flip=np.full(64, 20.0),
    TR=TR_MS,
    TI=0.0,
    states=20,
)
banded = balanced.simulate(T1=T1_MS, T2=T2_MS, B0=offsets_hz, repetitions="auto")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
axis.plot(offsets_hz, abs(banded[:, -1]))
for band in (-100.0, 0.0, 100.0):
    axis.axvline(band, color="0.6", linestyle=":", linewidth=1.2)
axis.set(
    xlabel="off resonance [Hz]",
    ylabel="signal magnitude [a.u.]",
    title=f"balanced, TR = {TR_MS:.0f} ms: bands every {1e3 / TR_MS:.0f} Hz",
)
axis.grid(alpha=0.3)
print(f"  nulls sit {1e3 / TR_MS:.0f} Hz apart, which is 1 / TR")
# sphinx_gallery_end_ignore

# %%
# An inversion that does not invert
# ---------------------------------
#
# ``inv_efficiency`` is how much of the magnetization an inversion pulse
# actually turns over. It is a property of the pulse and the transmit field
# rather than of the tissue, and it enters where the inversion does -- at the
# front of the train, on the repetitions whose contrast the inversion is there
# to create.
#
efficiencies = torch.tensor([1.0, 0.9, 0.8])
inverted = MRFSimulator(**SCHEDULE).simulate(
    T1=T1_MS, T2=T2_MS, inv_efficiency=efficiencies
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
for row, value in enumerate(efficiencies):
    axis.plot(abs(inverted[row][:80]), label=f"efficiency {float(value):.1f}")
axis.set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="an imperfect inversion is a transient, not a scaling",
)
axis.grid(alpha=0.3)
key(axis, ncols=3)
print(
    "  first repetition falls to "
    f"{float(inverted[2][0].abs() / inverted[0][0].abs()):.4f} of the ideal;"
)
print(
    "  by the four-hundredth the three agree to "
    f"{float((inverted[2][-1] - inverted[0][-1]).abs() / inverted[0][-1].abs()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
# A second pool the readouts can see
# ----------------------------------
#
# Tissue is not one pool of water. A second pool with its own T1, its own T2
# and its own resonance -- myelin water beside the intra- and extracellular
# water it exchanges with, fat beside water -- gives a signal that is not the
# sum of two signals, because magnetization crosses between them while the
# train plays.
#
# Five fields say what it is: how much of the magnetization it holds, how fast
# it exchanges, and its own T1, T2 and frequency shift. What the train records
# is one signal, and the exchange is what makes it irreducible to two.
#
exchanging = MRFSimulator(
    model=carrying(
        fB="pool_b_fraction",
        kB="pool_b_exchange_hz",
        T1B="t1_pool_b_ms",
        T2B="t2_pool_b_ms",
        dwB="pool_b_shift_hz",
    ),
    **SCHEDULE,
)
fractions = torch.tensor([0.0, 0.1, 0.2, 0.3])
two_pool = exchanging.simulate(
    T1=T1_MS,
    T2=T2_MS,
    fB=fractions,
    kB=20.0,
    T1B=500.0,
    T2B=20.0,
    dwB=100.0,
)

# %%
# A pool too broad to image
# -------------------------
#
# Protons bound to macromolecules have a T2 of tens of microseconds. They are
# gone before any readout, so there is nothing to record from them directly --
# but they exchange with the free water that is recorded, and saturating them
# is felt there. That is magnetization transfer, and it is three fields: how
# much of the magnetization is bound, how fast it exchanges, and its T1.
#
# It has no T2 among them, which is the point: the pool is modelled as
# longitudinal only, because on the timescale of a readout it has no
# transverse magnetization to carry.
#
bound = MRFSimulator(
    model=carrying(
        fM="bound_fraction",
        kM="bound_exchange_hz",
        T1M="t1_bound_ms",
    ),
    **SCHEDULE,
)
magnetization_transfer = bound.simulate(
    T1=T1_MS, T2=T2_MS, fM=fractions, kM=40.0, T1M=1000.0
)

# %%
# Both at once
# ------------
#
# The two are independent declarations, so a model can carry both: two free
# pools that exchange with each other, and a bound pool that exchanges with
# them. Nothing is written to combine them -- eight names in the property map
# is the whole of it.
#
combined = MRFSimulator(
    model=carrying(
        fB="pool_b_fraction",
        kB="pool_b_exchange_hz",
        T1B="t1_pool_b_ms",
        T2B="t2_pool_b_ms",
        dwB="pool_b_shift_hz",
        fM="bound_fraction",
        kM="bound_exchange_hz",
        T1M="t1_bound_ms",
    ),
    **SCHEDULE,
)
three_pool = combined.simulate(
    T1=T1_MS,
    T2=T2_MS,
    fB=0.2,
    kB=20.0,
    T1B=500.0,
    T2B=20.0,
    dwB=100.0,
    fM=0.2,
    kM=40.0,
    T1M=1000.0,
)

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6), sharey=True)
for row, value in enumerate(fractions):
    label = "one pool" if row == 0 else f"fraction {float(value):.1f}"
    axes[0].plot(abs(two_pool[row]), label=label)
    axes[1].plot(abs(magnetization_transfer[row]), label=label)
axes[1].plot(abs(three_pool.reshape(-1)), "k--", linewidth=1.4, label="both, 0.2 each")
axes[0].set(title="a second free pool", ylabel="signal magnitude [a.u.]")
axes[1].set(title="a bound pool")
for axis in axes:
    axis.set_xlabel("repetition")
    axis.grid(alpha=0.3)
key(axes, ncols=3)

print(
    "  a fifth of the magnetization in a second free pool moves the train by "
    f"{float((two_pool[2] - two_pool[0]).abs().max() / two_pool[0].abs().max()):.1%}"
)
print(
    "  a fifth bound moves it by "
    f"{float((magnetization_transfer[2] - magnetization_transfer[0]).abs().max() / magnetization_transfer[0].abs().max()):.1%}"
)
# sphinx_gallery_end_ignore

# %%
# Motion: diffusion and flow
# --------------------------
#
# Both are read off the same thing -- what a gradient has wound onto a
# configuration order -- so both need the sequence to say how much winding an
# order stands for. That is two arguments to the simulator rather than two
# tissue fields: ``crusher_dephasing_rad``, the turn one crusher puts across a
# voxel, and ``voxel_size_m``, the distance it puts it across. Without them an
# order is a bookkeeping index with no physical extent, and a voxel given a
# diffusivity or a velocity is attenuated by nothing at all.
#
MOMENT = dict(crusher_dephasing_rad=4.0 * math.pi, voxel_size_m=1e-3)
crushed = MRFSimulator(**SCHEDULE, **MOMENT).simulate(T1=T1_MS, T2=T2_MS)

diffusivities = torch.tensor([0.0, 1.0, 2.0, 3.0])
diffusing = MRFSimulator(
    model=carrying(D="diffusion_um2_per_ms"), **SCHEDULE, **MOMENT
).simulate(T1=T1_MS, T2=T2_MS, D=diffusivities)

velocities = torch.tensor([0.0, 0.01, 0.03, 0.05])
flowing = MRFSimulator(
    model=carrying(v="velocity_m_per_s"), **SCHEDULE, **MOMENT
).simulate(T1=T1_MS, T2=T2_MS, v=velocities)

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6))
for row in range(len(diffusivities)):
    axes[0].plot(abs(diffusing[row]), label=f"D = {float(diffusivities[row]):.0f}")
    axes[1].plot(
        abs(flowing[row]), label=f"v = {100 * float(velocities[row]):.0f} cm/s"
    )
axes[0].set(title=r"diffusion [$\mu$m$^2$/ms]", ylabel="signal magnitude [a.u.]")
axes[1].set(title="flow through the voxel")
for axis in axes:
    axis.set_xlabel("repetition")
    axis.grid(alpha=0.3)
key(axes, ncols=4)

print(
    "  diffusion attenuates the higher orders, so it costs most where the "
    "train is most coherent:"
)
print(
    f"    D = 3 departs from D = 0 by "
    f"{float((diffusing[3] - diffusing[0]).abs().max() / diffusing[0].abs().max()):.1%}"
)
print(
    "  flow both carries winding out of the voxel and brings unsaturated "
    "magnetization in, so it reshapes the train rather than scaling it:"
)
print(
    f"    5 cm/s departs by "
    f"{float((flowing[3] - flowing[0]).abs().max() / flowing[0].abs().max()):.1f}"
    f"x the unflowed signal"
)
# sphinx_gallery_end_ignore

# %%
# The state a scanner plays it in
# -------------------------------
#
# Everything above starts from equilibrium. A scanner does not: it plays the
# train over and over, and what it records is the state the train has settled
# into. Reaching that by running the train out is hundreds of playings, and
# every one of them is the full cost of the sequence.
#
# ``repetitions="auto"`` reads the limit off a handful of playings instead. A
# settled signal is a constant plus decaying modes, so finitely many terms fix
# where it is going, and the answer is the one running there arrives at.
#
settling = MRFSimulator(flip=np.full(48, 20.0), TR=10.0, TI=0.0, states=20)

start = time.perf_counter()
settled = settling.simulate(T1=T1_MS, T2=T2_MS, repetitions="auto")
auto_seconds = time.perf_counter() - start

start = time.perf_counter()
ran_out = settling.simulate(T1=T1_MS, T2=T2_MS, repetitions=400)
run_seconds = time.perf_counter() - start

first_pass = settling.simulate(T1=T1_MS, T2=T2_MS)

# sphinx_gallery_start_ignore
approaching = [
    abs(settling.simulate(T1=T1_MS, T2=T2_MS, repetitions=n).reshape(-1)[-1])
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)
]

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
axis.semilogx((1, 2, 4, 8, 16, 32, 64, 128, 256), approaching, "o-", label="played out")
axis.axhline(
    float(abs(settled.reshape(-1)[-1])),
    color="k",
    linestyle="--",
    label='repetitions="auto"',
)
axis.set(
    xlabel="repetitions played",
    ylabel="last echo [a.u.]",
    title="where the train is going, and what it costs to get there",
)
axis.grid(alpha=0.3)
key(axis, ncols=2)

print(
    f'  repetitions="auto"  {float(abs(settled.reshape(-1)[-1])):.6f}'
    f"   in {auto_seconds:.2f} s"
)
print(
    f"  400 playings        {float(abs(ran_out.reshape(-1)[-1])):.6f}"
    f"   in {run_seconds:.2f} s"
)
print(f"  one playing         {float(abs(first_pass.reshape(-1)[-1])):.6f}")
print(
    "  the two agree to "
    f"{float(np.abs(settled - ran_out).max() / np.abs(ran_out).max()):.1e}"
    f", at {run_seconds / auto_seconds:.0f}x less work"
)
# sphinx_gallery_end_ignore

# %%
#
# The first playing is wrong by a factor of several, which is what a
# simulation of a steady-state sequence gets wrong if it starts from
# equilibrium and stops.
#
# What declaring costs
# --------------------
#
# Nothing, until a voxel is given a value at which the term does something. A
# property held at the value where it has no effect -- unit transmit, no
# off resonance, an empty pool -- is reported absent, and its term stays out of
# the kernel that is compiled and run. So a model can name every field on this
# page and a call that leaves them alone pays what the two-parameter model
# pays.
#
everything = MRFSimulator(
    model=carrying(
        B0="b0_hz",
        fB="pool_b_fraction",
        kB="pool_b_exchange_hz",
        T1B="t1_pool_b_ms",
        T2B="t2_pool_b_ms",
        dwB="pool_b_shift_hz",
        fM="bound_fraction",
        kM="bound_exchange_hz",
        T1M="t1_bound_ms",
        D="diffusion_um2_per_ms",
        v="velocity_m_per_s",
    ),
    **SCHEDULE,
)
declared_only = everything.simulate(T1=T1_MS, T2=T2_MS)

# sphinx_gallery_start_ignore
print(
    "  a model naming eleven more fields, none of them given a value, agrees "
    "with the two-parameter one to "
    f"{float(np.abs(declared_only - baseline).max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# What is left is the vocabulary itself. Every term on this page is one the
# kernels carry; a term they do not carry -- a third free pool, a per-event
# gradient moment, an exchange rate that varies down the train -- is a change
# to the engine rather than to a declaration on top of it.
