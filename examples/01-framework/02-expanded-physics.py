"""
=========================================
More than T1 and T2: the physics on offer
=========================================

Two relaxation times describe a single pool of water in a uniform field, and
almost nothing in a scanner is that. The transmit field varies across the head
and is produced by an array; water sits in more than one compartment and moves
between them; some protons are bound so tightly that no readout ever sees them;
spins diffuse and flow; the field is off resonance.

Every one of those is a term the kernels already carry and skip. Naming the
property in a call is what turns it on, and what a voxel is not given costs
nothing -- so this page is a tour of the vocabulary, one effect at a time, each
shown on the sequence where it is easiest to see and checked against something
that was already true of it.

The pulse a scanner actually plays comes into it too: a shaped, slice-selective
excitation does not turn the same angle everywhere in the slice, and that is
neither a scaling nor a nuisance.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# Nothing here builds a model. A simulator that ships is handed more of the
# tissue than it declares, and answers.
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
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import torchsim
from torchsim import EpgEngine, ShimDefinition, TissueProperties, fse_description
from torchsim.model import BALANCED, SPOILED
from torchsim.simulators import FSESimulator, MRFSimulator

# %%
# One train, played throughout
# ----------------------------
#
# Four hundred repetitions after an inversion, at a fixed repetition time and a
# flip angle that rises and falls smoothly. A fingerprinting train drives every
# coherence pathway hard, which is what makes it a good place to watch a term
# being switched on. Where an effect needs a different readout to show at all,
# only the readout changes and it is said so.
#
FLIP_DEG = np.concatenate((np.linspace(5.0, 55.0, 200), np.linspace(55.0, 5.0, 200)))
TRAIN = dict(flip=FLIP_DEG, TR=10.0, TI=20.0, states=20)
WATER = dict(T1=1000.0, T2=80.0)

fingerprinting = MRFSimulator(**TRAIN)
baseline = fingerprinting.simulate(**WATER)

# %%
#
# ``MRFSimulator`` names T1, T2, M0, a transmit scaling and an inversion
# efficiency. Every other field a voxel has can be given to it anyway, and
# giving one is what asks for its physics:
#

# sphinx_gallery_start_ignore
print(f"  declared: {', '.join(fingerprinting.exposes)}")
print(f"  accepted: {', '.join(fingerprinting.accepts)}")
print(f"  the sequence is written in: {', '.join(fingerprinting.variables)}")
# sphinx_gallery_end_ignore

# %%
# The flip angle a voxel really turns
# -----------------------------------
#
# ``B1`` scales it. A fingerprinting train is sensitive to that in a way a
# single contrast is not: the nominal angle changes every repetition, so a
# transmit error distorts the trajectory rather than scaling it. That is what
# makes B1 estimable alongside the relaxation times, and what makes ignoring it
# a bias in both.
#
transmit = torch.tensor([0.7, 0.85, 1.0, 1.15])
scaled = fingerprinting.simulate(**WATER, B1=transmit)

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
# Eight coils making one field
# ----------------------------
#
# On a parallel-transmit system that field is what several channels put on the
# voxel together, so a voxel's transmit is a complex sum rather than a number.
# ``B1`` and ``B1phase`` then carry one row per channel, and a
# :class:`~torchsim.ShimDefinition` says how hard each channel is driven and at
# what phase.
#
# The check that matters is cancellation. Four channels whose sensitivities sit
# a quarter turn apart, all driven alike, put nothing on the voxel at all.
# Adding magnitudes and phases separately cannot produce that, which is why the
# array is resolved into the field before the state machine sees it.
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


def driven_apart(step_rad):
    """The array with each channel held one more step behind the last."""
    return ShimDefinition(
        0,
        (1.0,) * CHANNELS,
        tuple(float(-channel * step_rad) for channel in range(CHANNELS)),
    )


def first_echo(step_rad):
    """What that shim leaves on each voxel at the first echo."""
    return (
        EpgEngine()
        .simulate(
            replace(echo_train, shim_definitions={0: driven_apart(step_rad)}),
            array_tissue,
            nstates=12,
        )
        .signal[:, 0]
        .abs()
    )


steps_rad = torch.linspace(0.0, 2.0 * math.pi, 61)
swept = torch.stack([first_echo(float(step)) for step in steps_rad])

# sphinx_gallery_start_ignore
print(f"  driven alike     {[round(float(v), 4) for v in first_echo(0.0)]}")
print(
    f"  counter-rotated  "
    f"{[round(float(v), 4) for v in first_echo(2.0 * math.pi / CHANNELS)]}"
)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6))
degrees = torch.rad2deg(steps_rad).numpy()
for voxel in range(VOXELS):
    axis.plot(degrees, swept[:, voxel].numpy(), label=f"voxel {voxel + 1}")
for marked, text in ((0.0, "driven\nalike"), (360.0 / CHANNELS, "counter-\nrotated")):
    axis.axvline(marked, color="0.55", linestyle=":", linewidth=1.4)
    axis.annotate(text, xy=(marked + 8.0, 0.05), fontsize=13, color="0.35")
axis.set(
    xlabel="phase step between channels [deg]",
    ylabel="first echo [a.u.]",
    title="four channels a quarter turn apart, and the shim that finds them",
    xlim=(0.0, 360.0),
    xticks=[0, 90, 180, 270, 360],
)
axis.grid(alpha=0.3)
key(axis, ncols=3)
# sphinx_gallery_end_ignore

# %%
#
# A shim belongs to the *pulse*, not to the sequence: an event names the shim
# it is driven on, so an excitation and a refocusing pulse can sit on different
# ones. What every backend and every derivative sees is still the two per-voxel
# buffers, so nothing downstream knows that channels exist.
#
# %%
# The pulse the scanner actually plays
# -----------------------------------
#
# Everything above turned instantly. A real excitation is a shaped waveform
# played under a slice-selection gradient, and what it does depends on where in
# the slice a spin sits: the angle it turns at the edge is not the angle it
# turns at the centre. That is a Bloch response, not a scaling, so it cannot be
# folded into the flip angle -- the pulse has to be integrated.
#
# TorchSim takes the envelope itself: the complex waveform, one row per
# transmit channel, which is what comes off a Pulseq block or an MRD sequence
# description. This one is an SLR 90 degree pulse, 2 ms long over a 5 mm slice,
# designed elsewhere and saved beside this file.
#

# sphinx_gallery_start_ignore
# The gallery runs an example from its own directory; a shell may not.
DATA = Path("data") if Path("data").is_dir() else Path(__file__).parent / "data"
# sphinx_gallery_end_ignore
waveform = np.load(DATA / "slr90.npz")
excitation = torchsim.rf_definition(
    waveform["samples"],
    dwell_s=float(waveform["dwell_s"]),
    bandwidth_hz=float(waveform["bandwidth_hz"]),
)

# %%
#
# ``pulse`` is what the events drive and ``across_slice`` how many positions to
# work it out at. Leave the second off and the pulse is evaluated at the slice
# centre alone, which is the hard-pulse answer -- so the two together separate
# what the shape does from where in the slice you stand.
#
REFOCUSED = dict(ESP=5.0, TR=3000.0, T1=830.0, T2=80.0, states=48)
angles = torch.full((48,), 150.0)

hard = FSESimulator(**REFOCUSED).simulate(flip=angles)
centre = FSESimulator(**REFOCUSED, pulse=excitation).simulate(flip=angles)
across = FSESimulator(**REFOCUSED, pulse=excitation, across_slice=21).simulate(
    flip=angles
)

# %%
#
# The flip a spin turns, position by position, is what the table holds. It is
# flat across the passband and falls away outside it, and that shape is the
# whole of what a slice profile is.
#

# sphinx_gallery_start_ignore
from torchsim.sequence._transition import transition_table  # noqa: E402

positions = torch.linspace(-1.0, 1.0, 121, dtype=torch.float64)
table = transition_table(excitation, positions, bins=64, rf_raster_time_s=1e-6)
_a, b = table.at(
    torch.arange(positions.numel()),
    torch.full((positions.numel(),), 0.5 * math.pi, dtype=torch.float64),
)
turned_deg = torch.rad2deg(2.0 * torch.arcsin(b.abs().clamp(max=1.0)))

figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.4))
time_ms = 1e3 * np.arange(waveform["samples"].size) * float(waveform["dwell_s"])
axes[0].plot(time_ms, np.abs(waveform["samples"]) / np.abs(waveform["samples"]).max())
axes[0].set(xlabel="time [ms]", ylabel="envelope [a.u.]", title="the pulse")
axes[1].plot(positions.numpy(), turned_deg.numpy())
axes[1].axhline(90.0, color="0.6", linestyle=":")
axes[1].set(
    xlabel="position [slice thicknesses]",
    ylabel="flip turned [deg]",
    title="what it turns, where",
)
for axis in axes:
    axis.grid(alpha=0.3)

print(f"  at the slice centre     {float(hard.abs().max()):.4f} (hard pulse)")
print(f"  the shaped pulse there  {float(centre.abs().max()):.4f}")
print(f"  averaged over the slice {float(across.abs().max()):.4f}")
print(
    f"  the profile costs       "
    f"{100 * (1 - float(across.abs().max()) / float(hard.abs().max())):.0f}% "
    f"of the signal"
)
# sphinx_gallery_end_ignore

# %%
#
# The shaped pulse at the centre is the hard-pulse answer, which is the check
# that the envelope was scaled right. Averaged across the slice it is much
# smaller, and for a refocused train that is not a scaling: the edges of the
# slice see a smaller refocusing angle, which is a different balance of
# coherence pathways rather than a weaker version of the same one. A fit that
# assumed the nominal angle would read that as a tissue difference.
#
# Water in two places at once
# ---------------------------
#
# Tissue is not one pool. Myelin water sits beside the intra- and
# extracellular water it exchanges with; protons bound to macromolecules have a
# T2 of tens of microseconds and are gone before any readout, yet they exchange
# with the water that is not. The first pool is recorded along with the free
# water and the second is not, and that is the difference between them.
#
# Five names say what an exchanging free pool is and three what a bound pool
# is. Both are shown here on a spoiled train driven to its steady state, where
# the two differ in *direction* rather than only in size, on the white matter
# of Malik et al. (Magn. Reson. Med. 2018).
#
REPETITIONS, TR_MS, SPOILED_FLIP, SPOILING_STEP = 200, 5.0, 10.0, 117.0
index = np.arange(REPETITIONS)
SPOILED_TRAIN = dict(
    flip=np.full(REPETITIONS, SPOILED_FLIP),
    phases=SPOILING_STEP * index * (index + 1) / 2.0,
    TR=TR_MS,
    TI=0.0,
    states=40,
)
WHITE_MATTER = dict(T1=779.0, T2=45.0)
FREE = dict(poolB_exchange=2.0, poolB_T1=500.0, poolB_T2=20.0)
BOUND = dict(bound_exchange=4.3, bound_T1=779.0)

spoiled = MRFSimulator(
    model=replace(MRFSimulator.model, operators=SPOILED), **SPOILED_TRAIN
)
one_pool = spoiled.simulate(**WHITE_MATTER)
with_free = spoiled.simulate(**WHITE_MATTER, poolB_fraction=0.2, **FREE)
with_bound = spoiled.simulate(**WHITE_MATTER, bound_fraction=0.117, **BOUND)

# %%
#
# A fraction is a tissue property, so sweeping it is one call over a voxel axis
# rather than a loop over sequences. At a fraction of nothing both have to land
# back on the single-pool answer, which is the check that they are the same
# sequence.
#
fractions = torch.linspace(0.0, 0.3, 31)
free_sweep = spoiled.simulate(**WHITE_MATTER, poolB_fraction=fractions, **FREE)
bound_sweep = spoiled.simulate(**WHITE_MATTER, bound_fraction=fractions, **BOUND)

# sphinx_gallery_start_ignore
SERIES = (
    ("one pool", "0.25", one_pool),
    ("a second free pool", "tab:orange", with_free),
    ("a bound pool", "tab:green", with_bound),
)
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6))
for label, colour, values in SERIES:
    axes[0].plot(abs(np.asarray(values).reshape(-1)), color=colour, label=label)
axes[0].set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="approach to the steady state",
    xlim=(0, REPETITIONS),
)
settled = float(abs(np.asarray(one_pool).reshape(-1)[-1]))
axes[1].axhline(settled, color="0.25", label="one pool")
axes[1].plot(fractions.numpy(), abs(free_sweep[:, -1]), color="tab:orange")
axes[1].plot(fractions.numpy(), abs(bound_sweep[:, -1]), color="tab:green")
axes[1].set(
    xlabel="second-pool fraction",
    ylabel="settled signal [a.u.]",
    title="what the fraction does",
)
for axis in axes:
    axis.grid(alpha=0.3)
key(figure, ncols=3)

print(f"  one pool settles at   {settled:.5f}")
print(
    f"  a second free pool    {float(abs(np.asarray(with_free).reshape(-1)[-1])):.5f}"
)
print(
    f"  a bound pool          {float(abs(np.asarray(with_bound).reshape(-1)[-1])):.5f}"
)
print(
    "  at a fraction of nothing the two rejoin it, to "
    f"{float(max(abs(free_sweep[0, -1] - settled), abs(bound_sweep[0, -1] - settled))):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# They move the signal in opposite directions, and that is the whole reason for
# keeping them apart. A second free pool is recorded along with the first, so
# filling it raises the signal; a bound pool is not, so filling it parks
# magnetization where no readout will find it and the signal falls.
#
# The declarations are independent, so a voxel can have both -- two free pools
# exchanging with each other and a bound pool exchanging with them. Nothing is
# written to combine them; it is eight names in one call.
#
three_pool = spoiled.simulate(
    **WHITE_MATTER, poolB_fraction=0.2, **FREE, bound_fraction=0.117, **BOUND
)

# sphinx_gallery_start_ignore
print(
    f"  both pools together   {float(abs(np.asarray(three_pool).reshape(-1)[-1])):.5f}"
)
# sphinx_gallery_end_ignore

# %%
# Where the field is not what it should be
# ----------------------------------------
#
# ``B0`` turns the transverse states between one event and the next. Whether
# that reaches the signal is a question about the sequence rather than about
# the field: a train that dephases by a whole configuration order every
# repetition separates the orders completely and comes back insensitive to it,
# while a balanced train keeps them together and bands.
#
# So the demonstration is balanced -- the readout is the only thing that
# changes -- and the bands sit where a balanced train puts them, ``1 / TR``
# apart.
#
BALANCED_TR_MS = 10.0
offsets_hz = torch.linspace(-150.0, 150.0, 121)
banded = MRFSimulator(
    model=replace(MRFSimulator.model, operators=BALANCED),
    flip=np.full(64, 20.0),
    TR=BALANCED_TR_MS,
    TI=0.0,
    states=20,
).simulate(T1=1000.0, T2=80.0, B0=offsets_hz, repetitions="auto")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
axis.plot(offsets_hz, abs(banded[:, -1]))
for band in (-100.0, 0.0, 100.0):
    axis.axvline(band, color="0.6", linestyle=":", linewidth=1.2)
axis.set(
    xlabel="off resonance [Hz]",
    ylabel="signal magnitude [a.u.]",
    title=f"balanced, TR = {BALANCED_TR_MS:.0f} ms: bands every "
    f"{1e3 / BALANCED_TR_MS:.0f} Hz",
)
axis.grid(alpha=0.3)
print(f"  nulls sit {1e3 / BALANCED_TR_MS:.0f} Hz apart, which is 1 / TR")
# sphinx_gallery_end_ignore

# %%
# An inversion that does not quite invert
# ---------------------------------------
#
# ``inv_efficiency`` is how much of the magnetization the inversion pulse
# actually turns over -- a property of the pulse and the transmit field rather
# than of the tissue. It enters where the inversion does, at the front of the
# train, on the repetitions whose contrast the inversion exists to create.
#
efficiencies = torch.tensor([1.0, 0.9, 0.8])
inverted = fingerprinting.simulate(**WATER, inv_efficiency=efficiencies)

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
    f"  the first repetition falls to {float(inverted[2][0].abs() / inverted[0][0].abs()):.3f} of the ideal;"
)
print(
    f"  by the four-hundredth the three agree to "
    f"{float((inverted[2][-1] - inverted[0][-1]).abs() / inverted[0][-1].abs()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
# Spins that will not hold still
# ------------------------------
#
# Diffusion and flow are read off the same thing -- what a gradient has wound
# onto a configuration order -- so both need the sequence to say how much
# winding an order stands for. That is two arguments to the simulator rather
# than two tissue fields: ``crusher_dephasing_rad``, the turn one crusher puts
# across a voxel, and ``voxel_size_m``, the distance it puts it across. Without
# them an order is a bookkeeping index with no physical extent, and a voxel
# given a diffusivity or a velocity is attenuated by nothing at all.
#
MOMENT = dict(crusher_dephasing_rad=4.0 * math.pi, voxel_size_m=1e-3)
moving = MRFSimulator(**TRAIN, **MOMENT)

diffusivities = torch.tensor([0.0, 1.0, 2.0, 3.0])
velocities = torch.tensor([0.0, 0.01, 0.03, 0.05])
diffusing = moving.simulate(**WATER, D=diffusivities)
flowing = moving.simulate(**WATER, v=velocities)

# %%
#
# A fingerprinting train is only mildly diffusion-weighted, so what is drawn is
# the ratio to a voxel that does not diffuse. Flow is a large enough effect to
# read off the train itself.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
undiffused = diffusing[0].abs()
readable = undiffused > 0.15 * undiffused.max()
for row in range(1, len(diffusivities)):
    ratio = torch.where(readable, diffusing[row].abs() / undiffused, torch.nan)
    axis.plot(ratio.numpy(), label=f"D = {float(diffusivities[row]):.0f}")
axis.axhline(1.0, color="0.35", lw=1.2)
axis.set(
    xlabel="repetition",
    ylabel="relative to no diffusion",
    title=r"diffusion in $\mu$m$^2$/ms, over a "
    f"{MOMENT['crusher_dephasing_rad'] / math.pi:.0f}"
    r"$\pi$ crusher across a "
    f"{1e3 * MOMENT['voxel_size_m']:.0f} mm voxel",
    ylim=(0.75, 1.2),
)
axis.grid(alpha=0.3)
key(axis, ncols=3)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
for row in range(len(velocities)):
    axis.plot(abs(flowing[row]), label=f"v = {100 * float(velocities[row]):.0f} cm/s")
axis.set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="flow through the same voxel",
)
axis.grid(alpha=0.3)
key(axis, ncols=4)

print(
    "  diffusion damps the higher orders, so it costs signal where the train "
    "is built from them:"
)
print(
    f"    D = 3 departs from D = 0 by "
    f"{float((diffusing[3] - diffusing[0]).abs().max() / diffusing[0].abs().max()):.0%}"
)
print(
    "  flow carries winding out of the voxel and brings unsaturated "
    "magnetization in, so it reshapes the train:"
)
print(
    f"    5 cm/s departs by "
    f"{float((flowing[3] - flowing[0]).abs().max() / flowing[0].abs().max()):.1f}"
    f"x the unflowed signal"
)
# sphinx_gallery_end_ignore

# %%
# What naming a property costs
# ----------------------------
#
# Nothing, until a voxel is given a value at which the term does something. A
# property held where it has no effect -- unit transmit, no off resonance, an
# empty pool -- is reported absent and its term stays out of the kernel that is
# compiled and run. So a call naming every field on this page, none of them
# given a value that matters, costs what the two-parameter call costs.
#
idle = fingerprinting.simulate(
    **WATER,
    B0=0.0,
    D=0.0,
    v=0.0,
    poolB_fraction=0.0,
    bound_fraction=0.0,
    inv_efficiency=1.0,
)

# sphinx_gallery_start_ignore
print(
    "  six more fields named, none of them doing anything: agrees to "
    f"{float(np.abs(np.asarray(idle) - np.asarray(baseline)).max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# What is left is the vocabulary itself. Every term here is one the kernels
# carry; a term they do not -- a third free pool, a gradient moment that varies
# down the train -- is a change to the engine rather than to a name in a call.
