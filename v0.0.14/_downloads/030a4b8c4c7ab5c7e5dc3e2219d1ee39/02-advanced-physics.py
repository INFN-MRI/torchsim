"""
===================
2. Advanced physics
===================

In lesson 1 a tissue was T1 and T2. A voxel has more properties than that, and
each one adds a term to the physics. In this lesson you add them to a
fingerprinting train one at a time: the transmit field and the array that
produces it, the shaped pulse of a real slice selection, a second exchanging
pool and a bound pool, off resonance, an imperfect inversion, and diffusion and
flow.

**Learning objectives**

- Turn a physics term on by naming the tissue property in the call.
- Describe a transmit array with per-channel sensitivities and an RF shim.
- Simulate a shaped RF pulse across the slice.
- Add an exchanging pool and a bound pool to a voxel.
- Know that a property held at its neutral value costs nothing.

Previous: :doc:`01-first-simulation`. Next: :doc:`03-custom-signal-model`,
where you write a sequence of your own.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install blochsim matplotlib
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# sphinx_gallery_start_ignore
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")
# The gallery runs an example from its own directory; a shell may not.
DATA = Path("data") if Path("data").is_dir() else Path(__file__).parent / "data"
# sphinx_gallery_end_ignore

# %%
# Sequence
# --------
#
# The test sequence is an inversion-prepared fingerprinting train: 400
# repetitions at TR = 10 ms, with a flip angle that rises from 5 to 55 degrees
# and falls back. The changing flip angle drives every echo pathway, so most of
# the terms below show on it. Where a term needs another readout to be visible,
# only the readout changes.
import numpy as np
import torch

from blochsim.simulators import MRFSimulator

FLIP_DEG = np.concatenate((np.linspace(5.0, 55.0, 200), np.linspace(55.0, 5.0, 200)))
fingerprinting = MRFSimulator(flip=FLIP_DEG, TR=10.0, TI=20.0, states=20)
WATER = dict(T1=1000.0, T2=80.0)
baseline = fingerprinting.simulate(**WATER)

# %%
# A simulator declares the tissue properties its sequence is written in:
print(f"declared: {', '.join(fingerprinting.exposes)}")

# %%
# It accepts more than it declares. Every property in this lesson can be given
# to any simulator, and giving one is what asks for its physics:
print(f"accepted: {', '.join(fingerprinting.accepts)}")

# %%
# Transmit field
# --------------
#
# ``B1`` scales the flip angle a voxel turns. The nominal angle changes every
# repetition, so a transmit error reshapes the signal trajectory instead of
# scaling it. That is why B1 can be estimated together with T1 and T2, and why
# ignoring it biases both.
transmit = torch.tensor([0.7, 0.85, 1.0, 1.15])
scaled = fingerprinting.simulate(**WATER, B1=transmit)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
for row, value in enumerate(transmit):
    ax.plot(abs(scaled[row]), color=SERIES[row], label=f"B1 = {float(value):.2f}")
ax.set(xlabel="repetition", ylabel="|signal| (a.u.)", xlim=(0, 400), ylim=(0, None))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Transmit array and RF shim
# --------------------------
#
# On a parallel-transmit system the field in a voxel is the complex sum of what
# the channels put there. ``B1`` and ``B1phase`` then have one row per channel,
# and a :class:`~blochsim.ShimDefinition` gives the amplitude and phase each
# channel is driven at.
#
# Take four channels whose sensitivities are a quarter turn apart in phase.
# Driven alike they cancel. Delaying each channel by one more phase step than
# the one before it undoes the offsets, and the field returns:
import math

import blochsim
from blochsim import ShimDefinition
from blochsim.simulators import FSESimulator

CHANNELS, VOXELS = 4, 3
sensitivity = torch.full((CHANNELS, VOXELS), 1.0 / CHANNELS)
sensitivity_phase = (
    (torch.arange(CHANNELS)[:, None] * 2.0 * math.pi / CHANNELS)
    .expand(CHANNELS, VOXELS)
    .contiguous()
    .float()
)
ARRAY = dict(
    T1=torch.linspace(600.0, 1400.0, VOXELS),
    T2=torch.linspace(40.0, 120.0, VOXELS),
    B1=sensitivity,
    B1phase=sensitivity_phase,
)


def first_echo(step_rad):
    """First echo of a train driven with a given phase step between channels."""
    shim = ShimDefinition(
        0,
        (1.0,) * CHANNELS,
        tuple(float(-channel * step_rad) for channel in range(CHANNELS)),
    )
    train = FSESimulator(
        ESP=5.0, flip=torch.full((8,), 150.0), states=12, shims={0: shim}
    )
    return train.simulate(**ARRAY)[..., 0].abs()


steps_rad = torch.linspace(0.0, 2.0 * math.pi, 61)
swept = torch.stack([first_echo(float(step)) for step in steps_rad])

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
degrees = torch.rad2deg(steps_rad).numpy()
for voxel in range(VOXELS):
    ax.plot(
        degrees,
        swept[:, voxel].numpy(),
        color=SERIES[voxel],
        label=f"voxel {voxel + 1}",
    )
for marked, text in ((0.0, "driven alike"), (360.0 / CHANNELS, "counter-rotated")):
    ax.axvline(marked, color=MUTED, linestyle=":", linewidth=1.4)
    ax.annotate(text, xy=(marked + 4.0, 0.03), color=MUTED, rotation=90, va="bottom")
ax.set(
    xlabel="phase step between channels (deg)",
    ylabel="first echo (a.u.)",
    xlim=(0.0, 360.0),
    xticks=[0, 90, 180, 270, 360],
)
legend_outside(ax)
plt.show()
print(f"driven alike     {[round(float(v), 4) for v in first_echo(0.0)]}")
print(
    "counter-rotated  "
    f"{[round(float(v), 4) for v in first_echo(2.0 * math.pi / CHANNELS)]}"
)
# sphinx_gallery_end_ignore

# %%
# A shim belongs to the pulse, not to the sequence: each RF event names the shim
# it is driven on, so an excitation and a refocusing pulse can use different
# ones.
#
# Shaped RF pulse and slice profile
# ---------------------------------
#
# The pulses so far are instantaneous. A slice-selective pulse turns a different
# angle at each position across the slice. That is a Bloch response, not a
# scaling, so it cannot be folded into the flip angle: the pulse has to be
# integrated.
#
# blochsim takes the complex RF envelope as it comes off a Pulseq block or an
# MRD sequence description. The one used here is an SLR 90-degree pulse, 2 ms
# long, for a 5 mm slice, saved beside this file:
waveform = np.load(DATA / "slr90.npz")
excitation = blochsim.rf_definition(
    waveform["samples"],
    dwell_s=float(waveform["dwell_s"]),
    bandwidth_hz=float(waveform["bandwidth_hz"]),
)

# %%
# Give it to a simulator as ``pulse``. ``across_slice`` is the number of
# positions through the slice the pulse is integrated at. Without it the pulse
# is evaluated at the slice centre only, which reproduces the hard-pulse answer:
REFOCUSED = dict(ESP=5.0, TR=3000.0, T1=830.0, T2=80.0, states=48)
angles = torch.full((48,), 150.0)

hard = FSESimulator(**REFOCUSED).simulate(flip=angles)
centre = FSESimulator(**REFOCUSED, pulse=excitation).simulate(flip=angles)
across = FSESimulator(**REFOCUSED, pulse=excitation, across_slice=21).simulate(
    flip=angles
)

# sphinx_gallery_start_ignore
from blochsim.sequence._transition import transition_table  # noqa: E402

positions = torch.linspace(-1.0, 1.0, 121, dtype=torch.float64)
table = transition_table(excitation, positions, bins=64, rf_raster_time_s=1e-6)
_a, b = table.at(
    torch.arange(positions.numel()),
    torch.full((positions.numel(),), 0.5 * math.pi, dtype=torch.float64),
)
turned_deg = torch.rad2deg(2.0 * torch.arcsin(b.abs().clamp(max=1.0)))

fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.38 * PAGE_WIDTH))
time_ms = 1e3 * np.arange(waveform["samples"].size) * float(waveform["dwell_s"])
axes[0].plot(
    time_ms,
    np.abs(waveform["samples"]) / np.abs(waveform["samples"]).max(),
    color=SERIES[0],
)
axes[0].set(xlabel="time (ms)", ylabel="envelope (a.u.)", title="pulse")
axes[1].plot(positions.numpy(), turned_deg.numpy(), color=SERIES[0])
axes[1].axhline(90.0, color=MUTED, linestyle=":")
axes[1].set(
    xlabel="position (slice thicknesses)",
    ylabel="flip turned (deg)",
    title="flip across the slice",
)
plt.show()
print(f"at the slice centre     {float(hard.abs().max()):.4f} (hard pulse)")
print(f"the shaped pulse there  {float(centre.abs().max()):.4f}")
print(f"averaged over the slice {float(across.abs().max()):.4f}")
print(
    "the profile costs       "
    f"{100 * (1 - float(across.abs().max()) / float(hard.abs().max())):.0f}% "
    "of the signal"
)
# sphinx_gallery_end_ignore

# %%
# The flip is flat across the passband and falls away outside it. At the centre
# the shaped pulse gives the hard-pulse signal, which confirms the envelope is
# scaled correctly. Averaged across the slice the signal is much smaller. For a
# refocused train that is more than a scaling: the slice edges see a smaller
# refocusing angle, so they carry a different mix of echo pathways.
#
# Exchange and magnetization transfer
# -----------------------------------
#
# A second *free* pool, such as myelin water next to intra- and extracellular
# water, is recorded along with the first. A *bound* pool has a T2 of tens of
# microseconds, is never recorded, and exchanges magnetization with the water
# that is.
#
# Five properties describe an exchanging free pool and three a bound pool. The
# values below are the white matter model of Malik et al. (Magn. Reson. Med.
# 2018), on a spoiled gradient echo train driven to steady state. Changing the
# readout of a simulator is a one-line subclass:
from blochsim import SPGRReadout

REPETITIONS, SPOILING_STEP = 200, 117.0
index = np.arange(REPETITIONS)
WHITE_MATTER = dict(T1=779.0, T2=45.0)
FREE = dict(poolB_exchange=2.0, poolB_T1=500.0, poolB_T2=20.0)
BOUND = dict(bound_exchange=4.3, bound_T1=779.0)


class SpoiledMRF(MRFSimulator):
    readout = SPGRReadout


spoiled = SpoiledMRF(
    flip=np.full(REPETITIONS, 10.0),
    phases=SPOILING_STEP * index * (index + 1) / 2.0,
    TR=5.0,
    TI=0.0,
    states=40,
)
one_pool = spoiled.simulate(**WHITE_MATTER)
with_free = spoiled.simulate(**WHITE_MATTER, poolB_fraction=0.2, **FREE)
with_bound = spoiled.simulate(**WHITE_MATTER, bound_fraction=0.117, **BOUND)

# %%
# A pool fraction is a tissue property, so sweeping it is one call over many
# voxels. At a fraction of zero both must return the one-pool signal:
fractions = torch.linspace(0.0, 0.3, 31)
free_sweep = spoiled.simulate(**WHITE_MATTER, poolB_fraction=fractions, **FREE)
bound_sweep = spoiled.simulate(**WHITE_MATTER, bound_fraction=fractions, **BOUND)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))
for color, label, values in (
    (MUTED, "one pool", one_pool),
    (SERIES[1], "second free pool", with_free),
    (SERIES[2], "bound pool", with_bound),
):
    axes[0].plot(abs(np.asarray(values).reshape(-1)), color=color, label=label)
axes[0].set(
    xlabel="repetition",
    ylabel="|signal| (a.u.)",
    title="approach to steady state",
    xlim=(0, REPETITIONS),
)
settled = float(abs(np.asarray(one_pool).reshape(-1)[-1]))
axes[1].axhline(settled, color=MUTED)
axes[1].plot(fractions.numpy(), abs(free_sweep[:, -1]), color=SERIES[1])
axes[1].plot(fractions.numpy(), abs(bound_sweep[:, -1]), color=SERIES[2])
axes[1].set(
    xlabel="second-pool fraction",
    ylabel="steady-state |signal| (a.u.)",
    title="effect of the fraction",
)
legend_outside(fig)
plt.show()
print(f"one pool settles at   {settled:.5f}")
print(f"a second free pool    {float(abs(np.asarray(with_free).reshape(-1)[-1])):.5f}")
print(f"a bound pool          {float(abs(np.asarray(with_bound).reshape(-1)[-1])):.5f}")
print(
    "at a fraction of zero the two rejoin it, to "
    f"{float(max(abs(free_sweep[0, -1] - settled), abs(bound_sweep[0, -1] - settled))):.1e}"
)
# sphinx_gallery_end_ignore

# %%
# The two move the signal in opposite directions. A second free pool raises it,
# because that pool is recorded too. A bound pool lowers it, because the
# magnetization parked there is never read.
#
# The properties are independent, so one voxel can carry both, eight names in
# one call:
three_pool = spoiled.simulate(
    **WHITE_MATTER, poolB_fraction=0.2, **FREE, bound_fraction=0.117, **BOUND
)

# sphinx_gallery_start_ignore
print(f"both pools together   {float(abs(np.asarray(three_pool).reshape(-1)[-1])):.5f}")
# sphinx_gallery_end_ignore

# %%
# Off resonance
# -------------
#
# ``B0`` is the off-resonance frequency in Hz. It rotates the transverse states
# between events, and whether that reaches the signal depends on the sequence.
# A train that dephases by a whole configuration order every repetition
# separates the orders and is insensitive to it. A balanced train shows bands.
# Only the readout changes below:
from blochsim import bSSFPReadout


class BalancedMRF(MRFSimulator):
    readout = bSSFPReadout


offsets_hz = torch.linspace(-150.0, 150.0, 121)
banded = BalancedMRF(flip=np.full(64, 20.0), TR=10.0, TI=0.0, states=20).simulate(
    T1=1000.0, T2=80.0, B0=offsets_hz, repetitions="auto"
)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
ax.plot(offsets_hz, abs(banded[:, -1]), color=SERIES[0])
for band in (-100.0, 0.0, 100.0):
    ax.axvline(band, color=MUTED, linestyle=":", linewidth=1.2)
ax.set(xlabel="off resonance (Hz)", ylabel="|signal| (a.u.)", xlim=(-150, 150))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The signal repeats every 1 / TR = 100 Hz, with the dotted lines at the band
# centres.
#
# Inversion efficiency
# --------------------
#
# ``inv_efficiency`` is the fraction of the magnetization the inversion pulse
# turns over. It changes the start of the train, where the inversion sets the
# contrast, and the difference fades as the train goes on:
efficiencies = torch.tensor([1.0, 0.9, 0.8])
inverted = fingerprinting.simulate(**WATER, inv_efficiency=efficiencies)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
for row, value in enumerate(efficiencies):
    ax.plot(
        abs(inverted[row][:80]),
        color=SERIES[row],
        label=f"efficiency {float(value):.1f}",
    )
ax.set(xlabel="repetition", ylabel="|signal| (a.u.)", xlim=(0, 80))
legend_outside(ax)
plt.show()
first = float(inverted[2][0].abs() / inverted[0][0].abs())
last = float((inverted[2][-1] - inverted[0][-1]).abs() / inverted[0][-1].abs())
print(f"the first repetition falls to {first:.3f} of the ideal;")
print(f"by the four-hundredth the three agree to {last:.1e}")
# sphinx_gallery_end_ignore

# %%
# Diffusion and flow
# ------------------
#
# Diffusion and flow are read off the phase that a gradient winds onto each
# configuration order, so the sequence has to say how much winding an order
# stands for. Two simulator arguments do that: ``crusher_dephasing_rad``, the
# phase one crusher puts across a voxel, and ``voxel_size_m``, the distance it
# puts it across. Without them an order has no physical extent and neither term
# does anything. The tissue properties are ``D`` in µm²/ms and ``v`` in m/s:
moving = MRFSimulator(
    flip=FLIP_DEG,
    TR=10.0,
    TI=20.0,
    states=20,
    crusher_dephasing_rad=4.0 * math.pi,
    voxel_size_m=1e-3,
)
diffusivities = torch.tensor([0.0, 1.0, 2.0, 3.0])
velocities = torch.tensor([0.0, 0.01, 0.03, 0.05])
diffusing = moving.simulate(**WATER, D=diffusivities)
flowing = moving.simulate(**WATER, v=velocities)

# %%
# The train is only mildly diffusion-weighted, so the figure shows diffusion as
# the change from a voxel that does not diffuse, relative to its peak signal.
# Flow is large enough to read directly.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(
    2, 1, figsize=(0.8 * PAGE_WIDTH, 0.75 * PAGE_WIDTH), sharex=True
)
undiffused = diffusing[0].abs()
for row in range(1, len(diffusivities)):
    change_pct = 100 * (diffusing[row].abs() - undiffused) / undiffused.max()
    axes[0].plot(
        change_pct.numpy(),
        color=SERIES[row - 1],
        label=f"D = {float(diffusivities[row]):.0f} µm²/ms",
    )
axes[0].axhline(0.0, color=MUTED, lw=1.2)
axes[0].set(ylabel="change from D = 0 (%)", title="diffusion")
for row in range(len(velocities)):
    axes[1].plot(
        abs(flowing[row]),
        color=SERIES[row],
        label=f"v = {100 * float(velocities[row]):.0f} cm/s",
    )
axes[1].set(xlabel="repetition", ylabel="|signal| (a.u.)", title="flow")
for axis in axes:
    legend_outside(axis)
plt.show()
change = float((diffusing[3] - diffusing[0]).abs().max() / diffusing[0].abs().max())
flow_change = float((flowing[3] - flowing[0]).abs().max() / flowing[0].abs().max())
print(f"D = 3 departs from D = 0 by {change:.0%}")
print(f"5 cm/s departs by {flow_change:.1f}x the unflowed signal")
# sphinx_gallery_end_ignore

# %%
# Diffusion damps the higher orders, so it costs signal where the train is built
# from them. Flow carries winding out of the voxel and brings unsaturated
# magnetization in, so it reshapes the train.
#
# Unused properties
# -----------------
#
# None. A property held at the value where it has no effect (unit transmit, no
# off resonance, an empty pool) is treated as absent, and its term is left out
# of the kernel that runs. Naming six of them at their neutral values gives the
# baseline back:
idle = fingerprinting.simulate(
    **WATER,
    B0=0.0,
    D=0.0,
    v=0.0,
    poolB_fraction=0.0,
    bound_fraction=0.0,
    inv_efficiency=1.0,
)
print(
    f"largest difference: {np.abs(np.asarray(idle) - np.asarray(baseline)).max():.1e}"
)

# %%
# Every term in this lesson is one the kernels already carry. A term they do
# not, such as a third free pool or a gradient moment that varies along the
# train, is a change to blochsim itself and not a name in a call.
#
# Specification
# -------------
#
# What this lesson did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, simulate an inversion-prepared MRF train of 400 repetitions
#    (TR 10 ms, flip 5 to 55 to 5 degrees) in a voxel with T1 1 s and T2 80 ms.
#    Add, one at a time and by naming the tissue property: B1 scaling, a
#    four-channel transmit array with an RF shim, an SLR pulse integrated across
#    the slice, a second free pool and a bound pool (white matter, Malik 2018),
#    B0 on a balanced readout, inversion efficiency, and diffusion and flow.
#    Check that each property at its neutral value reproduces the baseline.
