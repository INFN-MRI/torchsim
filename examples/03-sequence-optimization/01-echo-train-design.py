"""
===============================
Echo train design for sharpness
===============================

In this example you design the refocusing flip angles of a fast spin echo for
image quality rather than for precision: first one echo train, then a whole
segmented 3D protocol in which each shot has its own repetition time, echo
train length and angles.

T2 decay along a long train modulates k-space. That modulation is a point
spread function, so the refocusing angles set the resolution of the image [1]_.
You design the angles against a cost that measures this blur, the contrast
between tissues and the RF power.

Prerequisites: Course lesson :doc:`../01-framework/01-first-simulation`.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install blochsim matplotlib
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# sphinx_gallery_start_ignore
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

# %%
# Design problem
# --------------
#
# A design has three pieces. A simulator with the tissue fixed on it, a cost
# written on the signal it records, and the bounded parameters that
# :class:`~blochsim.optim.SequenceDesign` adjusts to lower the cost.
#
# The tissues are those of a PD-weighted knee protocol, read for the
# separation between fluid and cartilage. The design uses all three at once, so
# that the train is not tailored to one of them. T1 and T2 are in
# milliseconds:
import torch

from blochsim.optim import Bounded, SequenceDesign
from blochsim.simulators import FSESimulator

TISSUES = {
    #            cartilage  muscle  synovial fluid
    "T1": [1200.0, 1420.0, 3600.0],
    "T2": [35.0, 30.0, 250.0],
}
CARTILAGE, MUSCLE, FLUID = 0, 1, 2

# %%
# Blur
# ----
#
# Let the echo index run along one k-space direction. The magnitude of the echo
# train is then the k-space modulation, and the width of its Fourier transform
# is the blur it adds. You can read that width off the modulation without a
# transform: the second moment of :math:`|\mathcal{F}w|^2` is the energy in the
# slope of the train :math:`w` divided by the energy in :math:`w`.
#
# A train that stops early contributes fewer terms, so trains of different
# lengths compare on the same footing. The second half of this example needs
# that. ``signal`` is ``(shots, tissues, echoes)`` and ``acquired`` is
# ``(shots, echoes)``, one where the shot is still acquiring. The result is the
# blur in pixels, ``(shots, tissues)``:


def blur(signal, acquired):
    pair = acquired[:, None, :-1] * acquired[:, None, 1:]
    step = torch.diff(signal, dim=-1) * pair
    energy = (signal * acquired[:, None, :]).square().sum(-1).clamp_min(1e-12)
    lines = acquired.sum(-1)[:, None]
    return lines / (2 * torch.pi) * (step.square().sum(-1) / energy).sqrt()


# %%
# Echo train
# ----------
#
# A 120-echo train has 120 angles but three degrees of freedom:
#
# - the minimum angle, which sets how much flow and motion spoil the train;
# - the angle at the centre of k-space, which sets the image contrast;
# - the maximum angle, which the deposited RF power limits.
#
# The train starts at the maximum, drops to the minimum as the pseudo steady
# state is established, rises to the centre-of-k-space angle where the centre of
# k-space is sampled, and ramps back up to the maximum:
ESP_MS = 5.0
ECHOES = 120
CENTRE_ECHO = 24

one_train = FSESimulator(ESP=ESP_MS, states=12, **TISSUES)
echo = torch.arange(1, ECHOES + 1, dtype=torch.float32)


def ramp(index, start, stop, first, last):
    """A smooth step from ``first`` to ``last`` between two echo indices."""
    span = ((index - start) / (stop - start).clamp_min(1e-3)).clamp(0.0, 1.0)
    return first + (last - first) * span.square() * (3.0 - 2.0 * span)


def shape(index, control, length, centre_echo):
    """Refocusing angles of a train of ``length`` echoes.

    ``control`` is ``(shots, 3)``: the minimum, centre-of-k-space and maximum
    angles. ``centre_echo`` is the echo that samples the centre of the shot's
    k-space band.
    """
    low, middle, high = control[:, 0:1], control[:, 1:2], control[:, 2:3]
    settled = torch.full_like(low, 5.0)
    sampled = torch.full_like(low, float(centre_echo))
    return torch.where(
        index <= 5,
        ramp(index, torch.ones_like(low), settled, high, low),
        torch.where(
            index <= centre_echo,
            ramp(index, settled, sampled, low, middle),
            ramp(index, sampled, length, middle, high),
        ),
    )


# %%
# Cost
# ----
#
# The image should be sharp, fluid should stand out from cartilage, and the RF
# power should stay where the scanner accepts it. Power is the refocusing
# energy, relative to a train of 180 degree pulses, divided by the time it is
# spread over. A train that ends early gets no credit for echoes it never
# played:
LOWEST = torch.tensor([20.0, 30.0, 60.0])
HIGHEST = torch.tensor([90.0, 160.0, 170.0])
PRESCRIBED = torch.tensor([[50.0, 90.0, 150.0]])

ALWAYS = torch.ones(1, ECHOES)


def power(flip, acquired, TR_ms):
    energy = ((flip / 180.0).square() * acquired).sum(-1)
    return energy / (TR_ms.squeeze(-1) * 1e-3)


# %%
# The scanner limits RF power, so the cost may spend up to what the prescribed
# train already deposits and no more. The weights set the trade between the
# three terms:
POWER_BUDGET = power(
    shape(echo, PRESCRIBED, torch.full((1, 1), float(ECHOES)), CENTRE_ECHO),
    ALWAYS,
    torch.full((1, 1), 1800.0),
).mean()


def single_train(control):
    flip = shape(echo, control, torch.full_like(control[:, :1], ECHOES), CENTRE_ECHO)
    signal = one_train.simulate(flip=flip, TR=1800.0).abs()
    at_centre = signal[:, :, CENTRE_ECHO - 1]
    contrast = at_centre[:, FLUID] - at_centre[:, CARTILAGE]
    deposited = power(flip, ALWAYS, torch.full_like(control[:, :1], 1800.0))
    return (
        blur(signal, ALWAYS).mean()
        - 12.0 * contrast.mean()
        + 20.0 * torch.relu(deposited.mean() / POWER_BUDGET - 1.0)
    )


# %%
# Optimization
# ------------
#
# Start from a conventional prescription: 50 degree minimum, 90 degree
# centre-of-k-space angle and 150 degree maximum. :class:`~blochsim.optim.Bounded`
# holds each angle inside the limits the scanner plays:
design = SequenceDesign(single_train, control=Bounded(PRESCRIBED, LOWEST, HIGHEST))

# sphinx_gallery_start_ignore
# One call resolves the structure of the protocol and holds it, so the clock
# below measures the design alone.
design.minimize(iterations=1)
start = time.perf_counter()
# sphinx_gallery_end_ignore
one = design.minimize(iterations=25, learning_rate=0.3)

# sphinx_gallery_start_ignore
one_elapsed = time.perf_counter() - start
print(
    f"one train of {ECHOES} echoes designed in {one_elapsed:.2f} s, "
    f"{1000 * one_elapsed / 25:.1f} ms per iteration"
)
# sphinx_gallery_end_ignore

# %%
# Simulate the prescribed and the designed trains, and compare the angles, the
# signal and the point spread:
LENGTH = torch.full((1, 1), float(ECHOES))
prescribed_flip = shape(echo, PRESCRIBED, LENGTH, CENTRE_ECHO)
designed_flip = shape(echo, one.parameters["control"], LENGTH, CENTRE_ECHO)
prescribed_signal = one_train.simulate(flip=prescribed_flip, TR=1800.0).abs()
designed_signal = one_train.simulate(flip=designed_flip, TR=1800.0).abs()

# sphinx_gallery_start_ignore
for label, angles, signal in (
    ("prescribed", PRESCRIBED, prescribed_signal),
    ("designed", one.parameters["control"], designed_signal),
):
    at_centre = signal[:, :, CENTRE_ECHO - 1]
    print(
        f"{label:<11}"
        f" min {float(angles[0, 0]):>5.1f}  centre {float(angles[0, 1]):>5.1f}"
        f"  max {float(angles[0, 2]):>5.1f} deg"
        f" | blur {float(blur(signal, ALWAYS).mean()):>4.2f} px"
        f" | fluid - cartilage "
        f"{float(at_centre[0, FLUID] - at_centre[0, CARTILAGE]):>5.3f}"
    )

fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))
echo_index = np.arange(1, ECHOES + 1)
pixel = np.arange(ECHOES) - ECHOES // 2

axes[0].plot(
    echo_index,
    prescribed_flip[0].numpy(force=True),
    "--",
    color=MUTED,
    label="prescribed",
)
axes[0].plot(
    echo_index,
    designed_flip[0].numpy(force=True),
    color=SERIES[0],
    label="designed",
)
axes[0].set(xlabel="echo", ylabel="refocusing angle (deg)", title="train")

# Dashed is prescribed and solid is designed, as in the panel beside it, so
# the colour carries the tissue.
for tissue, name, colour in (
    (CARTILAGE, "cartilage", SERIES[1]),
    (FLUID, "synovial fluid", SERIES[2]),
):
    axes[1].plot(
        echo_index,
        prescribed_signal[0, tissue].numpy(force=True),
        "--",
        color=colour,
        alpha=0.6,
    )
    axes[1].plot(
        echo_index,
        designed_signal[0, tissue].numpy(force=True),
        color=colour,
        label=name,
    )
axes[1].set(xlabel="echo", ylabel="|signal| (a.u.)", title="k-space modulation")


def spread(modulation):
    return torch.fft.fftshift(torch.fft.fft(modulation, dim=-1).abs().square(), dim=-1)


for label, signal, style, colour in (
    ("prescribed", prescribed_signal, "--", MUTED),
    ("designed", designed_signal, "-", SERIES[0]),
):
    psf = spread(signal[0, CARTILAGE])
    axes[2].semilogy(
        pixel, (psf / psf.max()).numpy(force=True), style, color=colour, label=label
    )
axes[2].set(
    xlabel="pixel", ylabel="PSF (normalized)", title="cartilage", ylim=(1e-5, 2.0)
)
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The designed train has a lower blur and keeps the contrast at the centre of
# k-space. The cartilage point spread narrows from the prescribed to the
# designed train.
#
# Image
# -----
#
# The echo index runs along one k-space direction, so forming the image is a
# multiplication. Transform each tissue's contribution along the phase-encode
# axis, weight every line by the train at the echo that sampled it, and
# transform back. A train that decays fast weights the edges of k-space down,
# and the image comes back smeared.
#
# The phantom is a cartoon knee: a cartilage band with a two-pixel joint line,
# and four fluid bars five, three, two and one pixels thick. No noise is added,
# so the two images differ only by the train.

# sphinx_gallery_start_ignore
SIZE = ECHOES

rows, columns = torch.meshgrid(
    torch.arange(SIZE, dtype=torch.float32),
    torch.arange(SIZE, dtype=torch.float32),
    indexing="ij",
)
middle = SIZE / 2 - 0.5
inside = ((rows - middle) / 54.0) ** 2 + ((columns - middle) / 44.0) ** 2 < 1.0


def band(low, high):
    return inside & (rows >= low) & (rows < high)


fluid_map = band(39, 41)
for top, thickness in ((70, 5), (81, 3), (90, 2), (97, 1)):
    fluid_map = fluid_map | band(top, top + thickness)
cartilage_map = band(26, 54) & ~fluid_map
muscle_map = inside & ~cartilage_map & ~fluid_map

# One mask per tissue, in the order the simulator returns them.
PHANTOM = torch.stack([cartilage_map, muscle_map, fluid_map]).to(torch.complex64)


def imaged(signal):
    """Image from ``(1, tissues, echoes)`` magnitudes, phase encoding down."""
    spectrum = torch.zeros(SIZE, SIZE, dtype=torch.complex64)
    for tissue, component in enumerate(PHANTOM):
        # The echo that samples the centre of k-space belongs at ky = 0.
        along_ky = torch.roll(signal[0, tissue].to(torch.complex64), -(CENTRE_ECHO - 1))
        spectrum = spectrum + torch.fft.fft(component, dim=0) * along_ky[:, None]
    return torch.fft.ifft(spectrum, dim=0).abs()


prescribed_image = imaged(prescribed_signal)
designed_image = imaged(designed_signal)
scale = float(prescribed_image.max())

VIEW = (slice(6, 114), slice(12, 108))

fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 0.4 * PAGE_WIDTH))
for axis, picture, title in (
    (axes[0], prescribed_image, "prescribed"),
    (axes[1], designed_image, "designed"),
):
    axis.imshow(
        (picture[VIEW] / scale).numpy(force=True), cmap="gray", vmin=0.0, vmax=0.85
    )
    axis.set(title=title, xticks=[], yticks=[])
    axis.set_box_aspect(1)

profile = slice(64, 104)
column = SIZE // 2
rows_shown = np.arange(profile.start, profile.stop)
axes[2].plot(
    rows_shown,
    (prescribed_image[profile, column] / scale).numpy(force=True),
    "--",
    color=MUTED,
    label="prescribed",
)
axes[2].plot(
    rows_shown,
    (designed_image[profile, column] / scale).numpy(force=True),
    color=SERIES[0],
    label="designed",
)
axes[2].set(xlabel="row", ylabel="signal (normalized)", title="through the bars")
axes[2].set_box_aspect(1)
legend_outside(axes[2])
plt.show()
# sphinx_gallery_end_ignore

# %%
# The joint line and the thin bars show a point spread of a pixel or two. The
# ringing at every edge comes from the finite matrix, not from the train. The
# design moves the depth of the troughs between the bars.
#
# Sharpness moved without giving up contrast: the fluid-to-cartilage difference
# at the centre of k-space is within a percent of where it started, and the
# point spread narrowed by nearly a fifth. The power term keeps the deposited
# power at or below the prescribed level.
#
# The one-pixel bar is the limit. A point spread narrower than a pixel is not
# available, so a bar that is flat in both images is flattened by the matrix.
#
# Segmented protocol
# ------------------
#
# Segmented 3D TSE splits k-space over many shots. The centre of k-space sets
# the contrast and the periphery sets the sharpness. With parameters specific to
# each shot, the protocol can use a long repetition time for the shots that
# determine contrast and a short one for the others [2]_.
#
# The prescription is two sets of numbers, one at the centre and one at the
# periphery, and a cubic transition fills in every shot between them. The
# repetition time, the echo train length and the three control angles
# transition:
ESP_SPACE_MS = 3.5
TE_MS = 28.0
TE_ECHO = round(TE_MS / ESP_SPACE_MS)
GRID = 64  # padded echo axis, at least as long as the longest train

# 320 x 240 phase-encode matrix, CAIPIRINHA 4, elliptical scanning.
LINES = round(320 * 240 / 4 * torch.pi / 4)
BUDGET_S = 300.0

SAMPLES = 16
protocol_shots = FSESimulator(ESP=ESP_SPACE_MS, states=12, **TISSUES)
grid_echo = torch.arange(1, GRID + 1, dtype=torch.float32)

# Each sampled radius stands for the shots at that distance from the centre of
# k-space. Their number grows with radius, which is the area element of the
# phase-encode plane.
radius = (torch.arange(SAMPLES, dtype=torch.float32) + 0.5) / SAMPLES
density = 2 * radius / (2 * radius).sum()
cubic = (3 * radius.square() - 2 * radius.pow(3))[:, None]


def transition(centre, periphery):
    return centre + (periphery - centre) * cubic


# %%
# The shots differ in length. A long train at the centre is acceptable because
# the contrast is set by one echo of it. A short train at the periphery limits
# the T2 blur of its lines. A train that has ended is masked out of the padded
# echo axis.
#
# A refocusing angle of exactly zero is a corner, not a point: what reaches the
# scanner is a magnitude, which has no sign there. The mask therefore floors at
# a negligible angle:
FLOOR = 1e-6


def protocol(
    centre_control, edge_control, centre_length, edge_length, centre_TR, edge_TR
):
    control = transition(centre_control, edge_control)
    length = transition(centre_length, edge_length)
    TR = transition(centre_TR, edge_TR)
    acquired = torch.sigmoid(length - grid_echo).clamp_min(FLOOR)
    flip = shape(grid_echo, control, length, TE_ECHO) * acquired
    return flip, TR, length, acquired


# %%
# Covering k-space ties the two ends together. A shot covers as many lines as
# its train is long, so the number of shots is the number of lines divided by
# the average train length, and the scan time is that many shots at the average
# repetition time. Longer trains at the centre allow a longer repetition time there:


def measure(**design):
    """Everything the cost reads, from one batched simulation of all shots."""
    flip, TR, length, acquired = protocol(**design)
    signal = protocol_shots.simulate(flip=flip, TR=TR).abs()
    shots = LINES / (density * acquired.sum(-1)).sum()
    scan_s = shots * (density * TR.squeeze(-1)).sum() * 1e-3
    return signal, flip, TR, length, acquired, shots, scan_s


def deposited(flip, acquired, TR):
    return (density * power(flip, acquired, TR)).sum()


# The exam may spend the power the prescription deposits.
PRESCRIBED_PROTOCOL = protocol(
    PRESCRIBED,
    PRESCRIBED,
    torch.tensor([[45.0]]),
    torch.tensor([[20.0]]),
    torch.tensor([[1800.0]]),
    torch.tensor([[150.0]]),
)
SPACE_POWER_BUDGET = deposited(
    PRESCRIBED_PROTOCOL[0], PRESCRIBED_PROTOCOL[3], PRESCRIBED_PROTOCOL[1]
)

# %%
# The cost asks for sharpness where it is decided (the outer shots) and for
# contrast where it is decided (the inner shots). It also penalizes a scan
# longer than the budget, a train that does not fit inside its repetition time
# with 60 ms for the excitation and fat saturation, and RF power above the
# prescription:


def image_quality(**design):
    signal, flip, TR, length, acquired, shots, scan_s = measure(**design)
    at_centre = signal[:, :, TE_ECHO - 1]
    contrast = at_centre[:, FLUID] - at_centre[:, CARTILAGE]
    outer, inner = density * radius, density * (1.0 - radius)
    infeasible = torch.relu(
        length.squeeze(-1) * ESP_SPACE_MS + 60.0 - TR.squeeze(-1)
    ).mean()
    return (
        0.7 * (outer * blur(signal, acquired).mean(-1)).sum() / outer.sum()
        - 12.0 * (inner * contrast).sum() / inner.sum()
        + 20.0 * torch.relu(scan_s - BUDGET_S) / BUDGET_S
        + 10.0 * infeasible / 60.0
        + 20.0 * torch.relu(deposited(flip, acquired, TR) / SPACE_POWER_BUDGET - 1.0)
    )


# %%
# Start from the prescription of the abstract in [2]_: a 45-echo train at
# 1800 ms at the centre and a 20-echo train at 150 ms at the periphery:
PRESCRIPTION = {
    "centre_control": Bounded(PRESCRIBED, LOWEST, HIGHEST),
    "edge_control": Bounded(PRESCRIBED, LOWEST, HIGHEST),
    "centre_length": Bounded(torch.tensor([[45.0]]), 12.0, 60.0),
    "edge_length": Bounded(torch.tensor([[20.0]]), 12.0, 60.0),
    "centre_TR": Bounded(torch.tensor([[1800.0]]), 150.0, 2600.0),
    "edge_TR": Bounded(torch.tensor([[150.0]]), 150.0, 2600.0),
}

design = SequenceDesign(image_quality, **PRESCRIPTION)

# sphinx_gallery_start_ignore
design.minimize(iterations=1)
start = time.perf_counter()
# sphinx_gallery_end_ignore
many = design.minimize(iterations=40, learning_rate=0.2)

# sphinx_gallery_start_ignore
many_elapsed = time.perf_counter() - start
print(
    f"a whole protocol designed in {many_elapsed:.2f} s, "
    f"{1000 * many_elapsed / 40:.1f} ms per iteration"
)
# sphinx_gallery_end_ignore

# %%
# The number of shots follows from the echo train lengths. Compare the
# prescribed and the designed protocol:

# sphinx_gallery_start_ignore
prescribed = {name: value.initial for name, value in PRESCRIPTION.items()}
print("\n                centre of k-space          periphery")
print("                ETL     TR         ETL     TR       shots    scan")
for label, values in (("prescribed", prescribed), ("designed", many.parameters)):
    _, _, TR, length, _, shots, scan_s = measure(**values)
    print(
        f"{label:<12}"
        f" {float(length[0, 0]):>5.1f}  {float(TR[0, 0]):>5.0f} ms"
        f"  {float(length[-1, 0]):>7.1f}  {float(TR[-1, 0]):>4.0f} ms"
        f"  {float(shots):>7.0f}  {float(scan_s) / 60:>5.2f} min"
    )

print("\nrefocusing angles      min  centre-of-band    max")
for label, values in (
    ("prescribed", {"centre_control": PRESCRIBED, "edge_control": PRESCRIBED}),
    ("designed", many.parameters),
):
    for where, name in (("centre_control", "centre"), ("edge_control", "periphery")):
        angles = values[where][0]
        print(
            f"  {label + ', ' + name:<22}"
            + "".join(f"{float(value):>7.1f}" for value in angles)
        )

signal, flip, TR, length, acquired, shots, scan_s = measure(**many.parameters)
start_signal, start_flip, start_TR, start_length, start_acquired, _, _ = measure(
    **prescribed
)
at_centre = signal[:, :, TE_ECHO - 1]
start_at_centre = start_signal[:, :, TE_ECHO - 1]
width = blur(signal, acquired).mean(-1)
start_width = blur(start_signal, start_acquired).mean(-1)

print(
    f"\nfluid - cartilage at the centre "
    f"{float(start_at_centre[0, FLUID] - start_at_centre[0, CARTILAGE]):.3f}"
    f" -> {float(at_centre[0, FLUID] - at_centre[0, CARTILAGE]):.3f}"
)
print(
    f"blur at the periphery          {float(start_width[-1]):.2f}"
    f" -> {float(width[-1]):.2f} px"
)
print(
    f"RF power                       "
    f"{float(deposited(start_flip, start_acquired, start_TR)):.2f}"
    f" -> {float(deposited(flip, acquired, TR)):.2f}"
    f"  (budget {float(SPACE_POWER_BUDGET):.2f})"
)
# sphinx_gallery_end_ignore

# %%
# Each curve in the first panel is one sampled distance from the centre of
# k-space. Shots between the curves read the curve at their own radius, so
# k-space has no discontinuities between shots.

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 0.7 * PAGE_WIDTH))
colours = [SERIES[index % len(SERIES)] for index in range(SAMPLES)]
grid_index = np.arange(1, GRID + 1)
shown = (0, 5, 10, 15)

for shot in shown:
    live = acquired[shot] > 0.5
    axes[0, 0].plot(
        grid_index[live.numpy(force=True)],
        flip[shot][live].numpy(force=True),
        color=colours[shot],
        label=f"r = {float(radius[shot]):.2f}",
    )
axes[0, 0].set(xlabel="echo", ylabel="refocusing angle (deg)", title="designed trains")

for axis, prescribed_values, designed_values, ylabel, title in (
    (axes[0, 1], start_length[:, 0], length[:, 0], "echo train length", "length"),
    (
        axes[0, 2],
        start_TR[:, 0],
        TR[:, 0],
        "TR (ms)",
        f"scan {float(scan_s) / 60:.1f} min",
    ),
    (
        axes[1, 2],
        start_at_centre[:, FLUID] - start_at_centre[:, CARTILAGE],
        at_centre[:, FLUID] - at_centre[:, CARTILAGE],
        "fluid - cartilage",
        "contrast",
    ),
):
    axis.plot(
        radius.numpy(force=True),
        prescribed_values.numpy(force=True),
        "--",
        color=MUTED,
        label="prescribed",
    )
    axis.plot(
        radius.numpy(force=True),
        designed_values.numpy(force=True),
        color=SERIES[0],
        label="designed",
    )
    axis.set(xlabel="k-space radius", ylabel=ylabel, title=title)

for axis, tissue, name in (
    (axes[1, 0], CARTILAGE, "cartilage"),
    (axes[1, 1], FLUID, "synovial fluid"),
):
    for shot in shown:
        live = acquired[shot] > 0.5
        axis.plot(
            grid_index[live.numpy(force=True)],
            signal[shot, tissue][live].numpy(force=True),
            color=colours[shot],
        )
    axis.axvline(TE_ECHO, color=MUTED, ls=":", lw=1)
    axis.set(xlabel="echo", ylabel="|signal| (a.u.)", title=name)
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Specification
# -------------
#
# What this example did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, design the refocusing angles of a 120-echo fast spin echo
#    for a knee (cartilage, muscle, synovial fluid; ESP 5 ms, TR 1800 ms).
#    Parameterize the train by its minimum, centre-of-k-space and maximum
#    angles, bounded to what the scanner plays. Minimize the blur of the echo
#    train, minus the fluid-to-cartilage contrast at the centre, under an RF
#    power budget. Show the point spread and a phantom image before and after.
#    Then design a segmented 3D protocol whose repetition time, train length
#    and angles transition cubically from the centre to the periphery of
#    k-space, under a scan-time budget of 300 s.
#
# References
# ----------
#
# .. [1] Busse, R. F., Brau, A. C. S., Vu, A., et al., "Effects of
#    refocusing flip angle modulation and view ordering in 3D fast spin
#    echo", Magnetic Resonance in Medicine 60.3 (2008), pp. 640-649.
#    https://doi.org/10.1002/mrm.21680
#
# .. [2] Buonincontri, G., Paul, D., Liu, W., Forman, C., Kluge, T.,
#    "Doubling the repetition time without paying the price: 3D turbo spin
#    echo with individually parameterized echo trains", Proceedings of the
#    International Society for Magnetic Resonance in Medicine (2025),
#    abstract 566-05-007.
