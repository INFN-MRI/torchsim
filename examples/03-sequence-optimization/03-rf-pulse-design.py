"""
=========================================
RF pulse design by optimal control
=========================================

In this example you design the samples of a slice-selective RF pulse by
gradient descent through a Bloch simulation of what they do [1]_. Across a body
at 3 T the transmit field B1 varies by about a fifth either way, and the flip
angle inside the slice varies with it. You reshape a 90 degree excitation so
that it stays close to 90 degrees over that range.

Prerequisites: Course lessons :doc:`../01-framework/01-first-simulation` and
:doc:`../01-framework/02-advanced-physics`.

.. [1] Conolly S, Nishimura D, Macovski A. Optimal control solutions to the
   magnetic resonance selective excitation problem. IEEE Trans Med Imaging
   1986;5(2):106-115.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install blochsim matplotlib
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# sphinx_gallery_start_ignore
import warnings

import matplotlib.pyplot as plt

from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

# %%
# Pulse and slice
# ---------------
#
# The pulse has 128 samples. Under the slice-select gradient, each sample
# rotates a spin at ``x`` slice thicknesses from the centre by
# ``2 pi TBW x / 128`` about z, where TBW is the time-bandwidth product. The
# drive of each sample is in radians, so the design needs no raster and no
# gradient amplitude: any pulse with this time-bandwidth product can play it.
#
# The starting point is a Hamming-windowed sinc with the area of a 90 degree
# flip. That is a small-tip design, played here at a large tip:
import math

import torch

SAMPLES, TBW = 128, 4.0
x = torch.linspace(-2.0, 2.0, 161, dtype=torch.float64)
turn = 2 * math.pi * TBW / SAMPLES * x

t = torch.arange(SAMPLES, dtype=torch.float64) - (SAMPLES - 1) / 2
sinc = torch.sinc(TBW * t / SAMPLES) * (
    0.54 + 0.46 * torch.cos(2 * math.pi * t / SAMPLES)
)
start = sinc / sinc.sum() * (math.pi / 2)

# %%
# Transmit field
# --------------
#
# Simulate every spin at five transmit scalings, from 0.8 to 1.2 of nominal.
# ``compose_spinor`` multiplies the rotations of all samples into one spinor
# ``(a, b)`` per spin, and the transverse magnetization after a pulse applied
# to magnetization along +z is ``2 conj(a) b``:
from blochsim import compose_spinor

B1 = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2], dtype=torch.float64)


def excited(real, imag):
    """|Mxy| after the pulse, shape (B1, x)."""
    drive = (real + 1j * imag)[:, None, None] * B1[None, :, None]
    a, b = compose_spinor(drive, turn.expand(len(B1), -1))
    return (2 * a.conj() * b).abs()


# %%
# Cost
# ----
#
# Inside the slice the magnetization should be fully transverse. Outside it
# should be untouched. The transition band between them is left free. A small
# penalty on the pulse energy prevents the optimizer from trading power for
# flatness:
inside = (x.abs() < 0.4).double()
outside = (x.abs() > 0.75).double()


def cost(real, imag):
    transverse = excited(real, imag)
    miss = inside * (transverse - 1.0) ** 2 + outside * transverse**2
    energy = (real**2 + imag**2).sum() / (start**2).sum()
    return miss.sum() / (inside.sum() + outside.sum()) / len(B1) + 1e-4 * energy


# %%
# Optimization
# ------------
#
# The real and imaginary parts of every sample are the design parameters, with
# no limits. To respect the scanner's peak B1, you would wrap them in
# :class:`~blochsim.optim.Bounded`:
from blochsim.optim import SequenceDesign

design = SequenceDesign(
    cost, real=start.clone(), imag=torch.zeros(SAMPLES, dtype=torch.float64)
)
result = design.minimize(iterations=100, learning_rate=2e-4)
real, imag = result.parameters["real"], result.parameters["imag"]

with torch.no_grad():
    before = excited(start, torch.zeros_like(start))
    after = excited(real, imag)

# sphinx_gallery_start_ignore
fig, (left, middle, right) = plt.subplots(
    1, 3, figsize=(PAGE_WIDTH, 0.36 * PAGE_WIDTH), sharey=False
)
for k, (scaling, row_before, row_after) in enumerate(
    zip(B1, before, after, strict=True)
):
    left.plot(x, row_before, color=SERIES[k], label=f"B1 {float(scaling):.1f}")
    middle.plot(x, row_after, color=SERIES[k])
for axis, title in ((left, "starting sinc"), (middle, "optimized")):
    axis.set(title=title, xlabel="position (slice thicknesses)", ylim=(-0.02, 1.05))
left.set_ylabel(r"$|M_{xy}|$")
right.semilogy(result.loss.numpy(), color=MUTED)
right.set(title="cost", xlabel="iteration")
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Inside the slice, the flip now varies less across the transmit range, most of
# all where B1 is low. Outside the slice the leakage stays at the level of the
# sinc. Compare the mean :math:`|M_{xy}|` inside the slice at each B1:
centre = inside.bool()
for label, profile in (("starting sinc", before), ("optimized", after)):
    mean = profile[:, centre].mean(dim=1)
    print(f"{label:>14}: {[round(float(v), 3) for v in mean]}")

# %%
# Specification
# -------------
#
# What this example did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, design a 128-sample slice-selective 90 degree excitation
#    with time-bandwidth product 4, starting from a Hamming-windowed sinc.
#    Simulate each spin at B1 scalings 0.8 to 1.2 with compose_spinor. Minimize
#    the squared miss from |Mxy| = 1 inside the slice and 0 outside, plus a
#    small energy penalty, with SequenceDesign over the real and imaginary
#    parts of the samples. Report the mean |Mxy| in the slice at each B1,
#    before and after.
