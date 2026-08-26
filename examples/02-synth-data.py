"""
=============================
Synthetic MR fingerprinting
=============================

Training a reconstruction needs pairs: what the scanner would measure, and
what the answer is. Neither is available together from a real scan, so both
are made.

This example builds one, end to end. A subject is segmented into tissue
classes; each class is given an M0, a T1 and a T2; **one voxel per class** is
simulated by extended phase graphs; every voxel of a class is handed its
class's signal evolution; the volume is weighted by birdcage coil
sensitivities and pushed through a frame-wise non-uniform Fourier transform;
and the k-space that comes out is brought back and coil-combined. The
undersampled series and the fully sampled one it came from are the pair. The
ground-truth maps and the segmentation go with them.

Only the third step is TorchSim's. The phantom, the coils and the encoding are
torchio, SimpleITK, SigPy and mri-nufft, and the pipeline is mostly a matter
of handing each of them the right array.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install torchsim torchio SimpleITK sigpy mri-nufft[finufft]

# %%
#
# The imports:
#
import warnings

warnings.filterwarnings("ignore")

import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mrinufft
import numpy as np
import SimpleITK as sitk
import sigpy.mri as smri
import torch
import torchio as tio
from mrinufft.trajectories import initialize_2D_spiral

from torchsim import Acquisition
from torchsim.simulators import MRFSimulator

SIZE = 128
FRAMES = 400
SAMPLES = 768
COILS = 8
SLICE = 60

device = "cuda" if torch.cuda.is_available() else "cpu"
backend = "cufinufft" if device == "cuda" else "finufft"

# %%
#
# A subject
# ---------
#
# One IXI subject, through torchio. What arrives here is proton-density and
# T2-weighted, so that is what gets segmented; the step below does not care
# which contrast it is handed, and a tool built on this would hand it the
# T1-weighted volume instead.
#
# A quantitative T2 map is derived from the two contrasts, and is used only to
# say which tissue each label turned out to be.
#
subject = tio.datasets.IXI(
    os.path.realpath("data"), modalities=("PD", "T2"), download=False
)[0]


def slab(image):
    """One axial slice, at the matrix this example works in."""
    values = image.numpy().astype(np.float32).squeeze()[:, :, SLICE].T
    grid = torch.as_tensor(np.flip(values).copy())[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


density, weighted = slab(subject.PD), slab(subject.T2)
brain = density > 0.10 * density.max()
measured_T2 = torch.nan_to_num(
    -92.0 / torch.log(weighted / density), nan=0.0, posinf=0.0, neginf=0.0
).clamp(0.0, 400.0)
print(f"{SIZE}x{SIZE} slice, {int(brain.sum())} brain voxels")

# %%
#
# Tissue classes
# --------------
#
# Otsu's multi-level threshold, three thresholds, over the brain. Nothing here
# is told what a tissue is -- the classes come out of the histogram, and which
# one is which is read afterwards from the T2 they turned out to have.
#
inside = torch.where(brain, weighted, torch.zeros(())).numpy()
labels = torch.as_tensor(
    sitk.GetArrayFromImage(
        sitk.OtsuMultipleThresholds(sitk.GetImageFromArray(inside), 3, 0, 128, False)
    ).astype(np.int64)
)
labels = torch.where(brain, labels, torch.full_like(labels, -1))
CLASSES = int(labels.max()) + 1

# %%
#
# Each class is now given the three numbers a simulation needs. M0 and T2 are
# the class's own median, so they come from the subject; T1 is tabulated,
# because nothing in a PD and T2 pair measures it. That is the split a digital
# twin usually lives with: what the data says, and what a table says.
#
NAMES = ("other", "white matter", "grey matter", "CSF")
TABULATED_T1 = torch.tensor([900.0, 650.0, 1200.0, 4000.0])  # ms, at 1.5 T

median = lambda values, chosen: float(values[chosen].median())
class_T2 = torch.tensor([median(measured_T2, labels == k) for k in range(CLASSES)])
class_M0 = torch.tensor(
    [median(density / density.max(), labels == k) for k in range(CLASSES)]
)
class_T1 = TABULATED_T1[:CLASSES]

print(f"{'':2} {'tissue':<13} {'voxels':>7}   {'M0':>4} {'T1 (ms)':>8} {'T2 (ms)':>8}")
for k in range(CLASSES):
    print(
        f"{k:2d} {NAMES[k]:<13} {int((labels == k).sum()):7d}   "
        f"{class_M0[k]:4.2f} {class_T1[k]:8.0f} {class_T2[k]:8.1f}"
    )

# %%
#
# T2 rises with the label, which is the ordering that lets the classes be
# named at all. If it did not on some other subject, the names above would be
# wrong and the table with them -- so it is printed rather than assumed.
#
# %%
#
# One simulation per class
# ------------------------
#
# The fingerprinting train: an inversion, then four hundred repetitions whose
# flip angle sweeps. :class:`~torchsim.simulators.MRFSimulator` takes arrays of
# tissue properties, so the whole tissue table is one call -- **four extended
# phase graph runs, not sixty-five thousand.** That is the saving a segmented
# phantom exists for, and it is why a thousand-frame train over a whole volume
# is a few seconds rather than an afternoon.
#
schedule = 5.0 + 55.0 * torch.sin(torch.linspace(0.0, 4 * torch.pi, FRAMES)).abs()
acquisition = Acquisition(MRFSimulator(TR=12.0, TI=20.0), T1=class_T1, T2=class_T2)

started = time.perf_counter()
per_class = torch.as_tensor(acquisition.simulate(flip=schedule))
print(
    f"{CLASSES} classes x {FRAMES} frames in "
    f"{time.perf_counter() - started:.1f}s -> {tuple(per_class.shape)}"
)

# %%
#
# The whole brain
# ---------------
#
# Every voxel of a class gets that class's evolution, scaled by the class's
# M0. That is an indexing operation, and it is the whole of step four.
#
# It is also an assumption worth naming: a hard label makes every voxel of a
# tissue carry exactly the same curve, so this phantom has no partial volume
# in it at all. A fuzzy segmentation would fix that, and the fix is to mix the
# **signals** -- a voxel that is half one tissue and half another produces the
# sum of what the two do, never the signal of their averaged relaxation times.
# Averaging the parameters and simulating once is the tempting mistake.
#
picked = labels.clamp_min(0)
series = per_class[picked] * class_M0[picked][..., None] * brain[..., None]
print(f"whole-brain series {tuple(series.shape)} {series.dtype}")

truth_M0 = class_M0[picked] * brain
truth_T1 = class_T1[picked] * brain
truth_T2 = class_T2[picked] * brain

# %%
#
# Coils, and the encoding
# -----------------------
#
# Birdcage sensitivities from SigPy, then one spiral arm per frame, rotated by
# the golden angle. A single arm of 768 samples against a 128 x 128 matrix is
# twenty-one-fold undersampled -- which is not an approximation of MRF, it is
# how MRF is run.
#
sensitivities = torch.as_tensor(smri.birdcage_maps((COILS, SIZE, SIZE))).to(
    torch.complex64
)
trajectory = initialize_2D_spiral(
    FRAMES, SAMPLES, tilt="golden", nb_revolutions=8
).astype(np.float32)

build = mrinufft.get_operator(backend)
started = time.perf_counter()
arms = [
    build(trajectory[frame], (SIZE, SIZE), n_coils=COILS, squeeze_dims=False, density=True)
    for frame in range(FRAMES)
]
print(f"{FRAMES} arms built in {time.perf_counter() - started:.1f}s")

coil_series = (
    sensitivities[:, None] * series.movedim(-1, 0)[None]
).to(device)

started = time.perf_counter()
kspace = torch.stack(
    [arms[frame].op(coil_series[:, frame][None])[0] for frame in range(FRAMES)]
)
print(
    f"forward NUFFT {time.perf_counter() - started:.1f}s -> {tuple(kspace.shape)}"
)

# %%
#
# Back again
# ----------
#
# Adjoint per frame, then a sensitivity-weighted coil combination -- which is
# available here because the maps are known, this being a phantom. A real
# pipeline would estimate them.
#
started = time.perf_counter()
folded = torch.stack(
    [arms[frame].adj_op(kspace[frame][None])[0] for frame in range(FRAMES)]
)
weights = sensitivities.to(device)
combined = (folded * weights.conj()[None]).sum(1) / (
    weights.abs().square().sum(0)[None] + 1e-6
)
print(f"adjoint and combine {time.perf_counter() - started:.1f}s")

# %%
#
# The pair
# --------
#
# One frame of this is a mess, and that is not a failure. Each frame is one
# spiral arm, so the aliasing is worse than the signal; what survives is the
# **time course**, and that is what a fingerprinting reconstruction reads.
#
reference = (series.movedim(-1, 0) * brain).to(device)
reference = reference / reference.abs().max()
undersampled = combined / combined.abs().max()
here = brain.to(device)

frame_error = float(
    (undersampled[:, here] - reference[:, here]).abs().mean()
    / reference[:, here].abs().mean()
)
courses = undersampled[:, here].movedim(0, -1)
truth_courses = reference[:, here].movedim(0, -1)
courses = courses / courses.norm(dim=-1, keepdim=True)
truth_courses = truth_courses / truth_courses.norm(dim=-1, keepdim=True)
agreement = (courses.conj() * truth_courses).sum(-1).abs()

print(f"per-frame error inside the brain : {100 * frame_error:5.1f}%")
print(
    f"time-course agreement            : median {float(agreement.median()):.3f}, "
    f"tenth percentile {float(agreement.quantile(0.1)):.3f}"
)

# %%
#
# Fifty percent wrong frame by frame, and the fingerprints still line up above
# 0.86 for nine voxels in ten. That gap is the whole premise of the method,
# and it is why the pair is worth training on: the input is what the scanner
# gives, artefacts and all, and the target is the curve underneath it.
#
# %%
#
# What the phantom is
# -------------------
#
figure, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for panel, (title, values, unit) in zip(
    axes,
    (
        ("tissue class", labels.float(), None),
        ("M0", truth_M0, None),
        ("T1", truth_T1, "ms"),
        ("T2", truth_T2, "ms"),
    ),
):
    shown = panel.imshow(values.cpu(), cmap="viridis")
    panel.set_title(title if unit is None else f"{title} ({unit})")
    panel.axis("off")
    figure.colorbar(shown, ax=panel, shrink=0.8)
plt.show()

# %%
#
# ...and what the pair looks like
# -------------------------------
#
figure, axes = plt.subplots(2, 3, figsize=(12, 7))
for column, frame in enumerate((0, FRAMES // 2)):
    axes[0, column].imshow(reference[frame].abs().cpu(), cmap="gray")
    axes[0, column].set_title(f"fully sampled, frame {frame}")
    axes[1, column].imshow(undersampled[frame].abs().cpu(), cmap="gray")
    axes[1, column].set_title(f"one spiral arm, frame {frame}")
for panel in axes[:, :2].ravel():
    panel.axis("off")

voxel = torch.nonzero(here)[int(here.sum()) // 2]
row, column = int(voxel[0]), int(voxel[1])
axes[0, 2].plot(reference[:, row, column].abs().cpu(), lw=1.0)
axes[0, 2].set_title("the target time course")
axes[1, 2].plot(undersampled[:, row, column].abs().cpu(), lw=0.7)
axes[1, 2].set_title("what one voxel actually measures")
for panel in axes[:, 2]:
    panel.set_xlabel("frame")
plt.tight_layout()
plt.show()

# %%
#
# Exporting
# ---------
#
# The pair, the ground truth, the segmentation, and the schedule and
# trajectory that produced them -- everything needed to reproduce the input or
# to score a reconstruction against it.
#
# It goes to a temporary directory here, because a documentation build should
# not leave an archive behind. A tool would write where it was told to.
#
contents = {
    "undersampled": undersampled.cpu().numpy(),
    "reference": reference.cpu().numpy(),
    "M0": truth_M0.numpy(),
    "T1": truth_T1.numpy(),
    "T2": truth_T2.numpy(),
    "segmentation": labels.numpy().astype(np.int16),
    "flip_angles_deg": schedule.numpy(),
    "trajectory": trajectory,
}
with tempfile.TemporaryDirectory() as folder:
    archive = Path(folder) / "synthetic_mrf.npz"
    np.savez_compressed(archive, **contents)
    reloaded = np.load(archive)
    print(f"{archive.name}  {archive.stat().st_size / 2**20:.1f} MiB")
    for name in contents:
        array = reloaded[name]
        print(f"  {name:<16} {str(array.shape):<20} {array.dtype}")

# %%
#
# Toward a command-line tool
# --------------------------
#
# Everything above is fixed except its inputs, and there are three:
#
# * **a T1-weighted NIfTI**, which replaces the torchio subject and goes
#   straight into the same segmentation -- Otsu is given intensities and does
#   not know what produced them;
# * **a matfile** carrying the trajectory and the flip-angle schedule, read
#   instead of the spiral and the sine generated here;
# * **an output path**, replacing the temporary directory.
#
# The tissue table is the one thing that needs deciding rather than reading:
# with a T1-weighted input, T1 is the measurable and T2 becomes the tabulated
# one, the opposite way round from this example.
