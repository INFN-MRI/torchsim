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
torchio, deepmriprep, SigPy and mri-nufft, and the pipeline is mostly a matter
of handing each of them the right array.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install torchsim torchio deepmriprep sigpy mri-nufft[finufft]

# %%
#
# Only one step of this pipeline is TorchSim's. Four other packages do the
# rest, and each has exactly one job here:
#
# * ``torchio`` fetches an IXI subject and gives it as tensors, which is where
#   the anatomy comes from;
# * ``deepmriprep`` segments that anatomy into grey matter, white matter and
#   CSF with a U-Net -- torch throughout, and it hands back probabilities
#   rather than labels;
# * ``sigpy.mri`` generates birdcage coil sensitivities to weight the volume
#   with;
# * ``mri-nufft`` supplies the spiral trajectory and the non-uniform Fourier
#   transform that plays it, forwards and back.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# sphinx_gallery_end_ignore
import mrinufft
from deepmriprep import Preprocess
import sigpy.mri as smri
import torchio as tio
from mrinufft.trajectories import initialize_2D_spiral

# %%
#
# TorchSim's part is the third step: one fingerprinting simulation per tissue
# class, rather than one per voxel.
#
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from torchsim.simulators import MRFSimulator


# %%
#
# What the pair will be: a 128 matrix, four hundred frames, one spiral arm of
# 768 samples per frame, and eight receive channels.
#
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
# One IXI subject, through torchio: proton-density and T2-weighted, acquired
# at 1.5 T.
#
# The proton-density image is what the phantom's M0 comes from. The relaxation
# times are tabulated, which is the split a digital twin usually lives with:
# what the data says, and what a table says.
#
subject = tio.datasets.IXI(
    str(Path("data").resolve()), modalities=("PD", "T2"), download=False
)[0]


def slab(volume):
    """One axial slice of a volume, at the matrix this example works in."""
    values = np.asarray(volume, dtype=np.float32).squeeze()[:, :, SLICE].T
    grid = torch.as_tensor(np.flip(values).copy())[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


density, weighted = slab(subject.PD.numpy()), slab(subject.T2.numpy())

# %%
#
# Tissue classes
# --------------
#
# ``deepmriprep`` segments the head into grey matter, white matter and CSF with
# a U-Net, in torch and on the same card everything else here runs on. It is
# trained on T1-weighted data and is being handed proton density, which is not
# what it was built for and is visible in the CSF map -- a tool would give it
# the T1-weighted volume of the same subject.
#
# What comes back is what matters: three *probability* maps rather than one
# label per voxel. A brain at this resolution is full of voxels that are part
# one tissue and part another, and a segmentation that says so is the
# difference between a phantom with partial volume in it and one without.
#
segmentation = Preprocess().run(
    str(subject.PD.path),
    output_paths={name: f"{tempfile.gettempdir()}/{name}.nii.gz"
                  for name in ("p1", "p2", "p3")},
    run_all=False,
)

NAMES = ("grey matter", "white matter", "CSF")
fractions = torch.stack(
    [slab(segmentation[key].get_fdata()) for key in ("p1", "p2", "p3")], dim=-1
).clamp(0.0, 1.0)
occupancy = fractions.sum(-1)
brain = occupancy > 0.5

mixed = int(((fractions.max(-1).values < 0.99) & brain).sum())
print(f"{SIZE}x{SIZE} slice, {int(brain.sum())} brain voxels")
print(f"{100 * mixed / int(brain.sum()):.0f}% of them are a mixture of two "
      f"tissues or more")

# %%
#
# The two contrasts the subject arrived as, and the three probabilities the
# network made of them:
#

# sphinx_gallery_start_ignore
def panel(axis, values, cmap, limits, label=None):
    """One map, with a colorbar every panel loses the same width to."""
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    bar = axis.figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
    if label is None:
        bar.ax.set_visible(False)
    else:
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


figure, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for axis, values, title in (
    (axes[0], density, "proton density"),
    (axes[1], weighted, "T2-weighted"),
):
    panel(axis, values.cpu(), "gray", (0, float(values.max())))
    axis.set_title(title, fontsize=10)
for column, name in enumerate(NAMES, start=2):
    panel(axes[column], fractions[..., column - 2].cpu(), "magma", (0, 1),
          label="probability" if column == 4 else None)
    axes[column].set_title(name, fontsize=10)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# Each class is now given the three numbers a simulation needs. M0 is read off
# the subject, because a proton-density image is what M0 is; T1 and T2 are
# tabulated at 1.5 T.
#
# The relaxation times could not be read off this subject even though a T2
# weighting is sitting right there. A PD and T2 pair measures T2 only where T2
# is comparable to the echo time that separated them -- true of parenchyma, and
# badly false of CSF, whose T2 is an order of magnitude longer than the echo
# time and so barely changes the ratio. A tool with a real relaxometry protocol
# behind it would fill this table from the data; this one is honest about which
# half it has.
#
NAMES_T1 = torch.tensor([1100.0, 650.0, 4000.0])  # ms, at 1.5 T
NAMES_T2 = torch.tensor([95.0, 70.0, 2000.0])  # ms, at 1.5 T

dominant = fractions > 0.5
normalized = density / density.max()
class_T1 = NAMES_T1
class_T2 = NAMES_T2
class_M0 = torch.tensor(
    [float(normalized[dominant[..., k]].median()) for k in range(len(NAMES))]
)

print(f"\n{'tissue':<14} {'voxels':>8}   {'M0':>4} {'T1 (ms)':>8} {'T2 (ms)':>8}")
for k, name in enumerate(NAMES):
    print(f"{name:<14} {int(dominant[..., k].sum()):8d}   "
          f"{class_M0[k]:4.2f} {class_T1[k]:8.0f} {class_T2[k]:8.1f}")

# %%
#
# One simulation per class
# ------------------------
#
# The fingerprinting train: an inversion, then four hundred repetitions whose
# flip angle sweeps. :class:`~torchsim.simulators.MRFSimulator` takes arrays of
# tissue properties, so the whole tissue table is one call -- **three extended
# phase graph runs, not sixteen thousand.** That is the saving a segmented
# phantom exists for, and it is why a thousand-frame train over a whole volume
# is a few seconds rather than an afternoon.
#
schedule = 5.0 + 55.0 * torch.sin(torch.linspace(0.0, 4 * torch.pi, FRAMES)).abs()
acquisition = MRFSimulator(TR=12.0, TI=20.0, T1=class_T1, T2=class_T2)

started = time.perf_counter()
per_class = torch.as_tensor(acquisition.simulate(flip=schedule))
print(
    f"{len(NAMES)} classes x {FRAMES} frames in "
    f"{time.perf_counter() - started:.1f}s -> {tuple(per_class.shape)}"
)

# %%
#
# The whole brain
# ---------------
#
# A voxel that is part grey matter and part CSF produces the **sum of what the
# two do**, weighted by how much of each is there -- never the signal of their
# averaged relaxation times. So the mixing happens on the signals, which is one
# matrix product against the per-class evolutions and is the whole of step
# four.
#
# Averaging the parameters and simulating once is the tempting mistake, and it
# is wrong wherever a voxel is not pure: an inversion-prepared train is
# markedly nonlinear in T1.
#
weights = (fractions * class_M0).to(per_class.dtype)
series = (weights @ per_class) * brain[..., None]
print(f"whole-brain series {tuple(series.shape)} {series.dtype}")

# The maps that go out as ground truth are the mixture averages, which is what
# a single-compartment fit of this data could return at best.
share = occupancy.clamp_min(1e-6)
truth_M0 = (fractions @ class_M0) * brain
truth_T1 = (fractions @ class_T1) / share * brain
truth_T2 = (fractions @ class_T2) / share * brain

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
# What the encoding is: eight birdcage sensitivities, and one spiral arm per
# frame rotated by the golden angle so that consecutive frames sample different
# parts of k-space.
#

# sphinx_gallery_start_ignore
figure = plt.figure(figsize=(12, 3.8))
grid = figure.add_gridspec(1, 5, width_ratios=(1, 1, 1, 1, 1.4))
for index in range(4):
    axis = figure.add_subplot(grid[0, index])
    handle = axis.imshow(sensitivities[index].abs().cpu(), cmap="magma")
    bar = figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
    if index == 3:
        bar.set_label("|sensitivity|", fontsize=8)
        bar.ax.tick_params(labelsize=7)
    else:
        bar.ax.set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_box_aspect(1)
    axis.set_title(f"coil {index}", fontsize=9)

axis = figure.add_subplot(grid[0, 4])
for frame in (0, FRAMES // 2, FRAMES - 1):
    axis.plot(
        trajectory[frame, :, 0],
        trajectory[frame, :, 1],
        lw=0.4,
        color=plt.cm.plasma(frame / (FRAMES - 1)),
        label=f"frame {frame}",
    )
axis.set(
    xlabel="$k_x$", ylabel="$k_y$", title=f"1 of {FRAMES} arms, 3 shown"
)
axis.set_box_aspect(1)
axis.legend(fontsize=7)
figure.tight_layout()
# sphinx_gallery_end_ignore

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

# sphinx_gallery_start_ignore
def panel(axis, values, cmap, limits, label=None):
    """One map, with a colorbar every panel loses the same width to."""
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    bar = axis.figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
    if label is None:
        bar.ax.set_visible(False)
    else:
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    axis.set_xticks([])
    axis.set_yticks([])


figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for axis, (title, values, limits) in zip(axes, (
    ("M0", truth_M0, (0, 1.1)),
    ("T1 [ms]", truth_T1, (0, 4200)),
    ("T2 [ms]", truth_T2, (0, 2100)),
)):
    panel(axis, values.cpu(), "viridis", limits, label=title)
    axis.set_title(title, fontsize=10)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
#
# ...and what the pair looks like
# -------------------------------
#

# sphinx_gallery_start_ignore
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
# An image keeps its own aspect and a line plot fills whatever it is given, so
# a row of both ends up with panels of different heights unless the box each
# one is drawn into is fixed.
for panel in axes.ravel():
    panel.set_box_aspect(1)
plt.tight_layout()
# sphinx_gallery_end_ignore

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
    "tissue_probabilities": fractions.numpy(),
    "flip_angles_deg": schedule.numpy(),
    "trajectory": trajectory,
}

# sphinx_gallery_start_ignore
with tempfile.TemporaryDirectory() as folder:
    archive = Path(folder) / "synthetic_mrf.npz"
    np.savez_compressed(archive, **contents)
    reloaded = np.load(archive)
    print(f"{archive.name}  {archive.stat().st_size / 2**20:.1f} MiB")
    for name in contents:
        array = reloaded[name]
        print(f"  {name:<16} {str(array.shape):<20} {array.dtype}")
# sphinx_gallery_end_ignore

# %%
#
# Toward a command-line tool
# --------------------------
#
# Everything above is fixed except its inputs, and there are three:
#
# * **a T1-weighted NIfTI**, which replaces the torchio subject and is what
#   the segmentation was trained on, so the CSF map stops being the weak one;
# * **a matfile** carrying the trajectory and the flip-angle schedule, read
#   instead of the spiral and the sine generated here;
# * **an output path**, replacing the temporary directory.
#
# The tissue table is the one thing that needs deciding rather than reading. A
# real relaxometry protocol on the same subject would fill it from the data --
# which is what the parameter-inference examples do, and where a pipeline like
# this one would get its numbers.
