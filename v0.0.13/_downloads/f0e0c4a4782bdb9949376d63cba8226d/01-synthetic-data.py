"""
=====================
Synthetic fingerprint
=====================

In this Application you build a training pair for MR fingerprinting: the
undersampled image series a scanner would produce, the fully sampled series it
approximates, and the T1, T2 and M0 maps they come from.

The pipeline segments a T1-weighted subject into grey matter, white matter and
CSF, assigns each class an M0, T1 and T2, simulates one fingerprint per class
with extended phase graphs, and mixes the fingerprints voxelwise by tissue
fraction. The volume is then weighted by birdcage coil sensitivities and
encoded with a frame-wise spiral NUFFT. The adjoint and a coil combination
return the image series. Only the simulation is blochsim's. torchio,
deepmriprep, SigPy and mri-nufft provide the subject, the segmentation, the
coils and the encoding.

Prerequisites: Course lessons :doc:`../01-framework/01-first-simulation`.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install blochsim matplotlib torchio deepmriprep sigpy cmap mri-nufft[finufft,cufinufft]
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# %%
# Setup
# -----
#
# Each package has one task here: torchio loads a T1-weighted IXI subject,
# deepmriprep segments it with a U-Net and returns tissue probabilities,
# ``sigpy.mri`` generates birdcage coil sensitivities, and mri-nufft provides the
# spiral trajectory and the non-uniform Fourier transform.
import tempfile
import time

import mrinufft
import numpy as np
import sigpy.mri as smri
import torch
import torchio as tio

from deepmriprep import Preprocess
from mrinufft.trajectories import initialize_2D_spiral
from pathlib import Path

from blochsim.simulators import MRFSimulator

# sphinx_gallery_start_ignore
import warnings

import matplotlib.pyplot as plt
from cmap import Colormap

from figure_style import PAGE_WIDTH, SERIES

warnings.filterwarnings("ignore")

NAVIA = Colormap("crameri:navia").to_matplotlib()
LIPARI = Colormap("crameri:lipari").to_matplotlib()
# Phase is cyclic, so its colormap is too.
PHASE = Colormap("colorcet:CET_C6").to_matplotlib()

# One colormap, window and label per parameter. The relaxation windows stop short
# of CSF so that white and grey matter fill the scale.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "T1 [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "T2 [ms]"),
    "M0": ("gray", (0.0, 1.0), "M0"),
}
BAR_WIDTH = 0.8
PANEL = (PAGE_WIDTH - BAR_WIDTH) / 5.5


def panel(axis, values, cmap, limits, title=None):
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title)
    return handle


def domain(axis, values, title=None):
    """A complex map with phase as colour and magnitude as opacity."""
    values = torch.as_tensor(values).cpu()
    rgba = PHASE(((values.angle() / (2 * np.pi)) + 0.5).numpy())
    magnitude = values.abs().numpy()
    rgba[..., -1] = magnitude / max(magnitude.max(), 1e-12)
    axis.imshow(rgba)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_box_aspect(1)
    if title is not None:
        axis.set_title(title)


def scalebar(handle, axes, label):
    axes = list(np.ravel(axes))
    axes[0].figure.colorbar(handle, ax=axes, label=label, shrink=0.92, aspect=20)


def canvas(rows, columns, shape, *, bars=1, extra=0.6):
    return plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(
            columns * PANEL + bars * BAR_WIDTH,
            PANEL * shape[0] / shape[1] * rows + extra,
        ),
    )


# sphinx_gallery_end_ignore

# %%
# The pair has a 128 x 128 matrix, 400 frames with one spiral arm of 768 samples
# each, and 8 receive channels.
SIZE = 128
FRAMES = 400
SAMPLES = 768
COILS = 8
SLICE = 110  # axial, at the level of the lateral ventricles

# The GPU transform is used when it is installed and usable, and the simulation
# follows it so that the images and the operator are on one device.
on_gpu = torch.cuda.is_available() and mrinufft.check_backend("cufinufft")
device = "cuda" if on_gpu else "cpu"
backend = "cufinufft" if on_gpu else "finufft"

# %%
# Subject
# -------
#
# The only measurement the pipeline starts from is one IXI subject loaded by
# torchio: a T1-weighted volume at 1.5 T. The segmentation reads it, and a table
# supplies the tissue properties that a single contrast cannot give.
# ``download=True`` fetches the archive once, a few hundred MB, into a cache under
# the home directory.
CACHE = Path.home() / ".cache" / "blochsim" / "ixi-tiny"
subject = tio.datasets.IXITiny(str(CACHE), download=True)[0]


# sphinx_gallery_start_ignore
def slab(image):
    """One axial slice of a volume, at the matrix this example works in."""
    values = np.asarray(image.numpy(), dtype=np.float32).squeeze()[:, :, SLICE].T
    grid = torch.as_tensor(np.flip(values, 0).copy())[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


# sphinx_gallery_end_ignore

# %%
# Tissue classes
# --------------
#
# deepmriprep segments the head into grey matter, white matter and CSF with a
# U-Net. It returns three probability maps, not one label per voxel, so the
# phantom contains partial volume.
NAMES = ("grey matter", "white matter", "CSF")

segmentation = Preprocess().run(
    str(subject.image.path),
    output_paths={
        name: f"{tempfile.gettempdir()}/{name}.nii.gz" for name in ("p1", "p2", "p3")
    },
    run_all=False,
)

# sphinx_gallery_start_ignore
# The probability maps carry the affine that places them in the subject's
# coordinates, so resampling by it aligns them. The pipeline then puts every
# volume in RAS on a 1 mm isotropic grid, so that one index is the same axial
# slice everywhere.
for key, name in zip(("p1", "p2", "p3"), NAMES, strict=False):
    subject.add_image(
        tio.ScalarImage(
            tensor=torch.as_tensor(segmentation[key].get_fdata())[None].float(),
            affine=segmentation[key].affine,
        ),
        name,
    )
subject = tio.Compose(
    [tio.ToCanonical(), tio.Resample("image"), tio.Resample(1), tio.CropOrPad(240)]
)(subject)

anatomy = slab(subject.image)
fractions = torch.stack([slab(subject[name]) for name in NAMES], dim=-1)
fractions = fractions.clamp(0.0, 1.0)
occupancy = fractions.sum(-1)
brain = occupancy > 0.5

mixed = int(((fractions.max(-1).values < 0.99) & brain).sum())
print(
    f"{SIZE}x{SIZE} slice, {int(brain.sum())} brain voxels; "
    f"{100 * mixed / int(brain.sum()):.0f}% are a mixture of two tissues or more"
)

figure, axes = canvas(1, 4, (SIZE, SIZE))
panel(axes[0, 0], anatomy.cpu(), "gray", (0, float(anatomy.max())), title="T1w")
for column, name in enumerate(("grey", "white", "CSF"), start=1):
    handle = panel(
        axes[0, column], fractions[..., column - 1].cpu(), "magma", (0, 1), title=name
    )
scalebar(handle, axes[0, 1:], "probability")
plt.show()
# sphinx_gallery_end_ignore

# %%
# The figure shows the T1-weighted contrast and the three probabilities.
#
# Each class takes an M0, a T1 and a T2 from a table, here values at 1.5 T. A
# T1-weighted volume is a contrast and not a map, so the table cannot be read
# from this subject: the measurement gives the location and fraction of each
# tissue, and the table gives its properties. A relaxometry protocol on the same
# subject would fill the table from data.
class_M0 = torch.tensor([0.80, 0.70, 1.00])  # relative proton density
class_T1 = torch.tensor([1100.0, 650.0, 4000.0])  # ms
class_T2 = torch.tensor([95.0, 70.0, 2000.0])  # ms

# sphinx_gallery_start_ignore
dominant = fractions > 0.5
print(f"\n{'tissue':<14} {'voxels':>8}   {'M0':>4} {'T1 (ms)':>8} {'T2 (ms)':>8}")
for k, name in enumerate(NAMES):
    print(
        f"{name:<14} {int(dominant[..., k].sum()):8d}   "
        f"{class_M0[k]:4.2f} {class_T1[k]:8.0f} {class_T2[k]:8.1f}"
    )
# sphinx_gallery_end_ignore

# %%
# Simulation per class
# --------------------
#
# The sequence is an inversion followed by 400 repetitions whose flip angle
# follows a sinusoidal schedule. :class:`~blochsim.simulators.MRFSimulator`
# accepts tensor-valued tissue properties, so the three classes are simulated in
# one call, as three EPG runs instead of one per voxel.
schedule = 5.0 + 55.0 * torch.sin(torch.linspace(0.0, 4 * torch.pi, FRAMES)).abs()
simulator = MRFSimulator(TR=12.0, TI=20.0, T1=class_T1, T2=class_T2)

# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
per_class = torch.as_tensor(simulator.simulate(flip=schedule))

# sphinx_gallery_start_ignore
print(
    f"{len(NAMES)} classes x {FRAMES} frames in "
    f"{time.perf_counter() - started:.1f}s -> {tuple(per_class.shape)}"
)
# sphinx_gallery_end_ignore

# %%
# Whole-brain volume
# ------------------
#
# A voxel that is part grey matter and part CSF produces the sum of the two
# fingerprints, weighted by the tissue fractions, and not the fingerprint of the
# averaged relaxation times. An inversion-prepared train is strongly nonlinear in
# T1, so averaging the parameters before simulating is wrong in every voxel that
# is not pure. The mixing is a matrix product over the class fingerprints:
weights = (fractions * class_M0).to(per_class.dtype)
series = (weights @ per_class) * brain[..., None]

# %%
# The ground-truth maps are the fraction-weighted averages of the class
# parameters, the best result a single-compartment fit of this data can return:
share = occupancy.clamp_min(1e-6)
truth_M0 = (fractions @ class_M0) * brain
truth_T1 = (fractions @ class_T1) / share * brain
truth_T2 = (fractions @ class_T2) / share * brain

# %%
# Coils and encoding
# ------------------
#
# The coil sensitivities are birdcage maps from SigPy. Each frame is encoded by
# one spiral arm, rotated by the golden angle from the previous frame, so
# consecutive frames sample different parts of k-space. One arm of 768 samples
# on a 128 x 128 matrix is about 21-fold undersampled, as in fingerprinting
# acquisitions.
sensitivities = torch.as_tensor(smri.birdcage_maps((COILS, SIZE, SIZE))).to(
    torch.complex64
)
trajectory = initialize_2D_spiral(
    FRAMES, SAMPLES, tilt="golden", nb_revolutions=8
).astype(np.float32)

build = mrinufft.get_operator(backend)
# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
arms = [
    build(
        trajectory[frame], (SIZE, SIZE), n_coils=COILS, squeeze_dims=False, density=True
    )
    for frame in range(FRAMES)
]
# sphinx_gallery_start_ignore
print(f"{FRAMES} arms built in {time.perf_counter() - started:.1f}s")
# sphinx_gallery_end_ignore

# %%
# The forward transform takes the coil-weighted image of each frame to k-space:
coil_series = (sensitivities[:, None] * series.movedim(-1, 0)[None]).to(device)

# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
kspace = torch.stack(
    [arms[frame].op(coil_series[:, frame][None])[0] for frame in range(FRAMES)]
)
# sphinx_gallery_start_ignore
print(f"forward NUFFT {time.perf_counter() - started:.1f}s -> {tuple(kspace.shape)}")

# The trajectory is a plot, not a map, so it takes a panel and a half.
figure = plt.figure(figsize=(5.5 * PANEL + BAR_WIDTH, PANEL + 1.05))
grid = figure.add_gridspec(1, 6, width_ratios=(1, 1, 1, 1, BAR_WIDTH / PANEL, 1.5))
for index in range(4):
    domain(
        figure.add_subplot(grid[0, index]), sensitivities[index], title=f"coil {index}"
    )
bar = figure.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), cmap=PHASE),
    cax=figure.add_subplot(grid[0, 4]),
)
bar.set_label("phase [rad]")
bar.set_ticks([-np.pi, 0.0, np.pi], labels=["$-\\pi$", "0", "$\\pi$"])

axis = figure.add_subplot(grid[0, 5])
for shade, frame in enumerate((0, FRAMES // 2, FRAMES - 1)):
    axis.plot(
        trajectory[frame, :, 0],
        trajectory[frame, :, 1],
        lw=0.5,
        color=SERIES[shade],
    )
axis.set(xlabel="$k_x$", ylabel="$k_y$", title=f"3 of {FRAMES} arms")
axis.set_box_aspect(1)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The figure shows four of the eight coil sensitivities and three of the arms.
#
# Reconstruction
# --------------
#
# Each frame is reconstructed by the adjoint transform, and the channels are
# combined with the sensitivity maps. The maps are known here; a real pipeline
# estimates them.

# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
folded = torch.stack(
    [arms[frame].adj_op(kspace[frame][None])[0] for frame in range(FRAMES)]
)
weights = sensitivities.to(device)
combined = (folded * weights.conj()[None]).sum(1) / (
    weights.abs().square().sum(0)[None] + 1e-6
)
# sphinx_gallery_start_ignore
print(f"adjoint and combine {time.perf_counter() - started:.1f}s")
# sphinx_gallery_end_ignore

# %%
# Data pair
# ---------
#
# The training input is the undersampled series ``combined`` and the target is the
# fully sampled series ``series``. Each frame contains one spiral arm, so
# aliasing in a single frame exceeds the signal, but the time course of each
# voxel retains the fingerprint, which is what a fingerprinting reconstruction
# uses. Both series are normalised to their peak magnitude.

# sphinx_gallery_start_ignore
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

figure, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 0.55 * PAGE_WIDTH))
for column, frame in enumerate((0, FRAMES // 2)):
    axes[0, column].imshow(reference[frame].abs().cpu(), cmap="gray")
    axes[1, column].imshow(undersampled[frame].abs().cpu(), cmap="gray")
    axes[0, column].set_title(f"frame {frame}")
for axis in axes[:, :2].ravel():
    axis.set_xticks([]), axis.set_yticks([])
    for side in axis.spines.values():
        side.set_visible(False)

voxel = torch.nonzero(here)[int(here.sum()) // 2]
row, column = int(voxel[0]), int(voxel[1])
axes[0, 2].plot(reference[:, row, column].abs().cpu(), lw=1.0, color=SERIES[0])
axes[0, 2].set_title("one voxel")
axes[1, 2].plot(undersampled[:, row, column].abs().cpu(), lw=0.7, color=SERIES[1])
for axis, name in ((axes[0, 0], "fully sampled"), (axes[1, 0], "one spiral arm")):
    axis.set_ylabel(name)
for axis in axes[:, 2]:
    axis.set_xlabel("frame")
for axis in axes.ravel():
    axis.set_box_aspect(1)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The printed error is the mean per-frame error in the brain. The time-course
# agreement is the normalised inner product between the undersampled and the
# fully sampled time course of each voxel.
#
# Ground-truth maps
# -----------------

# sphinx_gallery_start_ignore
figure, axes = canvas(1, 3, (SIZE, SIZE), bars=3)
for axis, name, values in (
    (axes[0, 0], "T1", truth_T1),
    (axes[0, 1], "T2", truth_T2),
    (axes[0, 2], "M0", truth_M0),
):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values.cpu(), cmap, limits), axis, label)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The T1 and T2 maps use perceptually uniform colormaps, one per relaxation
# parameter, so that a T1 map is never read as a T2 map [1]_.
#
# Export
# ------
#
# The archive holds the pair, the ground-truth maps, the segmentation, and the
# schedule and trajectory that produced them. It is written to a temporary
# directory so that a documentation build leaves no file behind.
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
# Command-line interface
# ----------------------
#
# To turn the script into a tool, replace three inputs and keep the rest:
#
# - the torchio subject by a T1-weighted NIfTI, which is the contrast the
#   segmentation was trained on;
# - the generated spiral and sine schedule by a trajectory and a flip-angle
#   schedule read from a matfile;
# - the temporary directory by an output path.
#
# The tissue table is the one input that is chosen and not read.
#
# References
# ----------
#
# .. [1] Fuderer M, et al. Color-map recommendation for MR relaxometry maps.
#        Magn Reson Med 93(2):490-506 (2025).
#
# Specification
# -------------
#
# What this Application did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Build a synthetic MR fingerprinting training pair. Load an IXI subject with
#    torchio and segment it into grey matter, white matter and CSF with
#    deepmriprep. Give each class M0, T1 and T2 from a table at 1.5 T and
#    simulate one inversion-prepared MRFSimulator train of 400 frames per class.
#    Mix the fingerprints by tissue fraction, weight by 8 birdcage coils, encode
#    with one golden-angle spiral arm per frame (mri-nufft), then reconstruct
#    each frame by the adjoint and a coil combination. Report the per-frame error
#    and the time-course agreement, and save the pair with the maps and schedule.
