"""
=================
Parameter Fitting
=================

This example shows how to use TorchSim to perform parameter inference.

We will build on the previous example: we synthesize an FSE echo series from
realistic maps, then recover T2 from it with two complementary estimators
shipped with TorchSim:

* :class:`torchsim.DictionaryMatcher`, the familiar exhaustive search;
* :class:`torchsim.PERK`, a kernel-regression estimator whose inference cost
  and memory footprint do not grow with the size of the parameter grid.

Both consume signals produced by the very same simulator, so the forward model
is written once and reused for training, matching, and validation.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim torchio sigpy

# %%
#
# We'll generate an FSE dataset from IXI database.
# We will neglect encoding and assume single coil for this case.
#
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import torchio as tio
import torchsim

path = os.path.realpath("data")
ixi_dataset = tio.datasets.IXI(
    path,
    modalities=("PD", "T2"),
    download=False,
)

# get subject 0
sample_subject = ixi_dataset[0]

M0 = sample_subject.PD.numpy().astype(np.float32).squeeze()[:, :, 60].T
T2w = sample_subject.T2.numpy().astype(np.float32).squeeze()[:, :, 60].T

T2 = -92.0 / np.log(T2w / M0)
T2 = np.nan_to_num(T2, neginf=0.0, posinf=0.0)
T2 = np.clip(T2, a_min=0.0, a_max=np.inf)

M0 = np.flip(M0)
T2 = np.flip(T2)

# %%
#
# The maps contain air voxels with ``T2 == 0``. TorchSim maps a non-positive
# relaxation time to a fully dephased signal, so those voxels simply produce
# zero rather than a NaN that would spread through the reconstruction.
#
device = "cuda" if torch.cuda.is_available() else "cpu"
flip = 180.0 * np.ones(32, dtype=np.float32)
ESP = 5.0
T1 = 1000.0


def simulate(T2, flip, ESP, device="cpu"):
    ishape = T2.shape
    output = torchsim.fse_sim(
        flip=flip, ESP=ESP, T1=T1, T2=T2.flatten(), device=device
    )

    return abs(output.T.reshape(-1, *ishape)).numpy(force=True)


# simulate acquisition
echo_series = M0 * simulate(T2, flip.copy(), ESP, device=device)

# %%
#
# Measured data are noisy, and an estimator is only interesting insofar as it
# tolerates that noise. We add Gaussian noise at roughly 2% of the peak signal:
#
rng = np.random.default_rng(42)
noise_level = 0.02 * echo_series.max()
echo_series = echo_series + noise_level * rng.standard_normal(echo_series.shape)
echo_series = echo_series.astype(np.float32)

# display
img = np.concatenate((echo_series[0], echo_series[16], echo_series[-1]), axis=1)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(abs(img), cmap="gray"), plt.axis("image"), plt.axis("off")
plt.title("echoes 1, 17 and 32")

# %%
#
# Both estimators need the same thing: a batch of signals simulated over a
# grid of candidate T2 values. We express that once, as a plain function of a
# parameter matrix shaped ``(natoms, nparams)``. This is exactly the signature
# :meth:`torchsim.PERK.fit_simulator` expects.
#


def signal_model(parameters, known=None):
    """Simulate magnitude FSE signals for a batch of T2 values."""
    return abs(
        torchsim.fse_sim(
            flip=flip,
            ESP=ESP,
            T1=T1,
            T2=parameters[:, 0],
            device=device,
        )
    )


# %%
#
# Dictionary matching
# -------------------
#
# The reference approach evaluates every atom and keeps the best normalized
# inner product. :class:`torchsim.DictionaryMatcher` chunks the score matrix so
# memory stays bounded, while each chunk is a BLAS/cuBLAS matrix product.
#
t2_grid = torch.linspace(1.0, 350.0, 1000, device=device)[:, None]
dictionary = signal_model(t2_grid)

matcher = torchsim.DictionaryMatcher(dictionary, t2_grid).to(device)

# %%
#
# The measured series is shaped ``(nechoes, ny, nx)``; the estimators expect
# the contrast axis last.
#
signals = torch.as_tensor(echo_series, device=device).permute(1, 2, 0)

match = matcher.match(signals)
T2_dict = match.parameters[..., 0, 0].numpy(force=True)
M0_dict = abs(match.scales[..., 0]).numpy(force=True)

# %%
#
# PERK
# ----
#
# PERK regresses the parameter directly from a random Fourier feature map of
# the signal. Training draws T2 from a prior, simulates the corresponding
# signals, and adds noise so that the estimator learns the *noisy* inverse
# mapping. ``normalize=True`` makes the estimate invariant to the unknown
# proton density.
#
generator = torch.Generator(device=device).manual_seed(11)
t2_train = 1.0 + 349.0 * torch.rand(20000, 1, generator=generator, device=device)

estimator = torchsim.PERK(
    n_features=1000,
    regularization=1e-6,
    complex_mode="magnitude",
    normalize=True,
    seed=4,
).to(device)
estimator.fit_simulator(
    signal_model,
    t2_train,
    simulation_chunk_size=4096,
    noise_std=0.02,
)

# %%
#
# Inference is now a fixed-cost feed-forward pass, independent of how finely
# the training prior was sampled:
#
T2_perk = estimator(signals)[..., 0].clamp(0.0, 350.0).numpy(force=True)

# %%
#
# PERK returns T2 only. The matching proton density follows from a single
# batched simulation at the estimated T2 and a least-squares projection onto
# the measured signal, which reuses the forward model a third time:
#
atoms_perk = signal_model(
    torch.as_tensor(T2_perk.reshape(-1, 1), device=device),
)
measured = signals.reshape(-1, len(flip))
M0_perk = (
    (measured * atoms_perk).sum(-1) / atoms_perk.square().sum(-1).clamp_min(1e-12)
).reshape(T2_perk.shape).numpy(force=True)

# %%
#
# Comparison
# ----------
#
# We restrict the display to voxels with signal, since T2 is not identifiable
# in air:
#
mask = M0 > 0.05 * M0.max()


def masked(x):
    return np.where(mask, x, np.nan)


# The reference T2 comes from a two-point log ratio, which is itself unstable:
# a few voxels (mostly CSF and partial-volume edges) land far outside the
# 1-350 ms range the estimators can represent. Quantitative comparison is only
# meaningful where the reference is inside that range.
valid = mask & (T2 > 1.0) & (T2 < 350.0)


figure, axes = plt.subplots(2, 3, figsize=(9.5, 6))
for ax, (data, title) in zip(
    axes[0],
    [(T2, "true T2 [ms]"), (T2_dict, "dictionary T2"), (T2_perk, "PERK T2")],
):
    handle = ax.imshow(masked(data), vmin=0.0, vmax=350.0, cmap="viridis")
    ax.set_title(title), ax.axis("off")
    figure.colorbar(handle, ax=ax, fraction=0.046)

for ax, (data, title) in zip(
    axes[1],
    [(M0, "true M0"), (M0_dict, "dictionary M0"), (M0_perk, "PERK M0")],
):
    handle = ax.imshow(masked(data), cmap="gray")
    ax.set_title(title), ax.axis("off")
    figure.colorbar(handle, ax=ax, fraction=0.046)
figure.tight_layout()

# %%
#
# Both estimators recover the same structure. Quantitatively, we compare them
# against the ground truth over the masked region:
#
reference = T2[valid]
for name, estimate in [("dictionary", T2_dict), ("PERK", T2_perk)]:
    error = estimate[valid] - reference
    print(
        f"{name:>10}: bias {error.mean():+6.2f} ms, "
        f"RMSE {np.sqrt((error**2).mean()):5.2f} ms"
    )

# %%
#
# The dictionary estimate is quantized onto the T2 grid, whereas PERK
# interpolates continuously between training samples. The practical difference
# is cost: matching scales with the number of atoms at every inference, while
# PERK pays that price once, during training.
#
plt.figure(figsize=(5, 4))
plt.plot(reference[::37], T2_dict[valid][::37], ".", markersize=2, label="dictionary")
plt.plot(reference[::37], T2_perk[valid][::37], ".", markersize=2, label="PERK")
plt.plot([0, 350], [0, 350], "k--", linewidth=1, label="identity")
plt.xlabel("true T2 [ms]"), plt.ylabel("estimated T2 [ms]")
plt.xlim([0, 350]), plt.ylim([0, 350])
plt.legend(), plt.tight_layout()
