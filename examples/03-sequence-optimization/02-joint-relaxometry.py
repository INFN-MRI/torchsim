"""
=========================================
Designing a joint relaxometry protocol
=========================================

DESPOT estimates T1 from a set of spoiled gradient-echo acquisitions at
different flip angles and T2 from a set of balanced SSFP acquisitions. Fitting
them jointly rather than one after the other uses all the data for both
parameters, and then the flip angles themselves can be chosen to make the joint
estimate as precise as possible [1]_.

The cost here is a Cramer-Rao bound: the lowest variance an unbiased estimate
of T1 and T2 can have, given the derivative of each sequence's signal with
respect to every parameter being estimated. Minimizing it chooses where on each
signal curve the scan time is spent.

.. [1] Teixeira RPAG, Malik SJ, Hajnal JV. Joint system relaxometry (JSR) and
   Cramer-Rao lower bound optimization of sequence parameters: a framework for
   enhanced precision of DESPOT T1 and T2 estimation.
   Magn Reson Med. 2018;79:234-245.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# The imports:
#
import warnings

warnings.filterwarnings("ignore")

import time

import matplotlib.pyplot as plt
import torch

import torchsim
from torchsim.optim import Acquisition, Bounded, SequenceDesign
from torchsim.simulators import SPGRSimulator, bSSFPSimulator

# %%
#
# The two sequences
# -----------------
#
# A design problem is stated in three pieces. An
# :class:`~torchsim.Acquisition` is a simulator with the tissue it is being
# designed for already in place, so only the parameters under design are left
# to give. Both sequences here are closed forms, and they are constructed and
# asked exactly as a state-machine sequence would be.
#

# White and grey matter at 3 T -- the design is for both at once.
T1_MS = torch.tensor([830.0, 1330.0])
T2_MS = torch.tensor([80.0, 110.0])
# Noise standard deviation, as a fraction of the fully relaxed magnetization.
NOISE = 0.005

spgr = Acquisition(
    SPGRSimulator(TE=2.0, TR=6.0), T1=T1_MS, T2star=T2_MS, M0=1.0, B0=0.0
)
ssfp = Acquisition(
    bSSFPSimulator(TE=2.5, TR=5.0), T1=T1_MS, T2=T2_MS, M0=1.0, B0=0.0
)

# %%
#
# The cost
# --------
#
# Four parameters are estimated jointly: T1, T2, the proton density and the
# off-resonance. The last two are nuisances -- they have to be estimated
# because they affect the data, but the design is not for them.
#
# The two sequences do not carry the same information, and neither does this
# implementation pretend they do: the spoiled steady state written in closed
# form depends on T2\* rather than T2, so its T2 row is exactly zero. That is
# the structure of joint relaxometry rather than a limitation -- each block is
# blind to something, and the Fisher matrix adds them up.
#
JOINT = ("T1", "T2", "M0", "B0")


def rows(acquisition, **design):
    """The Jacobian rows for every joint parameter, zero where the block is blind."""
    present = [name for name in JOINT if name in acquisition.exposes]
    _, jacobian = acquisition.jacobian(present, **design)
    placed = jacobian.new_zeros(
        jacobian.shape[:-2] + (len(JOINT), jacobian.shape[-1])
    )
    where = torch.tensor([JOINT.index(name) for name in present])
    return placed.index_copy(-2, where, jacobian)


def bounds(spgr_flip, ssfp_flip):
    """The Cramer-Rao bound on each joint parameter, for each design tissue."""
    together = torch.cat(
        (rows(spgr, flip=spgr_flip), rows(ssfp, flip=ssfp_flip)), dim=-1
    )
    return torchsim.crlb(together, noise_variance=NOISE**2)


# %%
#
# The cost is the whole of what makes this problem this problem, and it is
# four lines. Dividing each bound by its own parameter squared makes the two
# terms dimensionless, so a 100 ms T2 and a 1000 ms T1 are weighted by how
# well they are known rather than by how large they are; the logarithm makes
# the gradient relative, so the design does not depend on the noise level.
#


def precision(spgr_flip, ssfp_flip):
    """Relative variance of T1 and T2, averaged over the design tissues."""
    bound = bounds(spgr_flip, ssfp_flip)
    relative = bound[..., 0] / T1_MS**2 + bound[..., 1] / T2_MS**2
    return relative.mean().log()


# %%
#
# The design
# ----------
#
# Four acquisitions of each kind, starting from a spread of angles. The limits
# are what the scanner will play, and they are enforced exactly -- no iterate
# is ever outside them.
#
spgr_start = torch.tensor([2.0, 4.0, 8.0, 16.0])
ssfp_start = torch.tensor([10.0, 20.0, 40.0, 60.0])

design = SequenceDesign(
    precision,
    spgr_flip=Bounded(spgr_start, 1.0, 40.0),
    ssfp_flip=Bounded(ssfp_start, 1.0, 70.0),
)

start = time.time()
result = design.minimize(iterations=120, learning_rate=0.3)
design_time = time.time() - start

spgr_designed = result.parameters["spgr_flip"]
ssfp_designed = result.parameters["ssfp_flip"]

# %%
#
# What it buys, as the number a spectroscopist would quote: the standard
# deviation of each estimate as a percentage of the value itself.
#
for label, angles_pair in (
    ("published spread", (spgr_start, ssfp_start)),
    ("designed", (spgr_designed, ssfp_designed)),
):
    bound = bounds(*angles_pair)
    sigma_t1 = 100.0 * bound[..., 0].sqrt() / T1_MS
    sigma_t2 = 100.0 * bound[..., 1].sqrt() / T2_MS
    print(
        f"{label:18s} "
        f"sigma(T1)/T1 = {sigma_t1[0]:.1f}%, {sigma_t1[1]:.1f}%   "
        f"sigma(T2)/T2 = {sigma_t2[0]:.1f}%, {sigma_t2[1]:.1f}%"
    )
print(f"designed in {design_time:.1f} s")

# %%
#
# Where the angles went
# ---------------------
#
# The design collapses eight distinct angles onto three, and repeats them.
# That is what an optimal design does: the information sits at a few places on
# each curve, and the best use of a fixed number of acquisitions is to spend
# them there rather than to sample the curve evenly.
#
# Where those places are is worth reading off the figure. The SPGR angle lands
# above the Ernst angle of both tissues, on the side where the curve separates
# the two T1 values most sharply -- the peak itself is where the signal is
# largest and where it says least. The two bSSFP angles sit either side of the
# steady-state maximum, which is what makes the pair sensitive to T2. The
# upper one is against its limit rather than at an interior optimum, so
# raising the limit would move it; that limit is a real one, being what the
# deposited RF power allows.
#
sweep = torch.linspace(1.0, 70.0, 200)
figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))

for axis, acquisition, start_angles, designed, title in (
    (axes[0], spgr, spgr_start, spgr_designed, "SPGR"),
    (axes[1], ssfp, ssfp_start, ssfp_designed, "bSSFP"),
):
    curve = acquisition.simulate(flip=sweep).abs()
    axis.plot(sweep, curve[0], label="T1/T2 = 830/80 ms")
    axis.plot(sweep, curve[1], label="T1/T2 = 1330/110 ms")
    sampled = acquisition.simulate(flip=start_angles).abs()
    axis.plot(start_angles, sampled[0], "o", color="grey", label="start")
    sampled = acquisition.simulate(flip=designed).abs()
    axis.plot(designed, sampled[0], "*", ms=14, color="crimson", label="designed")
    axis.set(xlabel="Flip angle [deg]", ylabel="|signal|", title=title)
    axis.grid(alpha=0.3)
axes[0].legend(fontsize=7)

axes[2].plot(result.loss.cpu())
axes[2].set(
    xlabel="Iteration", ylabel="log relative CRLB", title="convergence"
)
axes[2].grid(alpha=0.3)
figure.tight_layout()

# %%
#
# What that means for the answers
# -------------------------------
#
# A bound is a promise about variance, and it is worth cashing. Both protocols
# are played on the same tissue with the same noise, fitted the same way, and
# repeated enough times that the spread of the answers is itself well measured.
#
# The fit has to be the one thing that does not differ between the two, so it
# is the same nonlinear least squares over the same four unknowns, started from
# the same guess. Both blocks are one experiment, so they are fitted as one:
# a :class:`~torchsim.model.SignalModel` that plays each and concatenates what
# they record.
#
from torchsim import ParameterMapping
from torchsim.estimators import NonlinearLeastSquares
from torchsim.model import SignalModel


class JointRelaxometry(SignalModel):
    """Both blocks at fixed flip angles, as one signal model."""

    properties = ("T1", "T2", "M0", "B0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence):
        """The two blocks, end to end along the contrast axis."""
        T1, T2 = properties["T1"], properties["T2"]
        M0 = properties.get("M0", 1.0)
        B0 = properties.get("B0", 0.0)
        return torch.cat(
            (
                self.spoiled.simulate(T1=T1, T2star=T2, M0=M0, B0=B0),
                self.balanced.simulate(T1=T1, T2=T2, M0=M0, B0=B0),
            ),
            dim=-1,
        )


# %%
#
# The noise is independent on the real and the imaginary channel, each at the
# standard deviation the bound was computed with -- which is what makes the two
# numbers comparable at all.
#
REPEATS = 4000
UNKNOWN = {
    "T1": (200.0, 5000.0),
    "T2": (20.0, 600.0),
    "M0": (0.5, 1.5),
    "B0": (-50.0, 50.0),
}
generator = torch.Generator().manual_seed(7)


def fitted(spgr_flip, ssfp_flip):
    """Map a few thousand noisy realizations of both design tissues."""
    joint = Acquisition(
        JointRelaxometry(spgr_flip, ssfp_flip), resolve=False
    )
    problem = ParameterMapping(
        joint, noise_std=NOISE, seed=0, **UNKNOWN
    ).train(
        NonlinearLeastSquares(
            bounds=UNKNOWN,
            initial={"T1": 1000.0, "T2": 100.0, "M0": 1.0, "B0": 0.0},
        )
    )

    clean = joint.simulate(T1=T1_MS, T2=T2_MS, M0=1.0, B0=0.0)
    repeated = clean.expand(REPEATS, *clean.shape).reshape(-1, clean.shape[-1])
    noise = torch.randn(
        (2, *repeated.shape), generator=generator, dtype=torch.float32
    )
    maps = problem(repeated + NOISE * torch.complex(noise[0], noise[1]))
    return {name: maps[name].reshape(REPEATS, -1) for name in ("T1", "T2")}


start = time.time()
before = fitted(spgr_start, ssfp_start)
after = fitted(spgr_designed, ssfp_designed)
print(f"{2 * REPEATS * len(T1_MS)} fits in {time.time() - start:.1f} s")

# %%
#
# The spread of the fitted values, against the bound that predicted it. No
# unbiased estimator can beat the bound and a good one comes close to it, so
# the two agreeing is the check that the design was optimizing the right thing.
#
print(f"\n{'':20}{'sigma(T1)/T1':>26}{'sigma(T2)/T2':>26}")
print(f"{'':20}{'measured':>13}{'bound':>13}{'measured':>13}{'bound':>13}")
for label, angles_pair, maps in (
    ("published spread", (spgr_start, ssfp_start), before),
    ("designed", (spgr_designed, ssfp_designed), after),
):
    bound = bounds(*angles_pair)
    line = f"{label:20}"
    for index, (name, truth) in enumerate((("T1", T1_MS), ("T2", T2_MS))):
        spread = 100.0 * maps[name].std(0) / truth
        predicted = 100.0 * bound[..., index].sqrt() / truth
        line += f"{float(spread.mean()):12.1f}%{float(predicted.mean()):12.1f}%"
    print(line)

# %%
#
# The same result as a histogram, one design tissue per column. The designed
# protocol is the narrower distribution, and it is narrower without being
# displaced: what the design bought is precision and not bias.
#
figure, axes = plt.subplots(2, 2, figsize=(11, 6))
for column in range(len(T1_MS)):
    for row, (name, truth) in enumerate((("T1", T1_MS), ("T2", T2_MS))):
        axis = axes[row, column]
        centre = float(truth[column])
        span = (0.6 * centre, 1.4 * centre)
        for label, maps, colour in (
            ("published spread", before, "grey"),
            ("designed", after, "crimson"),
        ):
            axis.hist(
                maps[name][:, column].clamp(*span).numpy(),
                bins=60,
                range=span,
                histtype="step",
                color=colour,
                label=label,
            )
        axis.axvline(centre, color="k", lw=1, ls="--")
        axis.set(xlabel=f"fitted {name} [ms]", yticks=[])
        axis.set_title(f"{name}, truth {centre:.0f} ms", fontsize=10)
axes[0, 0].legend(fontsize=8)
figure.tight_layout()

# %%
#
# Reading the result honestly
# ---------------------------
#
# The bound is a bound, not a prediction: an estimator can be worse than it and
# none can be better. The measured spread sits close to it here because the
# noise is small enough that the fit is nearly linear over the region it
# explores. Raise the noise and the fit becomes biased near the ends of its
# bounds, the histogram grows a tail, and the design's advantage shrinks --
# which is the honest limit of designing against a Cramer-Rao bound.
#
# Nothing above is specific to DESPOT except the cost. The acquisition, the
# bounded parameters and the loop are the same three pieces that design a
# sequence for image quality rather than for precision.
#
