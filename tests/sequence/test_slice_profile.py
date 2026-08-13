"""Simulating across a slice profile on the fused kernels.

A profile scales the flip angle and nothing else, exactly as the transmit field
does, and the recorded signal is its mean. So the state machine never has to
know what a slice is: a profile of n points is n copies of every voxel at
scaled transmit. These tests hold that identity against the operator loop,
which reaches the same answer by carrying a location axis through the states.
"""

import pytest
import torch

from torchsim import FSE, fse_description
from torchsim.sequence import offload
from torchsim.sequence._accelerators import _across_slice
from torchsim.sequence._simulation import TissueProperties

from test_offload import _peak_over_baseline  # noqa: E402

ECHOES = 10
STATES = 10
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


def _describe(flip):
    return fse_description(
        flip,
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )


def _flip(trains=None):
    generator = torch.Generator().manual_seed(0)
    shape = (ECHOES,) if trains is None else (trains, ECHOES)
    return torch.deg2rad(80.0 + 80.0 * torch.rand(shape, generator=generator))


def _tissue(**overrides):
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        **overrides,
    )


def _signal(backend, profile, flip=None, tissue=None, device=None):
    return FSE().simulate(
        _describe(_flip() if flip is None else flip),
        _tissue() if tissue is None else tissue,
        slice_profile=profile,
        nstates=STATES,
        backend=backend,
        device=device,
    ).signal


# --- the same answer as carrying a location axis ---


@pytest.mark.parametrize("points", [1, 2, 3, 9])
def test_a_profile_matches_the_operator_loop(points):
    profile = torch.linspace(0.4, 1.0, points)
    expected = _signal("torch", profile)
    actual = _signal("native", profile)

    assert actual.shape == expected.shape
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def test_a_batch_of_trains_keeps_its_profile():
    """One train at a time is the reference the batched run has to reproduce."""
    profile = torch.linspace(0.4, 1.0, 5)
    flip = _flip(trains=4)
    actual = _signal("native", profile, flip=flip)
    expected = torch.stack(
        [_signal("torch", profile, flip=row) for row in flip]
    )

    assert actual.shape == expected.shape
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def test_a_single_point_profile_changes_nothing():
    """Bitwise: a profile of one point is the transmit field and no more."""
    assert torch.equal(_signal("native", 1.0), _signal("native", torch.ones(1)))


def test_a_flat_profile_is_the_transmit_field():
    """Every point identical means every copy identical, so the mean is one run."""
    plain = _signal("native", 1.0)
    spread = _signal("native", torch.full((4,), 1.0))

    assert ((plain - spread).abs().max() / plain.abs().max()) < 1e-6


@cuda_only
def test_the_card_agrees_with_the_host():
    profile = torch.linspace(0.4, 1.0, 5)
    expected = _signal("native", profile)
    tissue = TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0], device="cuda"),
        t2_ms=torch.tensor([45.0, 120.0], device="cuda"),
    )
    actual = _signal("native", profile.cuda(), tissue=tissue)

    assert ((expected - actual.cpu()).abs().max() / expected.abs().max()) < 1e-5


@cuda_only
def test_a_profile_is_counted_by_the_memory_policy():
    """Spreading happens first, so streaming sees the copies as the voxels."""
    voxels, points = 40_000, 5
    budget = 8 << 20
    tissue = TissueProperties(
        t1_ms=torch.linspace(300.0, 2000.0, voxels),
        t2_ms=torch.linspace(20.0, 200.0, voxels),
    )
    profile = torch.linspace(0.4, 1.0, points)
    expected = _signal("native", profile, tissue=tissue)
    streamed = []

    def run():
        with offload(["cuda"], budget_bytes=budget):
            streamed.append(_signal("native", profile, tissue=tissue))

    resident = _peak_over_baseline(run)
    actual = streamed[0]

    assert resident <= budget * 1.1
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


# --- gradients through the spreading ---


def test_a_transmit_gradient_survives_the_spreading():
    """Every copy of a voxel carries its transmit, so the gradient sums back."""
    profile = torch.linspace(0.4, 1.0, 5)

    def gradient(backend):
        b1 = torch.tensor([1.0, 0.9], requires_grad=True)
        signal = _signal(backend, profile, tissue=_tissue(b1=b1))
        return torch.autograd.grad(signal.abs().square().sum(), b1)[0]

    expected = gradient("torch")
    actual = gradient("native")

    assert expected.abs().max() > 0.0
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-4


def test_the_profile_itself_can_be_differentiated():
    """It enters as a factor on the flip angle, so it is an input like any other."""
    profile = torch.linspace(0.4, 1.0, 5).requires_grad_(True)
    signal = _signal("native", profile)
    (gradient,) = torch.autograd.grad(signal.abs().square().sum(), profile)

    assert gradient.abs().max() > 0.0


# --- the spreading itself ---


def test_spreading_lays_the_copies_out_voxel_major():
    """The mean at the end folds the last axis, so the copies have to be there."""
    tissue = tuple(
        torch.tensor(value, dtype=torch.float32)
        for value in ([800.0, 1400.0], [45.0, 120.0], [1.0, 1.0], [1.0, 0.5],
                      [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    )
    profile = torch.tensor([1.0, 0.5, 0.25])
    spread, locations = _across_slice(tissue, profile)

    assert locations == 3
    assert spread[0].tolist() == [800.0, 800.0, 800.0, 1400.0, 1400.0, 1400.0]
    # Transmit is the one that varies across the profile.
    assert spread[3].tolist() == [1.0, 0.5, 0.25, 0.5, 0.25, 0.125]


def test_one_point_is_left_alone():
    """No copies to make, so nothing is widened."""
    tissue = tuple(
        torch.tensor([1.0, 2.0], dtype=torch.float32) for _ in range(7)
    )
    spread, locations = _across_slice(tissue, torch.tensor([0.5]))

    assert locations == 1
    assert spread[0].numel() == 2
    assert spread[3].tolist() == [0.5, 1.0]
