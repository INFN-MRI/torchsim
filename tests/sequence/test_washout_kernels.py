"""Washout and inflow through the fused kernels.

Flow dephasing needs a gradient to wind a phase across; washout needs only a
voxel for the spins to leave. So the two terms are driven by the same velocity
through different geometry, and these check that each reaches the signal on its
own as well as together.

The physics here is one compartment: the spins arriving are taken to be fully
relaxed and unexcited. That makes washout an affine map of exactly the shape
longitudinal recovery already has, which is why it appears in the kernels as a
scaling of the two relaxation factors and nowhere else.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import TissueProperties, fse_description
from utils.epg import washout_op
from torchsim.sequence._accelerators import (
    NO_GEOMETRY,
    Geometry,
    _pack_events,
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp,
    _run_packed_vjp_jvp,
    geometry_of,
)
from torchsim.sequence._parameters import FLOAT_NAMES, TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue
from utils.packed_reference import simulate_packed

ECHOES = 6
STATE_COUNT = 10
VELOCITY_INDEX = TISSUE_NAMES.index("velocity_m_per_s")
CRUSHER_RAD = 8.0 * torch.pi
VOXEL_M = 5e-4
# Fast enough to turn a good fraction of a half-millimetre voxel over during a
# five-millisecond echo spacing, so washout is visible above the relaxation it
# multiplies rather than lost in it.
VELOCITY = 0.02


def _description(crusher_rad: float = CRUSHER_RAD, voxel_m: float | None = VOXEL_M):
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        crusher_dephasing_rad=crusher_rad,
        voxel_size_m=voxel_m,
    )


# Washout alone: a voxel to leave, but no gradient to dephase across.
WASHOUT_ONLY = Geometry(flow_scale=0.0, washout_scale=1.0 / VOXEL_M)
BOTH = geometry_of(_description())


def _tissue(velocity: float = VELOCITY) -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 90.0]),
        m0=torch.tensor([1.0, 0.8]),
        b1=torch.tensor([1.0, 0.9]),
        b0_hz=torch.tensor([13.0, -27.0]),
        velocity_m_per_s=torch.tensor([velocity, -0.5 * velocity]),
    )


def _packed(velocity: float = VELOCITY):
    """Prepared tissue and packed events, as the kernels take them."""
    prepared, _, resolved = _prepare_tissue(_tissue(velocity), "cpu")
    packed = _pack_events(
                _description(),
        repetitions=1,
        record="all",
        device=resolved,
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    return prepared, events, packed.output_count


def test_the_washout_rate_reproduces_the_operator() -> None:
    """The scale the kernels read a velocity through is the operator's rate."""
    interval_s = 5e-3
    speed = torch.tensor([VELOCITY, -VELOCITY])
    inflow, outflow = washout_op(speed, torch.tensor(interval_s), VOXEL_M)

    rate = speed.abs() * WASHOUT_ONLY.washout_scale
    assert torch.allclose(inflow, rate * interval_s, atol=1e-6)
    assert torch.allclose(outflow, 1.0 - rate * interval_s, atol=1e-6)


def test_a_direction_of_travel_does_not_change_the_washout() -> None:
    """Spins leave the voxel at their speed, whichever way they are going."""
    _tissue_unused, events, output_count = _packed()
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )
    forward, _, _ = _prepare_tissue(_tissue(VELOCITY), "cpu")
    backward, _, _ = _prepare_tissue(_tissue(-VELOCITY), "cpu")

    assert torch.equal(
        _run_packed(forward, events, **arguments),
        _run_packed(backward, events, **arguments),
    )


def test_washout_actually_moves_the_signal() -> None:
    """Without this, every parity check below would hold trivially."""
    tissue, events, output_count = _packed()
    still = tuple(
        torch.zeros_like(value) if index == VELOCITY_INDEX else value
        for index, value in enumerate(tissue)
    )
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )
    moving_signal = _run_packed(tissue, events, **arguments)
    still_signal = _run_packed(still, events, **arguments)
    relative = (moving_signal - still_signal).abs().max() / still_signal.abs().max()
    assert relative > 0.01


def test_washout_only_shrinks_the_signal() -> None:
    """Inflowing spins arrive unexcited, so they can only dilute what is there."""
    tissue, events, output_count = _packed()
    still = tuple(
        torch.zeros_like(value) if index == VELOCITY_INDEX else value
        for index, value in enumerate(tissue)
    )
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )

    moving = _run_packed(tissue, events, **arguments).abs()
    resting = _run_packed(still, events, **arguments).abs()
    assert bool((moving <= resting + 1e-6).all())


def test_no_declared_voxel_leaves_the_signal_untouched() -> None:
    """A velocity with nothing to cross is inert, bit for bit."""
    assert geometry_of(_description(crusher_rad=0.0, voxel_m=None)) == NO_GEOMETRY
    tissue, events, output_count = _packed()
    still = tuple(
        torch.zeros_like(value) if index == VELOCITY_INDEX else value
        for index, value in enumerate(tissue)
    )
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=NO_GEOMETRY,
    )
    assert torch.equal(
        _run_packed(tissue, events, **arguments),
        _run_packed(still, events, **arguments),
    )


def test_a_crusher_needs_a_voxel_to_wind_across() -> None:
    with pytest.raises(ValueError, match="voxel_size_m"):
        geometry_of(_description(crusher_rad=CRUSHER_RAD, voxel_m=None))


@pytest.mark.parametrize("geometry", [WASHOUT_ONLY, BOTH], ids=["washout", "both"])
def test_the_forward_kernel_matches_the_reference(geometry) -> None:
    tissue, events, output_count = _packed()
    expected = simulate_packed(
        tissue,
        events,
        state_count=STATE_COUNT,
        output_count=output_count,
        geometry=geometry,
    )
    actual = _run_packed(
        tissue,
        events,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=geometry,
    )
    assert (expected - actual).abs().max() / expected.abs().max() < 1e-5


def test_a_voxel_that_turns_over_within_one_interval_is_clamped() -> None:
    """Past a full turnover the voxel holds nothing but freshly arrived spins.

    Without the clamp the surviving fraction would go negative and the states
    would come back with their signs flipped.
    """
    tissue, events, output_count = _packed()
    # The intervals here are five milliseconds, so this rate replaces the voxel
    # several times over within each one.
    torrent = tuple(
        torch.full_like(value, 1e3) if index == VELOCITY_INDEX else value
        for index, value in enumerate(tissue)
    )
    signal = _run_packed(
        torrent,
        events,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )
    assert bool(torch.isfinite(signal).all())
    assert signal.abs().max() < 1e-6


@pytest.mark.parametrize("geometry", [WASHOUT_ONLY, BOTH], ids=["washout", "both"])
def test_the_adjoint_matches_the_reference(geometry) -> None:
    """Every gradient, including the one w.r.t. the velocity itself."""
    tissue, events, output_count = _packed()
    seed = torch.randn(
        (tissue[0].numel(), output_count),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(0),
    )
    leaves = tuple(value.detach().clone().requires_grad_(True) for value in tissue)
    differentiable = (
        events[0].detach().clone().requires_grad_(True),
        events[1],
        events[2].detach().clone().requires_grad_(True),
        events[3].detach().clone().requires_grad_(True),
        events[4],
        events[5],
        events[6],
        events[7],
        events[8],
    )
    reference = simulate_packed(
        leaves,
        differentiable,
        state_count=STATE_COUNT,
        output_count=output_count,
        geometry=geometry,
    )
    reference.backward(seed)
    expected = tuple(value.grad for value in leaves) + (
        differentiable[0].grad,
        differentiable[2].grad,
        differentiable[3].grad,
    )
    actual = _run_packed_vjp(
        tissue,
        events,
        seed,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=geometry,
    )
    scale = max(g.abs().max().item() for g in expected if g is not None)
    compared = []
    for name, want, got in zip(FLOAT_NAMES, expected, actual, strict=True):
        if want is None or want.abs().max().item() <= 1e-6 * scale:
            continue
        assert (want - got).abs().max().item() / want.abs().max().item() < 1e-4
        compared.append(name)
    assert "velocity_m_per_s" in compared
    assert "duration" in compared


def test_the_velocity_gradient_matches_a_finite_difference() -> None:
    """An independent check on the chain, velocity through washout to signal."""
    tissue, events, output_count = _packed()
    seed = torch.randn(
        (tissue[0].numel(), output_count),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(1),
    )
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )

    def loss(velocity: torch.Tensor) -> torch.Tensor:
        shifted = tuple(
            velocity if index == VELOCITY_INDEX else value
            for index, value in enumerate(tissue)
        )
        return torch.real(
            torch.sum(_run_packed(shifted, events, **arguments).conj() * seed)
        )

    velocity = tissue[VELOCITY_INDEX]
    step = 1e-2 * velocity.abs().max()
    analytic = _run_packed_vjp(tissue, events, seed, **arguments)[VELOCITY_INDEX]
    for voxel in range(velocity.numel()):
        bump = torch.zeros_like(velocity)
        bump[voxel] = step
        numeric = (loss(velocity + bump) - loss(velocity - bump)) / (2.0 * step)
        assert abs(numeric - analytic[voxel]) < 5e-3 * abs(analytic[voxel])


def test_forward_mode_along_velocity_matches_the_reference() -> None:
    tissue, events, output_count = _packed()
    tissue_tangents = tuple(
        torch.ones_like(value) if index == VELOCITY_INDEX else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    )
    event_tangents = tuple(torch.zeros_like(events[i]) for i in (0, 2, 3))
    actual = _run_packed_jvp(
        tissue,
        events,
        tissue_tangents,
        event_tangents,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )

    def forward(*values):
        return simulate_packed(
            values,
            events,
            state_count=STATE_COUNT,
            output_count=output_count,
            geometry=WASHOUT_ONLY,
        )

    _signal, expected = torch.func.jvp(forward, tuple(tissue), tissue_tangents)
    assert expected.abs().max() > 0
    assert (expected - actual).abs().max() / expected.abs().max() < 1e-4


def test_a_still_voxel_has_no_washout_derivative() -> None:
    """``|v|`` has no derivative at the origin, and the kernels report none.

    Flow dephasing does have one there, which is why velocity leaves the real
    subspace at any value; washout is the term that genuinely vanishes.
    """
    tissue, events, output_count = _packed(velocity=0.0)
    seed = torch.randn(
        (tissue[0].numel(), output_count),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(3),
    )
    gradient = _run_packed_vjp(
        tissue,
        events,
        seed,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )[VELOCITY_INDEX]
    assert bool((gradient == 0).all())


def test_the_second_order_pass_matches_the_reference() -> None:
    """Forward-over-reverse, seeded along the velocity washout depends on."""
    tissue, events, output_count = _packed()
    seed = torch.randn(
        (tissue[0].numel(), output_count),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(4),
    )
    directions = tuple(
        torch.ones_like(value) if index == VELOCITY_INDEX else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    ) + tuple(torch.zeros_like(events[i]) for i in (0, 2, 3))
    actual, _ = _run_packed_vjp_jvp(
        tissue,
        events,
        directions,
        seed,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )

    leaves = tuple(value.detach().clone().requires_grad_(True) for value in tissue)
    differentiable = (
        events[0].detach().clone().requires_grad_(True),
        events[1],
        events[2].detach().clone().requires_grad_(True),
        events[3].detach().clone().requires_grad_(True),
        events[4],
        events[5],
        events[6],
        events[7],
        events[8],
    )
    inputs = leaves + (differentiable[0], differentiable[2], differentiable[3])
    signal = simulate_packed(
        leaves,
        differentiable,
        state_count=STATE_COUNT,
        output_count=output_count,
        geometry=WASHOUT_ONLY,
    )
    adjoint = torch.autograd.grad(
        signal,
        inputs,
        grad_outputs=seed,
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )
    expected = torch.autograd.grad(
        adjoint,
        inputs,
        grad_outputs=directions,
        allow_unused=True,
        materialize_grads=True,
    )

    largest = max(value.abs().max().item() for value in expected)
    compared = []
    for name, want, got in zip(FLOAT_NAMES, expected, actual, strict=True):
        if want.abs().max().item() <= 1e-6 * largest:
            continue
        assert (want - got).abs().max().item() / want.abs().max().item() < 2e-3
        compared.append(name)
    assert "velocity_m_per_s" in compared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("geometry", [WASHOUT_ONLY, BOTH], ids=["washout", "both"])
def test_the_cuda_forward_agrees_with_the_cpu_one(geometry) -> None:
    tissue, events, output_count = _packed()
    device = torch.device("cuda")
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=geometry,
    )
    expected = _run_packed(tissue, events, **arguments)
    actual = _run_packed(
        tuple(value.to(device) for value in tissue),
        tuple(value.to(device) for value in events),
        **arguments,
    )
    assert (expected - actual.cpu()).abs().max() / expected.abs().max() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_second_order_agrees_with_the_cpu_one() -> None:
    tissue, events, output_count = _packed()
    device = torch.device("cuda")
    directions = tuple(
        torch.ones_like(value) if index == VELOCITY_INDEX else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    ) + tuple(torch.zeros_like(events[i]) for i in (0, 2, 3))
    seed = torch.randn(
        (tissue[0].numel(), output_count),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(5),
    )
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=WASHOUT_ONLY,
    )

    expected, _ = _run_packed_vjp_jvp(tissue, events, directions, seed, **arguments)
    actual, _ = _run_packed_vjp_jvp(
        tuple(value.to(device) for value in tissue),
        tuple(value.to(device) for value in events),
        tuple(value.to(device) for value in directions),
        seed.to(device),
        **arguments,
    )
    # An echo train refocuses off-resonance, so its b0 gradient is rounding
    # noise about zero and carries no signal of its own.
    largest = max(value.abs().max().item() for value in expected)
    compared = []
    for name, want, got in zip(FLOAT_NAMES, expected, actual, strict=True):
        scale = want.abs().max().item()
        if scale <= 1e-6 * largest:
            continue
        assert (want - got.cpu()).abs().max().item() / scale < 1e-3
        compared.append(name)
    assert "velocity_m_per_s" in compared
