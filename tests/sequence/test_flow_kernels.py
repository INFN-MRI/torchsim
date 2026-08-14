"""Flow dephasing through the fused kernels.

Diffusion damps each dephasing order; flow turns each through a phase instead.
That difference is what takes the states out of any real subspace, and it is
why these check the subspace verdict alongside the arithmetic.

Washout is the other term a velocity drives, and it needs no gradient to act
through; ``test_washout_kernels`` covers it and the two together.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.epg import flow_op
from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp,
    _run_packed_vjp_jvp,
    Geometry,
    dephasing_per_m,
    geometry_of,
    real_subspace_axis,
)
from torchsim.sequence._parameters import FLOAT_NAMES, TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue
from utils.packed_reference import simulate_packed

ECHOES = 6
STATE_COUNT = 10
VELOCITY_INDEX = TISSUE_NAMES.index("velocity_m_per_s")
CRUSHER_RAD = 8.0 * torch.pi
VOXEL_M = 5e-4
# Centimetres per second, the scale of venous flow. Chosen so the phase one
# interval winds is not a whole number of turns: at 0.15 m/s this geometry
# gives exactly 12 pi per interval, where every order aliases back onto one
# and flow would look like it did nothing.
VELOCITY = 0.02


def _description(crusher_rad: float = CRUSHER_RAD):
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        crusher_dephasing_rad=crusher_rad,
        voxel_size_m=VOXEL_M,
    )


def _tissue(velocity: float = VELOCITY) -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 90.0]),
        m0=torch.tensor([1.0, 0.8]),
        b1=torch.tensor([1.0, 0.9]),
        b0_hz=torch.tensor([13.0, -27.0]),
        velocity_m_per_s=torch.tensor([velocity, -0.5 * velocity]),
    )


GEOMETRY = geometry_of(_description())


def _packed(velocity: float = VELOCITY):
    """Prepared tissue and packed events, as the kernels take them."""
    prepared, _, resolved = _prepare_tissue(_tissue(velocity), "cpu")
    packed = _pack_events(
        "fse",
        _description(),
        repetitions=1,
        record="all",
        device=resolved,
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )
    return prepared, events, packed.output_count


def test_the_flow_rate_reproduces_the_operator() -> None:
    """The rate folded in on the host gives the operator library's factors."""
    interval_s = 5e-3
    speed = torch.tensor([VELOCITY])
    longitudinal, transverse = flow_op(
        speed, torch.tensor(interval_s), STATE_COUNT, CRUSHER_RAD, VOXEL_M
    )

    turn = dephasing_per_m(_description()) * VELOCITY * interval_s
    order = torch.arange(STATE_COUNT, dtype=torch.float32)
    assert torch.allclose(
        longitudinal.reshape(-1), torch.exp(-1j * order * turn), atol=1e-5
    )
    assert torch.allclose(
        transverse.reshape(-1), torch.exp(-1j * (order + 0.5) * turn), atol=1e-5
    )


def test_flow_actually_moves_the_signal() -> None:
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
        geometry=GEOMETRY,
    )
    moving_signal = _run_packed(tissue, events, **arguments)
    still_signal = _run_packed(still, events, **arguments)
    relative = (moving_signal - still_signal).abs().max() / still_signal.abs().max()
    assert relative > 0.01


def test_the_forward_kernel_matches_the_reference() -> None:
    tissue, events, output_count = _packed()
    expected = simulate_packed(
        tissue, events, state_count=STATE_COUNT, output_count=output_count,
        geometry=GEOMETRY,
    )
    actual = _run_packed(
        tissue,
        events,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=GEOMETRY,
    )
    assert (expected - actual).abs().max() / expected.abs().max() < 1e-5


def test_no_crusher_leaves_no_phase_to_turn_through() -> None:
    """Flow dephasing needs a gradient; without one only washout is left.

    So a sequence with no crusher gives bit for bit what the same sequence
    gives when the two terms are separated and only washout is asked for.
    """
    assert geometry_of(_description(crusher_rad=0.0)).flow_scale == 0.0
    tissue, events, output_count = _packed()
    arguments = dict(state_count=STATE_COUNT, output_count=output_count, threads=1)

    assert torch.equal(
        _run_packed(
            tissue,
            events,
            geometry=geometry_of(_description(crusher_rad=0.0)),
            **arguments,
        ),
        _run_packed(
            tissue,
            events,
            geometry=Geometry(flow_scale=0.0, washout_scale=1.0 / VOXEL_M),
            **arguments,
        ),
    )


def test_flow_takes_the_states_out_of_the_real_subspace() -> None:
    """A per-order phase is a rotation off the axis, not a scaling along it."""
    description = fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
        crusher_dephasing_rad=CRUSHER_RAD,
        voxel_size_m=VOXEL_M,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )
    still = TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]), t2_ms=torch.tensor([50.0, 90.0])
    )
    moving = TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 90.0]),
        velocity_m_per_s=torch.tensor([VELOCITY, VELOCITY]),
    )

    assert real_subspace_axis(events, _prepare_tissue(still, "cpu")[0]) == 1
    assert real_subspace_axis(events, _prepare_tissue(moving, "cpu")[0]) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_forward_agrees_with_the_cpu_one() -> None:
    tissue, events, output_count = _packed()
    device = torch.device("cuda")
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=GEOMETRY,
    )
    expected = _run_packed(tissue, events, **arguments)
    actual = _run_packed(
        tuple(value.to(device) for value in tissue),
        tuple(value.to(device) for value in events),
        **arguments,
    )
    assert (expected - actual.cpu()).abs().max() / expected.abs().max() < 1e-5


def test_the_adjoint_matches_the_reference() -> None:
    """Every gradient, including the one w.r.t. the flow rate itself."""
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
    )
    reference = simulate_packed(
        leaves,
        differentiable,
        state_count=STATE_COUNT,
        output_count=output_count,
        geometry=GEOMETRY,
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
        geometry=GEOMETRY,
    )
    scale = max(g.abs().max().item() for g in expected if g is not None)
    compared = []
    for name, want, got in zip(FLOAT_NAMES, expected, actual, strict=True):
        if want is None or want.abs().max().item() <= 1e-6 * scale:
            continue
        assert (want - got).abs().max().item() / want.abs().max().item() < 1e-4
        compared.append(name)
    assert "velocity_m_per_s" in compared


def test_the_velocity_gradient_matches_a_finite_difference() -> None:
    """An independent check on the chain, velocity through to signal."""
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
        geometry=GEOMETRY,
    )

    def loss(rate: torch.Tensor) -> torch.Tensor:
        shifted = tuple(
            rate if index == VELOCITY_INDEX else value
            for index, value in enumerate(tissue)
        )
        return torch.real(torch.sum(_run_packed(shifted, events, **arguments).conj() * seed))

    rate = tissue[VELOCITY_INDEX]
    step = 1e-4 * rate.abs().max()
    analytic = _run_packed_vjp(tissue, events, seed, **arguments)[VELOCITY_INDEX]
    for voxel in range(rate.numel()):
        bump = torch.zeros_like(rate)
        bump[voxel] = step
        numeric = (loss(rate + bump) - loss(rate - bump)) / (2.0 * step)
        assert abs(numeric - analytic[voxel]) < 5e-3 * abs(analytic[voxel])


def test_forward_mode_along_velocity_matches_the_reference() -> None:
    """A still voxel still has a derivative along velocity, and it is not zero."""
    tissue, events, output_count = _packed(velocity=0.0)
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
        geometry=GEOMETRY,
    )

    def forward(*values):
        return simulate_packed(
            values,
            events,
            state_count=STATE_COUNT,
            output_count=output_count,
            geometry=GEOMETRY,
        )

    _signal, expected = torch.func.jvp(forward, tuple(tissue), tissue_tangents)
    assert expected.abs().max() > 0
    assert (expected - actual).abs().max() / expected.abs().max() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_adjoint_agrees_with_the_cpu_one() -> None:
    """Second order on the card, against the same pass on the host."""
    tissue, events, output_count = _packed()
    device = torch.device("cuda")
    tangents = tuple(
        torch.ones_like(value) if index == VELOCITY_INDEX else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    ) + tuple(torch.zeros_like(events[i]) for i in (0, 2, 3))
    seed = torch.randn(
        (tissue[0].numel(), output_count),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(2),
    )
    arguments = dict(
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
        geometry=GEOMETRY,
    )

    expected, _ = _run_packed_vjp_jvp(tissue, events, tangents, seed, **arguments)
    actual, _ = _run_packed_vjp_jvp(
        tuple(value.to(device) for value in tissue),
        tuple(value.to(device) for value in events),
        tuple(value.to(device) for value in tangents),
        seed.to(device),
        **arguments,
    )
    # These span twelve orders of magnitude -- an echo train refocuses
    # off-resonance, leaving that one as rounding noise about zero -- so the
    # ones far below the largest carry no signal of their own.
    largest = max(value.abs().max().item() for value in expected)
    compared = []
    for name, want, got in zip(FLOAT_NAMES, expected, actual, strict=True):
        scale = want.abs().max().item()
        if scale <= 1e-6 * largest:
            continue
        assert (want - got.cpu()).abs().max().item() / scale < 1e-4, name
        compared.append(name)
    assert "velocity_m_per_s" in compared


def test_the_operator_loop_refuses_flow() -> None:
    with pytest.raises(NotImplementedError, match="fused kernels"):
        FSE().simulate(
            _description(), _tissue(), nstates=STATE_COUNT, backend="torch"
        )
