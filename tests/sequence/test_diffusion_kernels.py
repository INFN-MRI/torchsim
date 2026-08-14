"""Diffusion damping through the fused kernels.

The damping is keyed to a state's dephasing order, so it is the first term
that cannot be folded into the per-interval relaxation scalars. These check
the factors themselves against the operator library, the derivatives against
the differentiable state machine, and the whole chain against a finite
difference in the apparent diffusion coefficient.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.epg import diffusion_op
from torchsim.sequence._accelerators import (
    NO_GEOMETRY,
    _NativeEpg,
    _pack_events,
    _run_packed,
    _run_packed_vjp,
    _run_packed_vjp_jvp,
    damping_scale,
)
from torchsim.sequence._parameters import FLOAT_NAMES, TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue
from utils.packed_reference import simulate_packed

ECHOES = 6
STATE_COUNT = 10
DIFFUSION_INDEX = TISSUE_NAMES.index("diffusion_um2_per_ms")
# A crusher winding four turns across half a millimetre. Strong enough that
# the highest orders carried here are damped by tens of percent per interval.
CRUSHER_RAD = 8.0 * torch.pi
VOXEL_M = 5e-4
FREE_WATER = 3.0


def _description(crusher_rad: float = CRUSHER_RAD):
    flip = torch.deg2rad(torch.full((ECHOES,), 140.0))
    return fse_description(
        flip,
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        crusher_dephasing_rad=crusher_rad,
        voxel_size_m=VOXEL_M,
    )


def _tissue(diffusion: float = FREE_WATER) -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 90.0]),
        m0=torch.tensor([1.0, 0.8]),
        b1=torch.tensor([1.0, 0.9]),
        b1_phase_rad=torch.tensor([0.05, -0.1]),
        b0_hz=torch.tensor([13.0, -27.0]),
        diffusion_um2_per_ms=torch.tensor([diffusion, 0.5 * diffusion]),
    )


def _packed(device: str = "cpu", diffusion: float = FREE_WATER, rate: bool = True):
    """Prepared tissue and packed events, as the kernels take them.

    With ``rate`` the diffusion entry carries the sequence's gradient geometry
    already folded in, which is what reaches a kernel.
    """
    prepared, _, resolved = _prepare_tissue(_tissue(diffusion), device)
    if rate:
        scale = damping_scale(_description())
        prepared = tuple(
            value * scale if index == DIFFUSION_INDEX else value
            for index, value in enumerate(prepared)
        )
    packed = _pack_events(
        "fse",
        _description(),
        repetitions=1,
        record="all",
        device=resolved,
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    return prepared, events, packed.output_count


def _seed(shape) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(shape, dtype=torch.complex64, generator=generator)


def _agree(expected, actual, names, tolerance: float = 1e-3) -> None:
    """Compare gradient tuples, skipping the entries float32 cannot resolve."""
    scales = {
        name: value.abs().max().item()
        for name, value in zip(names, expected, strict=True)
        if value is not None
    }
    floor = 1e-6 * max(scales.values())
    compared = []
    for name, want, got in zip(names, expected, actual, strict=True):
        if want is None or scales[name] <= floor:
            continue
        error = (want - got).abs().max().item() / scales[name]
        assert error < tolerance, f"{name} differs by {error:.2e}"
        compared.append(name)
    assert "diffusion_um2_per_ms" in compared


def test_the_damping_rate_reproduces_the_operator() -> None:
    """The rate folded in on the host gives the operator library's factors."""
    interval_s = 5e-3
    coefficient = torch.tensor([FREE_WATER])
    longitudinal, transverse = diffusion_op(
        coefficient,
        torch.tensor(interval_s * 1e3),
        STATE_COUNT,
        CRUSHER_RAD,
        VOXEL_M,
    )

    b_factor = damping_scale(_description()) * FREE_WATER * interval_s
    order = torch.arange(STATE_COUNT, dtype=torch.float32)
    assert torch.allclose(
        longitudinal.reshape(-1), torch.exp(-b_factor * order.square()), atol=1e-6
    )
    assert torch.allclose(
        transverse.reshape(-1),
        torch.exp(-b_factor * (order.square() + order + 1.0 / 3.0)),
        atol=1e-6,
    )


def test_a_declared_crusher_actually_damps() -> None:
    """Without this, every parity check below would hold trivially."""
    tissue, events, output_count = _packed()
    undamped = tuple(
        torch.zeros_like(value) if index == DIFFUSION_INDEX else value
        for index, value in enumerate(tissue)
    )
    arguments = dict(state_count=STATE_COUNT, output_count=output_count, threads=1)
    damped_signal = _run_packed(tissue, events, **arguments)
    plain_signal = _run_packed(undamped, events, **arguments)
    relative = (damped_signal - plain_signal).abs().max() / plain_signal.abs().max()
    assert relative > 0.01


def test_the_forward_kernel_matches_the_reference() -> None:
    tissue, events, output_count = _packed()
    expected = simulate_packed(
        tissue, events, state_count=STATE_COUNT, output_count=output_count
    )
    actual = _run_packed(
        tissue, events, state_count=STATE_COUNT, output_count=output_count, threads=1
    )
    assert (expected - actual).abs().max() / expected.abs().max() < 1e-5


def test_the_adjoint_matches_the_reference() -> None:
    """Every gradient, including the one w.r.t. the damping rate itself."""
    tissue, events, output_count = _packed()
    seed = _seed((tissue[0].numel(), output_count))

    leaves = tuple(value.detach().clone().requires_grad_(True) for value in tissue)
    differentiable = (
        events[0].detach().clone().requires_grad_(True),
        events[1],
        events[2].detach().clone().requires_grad_(True),
        events[3].detach().clone().requires_grad_(True),
        events[4],
        events[5],
        events[6],
    )
    reference = simulate_packed(
        leaves, differentiable, state_count=STATE_COUNT, output_count=output_count
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
    )
    _agree(expected, actual, FLOAT_NAMES)


def test_the_second_order_kernel_matches_the_reference() -> None:
    """Forward-over-reverse, seeded along the damping rate."""
    tissue, events, output_count = _packed()
    seed = _seed((tissue[0].numel(), output_count))
    tangents = tuple(
        torch.ones_like(value) if index == DIFFUSION_INDEX else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    ) + (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
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
    )
    inputs = leaves + (differentiable[0], differentiable[2], differentiable[3])
    signal = simulate_packed(
        leaves, differentiable, state_count=STATE_COUNT, output_count=output_count
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
        grad_outputs=tangents,
        allow_unused=True,
        materialize_grads=True,
    )

    curvature, _ = _run_packed_vjp_jvp(
        tissue,
        events,
        tangents,
        seed,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
    )
    _agree(expected, curvature, FLOAT_NAMES, tolerance=2e-3)


def test_the_diffusion_gradient_matches_a_finite_difference() -> None:
    """An independent check on the whole chain, coefficient to signal."""
    tissue, events, output_count = _packed()
    seed = _seed((tissue[0].numel(), output_count))
    arguments = dict(state_count=STATE_COUNT, output_count=output_count, threads=1)

    def loss(rate: torch.Tensor) -> torch.Tensor:
        shifted = tuple(
            rate if index == DIFFUSION_INDEX else value
            for index, value in enumerate(tissue)
        )
        signal = _run_packed(shifted, events, **arguments)
        return torch.real(torch.sum(signal.conj() * seed))

    rate = tissue[DIFFUSION_INDEX]
    step = 1e-3 * rate.abs().max()
    analytic = _run_packed_vjp(tissue, events, seed, **arguments)[DIFFUSION_INDEX]
    for voxel in range(rate.numel()):
        bump = torch.zeros_like(rate)
        bump[voxel] = step
        numeric = (loss(rate + bump) - loss(rate - bump)) / (2.0 * step)
        assert abs(numeric - analytic[voxel]) < 2e-3 * abs(analytic[voxel])


def test_an_undeclared_crusher_reaches_the_same_bits() -> None:
    """Damping switched off must not perturb the answer in its last place.

    A sequence with no unbalanced gradient gives a rate of zero however large
    the coefficient is, and a rate of zero has to leave both the signal and
    every gradient exactly where they were before diffusion existed.
    """
    assert damping_scale(_description(crusher_rad=0.0)) == 0.0
    tissue, events, output_count = _packed()
    arguments = dict(state_count=STATE_COUNT, output_count=output_count, threads=1)
    seed = _seed((tissue[0].numel(), output_count))

    # Free water through a sequence that winds no unbalanced gradient, and a
    # dry voxel through the crusher: both reach the kernels as a zero rate.
    prepared, _, _ = _prepare_tissue(_tissue(), "cpu")
    without_crusher = tuple(
        value * damping_scale(_description(crusher_rad=0.0))
        if index == DIFFUSION_INDEX
        else value
        for index, value in enumerate(prepared)
    )
    zero_coefficient = tuple(
        torch.zeros_like(value) if index == DIFFUSION_INDEX else value
        for index, value in enumerate(tissue)
    )
    assert torch.equal(
        _run_packed(without_crusher, events, **arguments),
        _run_packed(zero_coefficient, events, **arguments),
    )
    for expected, actual in zip(
        _run_packed_vjp(without_crusher, events, seed, **arguments),
        _run_packed_vjp(zero_coefficient, events, seed, **arguments),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_the_diffusion_gradient_survives_a_zero_coefficient() -> None:
    """dS/dD at D = 0 is not zero, and the kernels must not report it as one."""
    tissue, events, output_count = _packed(diffusion=0.0)
    seed = _seed((tissue[0].numel(), output_count))
    gradients = _run_packed_vjp(
        tissue,
        events,
        seed,
        state_count=STATE_COUNT,
        output_count=output_count,
        threads=1,
    )
    assert gradients[DIFFUSION_INDEX].abs().min() > 0


def test_the_operator_loop_refuses_diffusion() -> None:
    """The fallback path does not carry the damping, so it must say so."""
    simulator = FSE()
    with pytest.raises(NotImplementedError, match="fused kernels"):
        simulator.simulate(
            _description(), _tissue(), nstates=STATE_COUNT, backend="torch"
        )


def test_the_operator_loop_still_runs_without_diffusion() -> None:
    simulator = FSE()
    result = simulator.simulate(
        _description(), _tissue(0.0), nstates=STATE_COUNT, backend="torch"
    )
    assert torch.isfinite(result.signal).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_kernels_agree_with_the_cpu_ones() -> None:
    tissue, events, output_count = _packed()
    device = torch.device("cuda")
    moved_tissue = tuple(value.to(device) for value in tissue)
    moved_events = tuple(value.to(device) for value in events)
    seed = _seed((tissue[0].numel(), output_count))
    arguments = dict(state_count=STATE_COUNT, output_count=output_count, threads=1)

    expected = _run_packed(tissue, events, **arguments)
    actual = _run_packed(moved_tissue, moved_events, **arguments)
    assert (expected - actual.cpu()).abs().max() / expected.abs().max() < 1e-5

    expected_grads = _run_packed_vjp(tissue, events, seed, **arguments)
    actual_grads = _run_packed_vjp(
        moved_tissue, moved_events, seed.to(device), **arguments
    )
    _agree(expected_grads, tuple(g.cpu() for g in actual_grads), FLOAT_NAMES)


def test_autograd_reaches_the_diffusion_coefficient() -> None:
    """The public entry point differentiates the coefficient the user set."""
    tissue, events, output_count = _packed()
    coefficient = tissue[DIFFUSION_INDEX].detach().clone().requires_grad_(True)
    inputs = tuple(
        coefficient if index == DIFFUSION_INDEX else value
        for index, value in enumerate(tissue)
    )
    signal = _NativeEpg.apply(
        *inputs, *events, STATE_COUNT, output_count, 1, NO_GEOMETRY, None
    )
    signal.abs().square().sum().backward()
    assert coefficient.grad is not None
    assert coefficient.grad.abs().max() > 0
