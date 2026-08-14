"""The fused adjoints must agree with the state machine written out in torch."""

from __future__ import annotations

import pytest
import torch

from torchsim import (
    FSE,
    TissueProperties,
    fse_description,
    mprage_description,
    mrf_description,
    spgr_description,
)
from torchsim.sequence import _accelerators
from torchsim.sequence._accelerators import (
    NO_GEOMETRY,
    _NativeEpg,
    _pack_events,
    _run_packed_vjp,
)
from torchsim.sequence._parameters import FLOAT_NAMES, TISSUE_COUNT
from torchsim.sequence._simulation import _prepare_tissue
from utils.packed_reference import simulate_packed

# Gradient tuples arrive in the packing order, so name them from the same place.
PARAMETER_NAMES = list(FLOAT_NAMES)
ECHOES = 6


def _tissue():
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1200.0]),
        t2_ms=torch.tensor([50.0, 90.0]),
        m0=torch.tensor([1.0, 0.8]),
        b1=torch.tensor([1.0, 0.9]),
        b1_phase_rad=torch.tensor([0.05, -0.1]),
        b0_hz=torch.tensor([13.0, -27.0]),
        inversion_efficiency=torch.tensor([0.95, 0.9]),
    )


def _descriptions():
    flip = torch.deg2rad(torch.full((ECHOES,), 140.0))
    return {
        "fse": (
            "fse",
            fse_description(flip, echo_spacing_s=5e-3, phases_rad=torch.pi / 2),
        ),
        "spgr": (
            "spgr",
            spgr_description(torch.deg2rad(torch.full((ECHOES,), 15.0)), 8e-3, 3e-3),
        ),
        "mrf": (
            "ssfpfid",
            mrf_description(
                torch.deg2rad(torch.linspace(5.0, 60.0, ECHOES)),
                torch.full((ECHOES,), 10e-3),
                inversion_time_s=20e-3,
            ),
        ),
        "mprage": (
            "spgr",
            mprage_description(
                2, ECHOES - 2, torch.deg2rad(torch.tensor(8.0)), 8e-3, 20e-3
            ),
        ),
    }


def _packed(name: str = "fse", state_count: int = 8):
    """Prepared tissue and packed events, the pair both routes are fed."""
    policy, description = _descriptions()[name]
    prepared, _, device = _prepare_tissue(_tissue(), "cpu")
    packed = _pack_events(
        policy,
        description,
        repetitions=1,
        record="all",
        device=device,
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    return prepared, events, packed.output_count, state_count


def _leaves(prepared, events):
    """Differentiable copies of the tissue and of the three float event buffers."""
    tissue = tuple(value.detach().clone().requires_grad_(True) for value in prepared)
    differentiable_events = (
        events[0].detach().clone().requires_grad_(True),
        events[1],
        events[2].detach().clone().requires_grad_(True),
        events[3].detach().clone().requires_grad_(True),
        events[4],
        events[5],
        events[6],
    )
    return tissue, differentiable_events


def _route(route: str, tissue, events, output_count: int, state_count: int):
    if route == "kernel":
        return _NativeEpg.apply(
            *tissue, *events, state_count, output_count, 1, NO_GEOMETRY
        )
    return simulate_packed(
        tissue, events, state_count=state_count, output_count=output_count
    )


def _agree(expected, actual, names, tolerance: float = 1e-3) -> None:
    """Compare gradient tuples, skipping the entries float32 cannot resolve.

    These gradients span many orders of magnitude -- the one w.r.t. proton
    density is eight above the one w.r.t. transmit phase under CPMG phases --
    and a component that far below the largest is under the rounding of the
    sums that produced it, so its own relative error carries no signal.
    """
    scales = {
        name: value.abs().max().item()
        for name, value in zip(names, expected, strict=True)
        if value is not None
    }
    floor = 1e-6 * max(scales.values())
    compared = 0
    for name, want, got in zip(names, expected, actual, strict=True):
        if want is None or scales[name] <= floor:
            continue
        error = (want - got).abs().max().item() / scales[name]
        assert error < tolerance, f"{name} differs by {error:.2e}"
        compared += 1
    assert compared > 0


@pytest.mark.parametrize("name", sorted(_descriptions()))
@pytest.mark.parametrize("threads", [1, 4])
def test_fused_vjp_matches_the_reference(name: str, threads: int) -> None:
    policy, description = _descriptions()[name]
    prepared, _, device = _prepare_tissue(_tissue(), "cpu")
    packed = _pack_events(
        policy,
        description,
        repetitions=1,
        record="all",
        device=device,
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    state_count = 8
    output_count = packed.output_count

    leaves = tuple(value.detach().clone().requires_grad_(True) for value in prepared)
    differentiable_events = (
        events[0].detach().clone().requires_grad_(True),
        events[1],
        events[2].detach().clone().requires_grad_(True),
        events[3].detach().clone().requires_grad_(True),
        events[4],
        events[5],
        events[6],
    )
    output = simulate_packed(
        leaves,
        differentiable_events,
        state_count=state_count,
        output_count=output_count,
    )
    torch.manual_seed(0)
    seed = torch.randn(output.shape, dtype=torch.complex64)
    reference = torch.autograd.grad(
        output,
        (
            *leaves,
            differentiable_events[0],
            differentiable_events[2],
            differentiable_events[3],
        ),
        grad_outputs=seed,
        allow_unused=True,
    )
    fused = _run_packed_vjp(
        prepared,
        events,
        seed,
        state_count=state_count,
        output_count=output_count,
        threads=threads,
    )

    for parameter, expected, actual in zip(
        PARAMETER_NAMES, reference, fused, strict=True
    ):
        if expected is None:
            continue
        scale = expected.abs().max().item()
        if scale < 1e-7:
            # gradient is numerically zero; nothing meaningful to compare
            continue
        error = (expected - actual).abs().max().item() / scale
        assert error < 1e-4, f"{parameter} gradient differs by {error:.2e}"


def _gradients():
    description = fse_description(
        torch.deg2rad(torch.full((12,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )
    t1 = torch.full((64,), 1000.0, requires_grad=True)
    t2 = torch.linspace(40.0, 120.0, 64).requires_grad_(True)
    signal = FSE().simulate(
        description, TissueProperties(t1_ms=t1, t2_ms=t2), nstates=10
    ).signal
    return torch.autograd.grad(signal.abs().square().sum(), (t1, t2))


def test_fused_vjp_is_bitwise_deterministic(monkeypatch) -> None:
    first = _gradients()
    for _ in range(3):
        for expected, actual in zip(first, _gradients(), strict=True):
            assert torch.equal(expected, actual)

    monkeypatch.setenv("TORCHSIM_NUM_THREADS", "1")
    single = _gradients()
    monkeypatch.setenv("TORCHSIM_NUM_THREADS", "8")
    for expected, actual in zip(single, _gradients(), strict=True):
        assert torch.equal(expected, actual)


def _objective_gradient():
    from torchsim import FSET2Precision

    objective = FSET2Precision(
        torch.tensor([800.0, 1400.0]), torch.tensor([45.0, 120.0]), 5.0
    )
    flip = torch.linspace(150.0, 90.0, 12).requires_grad_(True)
    loss = objective(flip)
    (gradient,) = torch.autograd.grad(loss, flip)
    return loss.detach(), gradient


def _forward_over_reverse(route: str):
    """Differentiate a directional derivative, which is what backward fuses."""
    prepared, events, output_count, state_count = _packed()
    tissue, differentiable_events = _leaves(prepared, events)
    inputs = (*tissue, differentiable_events[0], differentiable_events[2])
    tangents = tuple(torch.ones_like(value) for value in inputs)

    def simulate(*values):
        # The tissue properties, then duration, the untouched kind buffer, and
        # flip, followed by the rest of the packed events.
        tissue_values = values[:TISSUE_COUNT]
        rebuilt = (
            values[TISSUE_COUNT],
            events[1],
            values[TISSUE_COUNT + 1],
            *events[3:],
        )
        return _route(route, tissue_values, rebuilt, output_count, state_count)

    _signal, derivative = torch.func.jvp(simulate, inputs, tangents)
    loss = derivative.abs().square().sum()
    return loss.detach(), torch.autograd.grad(loss, inputs, allow_unused=True)


def test_forward_over_reverse_matches_the_reference() -> None:
    """The second-order path drives sequence optimization, so it must agree."""
    reference_loss, reference = _forward_over_reverse("reference")
    fused_loss, fused = _forward_over_reverse("kernel")

    assert torch.allclose(reference_loss, fused_loss, rtol=1e-5)
    _agree(reference, fused, PARAMETER_NAMES[:-1])


def test_forward_over_reverse_is_deterministic() -> None:
    _, first = _objective_gradient()
    for _ in range(3):
        assert torch.equal(first, _objective_gradient()[1])


# --- the adjoint differentiated a second time ---


def _second_order(route: str):
    """``d<v, J^T w>`` w.r.t. both the inputs and the seed ``w``.

    Reverse over reverse: the first pass builds a graph through the adjoint,
    the second differentiates the contraction of that adjoint with a fixed
    direction. Returns the ten input gradients followed by the seed's.
    """
    prepared, events, output_count, state_count = _packed()
    tissue, differentiable_events = _leaves(prepared, events)
    inputs = (
        *tissue,
        differentiable_events[0],
        differentiable_events[2],
        differentiable_events[3],
    )
    generator = torch.Generator().manual_seed(3)
    seed = torch.randn(
        (prepared[0].numel(), output_count),
        generator=generator,
        dtype=torch.complex64,
    ).requires_grad_(True)
    directions = tuple(
        torch.randn(value.shape, generator=generator) for value in inputs
    )

    signal = _route(route, tissue, differentiable_events, output_count, state_count)
    gradients = torch.autograd.grad(
        signal, inputs, seed, create_graph=True, allow_unused=True
    )
    contraction = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(gradients, directions, strict=True)
        if gradient is not None
    )
    return torch.autograd.grad(contraction, (*inputs, seed), allow_unused=True)


def test_the_second_derivative_matches_the_reference() -> None:
    """Two kernel calls stand in for differentiating the adjoint's own graph."""
    _agree(
        _second_order("reference"),
        _second_order("kernel"),
        [*PARAMETER_NAMES, "seed"],
    )


def test_a_hessian_vector_product_matches_finite_differences() -> None:
    """An oracle that shares no code with either route.

    Stepping the inputs along the direction and differencing the adjoint gives
    the same curvature the second pass computes analytically.
    """
    prepared, events, output_count, state_count = _packed()
    generator = torch.Generator().manual_seed(3)
    seed = torch.randn(
        (prepared[0].numel(), output_count),
        generator=generator,
        dtype=torch.complex64,
    )
    direction = torch.zeros_like(prepared[1])
    direction[:] = 1.0

    def adjoint(t2):
        return _run_packed_vjp(
            (prepared[0], t2, *prepared[2:]),
            events,
            seed,
            state_count=state_count,
            output_count=output_count,
            threads=1,
        )[1]

    step = 1e-2 * prepared[1].abs().max()
    expected = (adjoint(prepared[1] + step * direction)
                - adjoint(prepared[1] - step * direction)) / (2.0 * step)

    leaves = tuple(value.detach().clone().requires_grad_(True) for value in prepared)
    signal = _route("kernel", leaves, events, output_count, state_count)
    (gradient,) = torch.autograd.grad(
        signal, (leaves[1],), seed.detach(), create_graph=True
    )
    (actual,) = torch.autograd.grad((gradient * direction).sum(), (leaves[1],))

    scale = expected.abs().max().item()
    assert scale > 0.0
    assert ((expected - actual).abs().max().item() / scale) < 1e-2


def test_building_a_graph_does_not_change_the_first_derivative() -> None:
    """One adjoint serves both, so asking to keep it cannot move the answer."""

    def gradient(create_graph):
        prepared, events, output_count, state_count = _packed()
        tissue, differentiable_events = _leaves(prepared, events)
        signal = _route(
            "kernel", tissue, differentiable_events, output_count, state_count
        )
        return torch.autograd.grad(
            signal,
            (*tissue, differentiable_events[2]),
            torch.ones_like(signal),
            create_graph=create_graph,
        )

    for plain, kept in zip(gradient(False), gradient(True), strict=True):
        assert torch.equal(plain, kept.detach())


def _order(route: str, order: int):
    """Differentiate a scalar of the signal ``order`` times over."""
    prepared, events, output_count, state_count = _packed()
    t2 = prepared[1].detach().clone().requires_grad_(True)
    tissue = (prepared[0], t2, *prepared[2:])
    signal = _route(route, tissue, events, output_count, state_count)
    value = signal.abs().square().sum()
    for _ in range(order):
        (value,) = torch.autograd.grad(value, t2, create_graph=True)
        value = value.sum()
    return value


@pytest.mark.parametrize("order", [1, 2])
def test_the_first_two_derivatives_match_the_reference(order: int) -> None:
    """As far as the kernels go, they must land where the reference does."""
    expected = _order("reference", order)
    actual = _order("kernel", order)
    assert torch.allclose(expected, actual, rtol=1e-4)


def test_a_third_derivative_is_refused() -> None:
    """Two passes reach two orders, and the third must not be quietly short.

    Autograd can route a third derivative around the missing term -- the inputs
    the second one was computed from are still in the graph -- and return an
    answer that looks like one.
    """
    with pytest.raises(RuntimeError, match="first and second derivatives"):
        _order("kernel", 3)


def test_the_second_pass_reaches_the_kernels() -> None:
    """Both halves of ``d<v, J^T w>`` are kernel calls, not a torch graph."""
    seen = {"curvature": 0, "seed": 0}
    original_curvature = _accelerators._run_packed_vjp_jvp
    original_seed = _accelerators._run_packed_jvp

    def curvature(*arguments, **keywords):
        seen["curvature"] += 1
        return original_curvature(*arguments, **keywords)

    def directional(*arguments, **keywords):
        seen["seed"] += 1
        return original_seed(*arguments, **keywords)

    _accelerators._run_packed_vjp_jvp = curvature
    _accelerators._run_packed_jvp = directional
    try:
        _second_order("kernel")
    finally:
        _accelerators._run_packed_vjp_jvp = original_curvature
        _accelerators._run_packed_jvp = original_seed

    assert seen == {"curvature": 1, "seed": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_card_has_an_analytic_first_order_adjoint() -> None:
    """No direction to follow makes the second-order kernel an adjoint."""
    prepared, events, output_count, state_count = _packed()
    generator = torch.Generator().manual_seed(5)
    seed = torch.randn(
        (prepared[0].numel(), output_count),
        generator=generator,
        dtype=torch.complex64,
    )
    expected = _run_packed_vjp(
        prepared,
        events,
        seed,
        state_count=state_count,
        output_count=output_count,
        threads=1,
    )
    card = torch.device("cuda")
    actual = _run_packed_vjp(
        tuple(value.to(card) for value in prepared),
        tuple(value.to(card) for value in events),
        seed.to(card),
        state_count=state_count,
        output_count=output_count,
        threads=1,
    )
    _agree(expected, tuple(value.cpu() for value in actual), PARAMETER_NAMES)
