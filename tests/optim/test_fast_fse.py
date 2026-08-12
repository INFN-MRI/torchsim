"""The direct kernel path must agree with the generic model stack."""

import pytest
import torch

from torchsim import fse_sim
from torchsim.optim import FSET2Precision
from torchsim.optim._fast_fse import FseT2Plan
from torchsim.sequence._accelerators import _pack_events
from torchsim.sequence._builders import fse_description
from torchsim.sequence._simulation import TissueProperties

ECHO_SPACING_MS = 5.0
PHASES_DEG = 90.0
T1_MS = torch.tensor([800.0, 1400.0])
T2_MS = torch.tensor([45.0, 120.0])


def _degrees(trains: int, echoes: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return 80.0 + 80.0 * torch.rand(trains, echoes, generator=generator)


def _plan(echoes: int) -> FseT2Plan:
    return FseT2Plan(
        echoes, ECHO_SPACING_MS * 1e-3, phases_rad=torch.pi * PHASES_DEG / 180.0
    )


def _generic_jacobian(degrees: torch.Tensor) -> torch.Tensor:
    _, jacobian = fse_sim(
        flip=degrees,
        phases=torch.full_like(degrees, PHASES_DEG),
        ESP=ECHO_SPACING_MS,
        T1=T1_MS,
        T2=T2_MS,
        diff="T2",
    )
    return jacobian


@pytest.mark.parametrize(("trains", "echoes"), [(1, 8), (3, 12), (7, 20)])
def test_plan_buffers_match_the_generic_packer(trains, echoes):
    """The plan hardcodes the FSE event layout; check it against the packer."""
    degrees = _degrees(trains, echoes)
    flip_rad = torch.deg2rad(degrees)
    description = fse_description(
        flip_rad,
        echo_spacing_s=ECHO_SPACING_MS * 1e-3,
        phases_rad=torch.pi * PHASES_DEG / 180.0,
    )
    generic = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    duration, kind, flip, phase, action, output_index = _plan(echoes).buffers(flip_rad)

    assert torch.equal(duration, generic.duration.expand(trains, -1))
    assert torch.equal(flip, generic.flip)
    assert torch.equal(phase, generic.phase)
    assert torch.equal(kind, generic.kind)
    assert torch.equal(action, generic.action)
    assert torch.equal(output_index, generic.output_index)


@pytest.mark.parametrize(("trains", "echoes"), [(1, 8), (3, 12), (7, 20)])
def test_jacobian_matches_generic_stack(trains, echoes):
    degrees = _degrees(trains, echoes)
    expected = _generic_jacobian(degrees)
    actual = _plan(echoes).t2_jacobian(
        torch.deg2rad(degrees), TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS)
    )
    assert actual.shape == expected.shape
    scale = expected.abs().max()
    assert ((expected - actual).abs().max() / scale) < 1e-5


def test_gradient_matches_generic_stack():
    degrees = _degrees(5, 16)

    reference = degrees.clone().requires_grad_(True)
    _generic_jacobian(reference).abs().square().sum().backward()

    actual = degrees.clone().requires_grad_(True)
    _plan(16).t2_jacobian(
        torch.deg2rad(actual), TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS)
    ).abs().square().sum().backward()

    scale = reference.grad.abs().max()
    assert ((reference.grad - actual.grad).abs().max() / scale) < 1e-5


def test_gradient_matches_finite_differences():
    """Guards the second-order adjoint independently of the generic stack."""
    degrees = _degrees(1, 6)
    objective = FSET2Precision(T1_MS, T2_MS, ECHO_SPACING_MS, phases_deg=PHASES_DEG)

    exact = degrees.clone().requires_grad_(True)
    objective(exact).backward()

    step = 1e-2
    numeric = torch.zeros_like(degrees)
    for echo in range(degrees.shape[-1]):
        up, down = degrees.clone(), degrees.clone()
        up[0, echo] += step
        down[0, echo] -= step
        numeric[0, echo] = (objective(up) - objective(down)) / (2.0 * step)

    scale = numeric.abs().max()
    assert ((numeric - exact.grad).abs().max() / scale) < 1e-3


def test_objective_reuses_one_plan_per_shape():
    objective = FSET2Precision(T1_MS, T2_MS, ECHO_SPACING_MS, phases_deg=PHASES_DEG)
    objective(_degrees(3, 12))
    objective(_degrees(5, 12))
    assert len(objective._plans) == 1
    objective(_degrees(3, 16))
    assert len(objective._plans) == 2


def test_single_train_accepts_one_dimensional_input():
    objective = FSET2Precision(T1_MS, T2_MS, ECHO_SPACING_MS, phases_deg=PHASES_DEG)
    flat = torch.full((12,), 120.0, requires_grad=True)
    objective(flat).backward()
    assert flat.grad.shape == (12,)
    assert torch.isfinite(flat.grad).all()


def test_wrong_echo_count_is_rejected():
    with pytest.raises(ValueError, match="expected 8 flip angles"):
        _plan(8).t2_jacobian(
            torch.deg2rad(_degrees(2, 9)), TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS)
        )


def _fused(echoes=12):
    from torchsim.optim._fast_fse import FseT2Optimizer

    plan = FseT2Plan(
        echoes, ECHO_SPACING_MS * 1e-3, phases_rad=torch.pi * PHASES_DEG / 180.0
    )
    return FseT2Optimizer(plan, T2_MS)


def _autograd(start, iterations):
    objective = FSET2Precision(T1_MS, T2_MS, ECHO_SPACING_MS, phases_deg=PHASES_DEG)
    flip = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([flip], lr=1.0)
    loss = None
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(flip)
        loss.backward()
        optimizer.step()
    return flip.detach(), float(loss)


@pytest.mark.parametrize("iterations", [1, 5, 25])
def test_fused_loop_tracks_the_autograd_loop(iterations):
    """The fused loop is the same optimization, not merely a similar one.

    It re-implements the objective, its cotangent, the penalties and the Adam
    step in C++, so the trajectory it produces has to match the one the
    autograd loop takes, step for step.
    """
    start = _degrees(6, 12)
    expected, expected_loss = _autograd(start, iterations)
    actual, actual_loss = _fused().run(
        start,
        TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS),
        iterations=iterations,
    )
    assert abs(expected_loss - actual_loss) < 1e-4
    # Flip angles are in degrees, so this is a small fraction of a degree.
    assert (expected - actual).abs().max() < 1e-3


def test_the_starting_train_is_left_alone():
    """The caller's tensor is an input, not scratch space."""
    start = _degrees(4, 12)
    reference = start.clone()
    optimizer = _fused()
    result, _ = optimizer.run(
        start, TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS), iterations=3
    )
    assert torch.equal(start, reference)
    assert not torch.equal(result, reference)


def test_no_iterations_returns_the_starting_train():
    start = _degrees(4, 12)
    result, _ = _fused().run(
        start, TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS), iterations=0
    )
    assert torch.equal(result, start.to(torch.float32))


def test_a_pulse_that_rescales_the_flip_angle_is_refused():
    """The fused loop writes flip angles straight into the event buffer.

    A pulse whose flip angle is not its own nominal value needs the plan's
    conversion, which lives in Python, so the loop must decline rather than
    silently optimize the wrong thing.
    """
    optimizer = _fused()
    definition = optimizer.plan._definition

    class Rescaling:
        def flip_angle(self, flip, **keywords):
            magnitude, phase = definition.flip_angle(flip, **keywords)
            return 0.5 * magnitude, phase

    optimizer.plan._definition = Rescaling()
    assert not optimizer.supports()
    with pytest.raises(RuntimeError, match="nominal"):
        optimizer.run(
            _degrees(2, 12),
            TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS),
            iterations=1,
        )
