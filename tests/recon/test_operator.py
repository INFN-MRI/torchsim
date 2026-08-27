"""The signal model as an operator a reconstruction can compose with."""

from __future__ import annotations

import pytest
import torch

from torchsim import Subspace, execution
from torchsim.recon import ModelOperator
from torchsim.simulators import FSESimulator, MultiEchoSimulator

TE_MS = torch.tensor([10.0, 20.0, 40.0, 80.0, 160.0])
BOUND = {"T2": (10.0, 300.0)}


@pytest.fixture
def operator():
    """A multi-echo decay over one bounded unknown, with an amplitude."""
    return ModelOperator(
        MultiEchoSimulator(TE=TE_MS), "T2", bounds=BOUND
    )


@pytest.fixture
def point(operator):
    """Maps to evaluate at, spread so no two voxels share an answer."""
    generator = torch.Generator().manual_seed(0)
    x = operator.initial((7,), T2=80.0)
    return x + 0.3 * torch.randn(x.shape, generator=generator)


# %% what the operator computes


def test_the_operator_is_the_model(operator, point) -> None:
    """``A`` is the simulator, evaluated at the maps and scaled."""
    maps = operator.split(point)

    recorded = operator.A(point)

    expected = torch.as_tensor(
        operator.acquisition.simulate(T2=maps["T2"])
    ).to(recorded.dtype) * maps["amplitude"][:, None]
    torch.testing.assert_close(recorded, expected)


def test_the_maps_come_back_named(operator, point) -> None:
    """Nobody counts channels: the amplitude is one complex map, not two."""
    maps = operator.split(point)

    assert set(maps) == {"T2", "amplitude"}
    assert torch.is_complex(maps["amplitude"])
    assert operator.names == ("T2", "amplitude.real", "amplitude.imag")
    assert operator.channels == 3


def test_a_starting_point_is_the_value_it_was_asked_for(operator) -> None:
    """What a reconstruction begins at, stated in the property's own units."""
    x = operator.initial((4, 5), T2=42.0, amplitude=2.0 - 1.0j)

    maps = operator.split(x)

    assert x.shape == (4, 5, 3)
    torch.testing.assert_close(maps["T2"], torch.full((4, 5), 42.0))
    torch.testing.assert_close(
        maps["amplitude"], torch.full((4, 5), 2.0 - 1.0j, dtype=torch.complex64)
    )


def test_a_starting_point_with_no_middle_must_be_given() -> None:
    """Half a bound has no midpoint to fall back to, so it says so."""
    operator = ModelOperator(
        MultiEchoSimulator(TE=TE_MS),
        "T2",
        bounds={"T2": (0.0, None)},
    )

    with pytest.raises(ValueError, match="two-sided"):
        operator.initial((3,))


def test_a_starting_point_on_a_bound_is_refused(operator) -> None:
    """A value sitting on a bound has no unconstrained image."""
    with pytest.raises(ValueError, match="strictly inside"):
        operator.initial((3,), T2=10.0)


# %% the derivatives


def test_the_directional_derivative_is_the_jacobian_contracted(
    operator, point
) -> None:
    """One forward pass gives what the built blocks give, at any width."""
    generator = torch.Generator().manual_seed(1)
    direction = torch.randn(point.shape, generator=generator)

    forward = operator.A_jvp(point, direction)

    blocks = operator.jacobian(point)
    torch.testing.assert_close(
        forward,
        torch.einsum("vpt,vp->vt", blocks, direction.to(blocks.dtype)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_the_directional_derivative_is_the_slope_of_the_model(
    operator, point
) -> None:
    """Against a central difference, which knows nothing about our chain rule."""
    generator = torch.Generator().manual_seed(2)
    direction = torch.randn(point.shape, generator=generator)
    step = 1e-3

    forward = operator.A_jvp(point, direction)

    difference = (
        operator.A(point + step * direction) - operator.A(point - step * direction)
    ) / (2 * step)
    torch.testing.assert_close(forward, difference, atol=1e-3, rtol=1e-3)


def test_the_adjoint_is_the_adjoint(operator, point) -> None:
    """``<J d, v> == <d, J^H v>`` under the real inner product.

    The one test that catches a conjugate in the wrong place, which nothing
    downstream would report as anything but slow convergence.
    """
    generator = torch.Generator().manual_seed(3)
    direction = torch.randn(point.shape, generator=generator)
    cotangent = torch.randn(
        (point.shape[0], TE_MS.numel()), generator=generator, dtype=torch.complex64
    )

    forward = operator.A_jvp(point, direction)
    backward = operator.A_vjp(point, cotangent)

    torch.testing.assert_close(
        (forward.conj() * cotangent).sum().real,
        (direction * backward).sum(),
        atol=1e-4,
        rtol=1e-5,
    )
    assert not torch.is_complex(backward)


def test_the_amplitude_costs_no_pass(operator, point) -> None:
    """Its two columns are the model output, times one and times i, exactly."""
    blocks = operator.jacobian(point)
    maps = operator.split(point)

    signal = torch.as_tensor(
        operator.acquisition.simulate(T2=maps["T2"])
    ).to(blocks.dtype)

    assert bool((blocks[:, -2] == signal).all())
    assert bool((blocks[:, -1] == 1j * signal).all())


def test_a_state_machine_model_differentiates_both_ways() -> None:
    """Forward and reverse both reach the fused engine, not only forward."""
    acquisition = FSESimulator(
        ESP=5.0,
        TR=1800.0,
        T1=1000.0,
        flip=torch.full((24,), 150.0),
    )
    operator = ModelOperator(acquisition, "T2", bounds={"T2": (20.0, 300.0)})
    x = operator.initial((5,), T2=90.0)
    generator = torch.Generator().manual_seed(4)
    direction = torch.randn(x.shape, generator=generator)

    forward = operator.A_jvp(x, direction)

    cotangent = torch.randn(
        forward.shape, generator=generator, dtype=torch.complex64
    )
    backward = operator.A_vjp(x, cotangent)
    torch.testing.assert_close(
        (forward.conj() * cotangent).sum().real,
        (direction * backward).sum(),
        atol=1e-4,
        rtol=1e-5,
    )


# %% bounds


def test_a_bound_holds_however_far_the_variable_goes(operator) -> None:
    """The interval is where the parameter lives, not a penalty on leaving it."""
    wild = 40.0 * torch.randn(4096, 3, generator=torch.Generator().manual_seed(5))

    maps = operator.split(wild)

    assert float(maps["T2"].min()) >= 10.0
    assert float(maps["T2"].max()) <= 300.0
    assert torch.isfinite(operator.A(wild)).all()


def test_a_scale_sets_the_size_of_a_step_in_an_unbounded_parameter() -> None:
    """Without a bound to normalize it, the caller says what a step is worth."""
    operator = ModelOperator(
        MultiEchoSimulator(TE=TE_MS),
        "T2",
        scale={"T2": 100.0},
        amplitude=False,
    )

    maps = operator.split(torch.tensor([[0.8]]))

    torch.testing.assert_close(maps["T2"], torch.tensor([80.0]))


@pytest.mark.parametrize(
    "settings,complaint",
    [
        (dict(bounds={"T1": (0.0, 1.0)}), "bounds names"),
        (dict(scale={"T1": 2.0}), "scale names"),
        (dict(bounds={"T2": (300.0, 10.0)}), "not increasing"),
        (dict(scale={"T2": 0.0}), "scale must be positive"),
    ],
)
def test_settings_that_make_no_sense(settings, complaint) -> None:
    """Caught where they are written, not at the first iterate."""
    with pytest.raises(ValueError, match=complaint):
        ModelOperator(
            MultiEchoSimulator(TE=TE_MS), "T2", **settings
        )


def test_an_operator_needs_something_to_solve_for() -> None:
    """An operator over no unknowns is a simulator, and there is one already."""
    with pytest.raises(ValueError, match="at least one"):
        ModelOperator(MultiEchoSimulator(TE=TE_MS))


# %% a subspace, and where the work runs


def test_a_subspace_shortens_what_comes_out() -> None:
    """A reconstruction that solves in the basis gets its prediction there."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 32))
    grid = torch.linspace(20.0, 300.0, 64)
    subspace = Subspace.fit(torch.as_tensor(acquisition.simulate(T2=grid)), 4)
    operator = ModelOperator(
        acquisition, "T2", bounds=BOUND, subspace=subspace
    )
    x = operator.initial((6,), T2=80.0)

    coefficients = operator.A(x)

    assert coefficients.shape == (6, 4)
    plain = ModelOperator(acquisition, "T2", bounds=BOUND)
    torch.testing.assert_close(
        coefficients, subspace.project(plain.A(x)), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_a_policy_changes_where_the_work_runs_and_nothing_else(
    operator, target
) -> None:
    """Chunked, streamed or spread, the answer is the answer."""
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator().manual_seed(6)
    x = operator.initial((20_000,), T2=80.0)
    x = x + 0.2 * torch.randn(x.shape, generator=generator)
    direction = torch.randn(x.shape, generator=generator)
    cotangent = torch.randn(
        (x.shape[0], TE_MS.numel()), generator=generator, dtype=torch.complex64
    )
    here = (
        operator.A(x),
        operator.A_jvp(x, direction),
        operator.A_vjp(x, cotangent),
    )

    with execution(target):
        there = (
            operator.A(x),
            operator.A_jvp(x, direction),
            operator.A_vjp(x, cotangent),
        )

    for ours, theirs in zip(here, there, strict=True):
        assert theirs.device == x.device
        torch.testing.assert_close(theirs, ours, atol=1e-5, rtol=1e-5)


def test_streaming_a_volume_too_big_for_the_budget(operator) -> None:
    """The chunking is the policy's, and the seams do not show."""
    x = operator.initial((20_000,), T2=80.0)
    here = operator.A(x)

    with execution("cpu", stream=True, budget_bytes=1 << 20):
        there = operator.A(x)

    torch.testing.assert_close(there, here)
