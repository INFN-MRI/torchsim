"""Inverting the forward operator, one linearized problem at a time."""

from __future__ import annotations

import pytest
import torch

from torchsim.estimators import NonlinearLeastSquares
from functools import partial

from torchsim.recon import (
    LeastSquares,
    Linearization,
    GaussNewton,
    ModelOperator,
    Schedule,
    TrustRegion,
    iterative,
    direct,
)
from torchsim.simulators import MultiEchoSimulator

TE_MS = torch.linspace(10.0, 200.0, 12)
BOUND = {"T2": (10.0, 300.0)}
TRUTH = torch.tensor([30.0, 60.0, 90.0, 150.0, 250.0])
AMPLITUDE = torch.tensor(
    [1.0, 0.8 - 0.2j, 1.2 + 0.5j, 0.5, 1.0], dtype=torch.complex64
)


class MaskedFourier:
    """An undersampled Cartesian encoding: what a reconstruction sees.

    Written here rather than imported so the loop is tested against a real
    encoding without the test depending on one.
    """

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def A(self, images: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(images, norm="ortho") * self.mask

    def A_adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(kspace * self.mask, norm="ortho")


@pytest.fixture
def problem():
    """A multi-echo decay, its operator, and data with a known answer."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    operator = ModelOperator(acquisition, "T2", bounds=BOUND)
    measured = torch.as_tensor(acquisition.simulate(T2=TRUTH)).to(
        torch.complex64
    ) * AMPLITUDE[:, None]
    return operator, measured, operator.initial((5,), T2=100.0)


# %% the two policies


@pytest.mark.parametrize(
    "damping,solve",
    [
        (TrustRegion(), direct),
        (Schedule(minimum=1e-8), direct),
        (Schedule(minimum=1e-8), iterative()),
    ],
)
def test_every_pairing_finds_the_answer(problem, damping, solve) -> None:
    """Which inner solve and which damping is a choice, not a different problem.

    The tolerance is a tenth of a percent rather than round-off because an
    iterative inner solve stops on one: deepinv's conjugate gradients works on
    the normal equations, whose condition number is the square of the
    system's, so it declares itself done while the direct solve is still
    tightening. That is a property of the solver the caller chose, not a
    disagreement about where the minimum is.
    """
    operator, measured, start = problem

    found = GaussNewton(damping, solve=solve, max_iterations=40).minimize(
        operator, measured, start
    )

    maps = operator.split(found.x)
    torch.testing.assert_close(maps["T2"], TRUTH, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        maps["amplitude"], AMPLITUDE, rtol=1e-3, atol=1e-4
    )


def test_the_two_policies_land_in_the_same_place(problem) -> None:
    """A trust region and a schedule differ in how they get there, not where."""
    operator, measured, start = problem

    region = GaussNewton(TrustRegion(), solve=direct, max_iterations=40).minimize(
        operator, measured, start
    )
    schedule = GaussNewton(
        Schedule(minimum=1e-8), solve=iterative(), max_iterations=40
    ).minimize(operator, measured, start)

    torch.testing.assert_close(
        operator.split(region.x)["T2"],
        operator.split(schedule.x)["T2"],
        rtol=1e-3,
        atol=1e-3,
    )


def test_a_trust_region_is_the_fit_that_ships(problem) -> None:
    """The generalization is exact: the loop reproduces the estimator.

    :class:`~torchsim.NonlinearLeastSquares` is this loop under a per-voxel
    trust region with the voxel-diagonal solve, and nothing else.
    """
    acquisition = MultiEchoSimulator(TE=TE_MS)
    bounds = {"T2": (10.0, 300.0), "M0": (0.0, 5.0)}
    mapping = NonlinearLeastSquares(
        acquisition, bounds=bounds, initial={"T2": 100.0, "M0": 1.0}
    ).fit(T2=(10.0, 300.0), M0=(0.2, 2.0), seed=0, samples=256)
    scaling = torch.tensor([1.0, 0.8, 1.2, 0.5, 1.0])
    measured = acquisition.simulate(T2=TRUTH, M0=scaling)

    estimator = mapping(measured)

    operator = ModelOperator(
        acquisition, "T2", "M0", bounds=bounds, amplitude=False
    )
    found = GaussNewton(TrustRegion(), solve=direct, max_iterations=20).minimize(
        operator, measured, operator.initial((5,), T2=100.0, M0=1.0)
    )
    maps = operator.split(found.x)
    torch.testing.assert_close(maps["T2"], estimator["T2"])
    torch.testing.assert_close(maps["M0"], estimator["M0"])


def test_the_residual_falls(problem) -> None:
    """Reported per step, so convergence is read rather than assumed."""
    operator, measured, start = problem

    found = GaussNewton(TrustRegion(), solve=direct, max_iterations=40).minimize(
        operator, measured, start
    )

    assert found.cost.numel() == found.iterations + 1
    assert float(found.cost[-1]) < 1e-6 * float(found.cost[0])
    assert bool((torch.diff(found.cost) <= 1e-6).all())
    assert found.unconverged == 0


def test_the_damping_of_a_schedule_falls_to_its_floor(problem) -> None:
    """The regularization is released as the iterate settles, and no further."""
    operator, measured, start = problem

    found = GaussNewton(
        Schedule(initial=1.0, factor=0.5, minimum=0.05),
        solve=iterative(),
        max_iterations=10,
    ).minimize(operator, measured, start)

    assert float(found.damping[0]) == 1.0
    assert float(found.damping[-1]) >= 0.05
    assert bool((torch.diff(found.damping) <= 0).all())


# %% an encoding operator in front of the model


@pytest.fixture
def phantom():
    """A small T2 phantom, its k-space, and a mask that undersamples it."""
    generator = torch.Generator().manual_seed(0)
    size = 32
    t2 = torch.full((size, size), 60.0)
    t2[8:24, 8:24] = 150.0
    amplitude = torch.zeros(size, size, dtype=torch.complex64)
    amplitude[4:28, 4:28] = 1.0
    acquisition = MultiEchoSimulator(TE=TE_MS)
    operator = ModelOperator(acquisition, "T2", bounds=BOUND)
    images = torch.as_tensor(acquisition.simulate(T2=t2)).to(
        torch.complex64
    ) * amplitude[..., None]
    mask = (
        torch.rand(
            (1, TE_MS.numel(), size, size), generator=generator
        )
        < 0.35
    )
    mask[:, :, :, size // 2 - 3 : size // 2 + 3] = True
    encoding = MaskedFourier(mask)
    kspace = encoding.A(images.movedim(-1, 0)[None])
    return operator, encoding, kspace, t2, amplitude


def test_solving_through_an_encoding_beats_reconstructing_first(
    phantom,
) -> None:
    """The claim the whole approach rests on, on data that actually needs it.

    Filling the gaps with zeros and fitting the images that come back puts the
    undersampling into the parameter maps. Keeping the encoding in the
    operator does not.
    """
    operator, encoding, kspace, t2, amplitude = phantom
    inside = amplitude.abs() > 0
    start = operator.initial((1, 32, 32), T2=100.0)

    found = GaussNewton(
        Schedule(initial=1e-2, factor=0.5, minimum=1e-5),
        solve=iterative(),
        max_iterations=12,
    ).minimize(operator, kspace, start, encoding=encoding)

    modelled = operator.split(found.x)["T2"][0]
    gridded = encoding.A_adjoint(kspace)[0].movedim(0, -1)
    fitted = GaussNewton(TrustRegion(), solve=direct, max_iterations=30).minimize(
        operator, gridded, operator.initial((32, 32), T2=100.0)
    )
    naive = operator.split(fitted.x)["T2"]

    error = lambda found: float((found[inside] - t2[inside]).abs().mean())
    assert error(modelled) < error(naive)


def test_the_encoded_loop_lowers_its_residual(phantom) -> None:
    """Data consistency in k-space, which is where the data is."""
    operator, encoding, kspace, _, _ = phantom
    start = operator.initial((1, 32, 32), T2=100.0)

    found = GaussNewton(
        Schedule(initial=1e-2, factor=0.5, minimum=1e-5),
        solve=iterative(),
        max_iterations=8,
    ).minimize(operator, kspace, start, encoding=encoding)

    assert float(found.cost[-1]) < 0.2 * float(found.cost[0])
    assert found.cost.ndim == 1


# %% what does not make sense


def test_a_per_voxel_damping_is_refused_under_an_encoding(phantom) -> None:
    """A trust region needs independent rows, and encoding leaves none."""
    operator, encoding, kspace, _, _ = phantom

    with pytest.raises(ValueError, match="each voxel on its own"):
        GaussNewton(TrustRegion(), solve=iterative()).minimize(
            operator,
            kspace,
            operator.initial((1, 32, 32), T2=100.0),
            encoding=encoding,
        )


def test_the_direct_solve_says_when_it_cannot_be_used(phantom) -> None:
    """It factorizes per-voxel blocks, and an encoding leaves no blocks."""
    operator, encoding, kspace, _, _ = phantom

    with pytest.raises(ValueError, match="no encoding"):
        GaussNewton(Schedule(), solve=direct, max_iterations=1).minimize(
            operator,
            kspace,
            operator.initial((1, 32, 32), T2=100.0),
            encoding=encoding,
        )


def test_a_loop_needs_a_step_to_take() -> None:
    """Caught where it is written."""
    with pytest.raises(ValueError, match="max_iterations"):
        GaussNewton(max_iterations=0)


# %% who solves the linearized problem


def test_the_direct_solve_is_exact(problem) -> None:
    """The damping goes in as extra rows, not as a ridge on normal equations.

    Same minimizer, without squaring the condition number -- and against a
    dense solve of the augmented system built here, which shares no code with
    it.
    """
    operator, measured, start = problem
    blocks = operator.jacobian(start)
    residual = operator.A(start) - measured
    damping = torch.full((start.shape[0], 1), 0.3)
    reference = torch.zeros_like(start)

    step = direct(
        Linearization(matvec=None, rmatvec=None, blocks=blocks),
        -residual,
        damping,
        reference,
    )

    rows = torch.cat((blocks.real, blocks.imag), dim=-1).mT
    target = torch.cat((-residual.real, -residual.imag), dim=-1)
    eye = torch.eye(start.shape[-1]).expand(start.shape[0], -1, -1)
    system = torch.cat((rows, damping.sqrt()[..., None] * eye), dim=-2)
    padded = torch.cat((target, torch.zeros_like(reference)), dim=-1)
    dense = torch.linalg.solve(
        system.mT @ system, (system.mT @ padded[..., None])
    ).squeeze(-1)
    torch.testing.assert_close(step, dense, atol=1e-4, rtol=1e-4)


def test_the_inner_solve_defaults_to_what_the_problem_needs(problem) -> None:
    """Direct where the voxels are separate, deepinv's where they are not.

    Naming a solver is a choice worth having and not one worth making every
    time.
    """
    operator, measured, start = problem

    found = GaussNewton(TrustRegion(), max_iterations=40).minimize(
        operator, measured, start
    )

    torch.testing.assert_close(
        operator.split(found.x)["T2"], TRUTH, rtol=1e-3, atol=1e-3
    )


def test_the_inner_solve_defaults_under_an_encoding(phantom) -> None:
    """The same, with an encoding operator: nothing to name, and it runs."""
    operator, encoding, kspace, _, _ = phantom

    found = GaussNewton(
        Schedule(initial=1e-2, factor=0.5, minimum=1e-5), max_iterations=6
    ).minimize(
        operator, kspace, operator.initial((1, 32, 32), T2=100.0), encoding=encoding
    )

    assert float(found.cost[-1]) < 0.3 * float(found.cost[0])




@pytest.mark.parametrize("name", ["CG", "lsqr", "BiCGStab", "minres"])
def test_one_of_deepinvs_solvers_is_that_function_with_its_argument_bound(
    problem, name
) -> None:
    """Which one suits is the caller's to measure, so all of them must run.

    Binding the argument is the caller's own composition -- TorchSim takes an
    object and never a name.
    """
    least_squares = pytest.importorskip("deepinv.optim.linear").least_squares
    operator, measured, start = problem

    found = GaussNewton(
        Schedule(minimum=1e-8),
        solve=iterative(partial(least_squares, solver=name)),
        max_iterations=25,
    ).minimize(operator, measured, start)

    assert float(found.cost[-1]) < 1e-3 * float(found.cost[0])


class ConjugateGradients:
    """A least-squares solve written here, matching :class:`LeastSquares`.

    Conjugate gradients on ``(A^H A + (1/gamma) I) d = A^H y + (1/gamma) z``,
    which is the normal equations of what the protocol asks to be minimized.
    Written in the test so the duck-typed path is exercised without deepinv.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, A, AT, y, z, gamma, max_iter, tol):
        self.calls += 1
        weight = 0.0 if gamma is None else 1.0 / gamma

        def normal(d):
            return AT(A(d)) + weight * d

        d = torch.zeros_like(z)
        residual = AT(y) + weight * z - normal(d)
        direction = residual.clone()
        squared = (residual.conj() * residual).real.sum()
        for _ in range(max_iter):
            if squared <= tol**2:
                break
            applied = normal(direction)
            step = squared / ((direction.conj() * applied).real.sum() + 1e-30)
            d = d + step * direction
            residual = residual - step * applied
            update = (residual.conj() * residual).real.sum()
            direction = residual + (update / (squared + 1e-30)) * direction
            squared = update
        return d


def test_a_solver_object_needs_no_deepinv(problem) -> None:
    """Anything matching LeastSquares is taken, so deepinv is never forced."""
    operator, measured, start = problem
    mine = ConjugateGradients()

    found = GaussNewton(
        Schedule(minimum=1e-8), solve=iterative(mine), max_iterations=25
    ).minimize(operator, measured, start)

    assert mine.calls > 0
    assert isinstance(mine, LeastSquares)
    assert float(found.cost[-1]) < 1e-3 * float(found.cost[0])


def test_a_solver_object_reaches_what_the_default_does(problem) -> None:
    """The two paths differ in who solves, not in what is solved."""
    operator, measured, start = problem

    loop = lambda solve: GaussNewton(  # noqa: E731
        Schedule(minimum=1e-8), solve=solve, max_iterations=25
    ).minimize(operator, measured, start)

    assert torch.allclose(
        loop(iterative(ConjugateGradients())).x, loop(iterative()).x, rtol=1e-2
    )
