"""Inverting the forward operator, one linearized problem at a time."""

from __future__ import annotations

import pytest
import torch

from torchsim import Acquisition, ParameterMapping
from torchsim.estimators import NonlinearLeastSquares
from torchsim.recon import (
    GaussNewton,
    ModelOperator,
    Schedule,
    TrustRegion,
    cg,
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
    acquisition = Acquisition(MultiEchoSimulator(TE=TE_MS))
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
        (Schedule(minimum=1e-8), cg),
    ],
)
def test_every_pairing_finds_the_answer(problem, damping, solve) -> None:
    """Which inner solve and which damping is a choice, not a different problem."""
    operator, measured, start = problem

    found = GaussNewton(damping, solve=solve, max_iterations=40).minimize(
        operator, measured, start
    )

    maps = operator.split(found.x)
    torch.testing.assert_close(maps["T2"], TRUTH, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(maps["amplitude"], AMPLITUDE, atol=1e-5, rtol=1e-5)


def test_the_two_policies_land_in_the_same_place(problem) -> None:
    """A trust region and a schedule differ in how they get there, not where."""
    operator, measured, start = problem

    region = GaussNewton(TrustRegion(), solve=direct, max_iterations=40).minimize(
        operator, measured, start
    )
    schedule = GaussNewton(
        Schedule(minimum=1e-8), solve=cg, max_iterations=40
    ).minimize(operator, measured, start)

    torch.testing.assert_close(
        operator.split(region.x)["T2"],
        operator.split(schedule.x)["T2"],
        atol=1e-3,
        rtol=1e-4,
    )


def test_a_trust_region_is_the_fit_that_ships(problem) -> None:
    """The generalization is exact: the loop reproduces the estimator.

    :class:`~torchsim.NonlinearLeastSquares` is this loop under a per-voxel
    trust region with the voxel-diagonal solve, and nothing else.
    """
    acquisition = Acquisition(MultiEchoSimulator(TE=TE_MS))
    bounds = {"T2": (10.0, 300.0), "M0": (0.0, 5.0)}
    mapping = ParameterMapping(
        acquisition, T2=(10.0, 300.0), M0=(0.2, 2.0), seed=0
    ).train(
        NonlinearLeastSquares(
            bounds=bounds, initial={"T2": 100.0, "M0": 1.0}
        ),
        samples=256,
    )
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
        solve=cg,
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
    acquisition = Acquisition(MultiEchoSimulator(TE=TE_MS))
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
        solve=cg,
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
        solve=cg,
        max_iterations=8,
    ).minimize(operator, kspace, start, encoding=encoding)

    assert float(found.cost[-1]) < 0.2 * float(found.cost[0])
    assert found.cost.ndim == 1


# %% what does not make sense


def test_a_per_voxel_damping_is_refused_under_an_encoding(phantom) -> None:
    """A trust region needs independent rows, and encoding leaves none."""
    operator, encoding, kspace, _, _ = phantom

    with pytest.raises(ValueError, match="each voxel on its own"):
        GaussNewton(TrustRegion(), solve=cg).minimize(
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
