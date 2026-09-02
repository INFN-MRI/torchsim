"""What an estimator says the noise leaves on its answer.

The claim is a number with a definition outside TorchSim: repeat the
measurement with fresh noise and the estimate moves, and the standard
deviation of that movement is what :meth:`map` reports when asked. So every
check here is against a Monte Carlo written out in the test -- draw the noise,
map each realization, take the spread -- rather than against another route to
the same formula.

A Monte Carlo over ``n`` draws knows its own answer only to about
``1/sqrt(2(n-1))``, which is why the tolerances below are percentages rather
than round-off, and why they are stated on the median over voxels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsim
from torchsim.estimators import PERK, DictionaryMatcher, NonlinearLeastSquares
from torchsim.simulators import MRFSimulator

CONTRASTS = 120
VOXELS = 24
DRAWS = 400


@pytest.fixture(scope="module")
def acquisition():
    repetition = torch.arange(CONTRASTS, dtype=torch.float32)
    return MRFSimulator(
        flip=10.0 + 50.0 * torch.sin(torch.pi * repetition / CONTRASTS) ** 2,
        TR=10.0,
        TI=20.0,
        states=20,
        M0=1.0,
    )


@pytest.fixture(scope="module")
def truth():
    return (
        torch.exp(torch.linspace(np.log(600.0), np.log(2000.0), VOXELS)),
        torch.exp(torch.linspace(np.log(60.0), np.log(200.0), VOXELS)),
    )


@pytest.fixture(scope="module")
def measurement(acquisition, truth):
    """The clean signal, and the noise level every check below is stated at."""
    clean = acquisition.simulate(T1=truth[0], T2=truth[1])
    return clean, float(0.01 * clean.abs().max())


def _realizations(clean, noise_std, seed=3):
    """Independent measurements of the same tissue, one per draw."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(DRAWS):
        yield clean + noise_std * torch.complex(
            torch.randn(clean.shape, generator=generator),
            torch.randn(clean.shape, generator=generator),
        )


def _sampled(estimator, clean, noise_std):
    """The spread of the estimate, measured rather than derived."""
    drawn: dict[str, list[torch.Tensor]] = {}
    for realization in _realizations(clean, noise_std):
        for name, values in estimator(realization).items():
            drawn.setdefault(name, []).append(values)
    return {name: torch.stack(values).std(0) for name, values in drawn.items()}


def _log_uniform(low, high, count, generator):
    span = torch.rand(count, generator=generator)
    return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))


def _perk(acquisition, noise_std, **settings):
    prior = torch.Generator().manual_seed(11)
    return PERK(acquisition, n_features=600, regularization=1e-6, **settings).fit(
        T1=_log_uniform(200.0, 5000.0, 8000, prior),
        T2=_log_uniform(20.0, 600.0, 8000, prior),
        noise_std=noise_std,
        seed=0,
        samples=8000,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"normalize": True},
        {"normalize": False},
        {"normalize": True, "complex_mode": "magnitude"},
    ],
    ids=["normalized", "raw", "magnitude"],
)
def test_perk_states_the_spread_a_repeated_scan_would_show(
    acquisition, measurement, settings
) -> None:
    """The whole claim, on each representation the features can be built in."""
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, uncertainty_draws=128, **settings)

    _, stated = estimator(clean, uncertainty=True)
    measured = _sampled(estimator, clean, noise_std)

    for name in ("T1", "T2"):
        # The Monte Carlo knows its own answer to a few percent, and the
        # bootstrap is centred on the estimate rather than on a truth it does
        # not have, so the band is percentages rather than round-off.
        ratio = stated[name] / measured[name]
        assert 0.8 < float(ratio.median()) < 1.2, name


def test_the_spread_follows_the_voxel_and_not_only_the_average(
    acquisition, measurement
) -> None:
    """It is a map, so it has to say which voxels are the uncertain ones.

    A constant answer of about the right size would satisfy a comparison of
    medians and carry no information at all. What is checked here is that the
    stated spread rises and falls across voxels with the spread a repeated
    measurement actually shows.
    """
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True, uncertainty_draws=128)

    stated = estimator.uncertainty_of(clean)
    measured = _sampled(estimator, clean, noise_std)

    for name in ("T1", "T2"):
        together = torch.stack((stated[name], measured[name]))
        assert float(torch.corrcoef(together)[0, 1]) > 0.8, name


def test_a_noiseless_fit_states_no_spread(acquisition, measurement) -> None:
    """Nothing to propagate, and no realizations to draw."""
    clean, _ = measurement
    exact = _perk(acquisition, 0.0, normalize=True)

    assert not exact.uncertainty_of(clean)["T1"].any()


def test_least_squares_states_its_own_standard_error(
    acquisition, measurement, truth
) -> None:
    """The inverse Fisher matrix at the solution, against repeated fits.

    This is the number a least-squares fit reports as a standard error, and it
    is a linearization about the answer -- so it is checked where the residual
    is small, which is where a fit is entitled to report one.
    """
    clean, noise_std = measurement
    estimator = NonlinearLeastSquares(
        acquisition,
        bounds={"T1": (200.0, 5000.0), "T2": (20.0, 600.0)},
        initial={"T1": 1000.0, "T2": 100.0},
    ).fit(T1=(200.0, 5000.0), T2=(20.0, 600.0), noise_std=noise_std)

    found, stated = estimator(clean, uncertainty=True)
    assert float((found["T1"] - truth[0]).abs().max()) < 1e-2

    drawn: dict[str, list[torch.Tensor]] = {}
    generator = torch.Generator().manual_seed(5)
    for _ in range(60):
        noisy = clean + noise_std * torch.complex(
            torch.randn(clean.shape, generator=generator),
            torch.randn(clean.shape, generator=generator),
        )
        for name, values in estimator(noisy).items():
            drawn.setdefault(name, []).append(values)

    for name in ("T1", "T2"):
        ratio = stated[name] / torch.stack(drawn[name]).std(0)
        assert 0.8 < float(ratio.median()) < 1.25, name


def test_a_standard_error_is_never_below_the_bound(
    acquisition, measurement, truth
) -> None:
    """No unbiased estimate beats the Cramer-Rao bound, and this one does not.

    The bound is computed here from the acquisition at the true relaxation
    times, which is the statement the standard error has to respect.
    """
    clean, noise_std = measurement
    estimator = NonlinearLeastSquares(
        acquisition,
        bounds={"T1": (200.0, 5000.0), "T2": (20.0, 600.0)},
        initial={"T1": 1000.0, "T2": 100.0},
    ).fit(T1=(200.0, 5000.0), T2=(20.0, 600.0), noise_std=noise_std)
    stated = estimator.uncertainty_of(clean)

    _, sensitivity = acquisition.jacobian(["T1", "T2"], T1=truth[0], T2=truth[1])
    floor = torchsim.crlb(sensitivity, noise_variance=noise_std**2).sqrt()

    for column, name in enumerate(("T1", "T2")):
        assert float((stated[name] / floor[:, column]).min()) > 0.99, name


def test_a_voxel_that_cannot_be_read_is_infinite_rather_than_a_failure() -> None:
    """One unidentifiable voxel does not take the rest of the map with it."""
    sensitivity = torch.zeros(3, 2, 4)
    sensitivity[:, 0, 0] = 1.0
    sensitivity[:, 1, 1] = 1.0
    sensitivity[1, 1] = 0.0  # nothing there responds to the second parameter

    with pytest.raises(torch.linalg.LinAlgError):
        torchsim.crlb(sensitivity)

    bound = torchsim.crlb(sensitivity, singular="infinite")
    assert torch.isinf(bound[1]).all()
    assert torch.isfinite(bound[[0, 2]]).all()


def test_a_method_that_has_no_uncertainty_says_so(acquisition) -> None:
    """A match answers with a grid point, which does not move a little."""
    grid_t1, grid_t2 = torch.meshgrid(
        torch.linspace(300.0, 3000.0, 40),
        torch.linspace(30.0, 300.0, 20),
        indexing="ij",
    )
    matcher = DictionaryMatcher(acquisition).fit(
        T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), noise_std=0.01
    )
    signal = acquisition.simulate(
        T1=torch.tensor([800.0, 1200.0]), T2=torch.tensor([80.0, 100.0])
    )

    assert matcher(signal)["T1"].numel() == 2
    with pytest.raises(NotImplementedError, match="does not state an uncertainty"):
        matcher(signal, uncertainty=True)


def test_the_noise_already_in_the_measurement_is_not_counted_twice(
    acquisition, measurement
) -> None:
    """A scan carries one realization; the answer is not the spread at two.

    Asked from a measurement and from the noiseless signal behind it, the same
    estimator has to state about the same spread -- it is a property of the
    tissue and the sequence, not of which realization arrived. Drawing on top
    of the measurement instead of on the fingerprint the answer predicts would
    report the spread at ``sqrt(2)`` times the noise, and more than that once
    the regression stops being linear in it.
    """
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True, uncertainty_draws=128)

    generator = torch.Generator().manual_seed(17)
    noisy = clean + noise_std * torch.complex(
        torch.randn(clean.shape, generator=generator),
        torch.randn(clean.shape, generator=generator),
    )

    exact = estimator.uncertainty_of(clean)
    measured = estimator.uncertainty_of(noisy)

    for name in ("T1", "T2"):
        ratio = float(measured[name].median()) / float(exact[name].median())
        assert 0.8 < ratio < 1.25, name
