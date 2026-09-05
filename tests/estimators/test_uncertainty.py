"""What an estimator says the answer is worth.

:meth:`map` asked with ``uncertainty=True`` returns a second set of maps: how
far the answer is expected to sit from the truth. PERK learns that during
training, from the residuals of its own fit, so reporting it is a matrix
multiply rather than a rerun; least squares reads it off the curvature at its
solution.

Both are checked here against a Monte Carlo written out in the test -- draw
the noise, map each realization, measure the scatter -- because that is the
definition, and because it shares nothing with the closed forms it checks.
A Monte Carlo over ``n`` draws knows its own answer to about
``1 / sqrt(2 (n - 1))``, which is why the tolerances are percentages.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsim
from torchsim.estimators import PERK, DictionaryMatcher, NonlinearLeastSquares
from torchsim.simulators import MRFSimulator

CONTRASTS, VOXELS, DRAWS = 400, 40, 300


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
        torch.exp(torch.linspace(np.log(600.0), np.log(2500.0), VOXELS)),
        torch.exp(torch.linspace(np.log(60.0), np.log(250.0), VOXELS)),
        torch.linspace(0.5, 1.5, VOXELS),
    )


@pytest.fixture(scope="module")
def measurement(acquisition, truth):
    """At the amplitude the training set is simulated at."""
    t1, t2, _density = truth
    clean = acquisition.simulate(T1=t1, T2=t2)
    return clean, float(0.02 * clean.abs().max())


@pytest.fixture(scope="module")
def scaled_measurement(acquisition, truth, measurement):
    """The same tissue at densities either side of it."""
    t1, t2, density = truth
    return measurement[0] * density[:, None], measurement[1]


def _noisy(clean, noise_std, generator):
    return clean + noise_std * torch.complex(
        torch.randn(clean.shape, generator=generator),
        torch.randn(clean.shape, generator=generator),
    )


def _scatter(estimator, clean, noise_std, names, seed=3):
    """What repeating the measurement actually does to each answer."""
    generator = torch.Generator().manual_seed(seed)
    drawn = {name: [] for name in names}
    for _ in range(DRAWS):
        answered = estimator(_noisy(clean, noise_std, generator))
        for name in names:
            drawn[name].append(answered[name])
    return {name: torch.stack(values).std(0) for name, values in drawn.items()}


def _perk(acquisition, noise_std, **settings):
    prior = torch.Generator().manual_seed(11)

    def log_uniform(low, high, count):
        span = torch.rand(count, generator=prior)
        return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))

    return PERK(
        acquisition, n_features=1000, regularization=1e-6, feature_seed=0, **settings
    ).fit(
        T1=log_uniform(200.0, 5000.0, 20_000),
        T2=log_uniform(20.0, 600.0, 20_000),
        noise_std=noise_std,
        seed=0,
        samples=20_000,
        rank=4,
    )


# --- what PERK answers for -------------------------------------------------


def test_perk_answers_with_the_proton_density(
    acquisition, scaled_measurement, truth
) -> None:
    """And without simulating a fingerprint to find it.

    Normalizing the features throws the amplitude away, which is what leaves
    the density unknown. What the regression learns alongside the relaxation
    times is one over the length of the fingerprint they imply, so the length
    the measurement has is the density -- one multiplication, no forward pass.
    """
    clean, noise_std = scaled_measurement
    estimator = _perk(acquisition, noise_std, normalize=True)

    maps = estimator(clean)

    assert set(maps) == {"T1", "T2", "M0"}
    error = (maps["M0"] - truth[2]).abs() / truth[2]
    assert float(error.median()) < 0.05


def test_the_density_follows_the_measurement_it_is_a_scale_of(
    acquisition, measurement
) -> None:
    """Twice the signal is twice the density, and nothing else moves.

    The relaxation times are what the *shape* says and the density what the
    *size* says, so scaling a measurement has to leave one alone and scale the
    other exactly.
    """
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True)

    once = estimator(clean)
    twice = estimator(2.0 * clean)

    assert torch.allclose(twice["M0"], 2.0 * once["M0"], rtol=1e-4)
    assert torch.allclose(twice["T1"], once["T1"], rtol=1e-4)


def test_a_regression_that_keeps_the_scale_answers_for_no_density(
    acquisition, measurement
) -> None:
    """There is nothing to learn where the features were never normalized."""
    clean, noise_std = measurement

    assert set(_perk(acquisition, noise_std, normalize=False)(clean)) == {"T1", "T2"}


# --- what it says the answer is worth --------------------------------------


def test_perk_states_a_spread_without_repeating_itself(
    acquisition, measurement, truth
) -> None:
    """The whole claim, against the scatter it is predicting.

    What the regression learned is what its own answers were wrong by on the
    training set, so the number is of the right size and rises and falls with
    the voxel -- which is what makes it a map rather than a constant.
    """
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True)

    _maps, stated = estimator(clean, uncertainty=True)
    measured = _scatter(estimator, clean, noise_std, ("T1", "T2"))
    truth_t1, truth_t2, _ = truth
    error = {
        "T1": (estimator(clean)["T1"] - truth_t1).abs(),
        "T2": (estimator(clean)["T2"] - truth_t2).abs(),
    }

    for name in ("T1", "T2"):
        # The shape of the map follows where the noise moves the answer.
        together = torch.stack((stated[name], measured[name]))
        assert float(torch.corrcoef(together)[0, 1]) > 0.7, name
        # Its size is the whole error, which is larger than the noise alone
        # because a regression trained on a prior is also pulled by it.
        assert float(stated[name].median()) > float(measured[name].median())
        assert float(stated[name].median()) < 4.0 * float(error[name].median())


def test_stating_it_costs_one_multiply_and_not_a_rerun(
    acquisition, measurement
) -> None:
    """Asked twice, it answers identically: there is no sampling in it."""
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True)

    first = estimator.uncertainty_of(clean)
    again = estimator.uncertainty_of(clean)

    for name in ("T1", "T2"):
        assert torch.equal(first[name], again[name])


def test_a_regression_not_asked_to_learn_it_says_so(acquisition, measurement) -> None:
    """Because the second training pass it needs was never walked."""
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True, uncertainty=False)

    assert estimator(clean)["T1"].numel() == VOXELS
    with pytest.raises(NotImplementedError, match="uncertainty=False"):
        estimator(clean, uncertainty=True)


def test_a_noiseless_fit_still_states_what_the_prior_costs(
    acquisition, measurement
) -> None:
    """What is reported is the whole error, and noise is only part of it.

    Trained without noise the regression is still a regression: it answers
    with the prior where the features are uninformative, and it is wrong by
    that. Reporting nothing there would be the misleading answer.
    """
    clean, noise_std = measurement
    noiseless = _perk(acquisition, 0.0, normalize=True)
    noisy = _perk(acquisition, noise_std, normalize=True)

    assert float(noiseless.uncertainty_of(clean)["T1"].median()) > 0.0
    assert float(noiseless.uncertainty_of(clean)["T1"].median()) < float(
        noisy.uncertainty_of(clean)["T1"].median()
    )


def test_a_measurement_of_another_amplitude_is_outside_what_was_learned(
    acquisition, measurement, scaled_measurement
) -> None:
    """The regression is trained at one signal amplitude, so this is a limit.

    Halving the density doubles the noise a voxel effectively carries, and a
    regression that never saw that regime cannot report on it. The spread it
    states barely moves, which is exactly why a measurement should be given at
    the amplitude the fit was told about.
    """
    clean, noise_std = measurement
    estimator = _perk(acquisition, noise_std, normalize=True)

    stated = estimator.uncertainty_of(clean)["T1"]
    halved = estimator.uncertainty_of(0.5 * clean)["T1"]

    assert 0.8 < float(halved.median()) / float(stated.median()) < 1.25


# --- least squares ---------------------------------------------------------


def _least_squares(acquisition, noise_std):
    return NonlinearLeastSquares(
        acquisition,
        bounds={"T1": (200.0, 5000.0), "T2": (20.0, 600.0)},
        initial={"T1": 1000.0, "T2": 100.0},
    ).fit(T1=(200.0, 5000.0), T2=(20.0, 600.0), noise_std=noise_std)


def test_least_squares_states_its_own_standard_error(
    acquisition, measurement, truth
) -> None:
    """The inverse Fisher matrix at the solution, against repeated fits.

    This is the number a fit reports as a standard error, and it is a
    linearization about the answer -- so it is checked where the residual is
    small, which is where a fit is entitled to report one.
    """
    clean, noise_std = measurement
    estimator = _least_squares(acquisition, noise_std)

    found, stated = estimator(clean, uncertainty=True)
    assert float((found["T1"] - truth[0]).abs().max()) < 1.0

    generator = torch.Generator().manual_seed(5)
    drawn: dict[str, list[torch.Tensor]] = {}
    for _ in range(40):
        for name, values in estimator(_noisy(clean, noise_std, generator)).items():
            drawn.setdefault(name, []).append(values)

    for name in ("T1", "T2"):
        ratio = stated[name] / torch.stack(drawn[name]).std(0)
        assert 0.7 < float(ratio.median()) < 1.4, name


def test_a_standard_error_is_never_below_the_bound(
    acquisition, measurement, truth
) -> None:
    """No unbiased estimate beats the Cramer-Rao bound, and this one does not."""
    clean, noise_std = measurement
    stated = _least_squares(acquisition, noise_std).uncertainty_of(clean)

    _signal, sensitivity = acquisition.jacobian(["T1", "T2"], T1=truth[0], T2=truth[1])
    floor = torchsim.crlb(sensitivity, noise_variance=noise_std**2).sqrt()

    for column, name in enumerate(("T1", "T2")):
        assert float((stated[name] / floor[:, column]).min()) > 0.99, name


# --- and where there is none to state --------------------------------------


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


def test_a_method_that_answers_with_a_grid_point_says_it_has_none(
    acquisition,
) -> None:
    """A match does not move a little when the noise does."""
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


# --- and the density a match works out on its way --------------------------


def test_a_match_answers_with_the_density_without_touching_an_atom(
    acquisition, truth
) -> None:
    """The score already carries it, once the two lengths are put back.

    Matching normalizes both sides, so the score is a cosine: multiply by the
    measurement's own length and divide by the atom's -- a number stored per
    atom rather than a signal per atom -- and that is the least-squares scale.
    It has to equal the scale computed the long way, from the atoms.
    """
    t1, t2, density = truth
    grid_t1, grid_t2 = torch.meshgrid(
        torch.logspace(np.log10(300.0), np.log10(3000.0), 60),
        torch.logspace(np.log10(30.0), np.log10(300.0), 30),
        indexing="ij",
    )
    matcher = DictionaryMatcher(acquisition).fit(
        T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), noise_std=0.0
    )
    signal = acquisition.simulate(T1=t1, T2=t2) * density[:, None]

    maps = matcher(signal)
    assert set(maps) == {"T1", "T2", "M0"}

    found = matcher.match(signal)
    assert torch.allclose(found.densities, found.scales.abs(), atol=1e-5)
    assert float(((maps["M0"] - density).abs() / density).median()) < 0.05


def test_a_match_fitted_from_bare_arrays_answers_in_the_caller_s_columns(
    acquisition,
) -> None:
    """No names, so no room to append one: the columns are the caller's."""
    grid = torch.stack(
        (torch.linspace(400.0, 2000.0, 32), torch.linspace(40.0, 200.0, 32)), dim=-1
    )
    signals = acquisition.simulate(T1=grid[:, 0], T2=grid[:, 1])
    matcher = DictionaryMatcher().fit(signals=signals, parameters=grid)

    assert matcher(signals[:4]).shape == (4, 2)
