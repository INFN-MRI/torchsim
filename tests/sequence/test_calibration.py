"""Measuring the machine instead of assuming it.

Nothing here asserts a number of seconds or a number of voxels: the whole point
is that those differ per machine, and a test that pinned them would be testing
the laptop the constants came off. What is testable is the shape of the answer
-- that a threshold is finite and positive, that it is measured once, that
turning probing off returns the documented fallbacks, and that work either side
of a threshold ends up on the side the threshold claims.
"""

import math

import pytest
import torch

from torchsim.sequence import calibrate
from torchsim.sequence._calibration import (
    _FALLBACK_CROSSOVER,
    _FALLBACK_DETECTION,
    _CROSSOVER,
    _RATES,
    _fit,
    crossover,
    detection,
)

STATES = 10
KINDS = ["forward", "jvp", "adjoint"]
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@pytest.fixture
def off(monkeypatch):
    monkeypatch.setenv("TORCHSIM_CALIBRATION", "off")
    calibrate(force=True)
    yield
    calibrate(force=True)


# --- the shape of an answer ---


@pytest.mark.parametrize("kind", KINDS)
def test_a_threshold_is_a_finite_amount_of_work(kind):
    card = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = crossover(kind, card, STATES)

    assert math.isfinite(value)
    assert value >= 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_the_probe_a_measurement_times_actually_runs(kind):
    """A probe that raises is caught, on purpose -- a measurement must never
    break a simulation. That also means a probe built short of a buffer the
    kernels expect measures nothing at all and the fallback quietly stands in,
    which is indistinguishable from a real answer everywhere else.

    So the probe is called here directly, where a refusal is an error.
    """
    from torchsim.sequence._calibration import _call

    _call(kind, 64, torch.device("cpu"), -1, STATES)()


@pytest.mark.parametrize("kind", KINDS)
def test_detection_stays_reachable(kind):
    """An unmeasurable saving must not lock the real kernels away entirely."""
    value = detection(kind, torch.device("cpu"), STATES)

    assert math.isfinite(value)
    assert value > 0.0


def test_a_machine_is_measured_once():
    """Probing is the expensive part, so a second ask must not repeat it."""
    card = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = crossover("forward", card, STATES)
    before = dict(_RATES)
    second = crossover("forward", card, STATES)

    assert first == second
    assert dict(_RATES) == before


def test_a_different_shape_is_measured_separately():
    """The crossover moves with the state count, so it is keyed by it."""
    card = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crossover("forward", card, STATES)
    crossover("forward", card, STATES + 6)

    keys = {key for key in _CROSSOVER if key[0] == "forward"}
    assert {key[2] for key in keys} >= {STATES, STATES + 6}


# --- the escape hatches ---


@pytest.mark.parametrize("kind", KINDS)
def test_switching_it_off_uses_the_fallbacks(off, kind):
    card = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert crossover(kind, card, STATES) == _FALLBACK_CROSSOVER[kind]
    assert detection(kind, card, STATES) == _FALLBACK_DETECTION[card.type]


def test_switching_it_off_measures_nothing(off):
    _RATES.clear()
    crossover("adjoint", torch.device("cpu"), STATES)

    assert not _RATES


def test_calibrating_again_discards_what_was_measured():
    crossover("forward", torch.device("cpu"), STATES)
    calibrate(force=True)

    assert not _CROSSOVER


def test_a_probe_that_fails_does_not_break_the_simulation(monkeypatch):
    """A measurement is an optimization; losing it must cost only speed."""
    import torchsim.sequence._calibration as module

    calibrate(force=True)
    monkeypatch.setattr(
        module, "_rates", lambda *arguments: (_ for _ in ()).throw(RuntimeError)
    )

    assert crossover("forward", torch.device("cuda"), STATES) == (
        _FALLBACK_CROSSOVER["forward"]
    )


# --- the fit ---


def test_the_fit_recovers_a_line_it_is_given():
    points = [(w, 3e-4 + 2e-9 * w) for w in (16, 4096, 16384, 65536)]
    rates = _fit(points)

    assert rates.fixed == pytest.approx(3e-4, rel=1e-3)
    assert rates.per_work == pytest.approx(2e-9, rel=1e-3)


def test_the_fit_hears_the_small_sizes():
    """Weighting by time keeps the largest point from being the only voice.

    The intercept is what the crossover turns on, and it lives at the small
    end, so a fit deaf to the small sizes reports the wrong one.
    """
    truth = [(w, 3e-4 + 2e-9 * w) for w in (16, 4096, 16384, 65536)]
    disturbed = [(w, t * (1.10 if w == 65536 else 1.0)) for w, t in truth]

    assert _fit(disturbed).fixed == pytest.approx(3e-4, rel=0.5)


def test_a_flat_measurement_reports_no_rate():
    """Timings that never grow mean a slope of zero, not a negative one."""
    rates = _fit([(16, 1e-3), (4096, 1e-3), (65536, 9e-4)])

    assert rates.per_work == 0.0
    assert rates.fixed > 0.0


# --- what it decides ---


@cuda_only
def test_the_measured_crossover_is_where_the_two_sides_meet():
    """The property that defines it: at that size, neither side is far ahead.

    Generously bounded, because the crossover is a ratio of small differences
    -- and an error there is cheap precisely because both sides cost the same.
    """
    import time

    from test_offload import _volume
    from torchsim.sequence._accelerators import _run_packed

    card = torch.device("cuda", 0)
    events, prepared, outputs = _volume(1)
    per_voxel = int(events[1].numel())
    voxels = max(1, int(crossover("forward", card, STATES) / per_voxel))
    events, prepared, outputs = _volume(voxels)

    def elapsed(tissue, packed):
        for _ in range(3):
            _run_packed(tissue, packed, STATES, outputs, 0, real_axis=-1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _run_packed(tissue, packed, STATES, outputs, 0, real_axis=-1)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    host = elapsed(prepared, events)
    device = elapsed(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in events),
    )

    assert 1 / 20 < host / device < 20
