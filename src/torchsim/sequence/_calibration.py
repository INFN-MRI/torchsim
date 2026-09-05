"""What this machine can actually do, measured instead of assumed.

Two decisions in the dispatcher are really questions about hardware: whether a
problem is big enough to repay a device launch, and whether testing a sequence
for a real subspace repays the kernel it would speed up. Both answers move by
an order of magnitude between a laptop card and a recon server, so they are
measured here on first use rather than written down.

Every probe runs the same entry point the real work runs, at two sizes, and
fits a straight line through them. A pass is timed only when something asks
about it, so a session that never differentiates never compiles an adjoint.

Measurements live for the process. Set ``TORCHSIM_CALIBRATION=off`` to skip
probing entirely and use the fallbacks below.
"""

from __future__ import annotations

__all__ = ["calibrate"]

import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

# Used when probing is switched off, and for a device that cannot be probed.
# These came off one laptop GPU, so they are a place to start rather than a
# description of any particular machine.
_FALLBACK_CROSSOVER = {"forward": 1500.0, "jvp": 400.0, "adjoint": 500.0}
_FALLBACK_DETECTION = {"cpu": 5_000.0, "cuda": 1_000_000.0}

_PROBE_ECHOES = 8
_SAMPLES = 3

# The slope is the difference between two points, so it is only as good as the
# gap between them: a device still paying off its launch cost at the larger
# size reports mostly noise. Grow the problem until the gap dominates, or until
# a single run costs more than it is worth spending.
_FIRST_VOXELS = 256
_GROWTH = 4
_SEPARATION = 8.0
_BUDGET_SECONDS = 0.025
_MOST_VOXELS = 1 << 18

_RATES: dict[tuple[Any, ...], _Rates] = {}
_CROSSOVER: dict[tuple[Any, ...], float] = {}
_DETECTION: dict[tuple[Any, ...], float] = {}


@dataclass(frozen=True)
class _Rates:
    """A straight line through two problem sizes: seconds = fixed + rate x work.

    ``work`` counts ``(voxel, train, event)`` triples, the same measure the
    dispatcher sizes problems by.
    """

    fixed: float
    per_work: float


def _disabled() -> bool:
    return os.environ.get("TORCHSIM_CALIBRATION", "").lower() in {"off", "0", "false"}


def calibrate(force: bool = False) -> None:
    """Discard what has been measured so far, so the next ask probes again.

    Useful after something else has taken over the machine, or to move the
    cost of probing to a moment of the caller's choosing.
    """
    if force or not _disabled():
        _RATES.clear()
        _CROSSOVER.clear()
        _DETECTION.clear()


def crossover(kind: str, device: torch.device, state_count: int) -> float:
    """Work below which the host beats this device for a ``kind`` of pass.

    Where the two lines cross: the device pays a fixed launch cost the host
    does not, and earns it back on the slope. ``inf`` if it never does.
    """
    key = (kind, _key(device), state_count)
    if key not in _CROSSOVER:
        _CROSSOVER[key] = _measure_crossover(kind, device, state_count)
    return _CROSSOVER[key]


def detection(kind: str, device: torch.device, state_count: int) -> float:
    """Work above which testing for a real subspace repays what it costs.

    The test is a fixed handful of reductions and one round trip; the saving is
    the gap between the complex and real kernels, which grows with the problem.
    """
    key = (kind, _key(device), state_count)
    if key not in _DETECTION:
        _DETECTION[key] = _measure_detection(kind, device, state_count)
    return _DETECTION[key]


def _key(device: torch.device) -> str:
    return device.type if device.type != "cuda" else f"cuda:{device.index or 0}"


def _measure_crossover(kind: str, device: torch.device, state_count: int) -> float:
    if _disabled() or device.type != "cuda" or not torch.cuda.is_available():
        return _FALLBACK_CROSSOVER.get(kind, 0.0)
    try:
        host = _rates(kind, torch.device("cpu"), -1, state_count)
        card = _rates(kind, device, -1, state_count)
    except Exception:  # noqa: BLE001 - a probe must never break a simulation
        return _FALLBACK_CROSSOVER.get(kind, 0.0)
    gain = host.per_work - card.per_work
    if gain <= 0.0:
        # No measurable advantage is a probe that failed to see one, not proof
        # there is none. Fall back rather than write the device off entirely.
        return _FALLBACK_CROSSOVER.get(kind, 0.0)
    return max(0.0, (card.fixed - host.fixed) / gain)


def _measure_detection(kind: str, device: torch.device, state_count: int) -> float:
    if _disabled():
        return _FALLBACK_DETECTION.get(device.type, 0.0)
    try:
        complex_rates = _rates(kind, device, -1, state_count)
        real_rates = _rates(kind, device, 1, state_count)
        cost = _summary_cost(device)
    except Exception:  # noqa: BLE001 - a probe must never break a simulation
        return _FALLBACK_DETECTION.get(device.type, 0.0)
    saving = complex_rates.per_work - real_rates.per_work
    if saving <= 0.0:
        return _FALLBACK_DETECTION.get(device.type, 0.0)
    return cost / saving


@contextmanager
def _unpoliced() -> Iterator[None]:
    """Run a probe as written, whatever policy the caller has in force."""
    from .. import _execution
    from . import _accelerators

    previous = _accelerators._DEVICES
    _accelerators._DEVICES = ()
    try:
        with _execution.unpoliced():
            yield
    finally:
        _accelerators._DEVICES = previous


def _problem(
    voxels: int, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], int]:
    """A small in-phase echo train, which every kernel family can run."""
    from ._accelerators import _pack_events
    from ._builders import fse_description
    from ._simulation import TissueProperties, _prepare_tissue

    flip = torch.full((_PROBE_ECHOES,), math.radians(120.0))
    packed = _pack_events(
        fse_description(
            flip,
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
            excitation_phase_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    # Every buffer the packing produces, in its order. A probe short of one is
    # refused by the extension, and the refusal is caught -- so a probe that
    # does not carry them all measures nothing and the fallback stands in.
    events = tuple(value.to(device) for value in packed.buffers)
    tissue = TissueProperties(
        t1_ms=torch.linspace(300.0, 2000.0, voxels),
        t2_ms=torch.linspace(20.0, 200.0, voxels),
        b0_hz=torch.zeros(voxels),
        b1_phase_rad=torch.zeros(voxels),
    )
    prepared, _, _ = _prepare_tissue(tissue, torch.device("cpu"))
    prepared = tuple(
        value.to(torch.float32).contiguous().to(device) for value in prepared
    )
    return prepared, events, packed.output_count


def _call(
    kind: str, voxels: int, device: torch.device, real_axis: int, state_count: int
) -> Any:
    """A zero-argument closure running one pass of ``kind`` at this size."""
    from ._accelerators import _run_packed, _run_packed_jvp, _run_packed_vjp_jvp

    tissue, events, outputs = _problem(voxels, device)
    shared = (tissue, events, state_count, outputs, 0)
    if kind == "forward":
        return lambda: _run_packed(*shared, real_axis=real_axis)

    seeds = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    )
    event_seeds = tuple(
        torch.zeros_like(value) for value in (events[0], events[2], events[3])
    )
    if kind == "jvp":
        return lambda: _run_packed_jvp(
            tissue,
            events,
            seeds,
            event_seeds,
            state_count,
            outputs,
            0,
            real_axis=real_axis,
        )

    cotangent = torch.ones((voxels, outputs), dtype=torch.complex64, device=device)
    return lambda: _run_packed_vjp_jvp(
        tissue,
        events,
        (*seeds, *event_seeds),
        cotangent,
        state_count,
        outputs,
        0,
        real_axis=real_axis,
    )


def _elapsed(call: Any, device: torch.device, samples: int = _SAMPLES) -> float:
    """The floor of a few runs, past whatever the first one had to compile.

    The floor rather than the middle: every disturbance a run can meet -- a
    scheduler slice, another process on the card -- only ever adds time.
    """
    call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    best = math.inf
    for _ in range(samples):
        start = time.perf_counter()
        call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        best = min(best, time.perf_counter() - start)
    return best


def _rates(kind: str, device: torch.device, real_axis: int, state_count: int) -> _Rates:
    """Fit ``seconds = fixed + rate x work`` through one voxel and many.

    The larger size grows until it costs enough more than a single voxel for
    the difference to mean something, which on a device means growing until it
    is past its launch cost. A run that already takes a useful fraction of what
    the whole probe is worth stops the search whether or not it got there.
    """
    key = (kind, real_axis, _key(device), state_count)
    if key in _RATES:
        return _RATES[key]
    per_voxel = _PROBE_ECHOES * 2  # every echo carries a pulse and a readout
    points: list[tuple[float, float]] = []
    with _unpoliced():
        # One voxel is all launch cost and all jitter, so it is sampled harder.
        smallest = _elapsed(
            _call(kind, 1, device, real_axis, state_count), device, _SAMPLES * 3
        )
        points.append((float(per_voxel), smallest))
        voxels = _FIRST_VOXELS
        while True:
            elapsed = _elapsed(
                _call(kind, voxels, device, real_axis, state_count), device
            )
            points.append((float(voxels * per_voxel), elapsed))
            grown = voxels * _GROWTH
            if (
                elapsed >= _SEPARATION * smallest
                or elapsed >= _BUDGET_SECONDS
                or grown > _MOST_VOXELS
            ):
                break
            voxels = grown
    _RATES[key] = _fit(points)
    return _RATES[key]


def _fit(points: list[tuple[float, float]]) -> _Rates:
    """Least squares through every size the search visited.

    Weighted by ``1 / time``, because a probe's error is a fraction of what it
    measured: unweighted, the largest size would be the only one heard, and the
    intercept -- which is what the crossover turns on -- would come from
    extrapolating a line fitted to a single point.
    """
    total = sum(1.0 / time for _work, time in points)
    mean_work = sum(work / time for work, time in points) / total
    mean_time = sum(1.0 for _work, _time in points) / total
    spread = sum((work - mean_work) ** 2 / time for work, time in points)
    covariance = sum(
        (work - mean_work) * (time - mean_time) / time for work, time in points
    )
    per_work = max(0.0, covariance / spread) if spread > 0.0 else 0.0
    return _Rates(max(0.0, mean_time - per_work * mean_work), per_work)


def _summary_cost(device: torch.device) -> float:
    """What one subspace test costs a call: the reductions plus the round trip.

    The half of the verdict that reads the event stream is settled once per
    sequence and reused, so what a call pays is the half that reads the tissue.
    A caller who leaves off-resonance, transmit phase and flow at their
    identities pays nothing at all; this measures the caller who gives one of
    them as a map, which is the case the threshold has to cover.
    """
    from ._accelerators import _tissue_stays_on_the_axis

    tissue, _events, _outputs = _problem(_FIRST_VOXELS, device)
    return _elapsed(lambda: _tissue_stays_on_the_axis(tissue, None), device)
