"""What this machine can do with a fitted regression, measured on it.

Whether a mapping is big enough to repay a device launch is a question about
hardware, and the answer moves by an order of magnitude between a laptop card
and a recon server. So it is measured on first use rather than written down,
the same way ``sequence/_calibration.py`` measures the simulation kernels.

Measurements live for the process. Set ``TORCHSIM_CALIBRATION=off`` to skip
probing and use the fallback below.
"""

from __future__ import annotations

__all__ = ["crossover"]

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

#: Where a probe cannot run. Off one laptop GPU, so a place to start rather
#: than a description of any particular machine.
_FALLBACK = 2_000_000.0

#: The two sizes a straight line is fitted through, in voxels.
_SMALL = 512
_LARGE = 32_768
_REPEATS = 3

_CROSSOVER: dict[tuple[Any, ...], float] = {}


@dataclass(frozen=True)
class _Rates:
    """One backend's cost as a launch plus a slope."""

    fixed: float
    per_voxel: float


def crossover(
    key: tuple[Any, ...],
    device: torch.device,
    build: Callable[[torch.device, int], Callable[[], Any]],
    work_per_voxel: int,
) -> float:
    """Work below which the host beats this device at whatever ``build`` runs.

    Parameters
    ----------
    key:
        What the answer depends on besides the device -- the shapes the
        problem is fixed at. Measurements are cached under it.
    device:
        The card in question.
    build:
        Called as ``build(device, voxels)`` and returns a closure running the
        real work once, at that size, on that device. Timing the thing itself
        is what keeps this honest.
    work_per_voxel:
        What one voxel counts as, in the unit the caller measures work in, so
        the crossover comes back in the same unit.

    Returns
    -------
    float
        The crossover in the caller's work units.
    """
    cached = (_name(device), *key)
    if cached not in _CROSSOVER:
        _CROSSOVER[cached] = _measure(device, build, work_per_voxel)
    return _CROSSOVER[cached]


def forget() -> None:
    """Discard what has been measured, so the next ask probes again."""
    _CROSSOVER.clear()


# %% private module subroutines


def _name(device: torch.device) -> str:
    return device.type if device.type != "cuda" else f"cuda:{device.index or 0}"


def _disabled() -> bool:
    return os.environ.get("TORCHSIM_CALIBRATION", "").lower() in {
        "off",
        "0",
        "false",
    }


def _measure(
    device: torch.device,
    build: Callable[[torch.device, int], Callable[[], Any]],
    work_per_voxel: int,
) -> float:
    if _disabled() or device.type != "cuda" or not torch.cuda.is_available():
        return _FALLBACK
    try:
        host = _rates(torch.device("cpu"), build)
        card = _rates(device, build)
    except Exception:  # noqa: BLE001 - a probe must never break a mapping
        return _FALLBACK
    gain = host.per_voxel - card.per_voxel
    if gain <= 0.0:
        # No measurable advantage is a probe that failed to see one, not proof
        # there is none. Fall back rather than write the device off entirely.
        return _FALLBACK
    voxels = max(0.0, (card.fixed - host.fixed) / gain)
    return voxels * work_per_voxel


def _rates(
    device: torch.device,
    build: Callable[[torch.device, int], Callable[[], Any]],
) -> _Rates:
    """A straight line through one backend's cost at two sizes."""
    small = _time(device, build(device, _SMALL))
    large = _time(device, build(device, _LARGE))
    per_voxel = (large - small) / (_LARGE - _SMALL)
    return _Rates(fixed=max(0.0, small - per_voxel * _SMALL), per_voxel=per_voxel)


def _time(device: torch.device, once: Callable[[], Any]) -> float:
    """Seconds for one run, the launch already paid for."""
    once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(_REPEATS):
        once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / _REPEATS
