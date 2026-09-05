"""What every benchmark backend shares: timing, memory, and the task itself.

The task is one MR fingerprinting dictionary: an inversion, then a train of
variable flip angles at a fixed repetition time, one sample per repetition,
simulated for a number of tissues at once. It is the workload every toolbox
compared here can express, which is what makes the numbers comparable at all.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def flip_train(length: int) -> np.ndarray:
    """The flip-angle pattern the dictionary is simulated over, in degrees.

    The ramp-up, ramp-down and constant tail of the original fingerprinting
    schedule, stretched to whatever length is asked for.
    """
    up = np.linspace(5.0, 60.0, round(0.375 * length))
    down = np.linspace(60.0, 2.0, round(0.375 * length))
    tail = np.full(length - up.size - down.size, 2.0)
    return np.concatenate((up, down, tail)).astype(np.float64)


def tissue_grid(atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """``atoms`` (T1, T2) pairs spanning the range a brain dictionary covers.

    Log-spaced in both, paired off rather than crossed, so the count is exactly
    what was asked for at every size.
    """
    t1 = np.geomspace(200.0, 3000.0, atoms)
    t2 = np.geomspace(10.0, 300.0, atoms)
    return t1.astype(np.float64), t2.astype(np.float64)


def peak_rss_mib() -> float:
    """Peak resident set size of this process so far, in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)


@dataclass
class Measurement:
    """One backend, one problem size, one mode."""

    backend: str
    mode: str
    atoms: int
    length: int
    states: int
    device: str
    threads: int
    repeats: int
    seconds: list[float] = field(default_factory=list)
    setup_seconds: float = 0.0
    baseline_rss_mib: float = 0.0
    peak_rss_mib: float = 0.0
    peak_device_mib: float = 0.0
    checksum: complex | None = None
    versions: dict[str, str] = field(default_factory=dict)
    machine: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def best(self) -> float:
        """The fastest timed run, in seconds."""
        return min(self.seconds)

    @property
    def median(self) -> float:
        """The median timed run, in seconds."""
        return statistics.median(self.seconds)

    def as_dict(self) -> dict[str, Any]:
        """This measurement as JSON-safe data."""
        payload = asdict(self)
        payload["best"] = self.best
        payload["median"] = self.median
        payload["atoms_per_second"] = self.atoms / self.best if self.best else None
        checksum = payload.pop("checksum")
        if checksum is not None:
            payload["checksum"] = [float(checksum.real), float(checksum.imag)]
        return payload


def machine() -> dict[str, str]:
    """Enough about where this ran that a number can be read later."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


WARMUP_SECONDS = 2.0


def timed(
    run: Callable[[], Any],
    *,
    repeats: int,
    synchronize: Callable[[], None] = lambda: None,
    warmup_seconds: float = WARMUP_SECONDS,
) -> tuple[list[float], Any]:
    """Warm up for ``warmup_seconds``, then run ``repeats`` times, timing each.

    The warm-up is a budget rather than a count because what has to be warm
    differs by orders of magnitude. A card idles at a low clock and takes the
    better part of a second of continuous work to reach its boost one -- a
    kernel of a few milliseconds measured over three runs reports the ramp, not
    the kernel. A pass that already takes tens of seconds is warm after one.
    So the budget buys hundreds of runs for the first and exactly one for the
    second.
    """
    deadline = time.perf_counter() + warmup_seconds
    while True:
        result = run()
        synchronize()
        if time.perf_counter() >= deadline:
            break
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        synchronize()
        seconds.append(time.perf_counter() - start)
    return seconds, result


def parser(backend: str) -> argparse.ArgumentParser:
    """The command line every backend script takes."""
    cli = argparse.ArgumentParser(description=f"{backend} fingerprinting benchmark")
    cli.add_argument("--atoms", type=int, default=1000)
    cli.add_argument("--length", type=int, default=500, help="repetitions in the train")
    cli.add_argument(
        "--states",
        type=int,
        default=32,
        help="configuration orders kept; BlochSimulators requires a multiple of 32",
    )
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--mode", choices=("forward", "jacobian"), default="forward")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--threads", type=int, default=0, help="0 leaves the default")
    cli.add_argument("--json", default="", help="write the measurement here")
    return cli


def report(measurement: Measurement, path: str) -> None:
    """Print one line a human reads, and write the record a table is built from."""
    print(
        f"{measurement.backend:>12s} {measurement.mode:<9s} "
        f"atoms={measurement.atoms:<7d} best={measurement.best * 1e3:9.2f} ms  "
        f"median={measurement.median * 1e3:9.2f} ms  "
        f"peak_rss={measurement.peak_rss_mib:7.1f} MiB  "
        f"(+{measurement.peak_rss_mib - measurement.baseline_rss_mib:6.1f})"
        + (
            f"  device={measurement.peak_device_mib:7.1f} MiB"
            if measurement.peak_device_mib
            else ""
        )
    )
    if path:
        # With the trailing newline a text file is expected to end on, since
        # these records are committed and the repository's hooks check for it.
        with open(path, "w") as stream:
            json.dump(measurement.as_dict(), stream, indent=2)
            stream.write("\n")
