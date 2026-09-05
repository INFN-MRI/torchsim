"""The same fingerprinting dictionary, on sycomore's regular EPG model.

Sycomore simulates one tissue at a time -- the loop over the dictionary is
Python, the loop over the train is C++ -- so the atom count multiplies the
whole cost rather than filling a wider kernel. That is the difference the
benchmark is there to show, and it is a property of the API rather than of the
arithmetic: the per-atom C++ inner loop is the same order of speed as anyone's.

Sycomore's regular model grows an order per shift instead of truncating, so
``--states`` is honoured by pruning states below ``--threshold`` relative to
the equilibrium magnetization; the achieved order count is recorded beside the
timings. Passing ``--threshold 0`` keeps every order, which is the accurate but
quadratic run.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    Measurement,
    flip_train,
    machine,
    parser,
    peak_rss_mib,
    report,
    timed,
    tissue_grid,
)


def installed_version() -> str:
    """Which sycomore this is, from whichever installer put it there.

    Conda-forge is what sycomore's own instructions reach for, and a conda
    package leaves no distribution metadata a virtual environment can read.
    """
    try:
        return version("sycomore")
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    """Time one dictionary, and say what it cost."""
    cli = parser("sycomore")
    cli.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="prune configuration states below this magnitude; 0 keeps all",
    )
    args = cli.parse_args()

    import sycomore
    from sycomore.units import deg, ms

    baseline = peak_rss_mib()

    flip = flip_train(args.length)
    t1, t2 = tissue_grid(args.atoms)
    orders = []

    def atom(T1: float, T2: float) -> np.ndarray:
        species = sycomore.Species(T1 * ms, T2 * ms)
        model = sycomore.epg.Regular(species, initial_size=args.states)
        if args.threshold:
            model.threshold = args.threshold
        model.apply_pulse(180 * deg)  # inversion
        signal = np.empty(flip.size, dtype=complex)
        for index, angle in enumerate(flip):
            model.apply_pulse(angle * deg)
            signal[index] = model.echo
            model.shift()
            model.apply_time_interval(10.0 * ms)
        orders.append(len(model.orders))
        return signal

    def dictionary() -> np.ndarray:
        orders.clear()
        return np.stack([atom(a, b) for a, b in zip(t1, t2, strict=True)])

    seconds, signal = timed(dictionary, repeats=args.repeats)

    measurement = Measurement(
        backend="sycomore",
        mode=args.mode,
        atoms=args.atoms,
        length=args.length,
        states=int(np.max(orders)) if orders else args.states,
        device="cpu",
        threads=1,
        repeats=args.repeats,
        seconds=seconds,
        baseline_rss_mib=baseline,
        peak_rss_mib=peak_rss_mib(),
        checksum=complex(signal.sum()),
        versions={"sycomore": installed_version()},
        machine=machine(),
        note=(
            f"threshold={args.threshold:g}; orders reached "
            f"{int(np.min(orders))}-{int(np.max(orders))}"
        ),
    )
    report(measurement, args.json)


if __name__ == "__main__":
    main()
