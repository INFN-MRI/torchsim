"""The same fingerprinting dictionary, on epgpy's operator-per-event EPG.

Epgpy is the closest thing to TorchSim in Python: NumPy under the hood, the
dictionary carried in the array shape rather than in a loop, analytic first-
and second-order derivatives, and CuPy for a card. What it does not do is fuse:
each operator is a pass over the whole state matrix, so a train of a thousand
events is a thousand passes over an array the size of the dictionary. That is
the axis this benchmark measures.

Install it from its repository -- it is not on PyPI:

    pip install git+https://github.com/py-baudin/epgpy
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


def main() -> None:
    """Time one dictionary, and say what it cost."""
    args = parser("epgpy").parse_args()

    from epgpy import epg

    baseline = peak_rss_mib()

    flip = flip_train(args.length)
    t1, t2 = tissue_grid(args.atoms)
    TR = 10.0
    order1 = ["T1", "T2"] if args.mode == "jacobian" else None

    def build() -> list:
        sequence = [epg.T(180.0, 0.0)]  # inversion
        for angle in flip:
            sequence += [
                epg.T(float(angle), 0.0),
                epg.ADC,
                epg.S(1),
                epg.E(TR, t1, t2, order1=order1),
            ]
        return sequence

    sequence = build()
    probe = epg.Jacobian(["T1", "T2"]) if args.mode == "jacobian" else None

    def run() -> np.ndarray:
        return epg.simulate(sequence, probe=probe, max_nstate=args.states)

    seconds, signal = timed(run, repeats=args.repeats)

    try:
        release = version("epgpy")
    except PackageNotFoundError:  # installed from a checkout without metadata
        release = "unknown"

    measurement = Measurement(
        backend="epgpy",
        mode="jacobian(T1,T2)" if args.mode == "jacobian" else "forward",
        atoms=args.atoms,
        length=args.length,
        states=args.states,
        device="cpu",
        threads=1,
        repeats=args.repeats,
        seconds=seconds,
        baseline_rss_mib=baseline,
        peak_rss_mib=peak_rss_mib(),
        checksum=complex(np.asarray(signal).sum()),
        versions={"epgpy": release, "numpy": np.__version__},
        machine=machine(),
        note="max_nstate matches the orders TorchSim keeps",
    )
    report(measurement, args.json)


if __name__ == "__main__":
    main()
