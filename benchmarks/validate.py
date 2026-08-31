"""Whether TorchSim, sycomore and epgpy compute the same fingerprint.

Three independent implementations of the same model -- a fused C++ or Triton
state machine over a packed event stream, a C++ EPG library driven one tissue
at a time from Python, and an operator-per-event NumPy library -- should agree
to the precision the coarsest of them carries. This says by how much they do,
over a tissue grid, and shows what truncating the configuration orders costs.

Where there is a card, TorchSim's two kernels are held against each other as
well: they are separate implementations of one recursion, and a run placed on a
card has to answer what the same run on the CPU answers.

Run as ``python benchmarks/validate.py``. Epgpy is optional; the comparison
against sycomore runs without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import flip_train  # noqa: E402

TISSUES = [
    (200.0, 10.0),
    (600.0, 40.0),
    (1000.0, 80.0),
    (1500.0, 120.0),
    (3000.0, 300.0),
    (4000.0, 2000.0),  # free water, where the states barely decay
]


def sycomore_signal(T1: float, T2: float, flip: np.ndarray, TR: float) -> np.ndarray:
    """The reference: every configuration order kept, in double precision."""
    import sycomore
    from sycomore.units import deg, ms

    model = sycomore.epg.Regular(sycomore.Species(T1 * ms, T2 * ms))
    model.apply_pulse(180 * deg)
    signal = np.empty(flip.size, dtype=complex)
    for index, angle in enumerate(flip):
        model.apply_pulse(angle * deg)
        signal[index] = model.echo
        model.shift()
        model.apply_time_interval(TR * ms)
    return signal


def epgpy_signal(
    T1: np.ndarray, T2: np.ndarray, flip: np.ndarray, TR: float, states: int
) -> np.ndarray:
    """The same train again, one NumPy operator per event."""
    from epgpy import epg

    sequence = [epg.T(180.0, 0.0)]
    for angle in flip:
        sequence += [epg.T(float(angle), 0.0), epg.ADC, epg.S(1), epg.E(TR, T1, T2)]
    return np.asarray(epg.simulate(sequence, max_nstate=states)).T


def torchsim_signal(
    T1: np.ndarray,
    T2: np.ndarray,
    flip: np.ndarray,
    TR: float,
    states: int,
    device: str = "cpu",
) -> np.ndarray:
    """The same train on the fused state machine, at a given order count."""
    import torch

    from torchsim.simulators import MRFSimulator

    where = torch.device(device)
    sequence = MRFSimulator(
        flip=torch.tensor(flip, dtype=torch.float32, device=where),
        TR=TR,
        states=states,
    )
    if where.type != "cpu":
        sequence = sequence.to(where)
    signal = sequence.simulate(
        T1=torch.tensor(T1, dtype=torch.float32, device=where),
        T2=torch.tensor(T2, dtype=torch.float32, device=where),
        inv_efficiency=1.0,
    )
    return signal.cpu().numpy()


def julia_signal(path: str) -> np.ndarray | None:
    """The signal a Julia backend dumped, as ``(tissues, samples)``.

    ``bench_blochsimulators.jl --tissues ... --dump <path>`` writes one line
    per sample: tissue index, sample index, real part, imaginary part.
    """
    if not path or not os.path.exists(path):
        return None
    rows = np.loadtxt(path, delimiter=",")
    tissues = int(rows[:, 0].max())
    samples = int(rows[:, 1].max())
    signal = np.zeros((tissues, samples), dtype=complex)
    signal[rows[:, 0].astype(int) - 1, rows[:, 1].astype(int) - 1] = (
        rows[:, 2] + 1j * rows[:, 3]
    )
    return signal


def main() -> None:
    """Print the agreement, and its dependence on the orders carried."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--julia",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="a signal dumped by a Julia backend, to compare against as well",
    )
    cli.add_argument(
        "--julia-tissues",
        default="",
        help="the tissues the Julia dumps were made for, as T1:T2,... in ms; "
        "defaults to the list this script uses",
    )
    cli.add_argument("--length", type=int, default=500)
    cli.add_argument("--states", type=int, default=32)
    arguments = cli.parse_args()

    length, TR = arguments.length, 10.0
    flip = flip_train(length)
    T1 = np.array([tissue[0] for tissue in TISSUES])
    T2 = np.array([tissue[1] for tissue in TISSUES])

    reference = np.stack([sycomore_signal(a, b, flip, TR) for a, b in TISSUES])
    record: dict[str, Any] = {
        "task": {"length": length, "TR_ms": TR, "states": arguments.states},
        "tissues": [{"T1_ms": a, "T2_ms": b} for a, b in TISSUES],
        "reference": "sycomore, every order kept, float64",
        "truncation": {},
        "against": {},
    }

    print(f"MRF-FISP, {length} repetitions, TR = {TR} ms, inversion prepared.")
    print("Reference: sycomore, every order kept, double precision.\n")
    print(f"{'states':>7s}  {'max |difference|':>17s}  {'relative':>10s}")
    for states in (4, 8, 16, 32, 64):
        signal = torchsim_signal(T1, T2, flip, TR, states)
        # The two differ by the constant phase each writes its echo with.
        turn = np.exp(1j * np.angle(np.sum(reference * np.conj(signal))))
        difference = np.abs(reference - signal * turn)
        print(
            f"{states:7d}  {difference.max():17.3e}  "
            f"{difference.max() / np.abs(reference).max():10.2e}"
        )
        record["truncation"][str(states)] = float(difference.max())

    import torch

    if torch.cuda.is_available():
        # The two kernels are separate implementations of the same recursion,
        # so this is a comparison of the C++ one against the Triton one rather
        # than a check that a tensor made the trip.
        on_cpu = torchsim_signal(T1, T2, flip, TR, arguments.states)
        on_card = torchsim_signal(T1, T2, flip, TR, arguments.states, device="cuda")
        difference = np.abs(on_cpu - on_card)
        print(
            f"\nTorchSim on {torch.cuda.get_device_name(0)} against its CPU kernel: "
            f"max {difference.max():.3e}, "
            f"relative {difference.max() / np.abs(on_cpu).max():.2e}"
        )
        record["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "max": float(difference.max()),
            "relative": float(difference.max() / np.abs(on_cpu).max()),
        }

    try:
        other = epgpy_signal(T1, T2, flip, TR, arguments.states)
    except ImportError:
        other = None
    if other is not None:
        signal = torchsim_signal(T1, T2, flip, TR, arguments.states)
        turn = np.exp(1j * np.angle(np.sum(other * np.conj(signal))))
        difference = np.abs(other - signal * turn)
        print(
            f"\nTorchSim against epgpy, both truncated to {arguments.states} orders: "
            f"max {difference.max():.3e}, "
            f"relative {difference.max() / np.abs(other).max():.2e}"
        )

    julia_tissues = (
        [
            tuple(float(v) for v in pair.split(":"))
            for pair in arguments.julia_tissues.split(",")
        ]
        if arguments.julia_tissues
        else TISSUES
    )
    julia_reference = (
        reference
        if julia_tissues == TISSUES
        else np.stack([sycomore_signal(a, b, flip, TR) for a, b in julia_tissues])
    )
    for entry in arguments.julia:
        label, _, path = entry.partition("=")
        other = julia_signal(path)
        if other is None:
            print(f"\n{label}: nothing at {path}")
            continue
        rows = other.shape[0]
        turn = np.exp(1j * np.angle(np.sum(julia_reference[:rows] * np.conj(other))))
        difference = np.abs(julia_reference[:rows] - other * turn)
        print(f"\n{label} against sycomore, per tissue:")
        record["against"][label] = []
        for row, (a, b) in enumerate(julia_tissues[:rows]):
            nrmse = float(
                np.sqrt(np.mean(difference[row] ** 2))
                / np.sqrt(np.mean(np.abs(julia_reference[row]) ** 2))
            )
            print(
                f"  T1 = {a:6.0f} ms, T2 = {b:6.0f} ms:  "
                f"max {difference[row].max():.3e}, NRMSE {nrmse:.3e}"
            )
            record["against"][label].append(
                {
                    "T1_ms": a,
                    "T2_ms": b,
                    "max": float(difference[row].max()),
                    "nrmse": nrmse,
                }
            )

    print(f"\nPer tissue, at {arguments.states} orders:")
    signal = torchsim_signal(T1, T2, flip, TR, arguments.states)
    turn = np.exp(1j * np.angle(np.sum(reference * np.conj(signal))))
    for row, (a, b) in enumerate(TISSUES):
        difference = np.abs(reference[row] - signal[row] * turn)
        print(
            f"  T1 = {a:6.0f} ms, T2 = {b:6.0f} ms:  "
            f"max {difference.max():.3e}, "
            f"NRMSE {np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(np.abs(reference[row]) ** 2)):.3e}"
        )
        record.setdefault("against", {}).setdefault("TorchSim", []).append(
            {
                "T1_ms": a,
                "T2_ms": b,
                "max": float(difference.max()),
                "nrmse": float(
                    np.sqrt(np.mean(difference**2))
                    / np.sqrt(np.mean(np.abs(reference[row]) ** 2))
                ),
            }
        )

    written = Path(__file__).resolve().parent / "results" / "validation.json"
    written.parent.mkdir(exist_ok=True)
    # Merge rather than overwrite: a second run comparing a different set of
    # dumps adds to what the first recorded.
    if written.exists():
        existing = json.loads(written.read_text())
        merged = {**existing, **record}
        merged["against"] = {**existing.get("against", {}), **record["against"]}
        record = merged
    written.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwritten to {written}")


if __name__ == "__main__":
    main()
