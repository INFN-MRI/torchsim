"""Whether TorchSim, sycomore and epgpy compute the same fingerprint.

Three independent implementations of the same model -- a fused C++ or Triton
state machine over a packed event stream, a C++ EPG library driven one tissue
at a time from Python, and an operator-per-event NumPy library -- should agree
to the precision the coarsest of them carries. This says by how much they do,
over a tissue grid, and shows what truncating the configuration orders costs.

Run as ``python benchmarks/validate.py``. Epgpy is optional; the comparison
against sycomore runs without it.
"""

from __future__ import annotations

import os
import sys

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
    T1: np.ndarray, T2: np.ndarray, flip: np.ndarray, TR: float, states: int
) -> np.ndarray:
    """The same train on the fused state machine, at a given order count."""
    import torch

    from torchsim.simulators import MRFSimulator

    sequence = MRFSimulator(
        flip=torch.tensor(flip, dtype=torch.float32), TR=TR, states=states
    )
    signal = sequence.simulate(
        T1=torch.tensor(T1, dtype=torch.float32),
        T2=torch.tensor(T2, dtype=torch.float32),
        inv_efficiency=1.0,
    )
    return signal.numpy()


def main() -> None:
    """Print the agreement, and its dependence on the orders carried."""
    length, TR = 500, 10.0
    flip = flip_train(length)
    T1 = np.array([tissue[0] for tissue in TISSUES])
    T2 = np.array([tissue[1] for tissue in TISSUES])

    reference = np.stack([sycomore_signal(a, b, flip, TR) for a, b in TISSUES])

    print(f"MRF-FISP, {length} repetitions, TR = {TR} ms, inversion prepared.")
    print("Reference: sycomore, every order kept, double precision.\n")
    print(f"{'states':>7s}  {'max |difference|':>17s}  {'relative':>10s}")
    for states in (5, 10, 20, 40, 80):
        signal = torchsim_signal(T1, T2, flip, TR, states)
        # The two differ by the constant phase each writes its echo with.
        turn = np.exp(1j * np.angle(np.sum(reference * np.conj(signal))))
        difference = np.abs(reference - signal * turn)
        print(
            f"{states:7d}  {difference.max():17.3e}  "
            f"{difference.max() / np.abs(reference).max():10.2e}"
        )

    try:
        other = epgpy_signal(T1, T2, flip, TR, 20)
    except ImportError:
        other = None
    if other is not None:
        signal = torchsim_signal(T1, T2, flip, TR, 20)
        turn = np.exp(1j * np.angle(np.sum(other * np.conj(signal))))
        difference = np.abs(other - signal * turn)
        print(
            "\nTorchSim against epgpy, both truncated to 20 orders: "
            f"max {difference.max():.3e}, "
            f"relative {difference.max() / np.abs(other).max():.2e}"
        )

    print("\nPer tissue, at 20 orders:")
    signal = torchsim_signal(T1, T2, flip, TR, 20)
    turn = np.exp(1j * np.angle(np.sum(reference * np.conj(signal))))
    for row, (a, b) in enumerate(TISSUES):
        difference = np.abs(reference[row] - signal[row] * turn)
        print(
            f"  T1 = {a:6.0f} ms, T2 = {b:6.0f} ms:  "
            f"max {difference.max():.3e}, "
            f"NRMSE {np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(np.abs(reference[row]) ** 2)):.3e}"
        )


if __name__ == "__main__":
    main()
