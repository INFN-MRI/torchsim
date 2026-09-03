"""The same sequences, run by another EPG library.

Every other test in this suite is TorchSim measured against a closed form, a
published figure, or an integration written out beside it. This one is
TorchSim measured against a second implementation of the whole formalism:
`epgpy <https://github.com/py-baudin/epgpy>`_, which shares no code, no author
and no array library with this one.

What that catches is a convention the closed forms are blind to -- which way a
phase turns, where a gradient winds, what a readout is taken relative to --
because two libraries agreeing on a hundred echoes of a spoiled train have to
agree on all of it.

epgpy is not on PyPI, so this is skipped unless it has been installed by hand:
``pip install git+https://github.com/py-baudin/epgpy``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from torchsim import (
    Delay,
    Dephase,
    EpgEngine,
    Excitation,
    Readout,
    Refocusing,
    TissueProperties,
    description,
)

epgpy = pytest.importorskip("epgpy", reason="the cross-check needs epgpy")

T1_MS, T2_MS = 1645.0, 50.0
ORDERS = 40


def _torchsim(parts, nstates=ORDERS):
    signal = (
        EpgEngine()
        .simulate(
            description(*parts),
            TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS),
            nstates=nstates,
        )
        .signal
    )
    return np.abs(np.asarray(signal).reshape(-1))


def _epgpy(operators, nstates=ORDERS):
    from epgpy import functions

    return np.abs(
        np.asarray(functions.simulate(operators, max_nstate=nstates)).reshape(-1)
    )


@pytest.mark.parametrize("flip_deg", [180.0, 120.0, 60.0])
def test_a_refocused_train_agrees_echo_for_echo(flip_deg: float) -> None:
    """A train below 180 degrees lives on its stimulated echoes.

    At a right angle and below, most of the signal has been stored along z and
    brought back, so the echo amplitudes are a sum over pathways rather than
    one decay -- which is the part of the formalism a closed form cannot
    check.
    """
    from epgpy import operators

    echoes, spacing_ms = 17, 5.0
    theirs = [operators.T(90, 90)] + [
        operators.S(1, duration=spacing_ms / 2),
        operators.E(spacing_ms / 2, T1_MS, T2_MS),
        operators.T(flip_deg, 0),
        operators.S(1, duration=spacing_ms / 2),
        operators.E(spacing_ms / 2, T1_MS, T2_MS),
        operators.ADC,
    ] * echoes

    ours = [Excitation(0.5 * math.pi, 0.5 * math.pi)]
    for _ in range(echoes):
        ours += [
            Delay(spacing_ms / 2 * 1e-3),
            Refocusing(math.radians(flip_deg), 0.0),
            Delay(spacing_ms / 2 * 1e-3),
            Readout(0.0),
        ]

    theirs, ours = _epgpy(theirs), _torchsim(ours)
    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() / theirs.max() < 1e-5


def test_a_spoiled_train_agrees_over_two_hundred_repetitions() -> None:
    """Quadratic RF spoiling, which is where a phase convention shows up.

    The transverse magnetization left over from one repetition is carried into
    the next on a configuration order that the phase increment is meant to
    scatter. Two libraries that turned a phase in opposite directions would
    still each settle, and to different places.
    """
    from epgpy import operators

    repetitions, tr_ms, flip_deg, step_deg = 200, 10.0, 20.0, 117.0
    phases = step_deg * np.arange(repetitions) * (np.arange(repetitions) + 1) / 2.0

    theirs = [
        part
        for index in range(repetitions)
        for part in (
            operators.T(flip_deg, float(phases[index])),
            operators.Adc(phase=float(-phases[index])),
            operators.E(tr_ms, T1_MS, T2_MS),
            operators.S(1),
        )
    ]
    ours = [
        part
        for index in range(repetitions)
        for part in (
            Excitation(math.radians(flip_deg), math.radians(float(phases[index]))),
            Readout(math.radians(float(phases[index]))),
            Delay(tr_ms * 1e-3),
            Dephase(),
        )
    ]

    theirs, ours = _epgpy(theirs, nstates=30), _torchsim(ours, nstates=30)
    assert np.abs(ours - theirs).max() / theirs.max() < 1e-3
    # And it is a settled answer rather than two curves that never started.
    assert theirs[-1] > 0.01
