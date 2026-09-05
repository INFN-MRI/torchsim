"""The two samples an unbalanced repetition can take.

A repetition that winds every order on once has two things worth recording: the
free induction decay of the pulse that just played, and the echo the next pulse
would refocus, which is the same state read after the gradient rather than
before it. :func:`SSFPFidReadout` takes the first and :func:`SSFPEchoReadout`
the second, and this pins both against an extended phase graph written out here
from the operators in the literature rather than from anything TorchSim does.

The reference is held honest by being asked for both: it has to reproduce the
FID sample, which is what the rest of the suite and the published comparisons
already agree on, with the same shift convention that produces the echo.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchsim import (
    SequenceDescription,
    SSFPEchoReadout,
    SSFPFidReadout,
)
from torchsim.sequence import (
    EpgEngine,
    TissueProperties,
    ideal_rf_definition,
)
from torchsim.sequence._operators import Excitation, compose

# More orders than the train can reach, so neither side truncates and the
# comparison is of the physics rather than of two truncation policies.
ORDERS = 32
REPETITIONS = 20
TR_MS = 8.0
FLIP_RAD = np.deg2rad(40.0)


def _reference(t1_ms: float, t2_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """The same train, one configuration state at a time, in double precision.

    Returns what an ADC placed before the unbalanced gradient reads, and what
    one placed after it reads.
    """
    plus = np.zeros(ORDERS + 2, dtype=complex)
    minus = np.zeros(ORDERS + 2, dtype=complex)
    longitudinal = np.zeros(ORDERS + 2, dtype=complex)
    longitudinal[0] = 1.0
    fid, echo = [], []

    cosine, sine = np.cos(FLIP_RAD), np.sin(FLIP_RAD)
    half_cosine = np.cos(0.5 * FLIP_RAD) ** 2
    half_sine = np.sin(0.5 * FLIP_RAD) ** 2
    relax1 = np.exp(-TR_MS / t1_ms)
    relax2 = np.exp(-TR_MS / t2_ms)

    for _ in range(REPETITIONS):
        # An instantaneous rotation about x, mixing the three families.
        turned_plus = half_cosine * plus + half_sine * minus - 1j * sine * longitudinal
        turned_minus = half_sine * plus + half_cosine * minus + 1j * sine * longitudinal
        turned_z = -0.5j * sine * plus + 0.5j * sine * minus + cosine * longitudinal
        plus, minus, longitudinal = turned_plus, turned_minus, turned_z

        fid.append(plus[0])

        # One unbalanced gradient: every order winds on, and the order that
        # crosses zero is the conjugate of its partner.
        plus = np.roll(plus, 1)
        plus[0] = 0.0
        minus = np.roll(minus, -1)
        minus[-1] = 0.0
        plus[0] = np.conj(minus[0])

        echo.append(plus[0])

        plus = plus * relax2
        minus = minus * relax2
        longitudinal = longitudinal * relax1
        longitudinal[0] += 1.0 - relax1

    return np.array(fid), np.array(echo)


def _simulate(readout, t1_ms: float, t2_ms: float) -> np.ndarray:
    """One unbalanced train, sampled by the given readout."""
    repetition = [
        Excitation(FLIP_RAD, 0.0),
        readout(0.0, duration_s=TR_MS * 1e-3),
    ]
    events, duration_s = compose(*(repetition * REPETITIONS))
    description = SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration_s,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )
    result = EpgEngine().simulate(
        description,
        TissueProperties(t1_ms=t1_ms, t2_ms=t2_ms),
        nstates=ORDERS,
    )
    return result.signal.reshape(-1).numpy()


@pytest.mark.parametrize(
    ("t1_ms", "t2_ms"), [(1000.0, 80.0), (600.0, 40.0), (1400.0, 250.0)]
)
def test_the_two_readouts_take_the_states_the_reference_takes(t1_ms, t2_ms):
    fid, echo = _reference(t1_ms, t2_ms)
    scale = np.abs(fid).max()
    assert np.abs(_simulate(SSFPFidReadout, t1_ms, t2_ms) - fid).max() < 1e-5 * scale
    assert np.abs(_simulate(SSFPEchoReadout, t1_ms, t2_ms) - echo).max() < 1e-5 * scale


def test_the_first_repetition_has_no_echo_to_read():
    """What tells the two readouts apart, in one sample.

    An echo readout can only see what an earlier excitation left behind, so the
    first repetition of a train reads zero where the free induction decay reads
    the whole of the excitation.
    """
    fid = _simulate(SSFPFidReadout, 1000.0, 80.0)
    echo = _simulate(SSFPEchoReadout, 1000.0, 80.0)
    assert abs(echo[0]) < 1e-7
    assert abs(abs(fid[0]) - np.sin(FLIP_RAD)) < 1e-6
    assert abs(echo[1]) > 1e-2


def test_it_is_reachable_by_name():
    from torchsim.sequence import (
        operator,
        operator_names,
    )

    assert "ssfp-echo-readout" in operator_names()
    assert operator("ssfp-echo-readout") is SSFPEchoReadout


def test_the_echo_lags_the_free_induction_decay_by_one_repetition():
    """The two readouts read the same train one gradient apart.

    Placing the ADC after the winding rather than before it moves what it sees
    back by a repetition, which is what makes the pair worth acquiring: they
    sample the same states at two ages.
    """
    fid, echo = _reference(1000.0, 80.0)
    assert abs(echo[0]) < 1e-12
    # The echo climbs while the free induction decay falls, over the transient
    # where the states the gradient has wound on are still accumulating.
    assert np.all(np.diff(np.abs(echo[:5])) > 0)
    assert np.all(np.diff(np.abs(fid[:5])) < 0)
