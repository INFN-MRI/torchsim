"""Playing a description into the state a scanner plays it in.

A description simulated once starts from equilibrium, which is a transient a
scanner plays at the beginning of an examination and never again: every later
playing starts from whatever the one before it left. ``repetitions`` plays the
stream that many times and records the last, which is what makes a simulated
dictionary the dictionary the scanner acquires.

The settled answer is held against a closed form written out here, so what is
being checked is the physics rather than TorchSim's agreement with itself.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from torchsim import EpgEngine, TissueProperties
from torchsim.sequence import _builders

FLIP_DEG, TR_S, TE_S = 20.0, 10e-3, 4e-3


def _spoiled():
    return _builders.spgr_description(
        torch.tensor([np.deg2rad(FLIP_DEG)]), torch.tensor([TR_S]), TE_S
    )


STREAMS = {
    "spoiled": _spoiled(),
    "refocused": _builders.fse_description(
        torch.full((8,), np.deg2rad(140.0)), 5e-3, phases_rad=0.0
    ),
    "unbalanced": _builders.mrf_description(
        torch.linspace(0.1, 1.0, 20), torch.full((20,), 10e-3)
    ),
}

TISSUE = TissueProperties(
    t1_ms=torch.tensor([600.0, 1400.0]), t2_ms=torch.tensor([40.0, 120.0])
)


def _written_out(stream, playings):
    """The same train, written out as one description that records throughout.

    An independent route to the same magnetization: rather than asking for
    ``playings`` of one stream, this concatenates the events by hand at the
    timestamps they fall on and records every one of them, so the last block of
    samples is what a settled run should hold.
    """
    events = []
    for playing in range(playings):
        offset = stream.tr_duration_us * playing
        events.extend(
            replace(event, timestamp_us=event.timestamp_us + offset)
            for event in stream.events
        )
    return replace(
        stream, events=tuple(events), tr_duration_us=stream.tr_duration_us * playings
    )


@pytest.mark.parametrize("name", list(STREAMS))
@pytest.mark.parametrize("playings", [1, 2, 3, 5])
def test_settling_records_what_the_last_playing_of_a_written_out_train_records(
    name, playings
):
    """The magnetization crosses a settling playing; only the recording stops."""
    stream = STREAMS[name]
    settled = EpgEngine().simulate(stream, TISSUE, repetitions=playings, nstates=16)
    written = EpgEngine().simulate(_written_out(stream, playings), TISSUE, nstates=16)

    kept = written.signal.reshape(*written.signal.shape[:-1], -1)
    kept = kept[..., -settled.signal.shape[-1] :]
    scale = kept.abs().max()
    assert ((settled.signal - kept).abs().max() / scale) < 1e-5
    # Its labels say what it is: one playing, numbered from zero.
    assert torch.equal(settled.repetition, torch.zeros_like(settled.repetition))


@pytest.mark.parametrize("playings", [1, 4, 16])
def test_a_settled_run_holds_one_playing_however_many_it_took(playings):
    """The settling playings cost their arithmetic and none of the signal."""
    stream = STREAMS["unbalanced"]
    settled = EpgEngine().simulate(stream, TISSUE, repetitions=playings, nstates=16)
    once = EpgEngine().simulate(stream, TISSUE, nstates=16)
    assert settled.signal.shape == once.signal.shape


@pytest.mark.parametrize(("t1_ms", "t2_ms"), [(600.0, 40.0), (1000.0, 80.0)])
def test_a_settled_spoiled_train_is_the_ernst_equation(t1_ms, t2_ms):
    """The closed form the sequence has one of, reached by running the states.

    An ideally spoiled train keeps nothing transverse across a repetition, so
    its steady state is the textbook expression -- which is what a state
    machine driven to the same place has to land on.
    """
    signal = (
        EpgEngine()
        .simulate(
            _spoiled(),
            TissueProperties(t1_ms=t1_ms, t2_ms=t2_ms),
            repetitions=512,
            nstates=4,
        )
        .signal.reshape(-1)[0]
    )
    recovery = np.exp(-TR_S * 1e3 / t1_ms)
    turn = np.deg2rad(FLIP_DEG)
    ernst = (
        np.sin(turn)
        * (1.0 - recovery)
        / (1.0 - recovery * np.cos(turn))
        * np.exp(-TE_S * 1e3 / t2_ms)
    )
    assert abs(abs(float(signal.abs())) - ernst) < 1e-5 * ernst


def test_settling_is_refused_where_it_makes_no_sense():
    with pytest.raises(ValueError, match="repetitions"):
        EpgEngine().simulate(_spoiled(), TISSUE, repetitions=0)


def test_a_sequence_may_declare_how_far_it_has_to_settle():
    """A simulator knows its own physics; a caller may still overrule it."""
    from torchsim.simulators import MRFSimulator

    flip = torch.linspace(5.0, 60.0, 20)
    assert MRFSimulator(flip=flip, TR=10.0, states=16).repetitions == 1

    declared = MRFSimulator(flip=flip, TR=10.0, states=16, repetitions=3)
    assert declared.repetitions == 3
    fixed = declared.simulate(T1=1000.0, T2=80.0)
    per_call = MRFSimulator(flip=flip, TR=10.0, states=16).simulate(
        T1=1000.0, T2=80.0, repetitions=3
    )
    assert torch.equal(fixed, per_call)
    assert not torch.equal(fixed, declared.simulate(T1=1000.0, T2=80.0, repetitions=1))
