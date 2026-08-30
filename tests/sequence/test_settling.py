"""Reading the settled signal off a few playings.

A description played over and over is an affine recursion, so its samples are a
constant plus decaying modes -- exactly, because the recursion is linear. That
form has its limit fixed by finitely many terms, which is what lets a sequence
that takes thousands of playings to arrive be answered by a handful.

The transform is checked first against a sequence whose limit is written down
here, and then against the sequences it exists for, held to the answer running
to them gives.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import (
    EpgEngine,
    SequenceDescription,
    TissueProperties,
    ideal_rf_definition,
    operator,
)
from torchsim.sequence import _builders
from torchsim.sequence._operators import Excitation, compose
from torchsim.sequence._settling import settled

STATES = 32


def test_the_transform_finds_the_limit_of_a_sequence_that_has_one():
    """Three geometric modes and a constant, which seven terms determine."""
    steps = torch.arange(15, dtype=torch.float64)
    limit = 0.37
    sequence = (
        limit + 0.8 * 0.97**steps - 0.5 * 0.8**steps + 0.2 * 0.5**steps
    ).unsqueeze(-1)

    found, residual = settled(sequence)
    assert abs(float(found[0]) - limit) < 1e-9
    assert residual < 1e-6


def test_a_sequence_that_has_arrived_is_left_alone():
    """Nothing to remove, and the transform would divide round-off by it."""
    constant = torch.full((9, 4), 0.25, dtype=torch.float64)
    found, residual = settled(constant)
    assert torch.equal(found, constant[-1])
    assert residual == 0.0


def _balanced():
    events, duration = compose(
        Excitation(np.deg2rad(20.0), 0.0),
        operator("bssfp-readout")(0.0, duration_s=5e-3),
    )
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )


def _long_train(frames=200):
    return _builders.mrf_description(
        torch.deg2rad(5.0 + 55.0 * torch.sin(torch.linspace(0, np.pi, frames))),
        torch.full((frames,), 10e-3),
        inversion_time_s=20e-3,
    )


# What it takes to run to the same answer, and how close settling gets. The
# tolerance is a property of the sequence rather than of the arithmetic: where
# one mode governs the train the transform removes it and lands within parts in
# a hundred thousand, and where several do, the order it stops at is chosen by
# a residual that is itself round-off before the modes are exhausted. Which
# order that turns out to be varies from one tissue to the next, so these are
# asked of a spread of them rather than of one: the worst of the spread and the
# middle of it are held separately, because they say different things.
FAMILIES = {
    "spoiled": (
        _builders.spgr_description(
            torch.tensor([np.deg2rad(20.0)]), torch.tensor([10e-3]), 4e-3
        ),
        {},
        2048,
        (5e-4, 2e-5),
    ),
    "unbalanced": (
        _builders.mrf_description(
            torch.tensor([np.deg2rad(20.0)]), torch.tensor([10e-3])
        ),
        {},
        4096,
        (1e-2, 6e-3),
    ),
    "balanced, off the band": (_balanced(), {"b0_hz": 40.0}, 4096, (1e-2, 2e-3)),
    "a train that arrives on its own": (_long_train(), {}, 8, (2e-4, 1e-6)),
}

TISSUES = dict(
    t1_ms=torch.linspace(300.0, 3000.0, 32), t2_ms=torch.linspace(20.0, 250.0, 32)
)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_settling_lands_where_running_to_it_lands(name):
    description, extra, playings, (worst, typical) = FAMILIES[name]
    spread = dict(TISSUES)
    spread.update(
        {key: torch.full_like(spread["t1_ms"], value) for key, value in extra.items()}
    )
    tissue = TissueProperties(**spread)
    engine = EpgEngine()

    reached = engine.simulate(
        description, tissue, repetitions=playings, nstates=STATES
    ).signal
    found = engine.simulate(
        description, tissue, repetitions="auto", nstates=STATES
    ).signal

    assert found.shape == reached.shape
    drift = (found - reached).abs() / reached.abs().max()
    assert float(drift.max()) < worst
    # Most of the spread does far better than the worst of it, which is what
    # says the bound is set by the hardest tissue rather than by the method.
    assert float(drift.median()) < typical


def test_a_settled_run_is_labelled_as_the_one_playing_it_stands_for():
    description, _extra, _playings, _bounds = FAMILIES["spoiled"]
    tissue = TissueProperties(t1_ms=1000.0, t2_ms=80.0)
    engine = EpgEngine()
    found = engine.simulate(description, tissue, repetitions="auto", nstates=4)
    once = engine.simulate(description, tissue, nstates=4)

    assert found.signal.shape == once.signal.shape
    assert torch.equal(found.time_us, once.time_us)
    assert torch.equal(found.event_index, once.event_index)
    assert torch.equal(found.repetition, once.repetition)
    assert torch.equal(found.echo, once.echo)


def test_the_settled_answer_carries_the_derivatives_of_the_one_it_stands_for():
    """The transform is arithmetic on the playings, so it differentiates."""
    description, _extra, playings, _bounds = FAMILIES["spoiled"]

    def gradients(repetitions):
        t1_ms = torch.tensor([600.0, 1400.0]).requires_grad_(True)
        t2_ms = torch.tensor([40.0, 120.0]).requires_grad_(True)
        signal = (
            EpgEngine()
            .simulate(
                description,
                TissueProperties(t1_ms=t1_ms, t2_ms=t2_ms),
                repetitions=repetitions,
                nstates=4,
            )
            .signal
        )
        signal.abs().sum().backward()
        return t1_ms.grad, t2_ms.grad

    for mine, theirs in zip(gradients("auto"), gradients(playings), strict=True):
        assert ((mine - theirs).abs().max() / theirs.abs().max()) < 1e-3
