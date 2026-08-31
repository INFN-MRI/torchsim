"""Solving for a settled state instead of running to it.

A train that winds nothing carries one configuration order, and one order is
three numbers: a transverse magnetization, one complex number there because
``F-`` is the conjugate of ``F+``, and a longitudinal one, which is real. Both
ends of the map are reachable with nothing but the sequence -- a state is
prepared by turning equilibrium through a pulse, and read by an ADC followed by
a pulse that tips the longitudinal part into the plane -- so four prepared
states determine the whole affine map, and its fixed point is a solve rather
than a limit.

The answers are held against closed forms written out here, which is what says
the machinery reproduces the physics rather than itself.
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
from torchsim.sequence._fixed_point import carries_one_order
from torchsim.sequence._operators import compose

TR_S, TE_S, FLIP_DEG = 5e-3, 2.5e-3, 30.0

Excitation = operator("excitation")
Readout = operator("readout")
Delay = operator("delay")


def _balanced(flip_deg=FLIP_DEG, alternating=True):
    """A balanced repetition, read half a repetition time after the pulse."""
    turn = np.deg2rad(flip_deg)
    half = [Delay(TE_S), Readout(0.0), Delay(TR_S - TE_S)]
    parts = [Excitation(turn, 0.0), *half]
    if alternating:
        parts += [Excitation(turn, np.pi), *half]
    events, duration = compose(*parts)
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )


def _spoiled():
    return _builders.spgr_description(
        torch.tensor([np.deg2rad(20.0)]), torch.tensor([10e-3]), 4e-3
    )


ONE_ORDER = {"balanced": _balanced(), "spoiled": _spoiled()}
MANY_ORDERS = {
    "unbalanced": _builders.mrf_description(
        torch.tensor([np.deg2rad(20.0)]), torch.tensor([10e-3])
    ),
    "refocused": _builders.fse_description(
        torch.full((4,), np.deg2rad(140.0)), 5e-3, phases_rad=0.0
    ),
}

SPREAD = TissueProperties(
    t1_ms=torch.linspace(300.0, 3000.0, 24), t2_ms=torch.linspace(20.0, 250.0, 24)
)


@pytest.mark.parametrize("name", list(ONE_ORDER))
def test_a_train_that_winds_nothing_is_recognised(name):
    assert carries_one_order(ONE_ORDER[name])


@pytest.mark.parametrize("name", list(MANY_ORDERS))
def test_a_train_that_winds_is_not(name):
    """A crusher or a shift puts states where three numbers cannot hold them."""
    assert not carries_one_order(MANY_ORDERS[name])


@pytest.mark.parametrize("name", list(ONE_ORDER))
def test_the_solved_state_is_the_one_running_to_it_reaches(name):
    engine = EpgEngine()
    reached = engine.simulate(
        ONE_ORDER[name], SPREAD, repetitions=8192, nstates=1
    ).signal
    solved = engine.simulate(
        ONE_ORDER[name], SPREAD, repetitions="auto", nstates=1
    ).signal

    assert solved.shape == reached.shape
    drift = (solved - reached).abs() / reached.abs().max()
    assert float(drift.max()) < 1e-4


@pytest.mark.parametrize(("t1_ms", "t2_ms"), [(600.0, 40.0), (1000.0, 80.0)])
def test_a_solved_balanced_train_is_the_closed_form(t1_ms, t2_ms):
    """The expression a balanced steady state has, reached by the states.

    Written out here rather than taken from the closed-form simulator, so that
    a change to either would have to break this to go unnoticed.
    """
    signal = (
        EpgEngine()
        .simulate(
            _balanced(),
            TissueProperties(t1_ms=t1_ms, t2_ms=t2_ms),
            repetitions="auto",
            nstates=1,
        )
        .signal.reshape(-1)[0]
    )
    recovery = np.exp(-TR_S * 1e3 / t1_ms)
    decay = np.exp(-TR_S * 1e3 / t2_ms)
    turn = np.deg2rad(FLIP_DEG)
    closed = (
        np.sin(turn)
        * (1.0 - recovery)
        / (1.0 - (recovery - decay) * np.cos(turn) - recovery * decay)
        # Half a repetition of transverse decay, to the echo.
        * np.sqrt(decay)
    )
    assert abs(float(signal.abs()) - closed) < 1e-4 * closed


def test_the_solve_carries_the_derivatives_of_what_it_stands_for():
    """Every step of it is a recorded signal, so it differentiates like one."""

    def gradients(repetitions):
        t1_ms = torch.tensor([600.0, 1400.0]).requires_grad_(True)
        t2_ms = torch.tensor([40.0, 120.0]).requires_grad_(True)
        signal = (
            EpgEngine()
            .simulate(
                _balanced(),
                TissueProperties(t1_ms=t1_ms, t2_ms=t2_ms),
                repetitions=repetitions,
                nstates=1,
            )
            .signal
        )
        signal.abs().sum().backward()
        return t1_ms.grad, t2_ms.grad

    for mine, theirs in zip(gradients("auto"), gradients(8192), strict=True):
        assert ((mine - theirs).abs().max() / theirs.abs().max()) < 1e-4
