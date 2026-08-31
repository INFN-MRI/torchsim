"""What a bulk off-resonance does to a sample, and when it can be said in one number.

A static field offset turns the transverse states through a phase that grows
with time; the gradients wind them through one that grows with area. A state's
configuration order stands for the second, so it stands for the first as well
-- and the turn can be applied to the sample rather than carried by the states
-- exactly where the two grow together.

These pin both halves of that against the kernel that carries the field
through the states: that the signed unrefocused time each sample carries
predicts what the kernel computes, and that the verdict saying when it does is
neither optimistic nor needlessly shy. The reference is the complex kernel,
which is a different implementation of the same physics rather than a restating
of this rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import EpgEngine, TissueProperties
from torchsim.sequence import _builders
from torchsim.sequence._accelerators import _pack_events

T1_MS, T2_MS = 1000.0, 80.0
# Deliberately not a whole number of turns per repetition or echo spacing: a
# frequency that wraps exactly would hide a wrong answer behind an alias.
B0_HZ = 53.0
STATES = 48
LENGTH = 24


def _packed(description):
    return _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )


def _signal(description, b0_hz):
    return (
        EpgEngine()
        .simulate(
            description,
            TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS, b0_hz=b0_hz),
            nstates=STATES,
        )
        .signal.reshape(-1)
        .numpy()
    )


def _refocused(degrees=180.0):
    return _builders.fse_description(
        torch.full((LENGTH,), np.deg2rad(degrees)),
        5e-3,
        phases_rad=0.0,
        excitation_phase_rad=np.pi / 2,
    )


def _spoiled(echo_time_s=3e-3, repetition_time_s=12e-3):
    return _builders.spgr_description(
        torch.full((LENGTH,), np.deg2rad(20.0)),
        torch.full((LENGTH,), repetition_time_s),
        echo_time_s,
    )


def _unbalanced(repetition_time_s, inversion_time_s=0.0):
    return _builders.mrf_description(
        torch.full((LENGTH,), np.deg2rad(40.0)),
        repetition_time_s,
        inversion_time_s=inversion_time_s,
    )


TAKES_IT = {
    "refocused, 180 deg": _refocused(180.0),
    "refocused, 60 deg": _refocused(60.0),
    "spoiled, TE 3 ms": _spoiled(3e-3),
    "spoiled, TE 6 ms": _spoiled(6e-3),
    "spoiled, varying TR": _builders.spgr_description(
        torch.full((LENGTH,), np.deg2rad(20.0)),
        torch.linspace(11e-3, 15e-3, LENGTH),
        3e-3,
    ),
    "unbalanced, one TR": _unbalanced(torch.full((LENGTH,), 10e-3)),
    "unbalanced, inversion prepared": _unbalanced(
        torch.full((LENGTH,), 10e-3), inversion_time_s=20e-3
    ),
}

REFUSES_IT = {
    "unbalanced, ramped TR": _unbalanced(torch.linspace(11e-3, 15e-3, LENGTH)),
    "unbalanced, random TR": _unbalanced(
        8e-3 + 8e-3 * torch.rand(LENGTH, generator=torch.Generator().manual_seed(0))
    ),
}


@pytest.mark.parametrize("name", list(TAKES_IT))
def test_the_unrefocused_time_reproduces_what_the_states_compute(name):
    """One turn per sample, against a field carried through every operator."""
    description = TAKES_IT[name]
    packed = _packed(description)
    assert bool(packed.analytic_dephasing)

    tau_s = packed.unrefocused_us.reshape(-1).numpy() * 1e-6
    on_resonance = _signal(description, 0.0)
    predicted = on_resonance * np.exp(-2j * np.pi * B0_HZ * tau_s)
    scale = np.abs(on_resonance).max()
    assert np.abs(_signal(description, B0_HZ) - predicted).max() < 1e-4 * scale


@pytest.mark.parametrize("name", list(REFUSES_IT))
def test_a_stream_the_verdict_refuses_is_one_that_would_be_wrong(name):
    """The refusals are not caution: the one-number answer is really wrong here.

    A repetition time that varies winds each stretch through the same orders in
    unequal times, so two pathways arriving at one order have dephased for
    different lengths of it. Their sum is then not a turn of the sample at all,
    and its magnitude moves.
    """
    description = REFUSES_IT[name]
    packed = _packed(description)
    assert not bool(packed.analytic_dephasing)

    tau_s = packed.unrefocused_us.reshape(-1).numpy() * 1e-6
    on_resonance = _signal(description, 0.0)
    off_resonance = _signal(description, B0_HZ)
    predicted = on_resonance * np.exp(-2j * np.pi * B0_HZ * tau_s)
    scale = np.abs(on_resonance).max()
    assert np.abs(off_resonance - predicted).max() > 1e-2 * scale


def test_a_refocused_train_carries_no_dephasing_to_its_echoes():
    """Which is why a spin echo is a spin echo."""
    packed = _packed(_refocused(180.0))
    # Against the echo spacing it refocuses, not against zero: the timestamps
    # are float32 microseconds and the cancellation is exact only in exact
    # arithmetic.
    assert np.abs(packed.unrefocused_us.numpy()).max() < 1e-5 * 5e3


def test_a_spoiled_train_carries_its_echo_time_to_every_sample():
    for echo_time_s in (3e-3, 6e-3):
        packed = _packed(_spoiled(echo_time_s))
        carried = packed.unrefocused_us.numpy() * 1e-6
        # Timestamps are float32 microseconds, so a millisecond is held to
        # about a nanosecond -- well under an RF raster tick.
        assert np.abs(carried - echo_time_s).max() < 1e-5 * echo_time_s


def _both_ways(description, tissue, wants_grad=False):
    """The same run with the field on the samples and carried by the states."""
    from torchsim.sequence import _accelerators

    answers = []
    for analytic in (True, False):
        patched = None
        if not analytic:
            patched = _accelerators._analytic_dephasing
            _accelerators._analytic_dephasing = lambda *_arguments: None
        try:
            properties = tissue()
            signal = (
                EpgEngine().simulate(description, properties, nstates=STATES).signal
            )
            if not wants_grad:
                answers.append((signal.detach(),))
                continue
            # Phase sensitive, so an off-resonance derivative is not zero by
            # symmetry the way a magnitude's would be.
            (signal.real.sum() + 0.5 * signal.imag.sum()).backward()
            answers.append(
                (
                    signal.detach(),
                    properties.t1_ms.grad,
                    properties.t2_ms.grad,
                    properties.b0_hz.grad,
                )
            )
        finally:
            if patched is not None:
                _accelerators._analytic_dephasing = patched
    return answers


def test_a_field_on_the_samples_answers_what_a_field_in_the_states_answers():
    """Including the derivative along it, which the kernels no longer produce."""
    voxels = 128
    description = _spoiled(4e-3)

    def tissue():
        return TissueProperties(
            t1_ms=torch.linspace(200.0, 3000.0, voxels).requires_grad_(True),
            t2_ms=torch.linspace(10.0, 300.0, voxels).requires_grad_(True),
            b0_hz=torch.full((voxels,), B0_HZ).requires_grad_(True),
        )

    analytic, in_states = _both_ways(description, tissue, wants_grad=True)
    for name, mine, theirs in zip(
        ("signal", "dT1", "dT2", "dB0"), analytic, in_states, strict=True
    ):
        scale = theirs.abs().max()
        assert scale > 0, name
        assert ((mine - theirs).abs().max() / scale) < 1e-5, name


def test_off_resonance_no_longer_keeps_a_run_off_the_real_kernels(
    always_worth_detecting,
):
    """Which is the point of taking it out of the states.

    A train whose pulses share an axis has a real subspace to run in, and a
    static field offset used to be enough to lose it. Applied to the samples it
    is not: the states never see it, so they stay on the axis and the reduced
    kernels stay available -- for the plain pass and for the derivatives alike.
    """
    from torchsim.sequence import _accelerators

    verdicts = []
    original = _accelerators._auto_real_axis

    def watched(kind, *arguments, **keywords):
        verdict = original(kind, *arguments, **keywords)
        verdicts.append((kind, verdict))
        return verdict

    voxels = 64
    description = _unbalanced(torch.full((LENGTH,), 10e-3))
    properties = TissueProperties(
        t1_ms=torch.linspace(200.0, 3000.0, voxels),
        t2_ms=torch.linspace(10.0, 300.0, voxels),
        b0_hz=torch.full((voxels,), B0_HZ),
    )
    _accelerators._auto_real_axis = watched
    try:
        EpgEngine().simulate(description, properties, nstates=STATES)
        assert verdicts == [("forward", 1)]

        verdicts.clear()
        _accelerators._analytic_dephasing = (
            patched := _accelerators._analytic_dephasing
        )
        _accelerators._analytic_dephasing = lambda *_arguments: None
        try:
            EpgEngine().simulate(description, properties, nstates=STATES)
            assert verdicts == [("forward", None)]
        finally:
            _accelerators._analytic_dephasing = patched
    finally:
        _accelerators._auto_real_axis = original
