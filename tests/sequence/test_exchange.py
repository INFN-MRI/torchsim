"""A chemically exchanging pool, through the fused kernels.

Two pools that both carry transverse magnetization -- fat beside water, myelin
water beside free water, a metabolite beside its solvent. What separates this
from the semisolid pool of :mod:`test_bound_pool` is that this one precesses:
it has a T2, it sits at its own chemical shift, and a pulse rotates it rather
than saturating it. So ``F+`` and ``F-`` double, and with them the shift, the
spoil, the RF operator and what the ADC reads.

A tissue may declare this pool beside the semisolid one; what that system does
is :mod:`test_both_pools`, and this file holds the exchanging pool on its own.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import EpgEngine, TissueProperties, fse_description
from torchsim.sequence._accelerators import (
    NO_GEOMETRY,
    _pack_events,
    _run_packed,
)
from torchsim.sequence._parameters import TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue
from utils.packed_reference import simulate_packed

ECHOES = 8
STATES = 8

# Fat sits about 3.4 ppm off water, which at 3 T is a little over 400 Hz.
FAT_SHIFT_HZ = -420.0
FRACTION = 0.2
RATE_HZ = 20.0


def _description():
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )


def _signal(**properties):
    return (
        EpgEngine()
        .simulate(
            _description(),
            TissueProperties(**{"t1_ms": 1000.0, "t2_ms": 80.0, **properties}),
            nstates=STATES,
        )
        .signal
    )


# --- the gate ---


def test_the_default_tissue_runs_the_single_pool_kernel_bit_for_bit():
    """The safety net for the whole stage.

    A fraction of zero leaves the exchange diagonal and the pool empty, so
    nothing it carries can reach the free water and nothing it holds can reach
    the coil. The answer is not merely close to the one-pool answer, it is
    that answer.
    """
    plain = _signal()
    gated = _signal(
        pool_b_fraction=0.0,
        pool_b_exchange_hz=RATE_HZ,
        t2_pool_b_ms=20.0,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    )

    assert torch.equal(plain, gated)


def test_a_second_pool_takes_its_share_of_the_first_echo():
    """Equilibrium is split between the pools, and both reach the coil.

    On resonance and with no exchange, two pools with the same relaxation are
    one pool: the fractions add back to unity and the recorded signal is
    exactly what a single pool gives.
    """
    plain = _signal().abs()
    twinned = _signal(
        pool_b_fraction=0.3,
        t1_pool_b_ms=1000.0,
        t2_pool_b_ms=80.0,
    ).abs()

    scale = float(plain.max())
    assert scale > 0.0
    assert float((plain - twinned).abs().max()) / scale < 1e-5


# --- what the second pool is for ---


def _free_induction(times_s, **properties):
    """Excite, then read the transverse magnetization at each time after it.

    No refocusing, so what the coil sees is the two pools precessing freely
    and interfering -- which is the whole point of a chemical shift.

    ``times_s`` are measured from the excitation; the events carry the
    intervals between them, which is what the state machine steps through.
    """
    from torchsim.sequence._accelerators import _EXCITATION, _RECORD

    intervals = [
        after - before for before, after in zip((0.0, *times_s), times_s, strict=False)
    ]
    events = (
        torch.tensor([0.0, *intervals], dtype=torch.float32),
        torch.tensor([1, *([2] * len(times_s))], dtype=torch.int32),
        torch.tensor([0.5 * torch.pi, *([0.0] * len(times_s))], dtype=torch.float32),
        torch.zeros(1 + len(times_s), dtype=torch.float32),
        torch.tensor([_EXCITATION, *([_RECORD] * len(times_s))], dtype=torch.uint8),
        torch.tensor([-1, *range(len(times_s))], dtype=torch.int32),
        torch.zeros(1 + len(times_s), dtype=torch.int32),
        torch.zeros(1 + len(times_s), dtype=torch.float32),
        torch.zeros(1 + len(times_s), dtype=torch.float32),
    )
    prepared, _, _ = _prepare_tissue(
        TissueProperties(**{"t1_ms": 1000.0, "t2_ms": 80.0, **properties}),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return _run_packed(
        prepared,
        events,
        STATES,
        len(times_s),
        1,
        geometry=NO_GEOMETRY,
        exchanging=True,
    ).flatten()


def test_a_chemical_shift_beats_against_the_free_water():
    """The signature of two pools at different offsets.

    Read at the period the shift sets, the two are back in phase and the
    magnitude recovers; read at half of it they oppose and it dips. A kernel
    that dropped the shift would decay monotonically instead.
    """
    period = 1.0 / abs(FAT_SHIFT_HZ)
    delays = (0.0, 0.5 * period, period)
    signal = _free_induction(
        delays,
        pool_b_fraction=FRACTION,
        pool_b_exchange_hz=0.0,
        t1_pool_b_ms=300.0,
        t2_pool_b_ms=80.0,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    ).abs()

    # Out of phase at half a period, back in phase at a whole one.
    assert float(signal[1]) < float(signal[0])
    assert float(signal[2]) > float(signal[1])
    # Opposed, the two pools nearly cancel down to what their difference
    # leaves: |(1 - f) - f| against the 1 they start at.
    assert abs(float(signal[1] / signal[0]) - abs(1.0 - 2.0 * FRACTION)) < 0.02


def test_exchange_washes_the_beat_out():
    """Fast exchange merges two lines into one.

    Past the coalescence rate a spin visits both environments many times in a
    period, so it precesses at the population-weighted average and the beat
    disappears. That is a property of the exchange term, and nothing else in
    the state machine can produce it.
    """
    period = 1.0 / abs(FAT_SHIFT_HZ)
    delays = (0.0, 0.5 * period)
    tissue = dict(
        pool_b_fraction=FRACTION,
        t1_pool_b_ms=300.0,
        t2_pool_b_ms=80.0,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    )
    slow = _free_induction(delays, pool_b_exchange_hz=0.0, **tissue).abs()
    fast = _free_induction(delays, pool_b_exchange_hz=2.0e5, **tissue).abs()

    assert float(slow[1] / slow[0]) < 0.7
    assert float(fast[1] / fast[0]) > 0.95


def test_the_transverse_step_reproduces_the_package_operator():
    """The kernel's closed form against ``epg``'s matrix exponential.

    The readout scale is not what is being checked, so it is measured on the
    single-pool run -- whose transverse state is the elementary ``exp(-t R2)``
    -- and applied to the two-pool prediction.
    """
    from utils.epg import transverse_relaxation_exchange_op
    from utils.exchange import build_two_pool_exchange_matrix

    t2_ms, t2_b_ms = 80.0, 20.0
    delays = (2e-3, 5e-3, 1e-2, 2e-2)
    tissue = dict(
        pool_b_fraction=FRACTION,
        pool_b_exchange_hz=RATE_HZ,
        t1_pool_b_ms=300.0,
        t2_pool_b_ms=t2_b_ms,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    )
    measured = _free_induction(delays, t2_ms=t2_ms, **tissue)
    scale = _free_induction((0.0,), t2_ms=t2_ms)[0]

    free = 1.0 - FRACTION
    weight = torch.tensor([free, FRACTION], dtype=torch.float64)
    matrix = build_two_pool_exchange_matrix(
        weight, torch.tensor(RATE_HZ, dtype=torch.float64)
    )
    rates = torch.tensor([1000.0 / t2_ms, 1000.0 / t2_b_ms], dtype=torch.float64)
    offsets = torch.tensor([0.0, FAT_SHIFT_HZ], dtype=torch.float64)
    start = torch.tensor([free, FRACTION], dtype=torch.complex128)

    for index, delay in enumerate(delays):
        operator = transverse_relaxation_exchange_op(
            matrix, rates, torch.tensor(delay, dtype=torch.float64), offsets
        )
        expected = complex((operator @ start).sum()) * complex(scale)
        assert abs(complex(measured[index]) - expected) < 2e-5


def test_a_pulse_turns_both_pools_alike():
    """A chemical shift moves where a pool precesses, not what a pulse does.

    Two pools with identical relaxation and no exchange, given different
    offsets, must each be the single-pool answer turned through its own phase
    -- so their sum is what interference predicts and nothing about the
    rotation has singled either out.
    """
    delay = 3e-3
    turned = _free_induction(
        (delay,),
        pool_b_fraction=0.5,
        pool_b_exchange_hz=0.0,
        t1_pool_b_ms=1000.0,
        t2_pool_b_ms=80.0,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    )[0]
    plain = _free_induction((delay,))[0]

    phase = torch.exp(
        torch.tensor(-2j * torch.pi * FAT_SHIFT_HZ * delay, dtype=torch.complex128)
    )
    expected = complex(plain) * 0.5 * (1.0 + complex(phase))
    assert abs(complex(turned) - expected) / abs(expected) < 1e-5


# --- forward mode ---


# One tissue whose every exchanging term is live at once: enough exchange to
# move magnetization over the read, relaxation the two pools do not share, and
# an offset far enough out to put the pools well apart in phase.
LIVE = dict(
    t1_ms=1000.0,
    t2_ms=80.0,
    pool_b_fraction=0.25,
    pool_b_exchange_hz=40.0,
    t1_pool_b_ms=300.0,
    t2_pool_b_ms=25.0,
    pool_b_shift_hz=FAT_SHIFT_HZ,
    b0_hz=90.0,
)
READ_TIMES_S = (1.0e-3, 4.0e-3)

# The probe excites once and reads the transverse magnetization; nothing puts
# anything back on the longitudinal axis, so neither pool's T1 reaches the
# answer and differencing along one measures float32 noise.
DIRECTIONS = sorted(set(LIVE) - {"t1_ms", "t1_pool_b_ms"})


def _prepared(device="cpu", **properties):
    prepared, _, _ = _prepare_tissue(TissueProperties(**properties), device)
    return tuple(value.to(torch.float32).contiguous() for value in prepared)


def _live_events():
    from torchsim.sequence._accelerators import _EXCITATION, _RECORD

    times = READ_TIMES_S
    intervals = [
        after - before for before, after in zip((0.0, *times), times, strict=False)
    ]
    return (
        torch.tensor([0.0, *intervals], dtype=torch.float32),
        torch.tensor([1, *([2] * len(times))], dtype=torch.int32),
        torch.tensor([0.5 * torch.pi, *([0.0] * len(times))], dtype=torch.float32),
        torch.zeros(1 + len(times), dtype=torch.float32),
        torch.tensor([_EXCITATION, *([_RECORD] * len(times))], dtype=torch.uint8),
        torch.tensor([-1, *range(len(times))], dtype=torch.int32),
        torch.zeros(1 + len(times), dtype=torch.int32),
        torch.zeros(1 + len(times), dtype=torch.float32),
        torch.zeros(1 + len(times), dtype=torch.float32),
    )


def _live_readout(prepared, seed=None):
    """The free-induction reading, or its directional derivative."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    events = _live_events()
    if seed is None:
        return _run_packed(
            prepared, events, STATES, len(READ_TIMES_S), 1, exchanging=True
        ).flatten()
    return _run_packed_jvp(
        prepared,
        events,
        seed,
        tuple(torch.zeros_like(events[0]) for _ in range(3)),
        STATES,
        len(READ_TIMES_S),
        1,
        exchanging=True,
    ).flatten()


@pytest.mark.parametrize("name", DIRECTIONS)
def test_forward_mode_matches_finite_differences(name: str) -> None:
    """Every direction the exchanging pool adds, and every one it changes.

    The step is a hundredth of the value: differencing a float32 kernel at a
    thousandth leaves the answer under the rounding of the two runs, and at a
    tenth the difference has curved away from the tangent.
    """
    index = TISSUE_NAMES.index(name)
    prepared = _prepared(**LIVE)
    seed = tuple(
        torch.ones_like(value) if position == index else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    )
    tangent = _live_readout(prepared, seed)

    step = abs(LIVE[name]) * 1e-2
    forward = _live_readout(_prepared(**{**LIVE, name: LIVE[name] + step}))
    backward = _live_readout(_prepared(**{**LIVE, name: LIVE[name] - step}))
    difference = (forward - backward) / (2.0 * step)

    scale = float(difference.abs().max())
    assert scale > 0.0, "the probe leaves this direction dead"
    assert float((tangent - difference).abs().max()) / scale < 5e-3


def test_the_shift_is_the_off_resonance_of_the_pool_it_belongs_to():
    """The two are the same knob, pointed at different pools.

    Give the whole voxel to pool b and switch the exchange off, and it is the
    only pool there is -- so a direction along its chemical shift must equal a
    direction along the voxel's own off-resonance, exactly. A kernel that put
    the shift on the wrong diagonal entry, or on both, still passes a
    finite-difference check on the operator and fails here.
    """
    alone = dict(LIVE, pool_b_fraction=1.0, pool_b_exchange_hz=0.0)
    prepared = _prepared(**alone)

    def along(name):
        index = TISSUE_NAMES.index(name)
        seed = tuple(
            torch.ones_like(value) if position == index else torch.zeros_like(value)
            for position, value in enumerate(prepared)
        )
        return _live_readout(prepared, seed)

    shifted = along("pool_b_shift_hz")
    detuned = along("b0_hz")

    scale = float(detuned.abs().max())
    assert scale > 0.0
    assert float((shifted - detuned).abs().max()) / scale < 1e-5


def test_forward_mode_leaves_the_single_pool_answer_untouched():
    """A tissue at the default fraction takes the single-pool kernel in forward
    mode too, so its directions are bit for bit what they were.
    """
    t2 = torch.tensor([80.0])

    def signal(value):
        return (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(t1_ms=torch.tensor([1000.0]), t2_ms=value),
                nstates=STATES,
            )
            .signal
        )

    def gated(value):
        return (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(
                    t1_ms=torch.tensor([1000.0]),
                    t2_ms=value,
                    pool_b_fraction=0.0,
                    pool_b_shift_hz=FAT_SHIFT_HZ,
                ),
                nstates=STATES,
            )
            .signal
        )

    _, plain = torch.func.jvp(signal, (t2,), (torch.ones_like(t2),))
    _, still = torch.func.jvp(gated, (t2,), (torch.ones_like(t2),))

    assert torch.equal(plain, still)


def test_forward_mode_reaches_an_exchanging_pool_through_the_public_api():
    """The path an optimizer takes, rather than the packed buffers directly."""
    fraction = torch.tensor([FRACTION])

    def signal(value):
        return (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(
                    t1_ms=torch.tensor([1000.0]),
                    t2_ms=torch.tensor([80.0]),
                    pool_b_fraction=value,
                    pool_b_exchange_hz=RATE_HZ,
                    t2_pool_b_ms=torch.tensor([25.0]),
                    pool_b_shift_hz=FAT_SHIFT_HZ,
                ),
                nstates=STATES,
            )
            .signal
        )

    reading, tangent = torch.func.jvp(signal, (fraction,), (torch.ones_like(fraction),))
    step = 1e-3
    difference = (signal(fraction + step) - signal(fraction - step)) / (2.0 * step)

    assert float(reading.abs().max()) > 0.0
    scale = float(difference.abs().max())
    assert float((tangent - difference).abs().max()) / scale < 5e-3


# --- the adjoint ---


def _live_adjoint(prepared, seed):
    """The gradients a cotangent on the free-induction reading leaves."""
    from torchsim.sequence._accelerators import _run_packed_vjp

    return _run_packed_vjp(
        prepared,
        _live_events(),
        seed,
        state_count=STATES,
        output_count=len(READ_TIMES_S),
        threads=1,
        exchanging=True,
    )


def _cotangent(seed=7):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, len(READ_TIMES_S))
    return torch.complex(
        torch.rand(shape, generator=generator) * 2.0 - 1.0,
        torch.rand(shape, generator=generator) * 2.0 - 1.0,
    )


@pytest.mark.parametrize("name", DIRECTIONS)
def test_the_adjoint_matches_finite_differences(name: str) -> None:
    """Every direction the exchanging pool adds, and every one it changes."""
    index = TISSUE_NAMES.index(name)
    seed = _cotangent()

    def reading(**properties):
        return float(
            (
                seed.conj() * _live_readout(_prepared(**properties)).reshape(seed.shape)
            ).real.sum()
        )

    gradient = float(_live_adjoint(_prepared(**LIVE), seed)[index].sum())
    step = abs(LIVE[name]) * 1e-2
    difference = (
        reading(**{**LIVE, name: LIVE[name] + step})
        - reading(**{**LIVE, name: LIVE[name] - step})
    ) / (2.0 * step)

    assert abs(difference) > 0.0, "the probe leaves this direction dead"
    assert abs(gradient - difference) / abs(difference) < 5e-3


def test_the_adjoint_transposes_the_forward_direction():
    """``<w, J v> == <J^T w, v>``, at a tolerance that separates float32 from
    a dropped term.

    Taken against the sum of the terms' magnitudes rather than their total, so
    one small gradient going wrong is not hidden by two large ones that did
    not.
    """
    prepared = _prepared(**LIVE)
    generator = torch.Generator().manual_seed(19)
    directions = tuple(
        torch.rand(value.shape, generator=generator) * 2.0 - 1.0 for value in prepared
    )
    seed = _cotangent(23)

    forward = _live_readout(prepared, directions).reshape(seed.shape)
    left = float((seed.conj() * forward).real.sum())
    terms = [
        float((gradient * direction).sum())
        for gradient, direction in zip(
            _live_adjoint(prepared, seed)[: len(prepared)], directions, strict=False
        )
    ]
    scale = sum(abs(term) for term in terms)

    assert scale > 0.0
    assert abs(left - sum(terms)) / scale < 1e-6


def test_a_refocused_train_carries_the_gradient_through_the_pulses():
    """The probe above never refocuses, so it leaves the rotation, the shifts
    and the spoil untested on the exchanging pool. A train exercises all three.
    """
    live = dict(
        pool_b_fraction=0.25,
        pool_b_exchange_hz=40.0,
        t1_pool_b_ms=300.0,
        t2_pool_b_ms=25.0,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    )

    def loss(**over):
        return float(_signal(**{**live, **over}).abs().square().sum())

    held = dict(live, t1_ms=1000.0, t2_ms=80.0)
    leaves = {
        name: torch.tensor([value], requires_grad=True) for name, value in held.items()
    }
    EpgEngine().simulate(
        _description(), TissueProperties(**leaves), nstates=STATES
    ).signal.abs().square().sum().backward()

    for name in ("t2_ms", "pool_b_fraction", "pool_b_exchange_hz", "t2_pool_b_ms"):
        step = abs(held[name]) * 1e-3
        difference = (
            loss(**{name: held[name] + step}) - loss(**{name: held[name] - step})
        ) / (2.0 * step)
        assert abs(difference) > 0.0, name
        gradient = float(leaves[name].grad)
        assert abs(gradient - difference) / abs(difference) < 5e-3, name


def test_the_second_order_pass_differentiates_the_adjoint():
    """A curvature the optimizers ask for: the gradient in one property,
    differentiated along another.
    """
    live = dict(
        t1_ms=1000.0,
        t2_ms=80.0,
        pool_b_exchange_hz=40.0,
        t1_pool_b_ms=300.0,
        t2_pool_b_ms=25.0,
        pool_b_shift_hz=FAT_SHIFT_HZ,
    )

    def gradient(fraction, *, graph):
        t2 = torch.tensor([live["t2_ms"]], requires_grad=True)
        leaves = {
            name: torch.tensor([value])
            for name, value in live.items()
            if name != "t2_ms"
        }
        signal = (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(t2_ms=t2, pool_b_fraction=fraction, **leaves),
                nstates=STATES,
            )
            .signal
        )
        return torch.autograd.grad(signal.abs().square().sum(), t2, create_graph=graph)[
            0
        ]

    fraction = torch.tensor([0.25], requires_grad=True)
    curvature = float(torch.autograd.grad(gradient(fraction, graph=True), fraction)[0])
    step = 1e-3
    difference = float(
        gradient(torch.tensor([0.25 + step]), graph=False)
        - gradient(torch.tensor([0.25 - step]), graph=False)
    ) / (2.0 * step)

    assert abs(difference) > 0.0
    assert abs(curvature - difference) / abs(difference) < 5e-3


def test_the_adjoint_leaves_the_single_pool_answer_untouched():
    """A tissue at the default fraction takes the single-pool kernel in reverse
    mode too, so its gradients are bit for bit what they were.
    """

    def gradient(**extra):
        t2 = torch.tensor([80.0], requires_grad=True)
        signal = (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(t1_ms=torch.tensor([1000.0]), t2_ms=t2, **extra),
                nstates=STATES,
            )
            .signal
        )
        signal.abs().square().sum().backward()
        return t2.grad

    plain = gradient()
    gated = gradient(pool_b_fraction=0.0, pool_b_shift_hz=FAT_SHIFT_HZ)

    assert torch.equal(plain, gated)


def test_the_single_pool_gradient_still_reaches_every_property():
    """A tissue that never mentions the exchanging pool runs the single-pool
    machine, and no gradient is disturbed by the pool's properties being in the
    list.
    """
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (("t1_ms", 1000.0), ("t2_ms", 80.0))
    }
    signal = (
        EpgEngine()
        .simulate(
            _description(),
            TissueProperties(**leaves, pool_b_fraction=0.0, pool_b_exchange_hz=0.0),
            nstates=STATES,
        )
        .signal
    )
    signal.abs().square().sum().backward()

    assert float(leaves["t1_ms"].grad.abs().max()) > 0.0
    assert float(leaves["t2_ms"].grad.abs().max()) > 0.0


def test_an_empty_pool_asked_for_its_gradient_gives_the_true_one():
    """A fraction of zero is where a two-pool fit starts, and the signal moves
    as it leaves: dropping the pool because its fraction sits at the identity
    would answer that fit with a zero that is not the derivative. So a value
    carrying a gradient keeps its term, checked here against a difference the
    state machine did not produce.

    The rate, the relaxation times and the shift describe the pool rather than
    gate it, and an empty pool genuinely does not depend on any of them.
    """

    def loss(fraction):
        signal = (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(
                    t1_ms=1000.0,
                    t2_ms=80.0,
                    pool_b_fraction=fraction,
                    pool_b_exchange_hz=20.0,
                    t1_pool_b_ms=600.0,
                    t2_pool_b_ms=40.0,
                    pool_b_shift_hz=90.0,
                ),
                nstates=STATES,
            )
            .signal
        )
        return float(signal.abs().square().sum())

    described = {
        "pool_b_exchange_hz": 20.0,
        "t1_pool_b_ms": 600.0,
        "t2_pool_b_ms": 40.0,
        "pool_b_shift_hz": 90.0,
    }
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (("pool_b_fraction", 0.0), *described.items())
    }
    signal = (
        EpgEngine()
        .simulate(
            _description(),
            TissueProperties(t1_ms=1000.0, t2_ms=80.0, **leaves),
            nstates=STATES,
        )
        .signal
    )
    signal.abs().square().sum().backward()

    # One-sided: a fraction is not defined below zero. The step is kept well
    # clear of where float32 cancellation swamps the difference -- below about
    # 1e-4 the quotient walks away from the derivative rather than toward it.
    step = 1e-3
    expected = (loss(step) - loss(0.0)) / step
    assert abs(expected) > 0.1
    measured = float(leaves["pool_b_fraction"].grad)
    assert abs(measured - expected) / abs(expected) < 2e-2, (measured, expected)

    for name in described:
        assert float(leaves[name].grad.abs().max()) < 1e-6 * abs(measured)


# --- against the state machine written out in torch ---


def _train_events():
    """A refocused train, which exercises the rotation, the shifts and the
    spoil that a free-induction probe leaves alone.
    """
    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    return (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
        packed.shim_index,
        packed.saturation,
        packed.rf_frequency_hz,
    )


def _instantaneous_table(device="cpu"):
    """A pulse with no gradient across it: one rotation, every position."""
    import numpy as np

    from torchsim.sequence._description import RfDefinition, RfShape
    from torchsim.sequence._transition import transition_table

    flat = RfDefinition(
        id=0,
        bandwidth_hz=0.0,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=0.0,
        total_b1sq_power=1.0,
        magnitude=RfShape(num_uncompressed=8, samples=np.ones(8, dtype=np.float32)),
    )
    return transition_table(
        flat, torch.zeros(1), bins=1024, rf_raster_time_s=1e-6, device=device
    )


def _routes(leaves, events, profile=None):
    """The kernels and the oracle, fed from one place."""
    from torchsim.sequence._accelerators import _NativeEpg

    fused = _NativeEpg.apply(
        *leaves,
        *events,
        STATES,
        ECHOES,
        1,
        NO_GEOMETRY,
        profile,
        None,
        None,
        None,
        True,
        None,
    )
    reference = simulate_packed(
        leaves,
        events,
        state_count=STATES,
        output_count=ECHOES,
        profile=profile,
        exchanging=True,
    )
    return fused, reference


def _oracle_leaves():
    return tuple(
        value.detach().clone().requires_grad_(True) for value in _prepared(**LIVE)
    )


def _agree(want, got, tolerance: float = 1e-3) -> None:
    """Compare gradient tuples, skipping what float32 cannot resolve.

    These span many orders of magnitude, and an entry far below the largest is
    under the rounding of the sums that produced it.
    """
    floor = 1e-6 * max(float(value.abs().max()) for value in want)
    compared = 0
    for index, (expected, measured) in enumerate(zip(want, got, strict=True)):
        scale = float(expected.abs().max())
        if scale <= floor:
            continue
        assert float((expected - measured).abs().max()) / scale < tolerance, index
        compared += 1
    assert compared > 0


def _inverted_events():
    """An inversion, a delay, then a hard 90 and a read.

    This is where the exchanging pool parts company with the semisolid one: a
    sweep saturates a bound pool and turns free water over, so the two must
    start at ``-(1 - f), -f`` rather than at ``-(1 - f), f``.
    """
    from torchsim.sequence._accelerators import (
        _EXCITATION,
        _INVERSION,
        _RECORD,
    )

    return (
        torch.tensor([0.0, 0.12, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1, 0, 1, 2], dtype=torch.int32),
        torch.tensor([0.0, 0.0, 0.5 * torch.pi, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.5 * torch.pi, 0.0], dtype=torch.float32),
        torch.tensor([_INVERSION, 0, _EXCITATION, _RECORD], dtype=torch.uint8),
        torch.tensor([-1, -1, -1, 0], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(4, dtype=torch.float32),
    )


def test_an_inversion_turns_both_pools_over():
    """Read against the oracle rather than against a sign: the exchange over
    the recovery is what tells a pool that was inverted from one that was not.
    """
    leaves = _oracle_leaves()
    events = _inverted_events()
    from torchsim.sequence._accelerators import _NativeEpg

    fused = _NativeEpg.apply(
        *leaves, *events, STATES, 1, 1, NO_GEOMETRY, None, None, None, None, True, None
    )
    reference = simulate_packed(
        leaves, events, state_count=STATES, output_count=1, exchanging=True
    )

    reference = reference.detach()
    assert float(reference.abs().max()) > 0.0
    assert (
        float((fused.detach() - reference).abs().max() / reference.abs().max()) < 1e-4
    )


def test_the_inversion_efficiency_carries_a_gradient_from_both_pools():
    """Both pools are turned over by the same number, so its gradient is the
    sum of what each leaves; a kernel that inverted one would still produce a
    plausible one.
    """
    leaves = _oracle_leaves()
    events = _inverted_events()
    from torchsim.sequence._accelerators import _NativeEpg

    seed = torch.full((1, 1), 1.0 + 1.0j, dtype=torch.complex64)
    fused = torch.autograd.grad(
        _NativeEpg.apply(
            *leaves,
            *events,
            STATES,
            1,
            1,
            NO_GEOMETRY,
            None,
            None,
            None,
            None,
            True,
            None,
        ),
        leaves,
        seed,
        allow_unused=True,
        materialize_grads=True,
    )
    mirrored = _oracle_leaves()
    reference = torch.autograd.grad(
        simulate_packed(
            mirrored,
            events,
            state_count=STATES,
            output_count=1,
            exchanging=True,
        ),
        mirrored,
        seed,
        allow_unused=True,
        materialize_grads=True,
    )

    index = TISSUE_NAMES.index("inversion_efficiency")
    assert float(reference[index].abs().max()) > 0.0
    _agree(reference, fused)


@pytest.mark.parametrize("tabulated", [False, True])
def test_the_fused_forward_matches_the_oracle(tabulated: bool):
    """The oracle reaches its exponential through Pade rather than through the
    closed form, so this is the transverse step checked against a different
    algorithm and not against a second copy of itself.

    Run with and without a tabulated rotation, because the two reach the
    pulse through different code even though both pools take the same one.
    """
    profile = _instantaneous_table() if tabulated else None
    fused, reference = _routes(_oracle_leaves(), _train_events(), profile)

    reference = reference.detach()
    assert float(reference.abs().max()) > 0.0
    assert (
        float((fused.detach() - reference).abs().max() / reference.abs().max()) < 1e-4
    )


@pytest.mark.parametrize("tabulated", [False, True])
def test_the_fused_forward_mode_matches_the_oracle(tabulated: bool):
    """A direction followed through both routes at once."""
    profile = _instantaneous_table() if tabulated else None
    events = _train_events()
    prepared = _prepared(**LIVE)
    generator = torch.Generator().manual_seed(37)
    direction = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for value in prepared
    )

    def followed(route: int):
        _, tangent = torch.func.jvp(
            lambda *values: _routes(values, events, profile)[route],
            prepared,
            direction,
        )
        return tangent

    fused, reference = followed(0), followed(1)

    assert float(reference.abs().max()) > 0.0
    assert float((fused - reference).abs().max() / reference.abs().max()) < 1e-3


@pytest.mark.parametrize("tabulated", [False, True])
def test_the_fused_derivatives_match_the_oracle(tabulated: bool):
    """Every pass at once: autograd differentiates the oracle to whatever order
    is asked of it, so the analytic kernels are checked against it rather than
    against a difference.

    Per parameter, not against one contracted scalar: a transposed adjoint
    still transposes when a small gradient is wrong.
    """
    profile = _instantaneous_table() if tabulated else None
    events = _train_events()
    generator = torch.Generator().manual_seed(31)
    seed = torch.randn(
        (1, ECHOES), generator=generator, dtype=torch.float32
    ) + 1j * torch.randn((1, ECHOES), generator=generator, dtype=torch.float32)
    # The direction the second pass follows is drawn once, so the two routes
    # are contracted against the same one rather than each against its own.
    direction = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for value in _oracle_leaves()
    )

    def gradients(route: int, order: int):
        leaves = _oracle_leaves()
        signal = _routes(leaves, events, profile)[route]
        first = torch.autograd.grad(
            signal,
            leaves,
            seed,
            create_graph=order > 1,
            allow_unused=True,
            materialize_grads=True,
        )
        if order == 1:
            return first
        return torch.autograd.grad(
            first, leaves, direction, allow_unused=True, materialize_grads=True
        )

    for order in (1, 2):
        _agree(gradients(1, order), gradients(0, order))


# --- the other backend ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_forward_mode_matches_the_cpu_kernel():
    """The tangent of the complex two-pool step on the card.

    The two backends share no code, so agreement is what keeps the second
    transverse pool's derivative honest there: a shift dropped from the
    tangent alone still produces a plausible direction.
    """
    from torchsim.sequence._accelerators import _run_packed_jvp

    voxels = 6
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    events = _live_events()

    def run(device):
        prepared = _prepared(device, **spread)
        return _run_packed_jvp(
            prepared,
            tuple(value.to(device) for value in events),
            tuple(torch.ones_like(value) for value in prepared),
            tuple(torch.zeros_like(events[0]).to(device) for _ in range(3)),
            STATES,
            len(READ_TIMES_S),
            1,
            exchanging=True,
        )

    host = run("cpu")
    card = run("cuda").cpu()

    assert float(host.abs().max()) > 0.0
    assert float((host - card).abs().max() / host.abs().max()) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_forward_mode_matches_the_whole_one():
    """Streaming cuts the voxel axis, which the second pool's seeds follow."""
    from torchsim.sequence._accelerators import _run_packed_jvp, offload

    voxels = 3000
    prepared = _prepared(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    events = _live_events()
    arguments = (
        prepared,
        events,
        tuple(torch.ones_like(value) for value in prepared),
        tuple(torch.zeros_like(events[0]) for _ in range(3)),
        STATES,
        len(READ_TIMES_S),
        1,
    )

    whole = _run_packed_jvp(*arguments, exchanging=True)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed_jvp(*arguments, exchanging=True)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 5e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_kernel_matches_the_cpu_kernel():
    """The two share no code, so agreement is what keeps the second transverse
    pool honest on the card: a shift dropped there would still produce a
    plausible train.
    """
    voxels = 6
    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )

    def run(device):
        prepared, _, _ = _prepare_tissue(tissue, device)
        prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
        events = tuple(value.to(device) for value in packed.buffers)
        return _run_packed(
            prepared, events, STATES, packed.output_count, 1, exchanging=True
        )

    host = run("cpu")
    card = run("cuda").cpu()

    assert float(host.abs().max()) > 0.0
    assert float((host - card).abs().max() / host.abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_volume_matches_the_whole_one():
    """Streaming cuts the voxel axis, which the second pool's buffers follow."""
    from torchsim.sequence._accelerators import offload

    voxels = 3000
    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        pool_b_fraction=torch.linspace(0.0, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    arguments = (prepared, packed.buffers, STATES, packed.output_count, 1)

    whole = _run_packed(*arguments, exchanging=True)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed(*arguments, exchanging=True)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 5e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_adjoint_matches_the_cpu_kernel():
    """The two backends share no code, so agreement is what keeps the second
    transverse pool's cotangent honest there.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp

    voxels = 6
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    generator = torch.Generator().manual_seed(11)
    shape = (voxels, len(READ_TIMES_S))
    seed = torch.complex(
        torch.rand(shape, generator=generator) * 2.0 - 1.0,
        torch.rand(shape, generator=generator) * 2.0 - 1.0,
    )
    events = _live_events()

    def run(device):
        prepared = _prepared(device, **spread)
        return _run_packed_vjp(
            prepared,
            tuple(value.to(device) for value in events),
            seed.to(device),
            state_count=STATES,
            output_count=len(READ_TIMES_S),
            threads=1,
            exchanging=True,
        )

    host = run("cpu")
    card = run("cuda")
    for expected, measured in zip(host, card, strict=True):
        scale = float(expected.abs().max())
        if scale == 0.0:
            assert float(measured.abs().max()) == 0.0
            continue
        assert float((expected - measured.cpu()).abs().max()) / scale < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_adjoint_matches_the_whole_one():
    """Streaming cuts the voxel axis, which the second pool's planes follow.

    The forward-over-reverse pass is the one that streams, and it is also what
    a first-order adjoint on the card runs with no direction to follow.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp, offload

    voxels = 3000
    prepared = _prepared(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    events = _live_events()
    generator = torch.Generator().manual_seed(5)
    shape = (voxels, len(READ_TIMES_S))
    seed = torch.complex(
        torch.rand(shape, generator=generator) * 2.0 - 1.0,
        torch.rand(shape, generator=generator) * 2.0 - 1.0,
    )
    directions = tuple(
        torch.rand(value.shape, generator=generator) * 2.0 - 1.0 for value in prepared
    ) + tuple(torch.zeros_like(events[0]) for _ in range(3))
    arguments = (prepared, events, directions, seed)
    options = dict(
        state_count=STATES,
        output_count=len(READ_TIMES_S),
        threads=1,
        exchanging=True,
    )

    whole = _run_packed_vjp_jvp(*arguments, **options)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed_vjp_jvp(*arguments, **options)

    for side, other in zip(whole, streamed, strict=True):
        for expected, measured in zip(side, other, strict=True):
            scale = float(expected.abs().max())
            if scale == 0.0:
                assert float(measured.abs().max()) == 0.0
                continue
            assert float((expected - measured).abs().max()) / scale < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("state_count", [8, 12, 17])
def test_an_exchanging_pool_takes_the_first_order_kernel_on_the_card(
    state_count: int,
) -> None:
    """An exchanging pool does not cost the kernel written for a gradient.

    Held against the host's own first-order adjoint rather than against the
    forward-over-reverse pass on the same card: two arms of one wrong kernel
    agree with each other, and the backends share no code.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _run_packed_vjp

    voxels = 64
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.30, voxels),
        pool_b_exchange_hz=torch.linspace(5.0, 60.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 60.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    outputs = int(packed.output_count)

    def side(device):
        prepared, _, _ = _prepare_tissue(tissue, device)
        prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
        return prepared, tuple(value.to(device) for value in packed.buffers)

    host_tissue, host_events = side("cpu")
    card_tissue, card_events = side("cuda")
    seed = torch.randn(
        (voxels, outputs),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(4),
    )
    arguments = dict(
        state_count=state_count,
        output_count=outputs,
        threads=1,
        exchanging=True,
    )

    reached = []
    original = _accelerators._run_packed_vjp_jvp

    def record(*args, **kwargs):
        reached.append(True)
        return original(*args, **kwargs)

    _accelerators._run_packed_vjp_jvp = record
    try:
        card = _run_packed_vjp(card_tissue, card_events, seed.cuda(), **arguments)
    finally:
        _accelerators._run_packed_vjp_jvp = original
    host = _run_packed_vjp(host_tissue, host_events, seed, **arguments)

    assert not reached
    largest = max(float(value.abs().max()) for value in host)
    assert largest > 0.0
    for name, reference, result in zip(
        (*TISSUE_NAMES, "duration", "flip", "phase"), host, card, strict=True
    ):
        assert reference.shape == result.shape, name
        difference = float((reference.cpu() - result.cpu()).abs().max())
        assert difference / largest < 1e-4, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_tabulated_rotation_beside_a_pool_takes_the_first_order_kernel() -> None:
    """The pool's own share of a shaped pulse's cotangent.

    A shaped rotation reaches both pools through one spinor pair, so the pool
    contributes to the flip and the phase through the table's slopes rather
    than through the instant rotation's derivative. Held against the host.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _run_packed_vjp

    voxels = 64
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.30, voxels),
        pool_b_exchange_hz=torch.linspace(5.0, 60.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 60.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    events = _train_events()
    seed = torch.randn(
        (voxels, ECHOES),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(6),
    )

    def side(device):
        prepared, _, _ = _prepare_tissue(tissue, device)
        return (
            tuple(value.to(torch.float32).contiguous() for value in prepared),
            tuple(value.to(device) for value in events),
        )

    host_tissue, host_events = side("cpu")
    card_tissue, card_events = side("cuda")
    arguments = dict(
        state_count=STATES, output_count=ECHOES, threads=1, exchanging=True
    )

    reached = []
    original = _accelerators._run_packed_vjp_jvp

    def record(*args, **kwargs):
        reached.append(True)
        return original(*args, **kwargs)

    _accelerators._run_packed_vjp_jvp = record
    try:
        card = _run_packed_vjp(
            card_tissue,
            card_events,
            seed.cuda(),
            profile=_instantaneous_table("cuda"),
            **arguments,
        )
    finally:
        _accelerators._run_packed_vjp_jvp = original
    host = _run_packed_vjp(
        host_tissue, host_events, seed, profile=_instantaneous_table(), **arguments
    )

    assert not reached
    largest = max(float(value.abs().max()) for value in host)
    assert largest > 0.0
    for name, reference, result in zip(
        (*TISSUE_NAMES, "duration", "flip", "phase"), host, card, strict=True
    ):
        difference = float((reference.cpu() - result.cpu()).abs().max())
        assert difference / largest < 1e-4, name
