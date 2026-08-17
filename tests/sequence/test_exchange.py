"""A chemically exchanging pool, through the fused kernels.

Two pools that both carry transverse magnetization -- fat beside water, myelin
water beside free water, a metabolite beside its solvent. What separates this
from the semisolid pool of :mod:`test_bound_pool` is that this one precesses:
it has a T2, it sits at its own chemical shift, and a pulse rotates it rather
than saturating it. So ``F+`` and ``F-`` double, and with them the shift, the
spoil, the RF operator and what the ADC reads.

The two second pools are alternatives rather than layers; a tissue declaring
both is a three-pool system, which is refused.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence._accelerators import (
    NO_GEOMETRY,
    _pack_events,
    _run_packed,
)
from torchsim.sequence._parameters import TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue

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
    return FSE().simulate(
        _description(),
        TissueProperties(t1_ms=1000.0, t2_ms=80.0, **properties),
        nstates=STATES,
    ).signal


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


def test_two_second_pools_at_once_are_refused():
    """Both together is a three-pool system, whose exchange matrix has no
    closed form of the shape the kernels evaluate. Carrying one of the two
    would answer a question nobody asked.
    """
    with pytest.raises(NotImplementedError, match="three pools"):
        _signal(bound_fraction=0.1, pool_b_fraction=0.1)


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
        after - before for before, after in zip((0.0, *times_s), times_s)
    ]
    events = (
        torch.tensor([0.0, *intervals], dtype=torch.float32),
        torch.tensor([1, *([2] * len(times_s))], dtype=torch.int32),
        torch.tensor(
            [0.5 * torch.pi, *([0.0] * len(times_s))], dtype=torch.float32
        ),
        torch.zeros(1 + len(times_s), dtype=torch.float32),
        torch.tensor(
            [_EXCITATION, *([_RECORD] * len(times_s))], dtype=torch.uint8
        ),
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
    from torchsim.base.config.relax_model import build_two_pool_exchange_matrix
    from torchsim.epg import transverse_relaxation_exchange_op

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
    rates = torch.tensor(
        [1000.0 / t2_ms, 1000.0 / t2_b_ms], dtype=torch.float64
    )
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
        after - before for before, after in zip((0.0, *times), times)
    ]
    return (
        torch.tensor([0.0, *intervals], dtype=torch.float32),
        torch.tensor([1, *([2] * len(times))], dtype=torch.int32),
        torch.tensor(
            [0.5 * torch.pi, *([0.0] * len(times))], dtype=torch.float32
        ),
        torch.zeros(1 + len(times), dtype=torch.float32),
        torch.tensor(
            [_EXCITATION, *([_RECORD] * len(times))], dtype=torch.uint8
        ),
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
            torch.ones_like(value) if position == index
            else torch.zeros_like(value)
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
        return FSE().simulate(
            _description(),
            TissueProperties(t1_ms=torch.tensor([1000.0]), t2_ms=value),
            nstates=STATES,
        ).signal

    def gated(value):
        return FSE().simulate(
            _description(),
            TissueProperties(
                t1_ms=torch.tensor([1000.0]),
                t2_ms=value,
                pool_b_fraction=0.0,
                pool_b_shift_hz=FAT_SHIFT_HZ,
            ),
            nstates=STATES,
        ).signal

    _, plain = torch.func.jvp(signal, (t2,), (torch.ones_like(t2),))
    _, still = torch.func.jvp(gated, (t2,), (torch.ones_like(t2),))

    assert torch.equal(plain, still)


def test_forward_mode_reaches_an_exchanging_pool_through_the_public_api():
    """The path an optimizer takes, rather than the packed buffers directly."""
    fraction = torch.tensor([FRACTION])

    def signal(value):
        return FSE().simulate(
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
        ).signal

    reading, tangent = torch.func.jvp(
        signal, (fraction,), (torch.ones_like(fraction),)
    )
    step = 1e-3
    difference = (signal(fraction + step) - signal(fraction - step)) / (2.0 * step)

    assert float(reading.abs().max()) > 0.0
    scale = float(difference.abs().max())
    assert float((tangent - difference).abs().max()) / scale < 5e-3


# --- what the derivative kernels will not do yet ---


def test_an_adjoint_of_an_exchanging_run_is_refused():
    """The forward carries the second transverse pool; the adjoint does not.

    Answering with the single-pool one would be a wrong number rather than a
    missing one, so it is refused instead.
    """
    t2 = torch.tensor([80.0], requires_grad=True)
    signal = FSE().simulate(
        _description(),
        TissueProperties(
            t1_ms=torch.tensor([1000.0]),
            t2_ms=t2,
            pool_b_fraction=FRACTION,
            pool_b_shift_hz=FAT_SHIFT_HZ,
        ),
        nstates=STATES,
    ).signal

    with pytest.raises(NotImplementedError, match="one transverse pool"):
        signal.abs().square().sum().backward()


def test_the_single_pool_gradient_still_reaches_every_property():
    """The exchanging pool's properties are arguments the single-pool machine
    does not take, so their gradients come back at zero -- and no other
    gradient is disturbed by their being in the list.
    """
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (
            ("t1_ms", 1000.0), ("t2_ms", 80.0), ("pool_b_fraction", 0.0),
            ("pool_b_exchange_hz", 0.0), ("t1_pool_b_ms", 1000.0),
            ("t2_pool_b_ms", 100.0), ("pool_b_shift_hz", 0.0),
        )
    }
    signal = FSE().simulate(
        _description(), TissueProperties(**leaves), nstates=STATES
    ).signal
    signal.abs().square().sum().backward()

    assert float(leaves["t1_ms"].grad.abs().max()) > 0.0
    assert float(leaves["t2_ms"].grad.abs().max()) > 0.0
    for name in TISSUE_NAMES:
        if name.startswith("pool_b") or name.endswith("pool_b_ms"):
            assert float(leaves[name].grad.abs().max()) == 0.0


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
        "fse",
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
        "fse",
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
