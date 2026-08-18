"""Free water beside both second pools at once, through the fused kernels.

Water with fat and a macromolecular pool, or myelin water beside free water and
a macromolecular pool. Each second pool exchanges with the free water and not
with the other, so the semisolid pool of :mod:`test_bound_pool` and the
chemically exchanging pool of :mod:`test_exchange` are limits of this system
rather than neighbours of it -- which is what lets both of their answers stay
bit for bit what they were.

Only the longitudinal axis sees all three pools. The semisolid pool has no
transverse magnetization for a gradient to dephase, so the transverse step is
the same 2x2 it already was and the 3x3 is confined to ``Z``.
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
from torchsim.sequence._lineshape import lineshape_table
from torchsim.sequence._simulation import _prepare_tissue

ECHOES = 8
STATES = 8
FAT_SHIFT_HZ = -420.0

# One tissue with every three-pool term live at once.
SEMISOLID = dict(
    bound_fraction=0.1, bound_exchange_hz=30.0, t1_bound_ms=1000.0
)
EXCHANGING = dict(
    pool_b_fraction=0.2, pool_b_exchange_hz=40.0, t1_pool_b_ms=300.0,
    t2_pool_b_ms=25.0, pool_b_shift_hz=FAT_SHIFT_HZ,
)
LIVE = dict(t1_ms=1000.0, t2_ms=80.0, **SEMISOLID, **EXCHANGING)


def _description():
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )


def _signal(device="cpu", **properties):
    held = {"t1_ms": 1000.0, "t2_ms": 80.0, **properties}
    return FSE().simulate(
        _description(),
        TissueProperties(
            **{name: torch.tensor([value], device=device) for name, value in held.items()}
        ),
        nstates=STATES,
    ).signal


# --- each two-pool system is a limit of this one ---


def test_an_empty_exchanging_pool_leaves_the_semisolid_answer():
    """The safety net for the whole stage, one half of it.

    A fraction of zero empties the exchanging pool and decouples it, so the
    tissue is the semisolid two-pool one and the kernels answer it exactly --
    not merely closely.
    """
    alone = _signal(**SEMISOLID)
    beside = _signal(**SEMISOLID, **{**EXCHANGING, "pool_b_fraction": 0.0})

    assert torch.equal(alone, beside)


def test_an_empty_semisolid_pool_leaves_the_exchanging_answer():
    """The other half. Neither two-pool answer moves when the pool that is not
    there is described.
    """
    alone = _signal(**EXCHANGING)
    beside = _signal(**EXCHANGING, **{**SEMISOLID, "bound_fraction": 0.0})

    assert torch.equal(alone, beside)


def test_fractions_past_the_whole_voxel_are_refused():
    """The free water is what the two second pools leave, so a tissue that
    hands out more than the voxel starts it at a negative magnetization -- a
    number every pass afterwards would carry as though it meant something.
    """
    with pytest.raises(ValueError, match="cannot sum past one"):
        _signal(bound_fraction=0.7, pool_b_fraction=0.5)


# --- what the third pool does ---


def test_both_second_pools_reach_the_answer():
    """Adding either pool to the other moves the train.

    Read against both two-pool runs rather than against one: a kernel that
    dropped the third pool would still reproduce whichever pair it kept.
    """
    semisolid = _signal(**SEMISOLID)
    exchanging = _signal(**EXCHANGING)
    both = _signal(**SEMISOLID, **EXCHANGING)

    scale = float(exchanging.abs().max())
    assert float((both - semisolid).abs().max()) / scale > 0.05
    assert float((both - exchanging).abs().max()) / scale > 0.01


def test_the_semisolid_pool_still_saturates_beside_the_exchanging_one():
    """The pulse deposits power in the semisolid pool whichever other pools are
    present, so raising its fraction has to cost signal here as it does alone.
    """

    def held(fraction):
        return float(
            _signal(
                **{**SEMISOLID, "bound_fraction": fraction}, **EXCHANGING
            ).abs().sum()
        )

    assert held(0.25) < held(0.1) < held(0.0)


def test_the_chemical_shift_still_reaches_the_coil():
    """And the exchanging pool keeps its offset beside the semisolid one."""
    on_resonance = _signal(
        **SEMISOLID, **{**EXCHANGING, "pool_b_shift_hz": 0.0}
    )
    shifted = _signal(**SEMISOLID, **EXCHANGING)

    scale = float(on_resonance.abs().max())
    assert float((shifted - on_resonance).abs().max()) / scale > 0.01


# --- against the package's own N-pool operator ---


def _inversion_recovery(delays_s, **properties):
    """Signal after an inversion and a delay, one entry per delay.

    An inversion turns the free water and the exchanging pool over and leaves
    the semisolid pool alone, so all three start far from equilibrium and the
    interval that follows exercises the exchange rather than sitting at the
    fixed point. The readout is a hard ninety degrees, which writes ``Z`` into
    the transverse plane.
    """
    from torchsim.sequence._accelerators import _EXCITATION, _INVERSION, _RECORD

    prepared, _, _ = _prepare_tissue(
        TissueProperties(**{"t1_ms": 1000.0, "t2_ms": 80.0, **properties}), "cpu"
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    signals = []
    for delay in delays_s:
        events = (
            torch.tensor([0.0, float(delay), 0.0, 0.0], dtype=torch.float32),
            torch.tensor([1, 0, 1, 2], dtype=torch.int32),
            torch.tensor([0.0, 0.0, 0.5 * torch.pi, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 0.0, 0.5 * torch.pi, 0.0], dtype=torch.float32),
            torch.tensor(
                [_INVERSION, 0, _EXCITATION, _RECORD], dtype=torch.uint8
            ),
            torch.tensor([-1, -1, -1, 0], dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        )
        signals.append(
            _run_packed(
                prepared,
                events,
                STATES,
                1,
                1,
                geometry=NO_GEOMETRY,
                lineshape=lineshape_table(),
                exchanging=True,
            ).flatten()[0]
        )
    return torch.stack(signals)


def test_the_longitudinal_step_reproduces_the_package_operator():
    """The kernel's closed form against ``epg``'s matrix exponential.

    ``longitudinal_relaxation_exchange_op`` is generic in the number of pools
    and says nothing about which exchanges with which, so handing it this
    file's exchange matrix settles that the rates are read the way the kernels
    read them -- and reaches the exponential through a different algorithm.

    The readout scale is not what is being checked, so it is measured on a
    single-pool run -- whose longitudinal state the inversion leaves at minus
    one -- and applied to the three-pool prediction. A ninety degree pulse
    writes both transverse pools' ``Z`` into the plane and the coil sums them,
    so what the reading follows is ``Z_a + Z_b`` and not the free water alone.
    """
    from torchsim.epg import longitudinal_relaxation_exchange_op

    delays = (0.05, 0.3, 1.0)
    tissue = dict(LIVE)
    measured = _inversion_recovery(delays, **tissue)
    readout = -complex(_inversion_recovery((0.0,))[0])

    free = 1.0 - tissue["pool_b_fraction"] - tissue["bound_fraction"]
    weight = torch.tensor(
        [free, tissue["pool_b_fraction"], tissue["bound_fraction"]],
        dtype=torch.float64,
    )
    rates = torch.tensor(
        [
            1000.0 / tissue["t1_ms"],
            1000.0 / tissue["t1_pool_b_ms"],
            1000.0 / tissue["t1_bound_ms"],
        ],
        dtype=torch.float64,
    )
    kab = tissue["pool_b_exchange_hz"] * weight[1]
    kba = tissue["pool_b_exchange_hz"] * weight[0]
    kac = tissue["bound_exchange_hz"] * weight[2]
    kca = tissue["bound_exchange_hz"] * weight[0]
    zero = torch.zeros((), dtype=torch.float64)
    matrix = torch.stack(
        (
            torch.stack((-kab - kac, kba, kca)),
            torch.stack((kab, -kba, zero)),
            torch.stack((kac, zero, -kca)),
        )
    )
    # An inversion turns the free water and the exchanging pool over and
    # leaves the semisolid pool where it was.
    start = torch.stack((-weight[0], -weight[1], weight[2])).to(torch.complex128)
    equilibrium = weight.to(torch.complex128)

    for index, delay in enumerate(delays):
        operator, recovery = longitudinal_relaxation_exchange_op(
            equilibrium, matrix, rates, torch.tensor(delay, dtype=torch.float64)
        )
        settled = operator @ start + recovery
        expected = complex(settled[0] + settled[1]) * readout
        assert abs(complex(measured[index]) - expected) / abs(readout) < 2e-5


# --- what the derivative kernels will not do yet ---


def test_a_derivative_of_a_three_pool_run_is_refused():
    """The forward and its forward mode carry all three pools; the reverse
    kernels carry two. Answering with a two-pool Jacobian would be a wrong
    number rather than a missing one, so it is refused instead.
    """
    t2 = torch.tensor([80.0], requires_grad=True)
    signal = FSE().simulate(
        _description(),
        TissueProperties(
            t1_ms=torch.tensor([1000.0]), t2_ms=t2, **SEMISOLID, **EXCHANGING
        ),
        nstates=STATES,
    ).signal

    with pytest.raises(NotImplementedError, match="carry two pools"):
        signal.abs().square().sum().backward()


# --- forward mode ---


def _live_events():
    """Invert, wait, excite, read, wait again and read again.

    Every property the three-pool system carries reaches this reading. A
    refocused train leaves both T1s and the off-resonance under the rounding
    of a difference, which is a property of the probe rather than of the
    kernel; here the recovery interval makes them live.
    """
    from torchsim.sequence._accelerators import _EXCITATION, _INVERSION, _RECORD

    return (
        torch.tensor([0.0, 0.35, 0.0, 0.0, 3e-3, 0.0], dtype=torch.float32),
        torch.tensor([1, 0, 1, 2, 0, 2], dtype=torch.int32),
        torch.tensor(
            [0.0, 0.0, 0.5 * torch.pi, 0.0, 0.0, 0.0], dtype=torch.float32
        ),
        torch.tensor(
            [0.0, 0.0, 0.5 * torch.pi, 0.0, 0.0, 0.0], dtype=torch.float32
        ),
        torch.tensor(
            [_INVERSION, 0, _EXCITATION, _RECORD, 0, _RECORD], dtype=torch.uint8
        ),
        torch.tensor([-1, -1, -1, 0, -1, 1], dtype=torch.int32),
        torch.zeros(6, dtype=torch.int32),
        torch.zeros(6, dtype=torch.float32),
        torch.zeros(6, dtype=torch.float32),
    )


def _prepared(device="cpu", **properties):
    prepared, _, _ = _prepare_tissue(
        TissueProperties(**{**LIVE, **properties}), device
    )
    return tuple(value.to(torch.float32).contiguous() for value in prepared)


def _live_readout(prepared, seed=None, device="cpu"):
    """The reading, or its directional derivative."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    events = tuple(value.to(device) for value in _live_events())
    table = lineshape_table(device=torch.device(device))
    if seed is None:
        return _run_packed(
            prepared, events, STATES, 2, 1, lineshape=table, exchanging=True
        )
    return _run_packed_jvp(
        prepared,
        events,
        seed,
        tuple(torch.zeros_like(events[0]) for _ in range(3)),
        STATES,
        2,
        1,
        lineshape=table,
        exchanging=True,
    )


@pytest.mark.parametrize("name", sorted(LIVE))
def test_forward_mode_matches_finite_differences(name: str) -> None:
    """Every direction the three-pool system carries, including the two that
    only the semisolid pool has and the five only the exchanging one does.
    """
    from torchsim.sequence._parameters import TISSUE_NAMES

    index = TISSUE_NAMES.index(name)
    prepared = _prepared()
    seed = tuple(
        torch.ones_like(value) if position == index else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    )
    tangent = _live_readout(prepared, seed)

    step = abs(LIVE[name]) * 1e-2
    forward = _live_readout(_prepared(**{name: LIVE[name] + step}))
    backward = _live_readout(_prepared(**{name: LIVE[name] - step}))
    difference = (forward - backward) / (2.0 * step)

    scale = float(difference.abs().max())
    assert scale > 0.0, "the probe leaves this direction dead"
    assert float((tangent - difference).abs().max()) / scale < 5e-3


def test_forward_mode_leaves_the_two_pool_answers_untouched():
    """A tissue at either default fraction takes the two-pool kernel in forward
    mode too, so its directions are bit for bit what they were.
    """
    t2 = torch.tensor([80.0])

    def along(**extra):
        def run(value):
            return FSE().simulate(
                _description(),
                TissueProperties(
                    t1_ms=torch.tensor([1000.0]), t2_ms=value, **extra
                ),
                nstates=STATES,
            ).signal

        return torch.func.jvp(run, (t2,), (torch.ones_like(t2),))[1]

    assert torch.equal(
        along(**SEMISOLID),
        along(**SEMISOLID, **{**EXCHANGING, "pool_b_fraction": 0.0}),
    )
    assert torch.equal(
        along(**EXCHANGING),
        along(**EXCHANGING, **{**SEMISOLID, "bound_fraction": 0.0}),
    )


def test_forward_mode_reaches_three_pools_through_the_public_api():
    """The path an optimizer takes, rather than the packed buffers directly."""
    fraction = torch.tensor([0.2])

    def run(value):
        return FSE().simulate(
            _description(),
            TissueProperties(
                t1_ms=torch.tensor([1000.0]),
                t2_ms=torch.tensor([80.0]),
                **SEMISOLID,
                **{**EXCHANGING, "pool_b_fraction": value},
            ),
            nstates=STATES,
        ).signal

    reading, tangent = torch.func.jvp(
        run, (fraction,), (torch.ones_like(fraction),)
    )
    step = 1e-3
    difference = (run(fraction + step) - run(fraction - step)) / (2.0 * step)

    assert float(reading.abs().max()) > 0.0
    scale = float(difference.abs().max())
    assert float((tangent - difference).abs().max()) / scale < 5e-3


# --- the other backend ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_kernel_matches_the_cpu_kernel():
    """The two share no code, so agreement is what keeps the 3x3 honest on the
    card: a pool dropped from the exchange there would still produce a
    plausible train.
    """
    voxels = 6
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.2, voxels),
        bound_exchange_hz=torch.linspace(5.0, 60.0, voxels),
        t1_bound_ms=torch.linspace(400.0, 1500.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )

    def run(device):
        return FSE().simulate(
            _description(),
            TissueProperties(
                **{name: value.to(device) for name, value in spread.items()}
            ),
            nstates=STATES,
        ).signal

    host = run("cpu")
    card = run("cuda").cpu()

    assert float(host.abs().max()) > 0.0
    assert float((host - card).abs().max() / host.abs().max()) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_volume_matches_the_whole_one():
    """Streaming cuts the voxel axis, which all three pools follow."""
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
    events = (
        packed.duration, packed.kind, packed.flip, packed.phase, packed.action,
        packed.output_index, packed.shim_index, packed.saturation,
        packed.rf_frequency_hz,
    )
    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(600.0, 1400.0, voxels),
            t2_ms=torch.linspace(40.0, 120.0, voxels),
            bound_fraction=torch.linspace(0.02, 0.2, voxels),
            bound_exchange_hz=torch.linspace(5.0, 60.0, voxels),
            pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
            pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
            t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
            pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
        ),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    arguments = (prepared, events, STATES, ECHOES, 1)
    options = dict(lineshape=lineshape_table(), exchanging=True)

    whole = _run_packed(*arguments, **options)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed(*arguments, **options)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 5e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_forward_mode_matches_the_cpu_kernel():
    """The tangent of the 3x3 on the card.

    The two backends share no code, so agreement is what keeps the direction
    honest there: a pool dropped from the tangent alone still produces a
    plausible one.
    """
    voxels = 6
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.2, voxels),
        bound_exchange_hz=torch.linspace(5.0, 60.0, voxels),
        t1_bound_ms=torch.linspace(400.0, 1500.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )

    def run(device):
        prepared = _prepared(device, **{name: value.to(device) for name, value in spread.items()})
        return _live_readout(
            prepared,
            tuple(torch.ones_like(value) for value in prepared),
            device=device,
        )

    host = run("cpu")
    card = run("cuda").cpu()

    assert float(host.abs().max()) > 0.0
    assert float((host - card).abs().max() / host.abs().max()) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_forward_mode_matches_the_whole_one():
    """Streaming cuts the voxel axis, which all three pools' seeds follow."""
    from torchsim.sequence._accelerators import offload

    voxels = 3000
    prepared = _prepared(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.2, voxels),
        bound_exchange_hz=torch.linspace(5.0, 60.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
        pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
    )
    seed = tuple(torch.ones_like(value) for value in prepared)

    whole = _live_readout(prepared, seed)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _live_readout(prepared, seed)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 5e-5
