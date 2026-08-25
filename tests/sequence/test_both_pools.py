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
from utils.packed_reference import simulate_packed

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


def _free_induction(times_s, **properties):
    """Excite, then read the transverse magnetization at each time after it.

    ``times_s`` are measured from the excitation; the events carry the
    intervals between them, which is what the state machine steps through.
    """
    from torchsim.sequence._accelerators import _EXCITATION, _RECORD

    intervals = [
        after - before for before, after in zip((0.0, *times_s), times_s)
    ]
    count = len(times_s)
    events = (
        torch.tensor([0.0, *intervals], dtype=torch.float32),
        torch.tensor([1, *([2] * count)], dtype=torch.int32),
        torch.tensor([0.5 * torch.pi, *([0.0] * count)], dtype=torch.float32),
        torch.zeros(1 + count, dtype=torch.float32),
        torch.tensor([_EXCITATION, *([_RECORD] * count)], dtype=torch.uint8),
        torch.tensor([-1, *range(count)], dtype=torch.int32),
        torch.zeros(1 + count, dtype=torch.int32),
        torch.zeros(1 + count, dtype=torch.float32),
        torch.zeros(1 + count, dtype=torch.float32),
    )
    prepared, _, _ = _prepare_tissue(
        TissueProperties(**{"t1_ms": 1000.0, "t2_ms": 80.0, **properties}), "cpu"
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return _run_packed(
        prepared, events, STATES, count, 1, geometry=NO_GEOMETRY,
        lineshape=lineshape_table(), exchanging=True,
    ).flatten()


def test_the_semisolid_pool_slows_the_transverse_exchange():
    """The free water is what *both* second pools leave.

    The semisolid pool carries no transverse magnetization, so it is absent
    from the transverse 2x2 -- but not from how much free water that 2x2's
    exchange sees. A kernel deriving the free fraction as ``1 - pool_b`` there
    reproduces both two-pool limits exactly and is still wrong in between,
    which is what this reads.

    Taken against the package's own operator, given the free fraction the
    three-pool system actually leaves.
    """
    from utils.exchange import build_two_pool_exchange_matrix
    from utils.epg import transverse_relaxation_exchange_op

    delay = 6e-3
    fraction_b, fraction_c, rate = 0.2, 0.3, 60.0
    t2_a, t2_b = 80.0, 25.0
    tissue = dict(
        t2_ms=t2_a, bound_fraction=fraction_c, bound_exchange_hz=0.0,
        t1_bound_ms=1000.0, pool_b_fraction=fraction_b,
        pool_b_exchange_hz=rate, t1_pool_b_ms=300.0, t2_pool_b_ms=t2_b,
        pool_b_shift_hz=0.0,
    )
    measured = complex(_free_induction((delay,), **tissue)[0])
    scale = complex(_free_induction((0.0,), t2_ms=t2_a)[0])

    free = 1.0 - fraction_b - fraction_c
    # The package's builder takes the two transverse pools' weights, and the
    # free water's is what the semisolid pool has already been taken out of.
    weight = torch.tensor([free, fraction_b], dtype=torch.float64)
    matrix = build_two_pool_exchange_matrix(
        weight, torch.tensor(rate, dtype=torch.float64)
    )
    operator = transverse_relaxation_exchange_op(
        matrix,
        torch.tensor([1000.0 / t2_a, 1000.0 / t2_b], dtype=torch.float64),
        torch.tensor(delay, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    start = torch.tensor([free, fraction_b], dtype=torch.complex128)
    expected = complex((operator @ start).sum()) * scale

    assert abs(measured - expected) / abs(scale) < 5e-5


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
    from utils.epg import longitudinal_relaxation_exchange_op

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


# --- the reverse passes ---


def _live_adjoint(prepared, seed):
    """The gradients a cotangent on the reading leaves."""
    from torchsim.sequence._accelerators import _run_packed_vjp

    return _run_packed_vjp(
        prepared,
        _live_events(),
        seed,
        state_count=STATES,
        output_count=2,
        threads=1,
        lineshape=lineshape_table(),
        exchanging=True,
    )


def _cotangent(seed=4):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.rand((1, 2), generator=generator) * 2.0 - 1.0,
        torch.rand((1, 2), generator=generator) * 2.0 - 1.0,
    )


@pytest.mark.parametrize("name", sorted(LIVE))
def test_the_adjoint_matches_finite_differences(name: str) -> None:
    """Every direction three pools carry, including the semisolid pool's own
    three and the exchanging pool's five.
    """
    from torchsim.sequence._parameters import TISSUE_NAMES

    index = TISSUE_NAMES.index(name)
    seed = _cotangent()

    def reading(**properties):
        return float(
            (seed.conj() * _live_readout(_prepared(**properties))).real.sum()
        )

    gradient = float(_live_adjoint(_prepared(), seed)[index].sum())
    step = abs(LIVE[name]) * 1e-3
    difference = (
        reading(**{name: LIVE[name] + step}) - reading(**{name: LIVE[name] - step})
    ) / (2.0 * step)

    assert abs(difference) > 0.0, "the probe leaves this direction dead"
    assert abs(gradient - difference) / abs(difference) < 5e-3


def test_the_adjoint_transposes_the_forward_direction():
    """``<w, J v> == <J^T w, v>``, taken against the sum of the terms'
    magnitudes rather than their total, so one small gradient going wrong is
    not hidden by two large ones that did not.
    """
    prepared = _prepared()
    generator = torch.Generator().manual_seed(19)
    directions = tuple(
        torch.rand(value.shape, generator=generator) * 2.0 - 1.0
        for value in prepared
    )
    seed = _cotangent(23)

    forward = _live_readout(prepared, directions)
    left = float((seed.conj() * forward).real.sum())
    terms = [
        float((gradient * direction).sum())
        for gradient, direction in zip(_live_adjoint(prepared, seed), directions)
    ]
    scale = sum(abs(term) for term in terms)

    assert scale > 0.0
    assert abs(left - sum(terms)) / scale < 1e-6


def test_the_second_order_pass_differentiates_the_adjoint():
    """Given no direction to follow, the forward-over-reverse kernel returns
    the adjoint on its own -- and given one, the adjoint's own derivative.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    prepared = _prepared()
    events = _live_events()
    seed = _cotangent()
    options = dict(
        state_count=STATES, output_count=2, threads=1,
        lineshape=lineshape_table(), exchanging=True,
    )
    still = tuple(torch.zeros_like(value) for value in prepared) + tuple(
        torch.zeros_like(events[0]) for _ in range(3)
    )
    first = _live_adjoint(prepared, seed)
    _, adjoint = _run_packed_vjp_jvp(prepared, events, still, seed, **options)
    for expected, measured in zip(first, adjoint, strict=True):
        scale = float(expected.abs().max())
        if scale > 1e-9:
            assert float((expected - measured).abs().max()) / scale < 1e-4

    from torchsim.sequence._parameters import TISSUE_NAMES

    # The direction is confined to the properties the difference below moves,
    # so the two are contracted against the same one.
    generator = torch.Generator().manual_seed(31)
    directions = tuple(
        torch.rand(value.shape, generator=generator) * 2.0 - 1.0
        if TISSUE_NAMES[position] in LIVE
        else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    ) + tuple(torch.zeros_like(events[0]) for _ in range(3))
    curvature, _ = _run_packed_vjp_jvp(prepared, events, directions, seed, **options)

    step = 1e-3
    def moved(sign):
        return _prepared(**{
            name: LIVE[name] + sign * step * float(
                directions[TISSUE_NAMES.index(name)][0]
            )
            for name in LIVE
        })

    ahead = _live_adjoint(moved(+1), seed)
    behind = _live_adjoint(moved(-1), seed)
    compared = 0
    for name in LIVE:
        index = TISSUE_NAMES.index(name)
        difference = float((ahead[index] - behind[index]).sum()) / (2.0 * step)
        if abs(difference) > 1e-6:
            got = float(curvature[index].sum())
            assert abs(got - difference) / abs(difference) < 5e-3, name
            compared += 1
    assert compared > 3



def test_the_second_order_pass_saturates_the_pool_the_pulse_deposits_into():
    """A refocused train, where every pulse both turns the free pools and
    deposits power in the semisolid one.

    Given no direction to follow the forward-over-reverse kernel has to return
    what the first-order adjoint does, and the two reach the saturation by
    different code.
    """
    from torchsim.sequence._accelerators import (
        _run_packed_vjp, _run_packed_vjp_jvp,
    )

    prepared = _prepared()
    events = _train_events()
    generator = torch.Generator().manual_seed(37)
    seed = torch.complex(
        torch.rand((1, ECHOES), generator=generator) * 2.0 - 1.0,
        torch.rand((1, ECHOES), generator=generator) * 2.0 - 1.0,
    )
    options = dict(
        state_count=STATES, output_count=ECHOES, threads=1,
        lineshape=lineshape_table(), exchanging=True,
    )
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )

    first = _run_packed_vjp(prepared, events, seed, **options)
    _, second = _run_packed_vjp_jvp(prepared, events, still, seed, **options)

    compared = 0
    for expected, measured in zip(first, second, strict=True):
        scale = float(expected.abs().max())
        if scale < 1e-9:
            continue
        assert float((expected - measured).abs().max()) / scale < 1e-4
        compared += 1
    assert compared > 8

def test_the_adjoint_leaves_the_two_pool_answers_untouched():
    """A tissue at either default fraction takes the two-pool kernel in reverse
    mode too, so its gradients are bit for bit what they were.
    """
    def gradient(**extra):
        t2 = torch.tensor([80.0], requires_grad=True)
        FSE().simulate(
            _description(),
            TissueProperties(t1_ms=torch.tensor([1000.0]), t2_ms=t2, **extra),
            nstates=STATES,
        ).signal.abs().square().sum().backward()
        return t2.grad

    assert torch.equal(
        gradient(**SEMISOLID),
        gradient(**SEMISOLID, **{**EXCHANGING, "pool_b_fraction": 0.0}),
    )
    assert torch.equal(
        gradient(**EXCHANGING),
        gradient(**EXCHANGING, **{**SEMISOLID, "bound_fraction": 0.0}),
    )


def test_a_three_pool_gradient_reaches_the_public_api():
    """The path an optimizer takes, rather than the packed buffers directly."""
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in dict(LIVE, t1_ms=1000.0, t2_ms=80.0).items()
    }
    FSE().simulate(
        _description(), TissueProperties(**leaves), nstates=STATES
    ).signal.abs().square().sum().backward()

    for name in ("t2_ms", "bound_fraction", "pool_b_fraction", "t2_pool_b_ms"):
        assert float(leaves[name].grad.abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_gradient_matches_the_cpu_gradient():
    """The two reverse sweeps share no code, so agreement is what keeps the
    3x3's transpose honest on the card.

    A gradient sums one term per echo through ``tl.atomic_add``, whose order is
    not fixed, so the bound is the accumulated one the parity suite uses.
    """
    voxels = 4
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
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
        leaves = {
            name: value.to(device).clone().requires_grad_(True)
            for name, value in spread.items()
        }
        signal = FSE().simulate(
            _description(), TissueProperties(**leaves), nstates=STATES
        ).signal
        signal.abs().square().sum().backward()
        return {name: leaf.grad.cpu() for name, leaf in leaves.items()}

    host, card = run("cpu"), run("cuda")

    # The floor is tied to the largest gradient in the set rather than to each
    # parameter's own. ``t1_bound_ms`` here is 4e-10 of its largest sibling --
    # a semisolid T1 barely moves a refocused train's echoes -- so what is left
    # in it is float32 round-off, and holding that to a relative bound asks for
    # a precision the representation does not carry. The absolute floor still
    # catches a gradient that is simply wrong.
    floor = 1e-6 * max(float(value.abs().max()) for value in host.values())
    for name in spread:
        scale = float(host[name].abs().max())
        assert scale > 0.0, name
        drift = float((host[name] - card[name]).abs().max())
        assert drift <= 1e-3 * scale + floor, f"{name}: {drift} vs {scale}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_second_order_pass_matches_the_cpu_one():
    """Forward over reverse, both pools live, on both backends."""
    voxels = 3
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        bound_fraction=torch.linspace(0.05, 0.2, voxels),
        bound_exchange_hz=torch.linspace(10.0, 60.0, voxels),
        t1_bound_ms=torch.linspace(400.0, 1200.0, voxels),
        pool_b_fraction=torch.linspace(0.05, 0.3, voxels),
        pool_b_exchange_hz=torch.linspace(5.0, 80.0, voxels),
        t1_pool_b_ms=torch.linspace(200.0, 900.0, voxels),
        t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
        pool_b_shift_hz=torch.linspace(-300.0, 300.0, voxels),
    )
    generator = torch.Generator().manual_seed(3)
    direction = {
        name: torch.rand(voxels, generator=generator) * 0.01 * value.abs().mean()
        for name, value in spread.items()
    }

    def run(device):
        leaves = {
            name: value.to(device).clone().requires_grad_(True)
            for name, value in spread.items()
        }
        signal = FSE().simulate(
            _description(), TissueProperties(**leaves), nstates=STATES
        ).signal
        loss = signal.abs().square().sum()
        gradients = torch.autograd.grad(loss, list(leaves.values()), create_graph=True)
        along = sum(
            (gradient * direction[name].to(device)).sum()
            for gradient, name in zip(gradients, leaves, strict=True)
        )
        return [
            curvature.cpu()
            for curvature in torch.autograd.grad(along, list(leaves.values()))
        ]

    host, card = run("cpu"), run("cuda")

    compared = 0
    for name, expected, measured in zip(spread, host, card, strict=True):
        scale = float(expected.abs().max())
        if scale == 0.0:
            continue
        assert float((expected - measured).abs().max()) / scale < 1e-2, name
        compared += 1
    assert compared > 6


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
        packed.duration, packed.kind, packed.flip, packed.phase, packed.action,
        packed.output_index, packed.shim_index, packed.saturation,
        packed.rf_frequency_hz,
    )


def _routes(leaves, events):
    """The kernels and the oracle, fed from one place."""
    from torchsim.sequence._accelerators import _NativeEpg

    fused = _NativeEpg.apply(
        *leaves, *events, STATES, ECHOES, 1, NO_GEOMETRY, None, None, None,
        lineshape_table(), True, None,
    )
    reference = simulate_packed(
        leaves,
        events,
        state_count=STATES,
        output_count=ECHOES,
        lineshape=lineshape_table(),
        exchanging=True,
    )
    return fused, reference


def _oracle_leaves():
    return tuple(
        value.detach().clone().requires_grad_(True) for value in _prepared()
    )


def test_the_fused_forward_matches_the_oracle():
    """The oracle reaches both exponentials through Pade rather than through
    the closed forms, so this is the three-pool step checked against a
    different algorithm and not against a second copy of itself.
    """
    fused, reference = _routes(_oracle_leaves(), _train_events())

    reference = reference.detach()
    assert float(reference.abs().max()) > 0.0
    assert float(
        (fused.detach() - reference).abs().max() / reference.abs().max()
    ) < 1e-4


def test_the_fused_adjoint_matches_the_oracle():
    """Per parameter, not against one contracted scalar: a transposed adjoint
    still transposes when a small gradient is wrong.
    """
    events = _train_events()
    generator = torch.Generator().manual_seed(31)
    seed = torch.randn(
        (1, ECHOES), generator=generator, dtype=torch.float32
    ) + 1j * torch.randn((1, ECHOES), generator=generator, dtype=torch.float32)

    def gradients(route: int):
        leaves = _oracle_leaves()
        return torch.autograd.grad(
            _routes(leaves, events)[route],
            leaves,
            seed,
            allow_unused=True,
            materialize_grads=True,
        )

    want, got = gradients(1), gradients(0)
    floor = 1e-6 * max(float(value.abs().max()) for value in want)
    compared = 0
    for index, (expected, measured) in enumerate(zip(want, got, strict=True)):
        scale = float(expected.abs().max())
        if scale <= floor:
            continue
        assert float((expected - measured).abs().max()) / scale < 1e-3, index
        compared += 1
    assert compared > 8


def test_the_fused_second_order_matches_the_oracle():
    """Autograd differentiates the oracle to whatever order is asked of it, so
    the analytic kernels are read against it rather than against a difference.

    Held an order looser than the first-order comparison beside it. A second
    derivative in float32 keeps about half the digits its value does, and the
    contraction against a direction cancels: taken one direction at a time the
    cross derivatives here agree to 6e-3, and the terms that look worse than
    that sit at 1e-11 against a row whose largest entry is 1e-3.
    """
    events = _train_events()
    generator = torch.Generator().manual_seed(31)
    seed = torch.randn(
        (1, ECHOES), generator=generator, dtype=torch.float32
    ) + 1j * torch.randn((1, ECHOES), generator=generator, dtype=torch.float32)
    # Drawn once, so the two routes are contracted against the same direction.
    direction = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for value in _oracle_leaves()
    )

    def curvature(route: int):
        leaves = _oracle_leaves()
        first = torch.autograd.grad(
            _routes(leaves, events)[route],
            leaves,
            seed,
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        return torch.autograd.grad(
            first, leaves, direction, allow_unused=True, materialize_grads=True
        )

    want, got = curvature(1), curvature(0)
    floor = 1e-6 * max(float(value.abs().max()) for value in want)
    compared = 0
    for index, (expected, measured) in enumerate(zip(want, got, strict=True)):
        scale = float(expected.abs().max())
        if scale <= floor:
            continue
        assert float((expected - measured).abs().max()) / scale < 1e-2, index
        compared += 1
    assert compared > 8


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
def test_a_streamed_adjoint_matches_the_whole_one():
    """Streaming has broken by a flag dropped on one of the two routes to the
    same kernel, which backend parity cannot see -- so the reverse pass takes
    the streamed-versus-whole check too.

    Given no direction to follow the forward-over-reverse kernel returns the
    adjoint on its own, and that is the reverse route the offload plan reaches.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp, offload

    voxels = 3000
    packed = _pack_events(
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
    generator = torch.Generator().manual_seed(29)
    seed = (
        torch.rand(voxels, ECHOES, generator=generator) * 2.0 - 1.0
    ).to(torch.complex64)
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )
    options = dict(
        state_count=STATES, output_count=ECHOES, threads=1,
        lineshape=lineshape_table(), exchanging=True,
    )

    # The whole volume runs on the card as well, so the only thing between the
    # two is where the voxel axis was cut.
    _, whole = _run_packed_vjp_jvp(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in events),
        tuple(value.cuda() for value in still),
        seed.cuda(),
        **options,
    )
    with offload(["cuda"], budget_bytes=1 << 22):
        _, streamed = _run_packed_vjp_jvp(prepared, events, still, seed, **options)

    compared = 0
    for expected, measured in zip(whole, streamed, strict=True):
        expected = expected.cpu()
        scale = float(expected.abs().max())
        if scale < 1e-6:
            continue
        assert float((expected - measured.cpu()).abs().max()) / scale < 1e-4
        compared += 1
    assert compared > 6


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

    # Both sides run the same kernel on the same card, so what is left between
    # them is the chunking and nothing else. Read against a host run instead
    # and the comparison is really one of backends, whose float32 tail is two
    # orders wider than anything streaming does.
    whole = _live_readout(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in seed),
        device="cuda",
    ).cpu()
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _live_readout(prepared, seed)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 1e-6


# --- a tabulated rotation beside both pools ---


def _instantaneous_table():
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
    return transition_table(flat, torch.zeros(1), bins=1024, rf_raster_time_s=1e-6)


def test_a_tabulated_rotation_reaches_both_pools() -> None:
    """The kernels are templated on the rotation mode and the pool count
    together, so a table beside three pools is an instantiation of its own.
    """
    from torchsim.sequence._accelerators import _NativeEpg

    leaves = _prepared()
    events = _train_events()
    profile = _instantaneous_table()

    fused = _NativeEpg.apply(
        *leaves, *events, STATES, ECHOES, 1, NO_GEOMETRY, profile, None, None,
        lineshape_table(), True, None,
    )
    reference = simulate_packed(
        leaves,
        events,
        state_count=STATES,
        output_count=ECHOES,
        profile=profile,
        lineshape=lineshape_table(),
        exchanging=True,
    )

    assert float(reference.abs().max()) > 0.0
    worst = float((fused - reference).abs().max() / reference.abs().max())
    assert worst < 1e-4, worst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("state_count", [8, 12, 17])
def test_three_pools_take_the_first_order_kernel_on_the_card(
    state_count: int,
) -> None:
    """Both second pools at once do not cost the kernel written for a gradient.

    Held against the host's own first-order adjoint rather than against the
    forward-over-reverse pass on the same card: two arms of one wrong kernel
    agree with each other, and the backends share no code.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _run_packed_vjp
    from torchsim.sequence._parameters import TISSUE_NAMES

    voxels = 64
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-200.0, 200.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.20, voxels),
        bound_exchange_hz=torch.linspace(5.0, 60.0, voxels),
        t1_bound_ms=torch.linspace(400.0, 1200.0, voxels),
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
        prepared = tuple(
            value.to(torch.float32).contiguous() for value in prepared
        )
        return prepared, tuple(value.to(device) for value in packed.buffers)

    host_tissue, host_events = side("cpu")
    card_tissue, card_events = side("cuda")
    seed = torch.randn(
        (voxels, outputs),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(13),
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
            state_count=state_count,
            output_count=outputs,
            threads=1,
            exchanging=True,
            lineshape=lineshape_table(device=torch.device("cuda")),
        )
    finally:
        _accelerators._run_packed_vjp_jvp = original
    host = _run_packed_vjp(
        host_tissue,
        host_events,
        seed,
        state_count=state_count,
        output_count=outputs,
        threads=1,
        exchanging=True,
        lineshape=lineshape_table(),
    )

    assert not reached
    largest = max(float(value.abs().max()) for value in host)
    assert largest > 0.0
    for name, reference, result in zip(
        (*TISSUE_NAMES, "duration", "flip", "phase"), host, card, strict=True
    ):
        assert reference.shape == result.shape, name
        # The semisolid fraction reaches the answer through the free share the
        # transverse operator sees as well as through its own plane, so it is
        # the one a missing term shows up in first.
        difference = float((reference.cpu() - result.cpu()).abs().max())
        assert difference / largest < 1e-4, name


# --- the series branch, and the bound that says it is enough ---


def _three_pool_columns(voxels, *, seconds, seed):
    """A physical three-pool tissue and one interval, as the helper reads them."""
    generator = torch.Generator().manual_seed(seed)

    def spread(low, high):
        return torch.rand(voxels, generator=generator) * (high - low) + low

    return (
        1000.0 / spread(200.0, 3000.0),
        1000.0 / spread(50.0, 2000.0),
        1000.0 / spread(100.0, 2000.0),
        spread(1.0, 200.0),
        spread(1.0, 200.0),
        spread(0.01, 0.40),
        spread(0.01, 0.35),
        torch.full((voxels,), float(seconds)),
        spread(0.2, 1.0),
    )


def _matrix_exp_oracle(columns):
    """``expm((K - diag(R1)) t)`` and the recoveries, built and exponentiated.

    Shares no code with the closed form under test: the generator is assembled
    and exponentiated by :func:`torch.linalg.matrix_exp` in double.
    """
    r1a, r1b, r1c, exb, exc, fb, fc, dt, att = (
        value.double() for value in columns
    )
    free = 1.0 - fb - fc
    zero = torch.zeros_like(free)
    generator = torch.stack(
        [
            torch.stack([-exb * fb - exc * fc - r1a, exb * free, exc * free], -1),
            torch.stack([exb * fb, -exb * free - r1b, zero], -1),
            torch.stack([exc * fc, zero, -exc * free - r1c], -1),
        ],
        dim=-2,
    ) * dt[..., None, None]
    step = att[..., None, None] * torch.linalg.matrix_exp(generator)
    start = torch.stack([free, fb, fc], dim=-1)
    grow = start - (step @ start[..., None]).squeeze(-1)
    return torch.cat([step.reshape(-1, 9), grow], dim=-1)


def _spread_over(columns):
    """The bound :func:`narrow_three_pool` tests, computed the long way."""
    r1a, r1b, r1c, exb, exc, fb, fc, dt, _att = (
        value.double() for value in columns
    )
    free = 1.0 - fb - fc
    a00 = (-exb * fb - exc * fc - r1a) * dt
    a11 = (-exb * free - r1b) * dt
    a22 = (-exc * free - r1c) * dt
    third = (a00 + a11 + a22) / 3.0
    s00, s11, s22 = a00 - third, a11 - third, a22 - third
    minors = (
        s00 * s11 - (exb * free * dt) * (exb * fb * dt)
        + s00 * s22 - (exc * free * dt) * (exc * fc * dt)
        + s11 * s22
    )
    return torch.sqrt(torch.clamp(-2.0 * minors, min=0.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_series_carries_the_answer_up_to_the_spread_it_is_trusted_to() -> None:
    """``NARROW_SPREAD`` is a measurement, so it is pinned as one.

    The series branch alone, in float32, against ``matrix_exp`` in double, at
    the widest spread the gate will let through. Above the bound it is expected
    to fail -- which is what makes the bound a choice rather than a hope.
    """
    import triton
    import triton.language as tl

    from torchsim.sequence._epg_triton import _three_pool_step
    from torchsim.sequence._parameters import NARROW_SPREAD

    @triton.jit
    def only_the_series(
        a, b, c, d, e, f, g, h, i, out, n,
        NARROW: tl.constexpr, BLOCK: tl.constexpr,
    ):
        j = tl.arange(0, BLOCK)
        mask = j < n
        entries = _three_pool_step(
            tl.load(a + j, mask=mask, other=1.0),
            tl.load(b + j, mask=mask, other=1.0),
            tl.load(c + j, mask=mask, other=1.0),
            tl.load(d + j, mask=mask, other=1.0),
            tl.load(e + j, mask=mask, other=1.0),
            tl.load(f + j, mask=mask, other=0.1),
            tl.load(g + j, mask=mask, other=0.1),
            tl.load(h + j, mask=mask, other=0.005),
            tl.load(i + j, mask=mask, other=1.0),
            NARROW,
        )
        for entry in tl.static_range(12):
            tl.store(out + entry * n + j, entries[entry], mask=mask)

    voxels = 4096
    worst = {}
    for label, seconds in (("inside", 20e-3), ("at the bound", None), ("past", 1.0)):
        columns = _three_pool_columns(voxels, seconds=seconds or 1.0, seed=5)
        if seconds is None:
            # Stretch the interval until the widest voxel sits on the bound.
            rate = float(_spread_over(columns).max())
            columns = _three_pool_columns(
                voxels, seconds=NARROW_SPREAD / rate, seed=5
            )
        reached = float(_spread_over(columns).max())
        expected = _matrix_exp_oracle(columns)
        scale = expected.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        out = torch.zeros(12 * voxels, device="cuda", dtype=torch.float32)
        only_the_series[(1,)](
            *[value.to("cuda", torch.float32) for value in columns],
            out, voxels, NARROW=True, BLOCK=triton.next_power_of_2(voxels),
        )
        got = out.reshape(12, voxels).T.double().cpu()
        worst[label] = (
            reached, float(((got - expected).abs() / scale).amax(dim=1).max())
        )

    inside_spread, inside = worst["inside"]
    bound_spread, at_bound = worst["at the bound"]
    past_spread, past = worst["past"]
    assert inside_spread < NARROW_SPREAD
    assert abs(bound_spread - NARROW_SPREAD) < 1e-6 * NARROW_SPREAD
    assert past_spread > 2.0 * NARROW_SPREAD
    # Float32 holds up to the bound: five ulps of the largest entry.
    assert inside < 1e-6, inside
    assert at_bound < 1e-6, at_bound
    # And does not past it, which is why the roots are still there. Far enough
    # out the series overflows rather than merely drifting, so the check is
    # that it fails, not that it fails by a particular amount.
    assert not past <= 1e-4, past


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_long_interval_declines_the_series_branch() -> None:
    """The gate is refused where it must be, and taken where it may be.

    A preparation delay spreads the eigenvalues past what the series carries,
    so the launch has to fall back to the roots -- checked on the predicate the
    launchers call, not inferred from the answer agreeing.
    """
    from torchsim.sequence._parameters import narrow_three_pool

    voxels = 256
    tissue, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(300.0, 2000.0, voxels),
            t2_ms=torch.linspace(20.0, 200.0, voxels),
            **{
                name: torch.full((voxels,), value)
                for name, value in {**SEMISOLID, **EXCHANGING}.items()
            },
        ),
        "cpu",
    )
    tissue = tuple(value.to(torch.float32) for value in tissue)

    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    assert narrow_three_pool(tissue, packed.duration, pools=3)
    assert not narrow_three_pool(
        tissue, torch.tensor([2.5], dtype=torch.float32), pools=3
    )
    # One pool cannot reach the branch at all, so it never pays for the bound.
    assert not narrow_three_pool(tissue, packed.duration, pools=2)
    assert not narrow_three_pool(tissue, packed.duration, pools=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("state_count", [8, 16])
def test_the_series_branch_gives_the_answer_the_roots_give(state_count) -> None:
    """Every pass, the flag forced both ways under one fixed input.

    The gate cannot be measured against the host: that comparison moves for
    reasons of its own. What it has to be held to is the branch it replaces.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence import _epg_triton

    voxels = 256
    tissue, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(300.0, 2000.0, voxels),
            t2_ms=torch.linspace(20.0, 200.0, voxels),
            b0_hz=torch.linspace(-30.0, 30.0, voxels),
            **{
                name: torch.full((voxels,), value)
                for name, value in {**SEMISOLID, **EXCHANGING}.items()
            },
        ),
        "cuda",
    )
    tissue = tuple(value.to(torch.float32).contiguous() for value in tissue)
    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cuda"),
        rf_raster_time_s=1e-6,
    )
    events = tuple(packed.buffers)
    outputs = int(packed.output_count)
    table = lineshape_table(device=torch.device("cuda"))
    seed = torch.randn(
        (voxels, outputs),
        dtype=torch.complex64,
        generator=torch.Generator().manual_seed(3),
    ).cuda()
    still = tuple(
        torch.zeros_like(value)
        for value in (*tissue, events[0], events[2], events[3])
    )
    options = dict(
        state_count=state_count, output_count=outputs, threads=1,
        exchanging=True, lineshape=table,
    )

    def both_ways(run):
        settled = _epg_triton.narrow_three_pool
        sides = []
        for forced in (True, False):
            _epg_triton.narrow_three_pool = (
                lambda *a, taken=forced, **k: taken
            )
            try:
                sides.append(run())
            finally:
                _epg_triton.narrow_three_pool = settled
        return sides

    def leaves(value):
        if isinstance(value, torch.Tensor):
            return [value]
        return [leaf for part in value for leaf in leaves(part)]

    runs = {
        "forward": lambda: _accelerators._run_packed(
            tissue, events, state_count, outputs, 1,
            exchanging=True, lineshape=table,
        ),
        "forward mode": lambda: _accelerators._run_packed_jvp(
            tissue, events,
            tuple(torch.full_like(value, 1e-2) for value in tissue),
            tuple(
                torch.zeros_like(value)
                for value in (events[0], events[2], events[3])
            ),
            state_count, outputs, 1, exchanging=True, lineshape=table,
        ),
        "adjoint": lambda: _accelerators._run_packed_vjp(
            tissue, events, seed, **options
        ),
        "second order": lambda: _accelerators._run_packed_vjp_jvp(
            tissue, events, still, seed, **options
        ),
    }
    for name, run in runs.items():
        narrow, roots = (leaves(side) for side in both_ways(run))
        largest = max(float(value.abs().max()) for value in roots)
        assert largest > 0.0, name
        worst = max(
            float((left - right).abs().max())
            for left, right in zip(narrow, roots, strict=True)
        )
        assert worst / largest < 1e-5, (name, worst / largest)
