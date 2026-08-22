"""A bound proton pool on the longitudinal axis, through the fused kernels.

Every tissue holds a share of its protons bound to macromolecules. That pool
has essentially no T2, so it never shows up in a signal directly; it reaches
one by exchanging magnetization with the free water, and by absorbing RF power
that the free water at the same offset does not. What the kernels carry is
therefore asymmetric -- ``F+`` and ``F-`` stay single-pool and only the ``Z``
step and the RF operator change -- and these tests pin each half of that.

The same asymmetry runs through all four passes: forward, forward mode, the
adjoint and the pass that differentiates the adjoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence._accelerators import (
    NO_GEOMETRY,
    _pack_events,
    _run_packed,
)
from torchsim.sequence._description import EventType, RfDefinition, RfShape
from torchsim.sequence._lineshape import lineshape_table
from torchsim.sequence._parameters import TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue
from utils.packed_reference import simulate_packed

ECHOES = 8
STATES = 8
BOUND = TISSUE_NAMES.index("bound_fraction")

# A bound fraction and an exchange rate in the range white matter is quoted at.
FRACTION = 0.1
RATE_HZ = 30.0


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

    A bound fraction of zero leaves the exchange diagonal and the pool empty,
    so nothing it drives can reach the free water. Nothing then builds a
    lineshape either, which is what selects the single-pool kernel -- so the
    answer is not merely close to the one-pool answer, it is that answer.
    """
    plain = _signal()
    gated = _signal(bound_fraction=0.0, bound_exchange_hz=RATE_HZ, t1_bound_ms=400.0)

    assert torch.equal(plain, gated)


def test_a_bound_pool_takes_its_share_of_the_first_echo():
    """Equilibrium is split between the pools, so the free water starts lower.

    The first echo is early enough that neither exchange nor saturation has
    moved much, so what it measures is the initial condition: the free pool
    holds ``1 - f`` of the magnetization rather than all of it.
    """
    plain = _signal().abs().flatten()[0]
    for fraction in (0.05, 0.1, 0.2):
        bound = _signal(
            bound_fraction=fraction, bound_exchange_hz=RATE_HZ
        ).abs().flatten()[0]
        assert abs(float(bound / plain) - (1.0 - fraction)) < 1e-6


def test_the_signal_falls_as_the_bound_pool_grows():
    """Monotone, and by more than the initial split alone accounts for.

    Later echoes have had exchange draining the free pool into a saturated one,
    so the ratio keeps falling along the train rather than sitting at ``1 - f``.
    """
    plain = _signal().abs().flatten()
    # The train passes through a near-null, where a magnitude is the modulus of
    # a number close to zero and can move either way for reasons that are not
    # the bound pool; the echoes carrying signal are what is compared.
    live = plain > 0.01 * float(plain.max())
    previous = plain
    for fraction in (0.05, 0.1, 0.2):
        bound = _signal(
            bound_fraction=fraction, bound_exchange_hz=RATE_HZ
        ).abs().flatten()
        assert bool((bound[live] < previous[live] + 1e-6).all())
        previous = bound

    ratio = (bound / plain)[live]
    assert float(ratio[-1]) < float(ratio[0])


# --- the two-pool longitudinal step ---


def _inversion_recovery(delays_s, **properties):
    """Signal after an inversion and a delay, one entry per delay.

    An inversion turns the free pool over and leaves the bound pool alone, so
    the pools start far from equilibrium and the interval that follows exercises
    the exchange rather than sitting at the fixed point. The readout is a hard
    90 degrees, which writes ``Z`` into the transverse plane, so the recorded
    signal is proportional to the free pool's longitudinal state.
    """
    from torchsim.sequence._accelerators import (
        _EXCITATION,
        _INVERSION,
        _RECORD,
    )

    signals = []
    tissue = TissueProperties(t1_ms=1000.0, t2_ms=80.0, **properties)
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    lineshape = (
        lineshape_table()
        if float(prepared[BOUND].max()) > 0.0
        else None
    )
    for delay in delays_s:
        events = (
            torch.tensor([[0.0, float(delay), 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([1, 0, 1, 2], dtype=torch.int32),
            torch.tensor([[0.0, 0.0, 0.5 * torch.pi, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 0.5 * torch.pi, 0.0]], dtype=torch.float32),
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
                lineshape=lineshape,
            ).flatten()[0]
        )
    return torch.stack(signals)


def _oracle(delays_s, fraction, rate_hz, t1_ms, t1_bound_ms):
    """``Z_a`` after an inversion and a delay, from the package's operator.

    The bound pool is untouched by the inversion, so the two pools start at
    ``(-(1 - f), f)`` and relax toward ``(1 - f, f)`` while exchanging.
    """
    from torchsim.epg import longitudinal_relaxation_exchange_op

    free = 1.0 - fraction
    weight = torch.tensor([free, fraction], dtype=torch.float64)
    exchange = torch.tensor(
        [[-rate_hz * fraction, rate_hz * free],
         [rate_hz * fraction, -rate_hz * free]],
        dtype=torch.float64,
    )
    rates = torch.tensor(
        [1000.0 / t1_ms, 1000.0 / t1_bound_ms], dtype=torch.float64
    )
    start = torch.tensor([-free, fraction], dtype=torch.complex64)
    answers = []
    for delay in delays_s:
        e1, re1 = longitudinal_relaxation_exchange_op(
            weight, exchange, rates, torch.tensor(delay, dtype=torch.float64)
        )
        answers.append((e1.to(torch.complex64) @ start + re1)[0].real)
    return torch.stack(answers).to(torch.float32)


DELAYS_S = (0.05, 0.2, 0.5, 1.0, 2.0)


def test_the_longitudinal_step_reproduces_the_package_operator():
    """The kernel's closed form against ``epg``'s matrix exponential.

    The readout scale is not part of what is being checked, so it is measured
    on the single-pool run -- whose longitudinal state is the elementary
    ``1 - 2 exp(-t R1)`` -- and applied to the two-pool prediction.
    """
    t1_ms, t1_bound_ms = 1000.0, 400.0
    single = _inversion_recovery(DELAYS_S).abs()
    elementary = torch.tensor(
        [1.0 - 2.0 * np.exp(-delay * 1000.0 / t1_ms) for delay in DELAYS_S],
        dtype=torch.float32,
    ).abs()
    scales = single / elementary
    assert float((scales / scales[0] - 1.0).abs().max()) < 1e-5

    bound = _inversion_recovery(
        DELAYS_S,
        bound_fraction=FRACTION,
        bound_exchange_hz=RATE_HZ,
        t1_bound_ms=t1_bound_ms,
    ).abs()
    expected = scales * _oracle(
        DELAYS_S, FRACTION, RATE_HZ, t1_ms, t1_bound_ms
    ).abs()

    assert float(((bound - expected).abs() / expected.abs().max()).max()) < 1e-5


def test_exchange_pulls_the_free_pool_back_faster_than_its_own_t1():
    """A bound pool with a short T1 is a relaxation sink for the free water.

    Nothing about the closed form guarantees this; it is what makes MT worth
    modelling, and it separates a kernel that exchanges from one that runs two
    independent pools side by side.
    """
    alone = _inversion_recovery(
        DELAYS_S, bound_fraction=FRACTION, bound_exchange_hz=0.0, t1_bound_ms=50.0
    )
    exchanging = _inversion_recovery(
        DELAYS_S, bound_fraction=FRACTION, bound_exchange_hz=200.0, t1_bound_ms=50.0
    )
    # Early in the recovery the free pool is still inverted, so faster recovery
    # means a smaller magnitude.
    assert float(exchanging[0].abs()) < float(alone[0].abs())


def test_the_inversion_leaves_the_bound_pool_alone():
    """A modelling choice, and one a kernel could plausibly get either way.

    A bound pool's T2 is short enough that an adiabatic sweep saturates it
    rather than turning it over. Were the kernel to invert it too, both pools
    would start at ``-(1 - f), -f`` and relax as a scaled copy of the
    single-pool curve, so the exchange would leave no trace at all.
    """
    both_inverted = -_oracle(DELAYS_S, FRACTION, RATE_HZ, 1000.0, 400.0)
    free_only = _inversion_recovery(
        DELAYS_S,
        bound_fraction=FRACTION,
        bound_exchange_hz=RATE_HZ,
        t1_bound_ms=400.0,
    ).abs()

    assert float((free_only / free_only[0] - both_inverted / both_inverted[0])
                 .abs().max()) > 1e-3


# --- RF saturation of the bound pool ---


def _hard_pulse(duration_s: float, samples: int = 256) -> RfDefinition:
    """A rectangular pulse, whose saturation has a closed form."""
    return RfDefinition(
        id=0,
        bandwidth_hz=1.0 / duration_s,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=1.0 / duration_s,
        total_b1sq_power=1.0,
        magnitude=RfShape(samples, np.ones(samples, dtype=np.float32)),
    )


def test_a_rectangular_pulse_deposits_pi_over_its_duration():
    """``-pi int |w|^2 dt / |int w dt|^2``, which for a constant envelope is
    ``-pi / tau``. The gyromagnetic ratio cancels out of that expression, so
    the number depends on the shape and its length and on nothing else.
    """
    duration_s, samples = 1e-3, 256
    raster = duration_s / samples
    deposited = _hard_pulse(duration_s, samples).saturation(rf_raster_time_s=raster)

    # The trapezoid drops half a sample at each end of a rectangle, so the
    # effective length is one raster step short.
    effective = duration_s - raster
    assert abs(deposited - (-np.pi / effective)) < 1e-6 * np.pi / effective


def test_the_canonical_saturation_matches_the_literature_number():
    """The classic bSSFP-MT figure: ``b1rms^2 tau`` of 32.7 uT^2 ms on
    resonance saturates a white-matter bound pool by about a tenth.

    ``epg.initialize_mt_sat`` computes this a thousand times too small -- it
    multiplies a rate written per millisecond by a lineshape in seconds -- so
    the number is held against the physics rather than against that function.
    """
    duration_s, samples = 1e-3, 4096
    raster = duration_s / samples
    gamma = 2.0 * np.pi * 42.577e6  # rad/s/T
    b1rms_t = 1e-6 * np.sqrt(32.7)  # 32.7 uT^2 ms over a 1 ms pulse
    flip_rad = gamma * b1rms_t * duration_s

    deposited = _hard_pulse(duration_s, samples).saturation(rf_raster_time_s=raster)
    table = lineshape_table()
    absorbed = np.exp(
        deposited * flip_rad**2 * float(table.at(torch.tensor(0.0)))
    )

    assert 0.85 < absorbed < 0.92


def test_an_off_resonance_pulse_saturates_less():
    """The lineshape falls away from resonance, so a pulse played further off
    it deposits less in the bound pool for the same power.
    """
    table = lineshape_table()
    near = float(table.at(torch.tensor(2.0e3)))
    far = float(table.at(torch.tensor(20.0e3)))

    assert near > 10.0 * far


def _saturation_events(saturation, offset_hz, delay_s=0.3):
    """Saturate the bound pool, wait, then read the free pool's ``Z``.

    A hard pulse tips the free water and saturates the bound pool; the spoiler
    after it removes the transverse magnetization the pulse created, so what
    survives the delay is longitudinal alone. The 90 degrees that follows
    writes it into the transverse plane, and the ADC records it -- which makes
    the reading proportional to ``Z_a``, unlike an echo train's magnitudes,
    which mix pathways and are not monotone in it.
    """
    from torchsim.sequence._accelerators import (
        _EXCITATION,
        _RECORD,
        _SPOIL_AFTER,
    )

    return (
        torch.tensor([0.0, delay_s, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1, 0, 1, 2], dtype=torch.int32),
        torch.tensor(
            [0.5 * torch.pi, 0.0, 0.5 * torch.pi, 0.0], dtype=torch.float32
        ),
        torch.zeros(4, dtype=torch.float32),
        torch.tensor(
            [_EXCITATION | _SPOIL_AFTER, 0, _EXCITATION, _RECORD],
            dtype=torch.uint8,
        ),
        torch.tensor([-1, -1, -1, 0], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([saturation, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([offset_hz, 0.0, 0.0, 0.0], dtype=torch.float32),
    )


def _prepared(device="cpu", **properties):
    prepared, _, _ = _prepare_tissue(
        TissueProperties(**properties), device
    )
    return tuple(value.to(torch.float32).contiguous() for value in prepared)


def _saturate_then_read(saturation, offset_hz, delay_s=0.3, **properties):
    """The reading that :func:`_saturation_events` records."""
    return float(
        _run_packed(
            _prepared(t1_ms=1000.0, t2_ms=80.0, **properties),
            _saturation_events(saturation, offset_hz, delay_s),
            STATES,
            1,
            1,
            lineshape=lineshape_table(),
        ).abs().flatten()[0]
    )


# A millisecond-long rectangular pulse deposits ``-pi / tau`` per radian
# squared, which at a 90 degree flip and on resonance saturates the pool by
# about a quarter.
MILLISECOND_PULSE = -np.pi / 1e-3


def test_saturating_the_bound_pool_drains_the_free_water():
    """What makes MT visible: the free pool recovers through a pool that the
    RF has emptied, so it recovers to less.

    The saturation reaches the kernel through the event buffer, so turning that
    buffer off with everything else held fixed isolates it from the initial
    split and from the exchange.
    """
    pools = dict(bound_fraction=FRACTION, bound_exchange_hz=200.0)
    quiet = _saturate_then_read(0.0, 0.0, **pools)
    driven = _saturate_then_read(MILLISECOND_PULSE, 0.0, **pools)
    harder = _saturate_then_read(4.0 * MILLISECOND_PULSE, 0.0, **pools)

    assert driven < quiet
    assert harder < driven
    assert (quiet - driven) / quiet > 1e-3


def test_saturation_falls_away_from_resonance():
    """The lineshape is why the offset is a per-voxel read rather than a
    constant: the same pulse played further off resonance deposits less, so it
    leaves more free water behind.
    """
    pools = dict(bound_fraction=FRACTION, bound_exchange_hz=200.0)
    quiet = _saturate_then_read(0.0, 0.0, **pools)
    readings = [
        _saturate_then_read(MILLISECOND_PULSE, offset, **pools)
        for offset in (0.0, 2.0e3, 20.0e3)
    ]

    assert readings[0] < readings[1] < readings[2] < quiet


def test_the_voxel_reads_the_lineshape_at_its_own_offset():
    """``df`` is the pulse's frequency less the voxel's off-resonance, so two
    voxels under one pulse absorb differently. Without that the lineshape could
    be folded into the event buffer on the host.
    """
    pools = dict(bound_fraction=FRACTION, bound_exchange_hz=200.0)
    # The pulse sits at 4 kHz; the voxel that is 4 kHz off-resonance sees it on
    # resonance and absorbs the most.
    on_resonance = _saturate_then_read(
        MILLISECOND_PULSE, 4.0e3, b0_hz=4.0e3, **pools
    )
    off_resonance = _saturate_then_read(
        MILLISECOND_PULSE, 4.0e3, b0_hz=0.0, **pools
    )

    assert on_resonance < off_resonance


def test_the_builders_idealized_pulse_declares_almost_no_power():
    """A sequence from :mod:`._builders` saturates almost nothing.

    Its RF definition exists to turn an amplitude in radians into that flip
    angle, and reaches it by declaring an envelope whose integral is
    ``1 / (2 pi)`` seconds -- a pulse a sixth of a second long. The saturation
    that shape implies is a very long, very weak pulse's, which is a fifth of a
    percent at 140 degrees. A description carrying a real waveform is what
    gives a real number; this is pinned so that a sequence which appears to
    model MT and barely does says so here rather than in a result.
    """
    definition = _description().rf_definitions[0]
    deposited = definition.saturation(rf_raster_time_s=1e-6)
    flip_rad = np.deg2rad(140.0)
    absorbed = np.exp(
        deposited * flip_rad**2 * float(lineshape_table().at(torch.tensor(0.0)))
    )

    assert 0.99 < absorbed < 0.999


# --- forward mode ---


# One tissue whose every bound-pool term is live at once: enough exchange to
# move magnetization over the delay, a bound T1 short enough that the pool it
# exchanges with matters, and a voxel far enough off resonance that the pulse's
# offset lands on a sloped part of the lineshape.
LIVE = dict(
    t1_ms=1000.0,
    t2_ms=80.0,
    bound_fraction=0.15,
    bound_exchange_hz=200.0,
    t1_bound_ms=60.0,
    b0_hz=1500.0,
)
PULSE_OFFSET_HZ = 4.0e3


def _live_readout(prepared, seed=None):
    """The saturate-then-read signal, or its directional derivative."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    events = _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)
    table = lineshape_table()
    if seed is None:
        return complex(
            _run_packed(prepared, events, STATES, 1, 1, lineshape=table)
            .flatten()[0]
        )
    return complex(
        _run_packed_jvp(
            prepared,
            events,
            seed,
            tuple(torch.zeros_like(events[0]) for _ in range(3)),
            STATES,
            1,
            1,
            lineshape=table,
        ).flatten()[0]
    )


# The probe spoils its transverse magnetization before the interval that
# matters, so T2 has nothing to act on and is not swept here; every other
# property it declares reaches the reading.
DIRECTIONS = sorted(set(LIVE) - {"t2_ms"})


@pytest.mark.parametrize("name", DIRECTIONS)
def test_forward_mode_matches_finite_differences(name: str) -> None:
    """Every direction the bound pool adds, and every one it changes.

    The step is a hundredth of the value: differencing a float32 kernel at a
    thousandth leaves the answer under the rounding of the two runs, and at a
    tenth the difference has curved away from the tangent. In between the two
    agree to a part in a thousand, which is what a float32 central difference
    can resolve.
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

    assert abs(difference) > 0.0, "the probe leaves this direction dead"
    assert abs(tangent - difference) / abs(difference) < 5e-3


def test_a_direction_along_the_bound_fraction_splits_the_equilibrium():
    """The fraction reaches the answer before a single event has run.

    Raising it takes magnetization out of the free water at once, so the
    tangent is large and negative where every other one is small. A kernel that
    seeded both pools at the primal equilibrium would still pass a
    finite-difference check on the operator and fail here.
    """
    prepared = _prepared(**LIVE)
    index = TISSUE_NAMES.index("bound_fraction")
    seed = tuple(
        torch.ones_like(value) if position == index else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    )
    tangent = _live_readout(prepared, seed)
    reading = _live_readout(prepared)

    # Most of it is the split: the free pool starts at ``1 - f``, so the
    # reading is nearly proportional to it.
    assert tangent.imag < 0.0
    assert abs(tangent.imag) > 0.5 * abs(reading.imag) / (1.0 - LIVE["bound_fraction"])


def test_the_offset_is_the_pulse_less_the_voxel():
    """``b0`` reaches the bound pool only through the lineshape's slope.

    The pulse's frequency and the voxel's off-resonance enter the saturation
    only through their difference, so a direction along the voxel's
    off-resonance must be exactly minus a step in the pulse's frequency. That
    pins the convention and the slope at once: a kernel reading ``G`` flat
    would leave the left side at zero, and one taking the offset the other way
    round would flip its sign.
    """
    prepared = _prepared(**LIVE)
    index = TISSUE_NAMES.index("b0_hz")
    seed = tuple(
        torch.ones_like(value) if position == index else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    )
    tangent = _live_readout(prepared, seed)

    table = lineshape_table()
    step = 40.0

    def moved(offset_hz):
        return complex(
            _run_packed(
                prepared,
                _saturation_events(MILLISECOND_PULSE, offset_hz, delay_s=0.4),
                STATES,
                1,
                1,
                lineshape=table,
            ).flatten()[0]
        )

    through_the_pulse = (
        moved(PULSE_OFFSET_HZ + step) - moved(PULSE_OFFSET_HZ - step)
    ) / (2.0 * step)

    assert abs(tangent) > 0.0
    assert abs(tangent + through_the_pulse) / abs(through_the_pulse) < 5e-3


def test_forward_mode_leaves_the_single_pool_answer_untouched():
    """A tissue at the default bound fraction takes the single-pool kernel in
    forward mode too, so its directions are bit for bit what they were.
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
                bound_fraction=0.0,
                bound_exchange_hz=RATE_HZ,
            ),
            nstates=STATES,
        ).signal

    _, plain = torch.func.jvp(signal, (t2,), (torch.ones_like(t2),))
    _, still = torch.func.jvp(gated, (t2,), (torch.ones_like(t2),))

    assert torch.equal(plain, still)


# --- reverse mode ---


def _live_events():
    return _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)


def _live_adjoint(prepared, seed, profile=None):
    """Gradients of ``Re(<seed, y>)`` w.r.t. every differentiable input."""
    from torchsim.sequence._accelerators import _run_packed_vjp

    return _run_packed_vjp(
        prepared,
        _live_events(),
        seed,
        state_count=STATES,
        output_count=1,
        threads=1,
        profile=profile,
        lineshape=lineshape_table(),
    )


def _directions(prepared, events, generator):
    """One random direction per differentiable input, tissue then event."""
    return tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for value in (*prepared, events[0], events[2], events[3])
    )


@pytest.mark.parametrize("tabulated", [False, True])
def test_the_adjoint_is_the_transpose_of_the_forward_direction(
    tabulated: bool,
) -> None:
    """``<w, J v> == <J^T w, v>``, which no finite difference can fake.

    Both sides are exact linear algebra on the same Jacobian, so this pins the
    whole reverse sweep -- the cotangent of the closed form, of the saturation,
    and of the split the fraction starts the two pools at -- against a forward
    mode that finite differences have already checked. A sign or a factor
    dropped anywhere in the adjoint breaks it; a tolerance cannot hide it.

    Run with and without a tabulated rotation, because the transmit field
    reaches the answer through both the rotation and the power the pulse
    deposits, and the two arrive at its gradient by different routes.
    """
    from torchsim.sequence._accelerators import _run_packed_jvp

    profile = _instantaneous_table() if tabulated else None
    prepared = _prepared(**LIVE)
    events = _live_events()
    generator = torch.Generator().manual_seed(11)
    direction = _directions(prepared, events, generator)
    seed = torch.randn(
        (1, 1), generator=generator, dtype=torch.float32
    ) + 1j * torch.randn((1, 1), generator=generator, dtype=torch.float32)

    tangent = _run_packed_jvp(
        prepared,
        events,
        direction[:len(prepared)],
        direction[len(prepared):],
        STATES,
        1,
        1,
        profile=profile,
        lineshape=lineshape_table(),
    )
    forward = float((seed.conj() * tangent).real.sum())

    gradients = _live_adjoint(prepared, seed, profile)
    terms = [
        float((gradient * value).sum())
        for gradient, value in zip(gradients, direction, strict=True)
    ]
    reverse = sum(terms)

    # Measured against the size of the terms rather than of their sum, which
    # can cancel. The tolerance is two orders above the residual float32
    # leaves and two below what dropping one contribution to the transmit
    # field's gradient costs, so it is tight enough to see that happen.
    scale = sum(abs(term) for term in terms)
    assert scale > 0.0
    assert abs(forward - reverse) / scale < 1e-6


@pytest.mark.parametrize("name", DIRECTIONS)
def test_reverse_mode_matches_finite_differences(name: str) -> None:
    """The adjoint against the same central differences forward mode took.

    The transpose identity says the two modes agree with each other; this says
    they agree with the simulator, which is what makes a wrong forward
    unable to certify a wrong reverse.
    """
    prepared = _prepared(**LIVE)
    # A seed of ``1 + i`` makes the loss ``Re(y) + Im(y)``, so the check does
    # not depend on which quadrature the probe happens to read in.
    seed = torch.full((1, 1), 1.0 + 1.0j, dtype=torch.complex64)
    gradient = float(_live_adjoint(prepared, seed)[TISSUE_NAMES.index(name)])

    step = abs(LIVE[name]) * 1e-2
    forward = _live_readout(_prepared(**{**LIVE, name: LIVE[name] + step}))
    backward = _live_readout(_prepared(**{**LIVE, name: LIVE[name] - step}))
    moved = (forward - backward) / (2.0 * step)
    difference = moved.real + moved.imag

    assert abs(difference) > 0.0, "the probe leaves this direction dead"
    assert abs(gradient - difference) / abs(difference) < 5e-3


def test_an_adjoint_of_a_bound_pool_run_reaches_the_public_api():
    """What an optimizer fitting a bound fraction actually calls."""
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (
            ("t1_ms", 1000.0), ("t2_ms", 80.0), ("bound_fraction", FRACTION),
            ("bound_exchange_hz", RATE_HZ), ("t1_bound_ms", 200.0),
        )
    }
    signal = FSE().simulate(
        _description(), TissueProperties(**leaves), nstates=STATES
    ).signal
    signal.abs().square().sum().backward()

    for name in leaves:
        assert float(leaves[name].grad.abs().max()) > 0.0, name


def test_the_second_order_pass_carries_the_bound_pool():
    """A Hessian-vector product along the bound fraction.

    Stepping the fraction and differencing the whole adjoint gives an oracle
    that shares no code with the forward-over-reverse kernel. The row is
    checked contracted against a random direction as well as at its diagonal,
    so an error in any entry of it shows up rather than only in the one the
    bound pool is most obviously responsible for.
    """
    from torchsim.sequence._accelerators import _NativeEpg

    prepared = _prepared(**LIVE)
    events = _live_events()
    index = TISSUE_NAMES.index("bound_fraction")
    seed = torch.full((1, 1), 1.0 + 1.0j, dtype=torch.complex64)

    def moved(fraction):
        return tuple(
            fraction if position == index else value
            for position, value in enumerate(prepared)
        )

    step = LIVE["bound_fraction"] * 1e-2
    base = prepared[index]
    expected = tuple(
        (ahead - behind) / (2.0 * step)
        for ahead, behind in zip(
            _live_adjoint(moved(base + step), seed),
            _live_adjoint(moved(base - step), seed),
            strict=True,
        )
    )

    leaves = tuple(
        value.detach().clone().requires_grad_(True) for value in prepared
    )
    signal = _NativeEpg.apply(
        *leaves, *events, STATES, 1, 1, NO_GEOMETRY, None, None, None,
        lineshape_table(), False, None
    )
    gradients = torch.autograd.grad(signal, leaves, seed, create_graph=True)
    generator = torch.Generator().manual_seed(23)
    weights = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for value in gradients
    )
    (contracted,) = torch.autograd.grad(
        gradients, leaves[index], weights, retain_graph=True
    )
    (diagonal,) = torch.autograd.grad(gradients[index], leaves[index])

    reference = sum(
        float((value * weight).sum())
        for value, weight in zip(expected[:len(leaves)], weights, strict=True)
    )
    assert abs(reference) > 0.0
    assert abs(float(contracted) - reference) / abs(reference) < 1e-2
    assert float(expected[index]) != 0.0
    assert abs(float(diagonal) - float(expected[index])) / abs(
        float(expected[index])
    ) < 1e-2


def test_forward_mode_reaches_a_bound_pool_through_the_public_api():
    """The path an optimizer takes, rather than the packed buffers directly."""
    fraction = torch.tensor([FRACTION])

    def signal(value):
        return FSE().simulate(
            _description(),
            TissueProperties(
                t1_ms=torch.tensor([1000.0]),
                t2_ms=torch.tensor([80.0]),
                bound_fraction=value,
                bound_exchange_hz=RATE_HZ,
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


def test_the_single_pool_gradient_still_reaches_every_property():
    """A tissue that never mentions the bound pool runs the single-pool machine,
    and no gradient is disturbed by the pool's properties being in the list.
    """
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (("t1_ms", 1000.0), ("t2_ms", 80.0))
    }
    signal = FSE().simulate(
        _description(),
        TissueProperties(**leaves, bound_fraction=0.0, bound_exchange_hz=0.0),
        nstates=STATES,
    ).signal
    signal.abs().square().sum().backward()

    assert float(leaves["t1_ms"].grad.abs().max()) > 0.0
    assert float(leaves["t2_ms"].grad.abs().max()) > 0.0


def test_an_empty_pool_asked_for_its_gradient_gives_the_true_one():
    """Sitting at the identity is not the same as having no derivative there.

    A fraction of zero is where a bound-pool fit starts, and the signal moves
    as it leaves: dropping the pool because its fraction is at the identity
    would answer that fit with a zero that is not the derivative. So a value
    carrying a gradient keeps its term, and what comes back is checked against
    a difference the state machine did not produce.

    The rate and the relaxation time describe the pool rather than gate it, and
    an empty pool genuinely does not depend on either. Their zeros are checked
    against the gate's own gradient rather than against nothing: the pool is
    carried here, so those are zeros the kernel worked out in float32 rather
    than rows it never touched.
    """

    def loss(fraction):
        signal = FSE().simulate(
            _description(),
            TissueProperties(
                t1_ms=1000.0, t2_ms=80.0, bound_fraction=fraction,
                bound_exchange_hz=25.0, t1_bound_ms=400.0,
            ),
            nstates=STATES,
        ).signal
        return float(signal.abs().square().sum())

    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (
            ("bound_fraction", 0.0), ("bound_exchange_hz", 25.0),
            ("t1_bound_ms", 400.0),
        )
    }
    signal = FSE().simulate(
        _description(),
        TissueProperties(t1_ms=1000.0, t2_ms=80.0, **leaves),
        nstates=STATES,
    ).signal
    signal.abs().square().sum().backward()

    # One-sided: a fraction is not defined below zero. The step is kept well
    # clear of where float32 cancellation swamps the difference -- below about
    # 1e-4 the quotient walks away from the derivative rather than toward it.
    step = 1e-3
    expected = (loss(step) - loss(0.0)) / step
    assert abs(expected) > 1.0
    measured = float(leaves["bound_fraction"].grad)
    assert abs(measured - expected) / abs(expected) < 2e-2, (measured, expected)

    for name in ("bound_exchange_hz", "t1_bound_ms"):
        assert float(leaves[name].grad.abs().max()) < 1e-6 * abs(measured)


# --- sizing the table to what the sequence asks of it ---


def _tuned(offset_hz: float):
    """The echo train with every pulse driven this far off centre."""
    from dataclasses import replace

    description = _description()
    return replace(
        description,
        events=tuple(
            replace(event, params=(*event.params[:4], offset_hz, *event.params[5:]))
            if event.type is EventType.RF
            else event
            for event in description.events
        ),
    )


def test_the_offset_a_sequence_drives_its_pulses_to_is_read_off_the_events():
    from torchsim.sequence._accelerators import largest_pulse_offset

    assert largest_pulse_offset(_description()) == 0.0
    assert largest_pulse_offset(_tuned(-6.0e3)) == 6.0e3


def test_a_table_stopping_short_cannot_tell_two_far_pulses_apart():
    """What the sizing is for.

    The lineshape falls steeply, so a read past the last knot returns the
    value at the table's edge -- which saturates the bound pool by orders of
    magnitude more than the pulse actually does. A default-sized table gives
    a pulse at 60 kHz and one at 90 kHz the same answer; one sized to reach
    them does not.
    """
    from torchsim.sequence._lineshape import lineshape_reaching

    near, far = torch.tensor(60.0e3), torch.tensor(90.0e3)
    stops_short = lineshape_table()
    assert float(stops_short.at(near)) == float(stops_short.at(far))

    reaches = lineshape_reaching(90.0e3)
    assert float(reaches.at(near)) > float(reaches.at(far)) > 0.0
    assert float(reaches.at(near)) < float(stops_short.at(near))


def test_reaching_further_adds_knots_rather_than_spreading_them():
    """A sequence played further out gets a longer table, not a coarser one."""
    from torchsim.sequence._lineshape import lineshape_reaching

    default = lineshape_table()
    stretched = lineshape_reaching(90.0e3)

    assert stretched.offset_max_hz >= 90.0e3
    assert stretched.bins > default.bins
    assert abs(stretched.step - default.step) < 1e-9 * default.step


def test_sequences_asking_for_similar_ranges_share_one_table():
    """The integral costs far more than a simulation does, so runs that can use
    the same table have to get the same object rather than each paying for it.

    Rounding the extent up to a whole number of default tables is what makes
    that happen for ranges that merely differ, rather than only for ones that
    match to the hertz.
    """
    from torchsim.sequence._lineshape import OFFSET_MAX_HZ, lineshape_reaching

    assert lineshape_reaching(0.0) is lineshape_reaching(30.0e3)
    assert lineshape_reaching(0.0).offset_max_hz == OFFSET_MAX_HZ
    assert lineshape_reaching(40.0e3) is lineshape_reaching(60.0e3)
    assert lineshape_reaching(40.0e3) is not lineshape_reaching(30.0e3)


def test_the_voxel_is_counted_beside_the_pulse():
    """The read is the pulse's frequency less the voxel's own, so a table sized
    from the sequence alone still stops short of what an off-resonant voxel
    reads.
    """
    from torchsim.sequence._simulation import _absorption_table

    still = _absorption_table(_tuned(33.0e3), torch.zeros(1), None)
    spread = _absorption_table(
        _tuned(33.0e3), torch.tensor([-8.0e3, 8.0e3]), None
    )

    assert still.offset_max_hz >= 33.0e3
    assert spread.offset_max_hz >= 41.0e3


def test_a_far_off_resonance_prep_reaches_the_public_api():
    """The whole path: a description whose pulses sit past the default table,
    simulated through the public API rather than through the packed buffers.
    """
    def signal(offset_hz):
        return FSE().simulate(
            _tuned(offset_hz),
            TissueProperties(
                t1_ms=torch.tensor([1000.0]),
                t2_ms=torch.tensor([80.0]),
                bound_fraction=FRACTION,
                bound_exchange_hz=RATE_HZ,
            ),
            nstates=STATES,
        ).signal

    assert float(signal(60.0e3).abs().max()) > 0.0
    assert not torch.equal(signal(60.0e3), signal(0.0))


# --- the bound pool beside a tabulated rotation ---


def _instantaneous_table():
    """A pulse with no gradient across it: one rotation, every position."""
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


def test_a_tabulated_pulse_leaves_the_bound_pool_reading_the_bare_flip():
    """The two pools read the pulse differently, which only a table shows.

    The free pool takes its rotation from the table; the bound pool takes the
    power the pulse deposits, which the slice-select gradient does not change.
    A table built from a pulse with no gradient across it holds the
    instantaneous rotation, so the whole run must reproduce the unprofiled one
    -- and it does only if the saturation ignored the table and squared the
    bare flip instead.
    """
    prepared = _prepared(**LIVE)
    events = _live_events()
    arguments = (prepared, events, STATES, 1, 1)
    table = lineshape_table()

    plain = _run_packed(*arguments, lineshape=table)
    tabulated = _run_packed(
        *arguments, lineshape=table, profile=_instantaneous_table()
    )

    assert float(plain.abs().max()) > 0.0
    assert float((plain - tabulated).abs().max() / plain.abs().max()) < 1e-5


def test_a_tabulated_bound_pool_run_is_differentiable_both_ways():
    """The profiled and bound-pool kernels are one instantiation, so the two
    features have to be checked together rather than each beside a plain run.
    """
    from torchsim.sequence._accelerators import _run_packed_jvp, _run_packed_vjp

    prepared = _prepared(**LIVE)
    events = _live_events()
    profile = _instantaneous_table()
    table = lineshape_table()
    index = TISSUE_NAMES.index("bound_fraction")
    seed = torch.full((1, 1), 1.0 + 1.0j, dtype=torch.complex64)
    direction = tuple(
        torch.ones_like(value) if position == index else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    )

    tangent = _run_packed_jvp(
        prepared,
        events,
        direction,
        tuple(torch.zeros_like(events[0]) for _ in range(3)),
        STATES,
        1,
        1,
        profile=profile,
        lineshape=table,
    )
    forward = float((seed.conj() * tangent).real.sum())
    gradient = float(
        _run_packed_vjp(
            prepared,
            events,
            seed,
            state_count=STATES,
            output_count=1,
            threads=1,
            profile=profile,
            lineshape=table,
        )[index]
    )

    def moved(step):
        return _run_packed(
            _prepared(**{**LIVE, "bound_fraction": LIVE["bound_fraction"] + step}),
            events,
            STATES,
            1,
            1,
            profile=profile,
            lineshape=table,
        )

    step = LIVE["bound_fraction"] * 1e-2
    difference = float(
        (seed.conj() * (moved(step) - moved(-step)) / (2.0 * step)).real.sum()
    )

    assert abs(difference) > 0.0
    assert abs(forward - difference) / abs(difference) < 5e-3
    assert abs(gradient - difference) / abs(difference) < 5e-3


# --- against the state machine written out in torch ---


def _routes(leaves, events, profile=None):
    """The kernels and the oracle, fed from one place."""
    from torchsim.sequence._accelerators import _NativeEpg

    table = lineshape_table()
    fused = _NativeEpg.apply(
        *leaves, *events, STATES, 1, 1, NO_GEOMETRY, profile, None, None, table,
        False, None
    )
    reference = simulate_packed(
        leaves,
        events,
        state_count=STATES,
        output_count=1,
        profile=profile,
        lineshape=table,
    )
    return fused, reference


def _live_leaves():
    return tuple(
        value.detach().clone().requires_grad_(True)
        for value in _prepared(**LIVE)
    )


@pytest.mark.parametrize("tabulated", [False, True])
def test_the_fused_forward_matches_the_oracle(tabulated: bool):
    """The oracle reaches its exponential through Pade rather than through the
    closed form, so this is the two-pool step checked against a different
    algorithm and not against a second copy of itself.

    Run with and without a tabulated rotation, because the two pools read a
    pulse differently and only a table separates the readings.
    """
    profile = _instantaneous_table() if tabulated else None
    fused, reference = _routes(_live_leaves(), _live_events(), profile)

    reference = reference.detach()
    assert float(reference.abs().max()) > 0.0
    assert float(
        (fused.detach() - reference).abs().max() / reference.abs().max()
    ) < 1e-4


@pytest.mark.parametrize("tabulated", [False, True])
def test_the_fused_forward_mode_matches_the_oracle(tabulated: bool):
    """A direction followed through both routes at once."""
    profile = _instantaneous_table() if tabulated else None
    events = _live_events()
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
    """
    profile = _instantaneous_table() if tabulated else None
    seed = torch.full((1, 1), 1.0 + 1.0j, dtype=torch.complex64)
    generator = torch.Generator().manual_seed(31)
    events = _live_events()
    # The direction the second pass follows is drawn once, so the two routes
    # are contracted against the same one rather than each against its own.
    direction = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float32)
        for value in _live_leaves()
    )

    def gradients(route: int, order: int):
        leaves = _live_leaves()
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
        _agree(gradients(1, order), gradients(0, order), tolerance=1e-3)


# --- the other backends ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_kernel_matches_the_cpu_kernel():
    """The two share no code, so agreement is what keeps the second pool honest
    on the card: a ``Z`` step written for one pool there would still produce a
    plausible train.
    """
    packed = _pack_events(
        "fse",
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, 6),
        t2_ms=torch.linspace(40.0, 120.0, 6),
        b0_hz=torch.linspace(-200.0, 200.0, 6),
        bound_fraction=torch.linspace(0.02, 0.25, 6),
        bound_exchange_hz=torch.linspace(5.0, 80.0, 6),
        t1_bound_ms=torch.linspace(200.0, 900.0, 6),
    )

    def run(device):
        prepared, _, _ = _prepare_tissue(tissue, device)
        prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
        events = tuple(value.to(device) for value in packed.buffers)
        return _run_packed(
            prepared,
            events,
            STATES,
            packed.output_count,
            1,
            lineshape=lineshape_table(device=device),
        )

    host = run("cpu")
    card = run("cuda").cpu()

    assert float((host - card).abs().max() / host.abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_forward_mode_matches_the_cpu_kernel():
    """The tangent of the two-pool step, and of the saturation, on the card."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    voxels = 6
    spread = dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-2000.0, 2000.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.25, voxels),
        bound_exchange_hz=torch.linspace(5.0, 200.0, voxels),
        t1_bound_ms=torch.linspace(50.0, 900.0, voxels),
    )
    events = _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)

    def run(device):
        prepared = _prepared(device, **spread)
        seed = tuple(torch.ones_like(value) for value in prepared)
        return _run_packed_jvp(
            prepared,
            tuple(value.to(device) for value in events),
            seed,
            tuple(torch.zeros_like(events[0]).to(device) for _ in range(3)),
            STATES,
            1,
            1,
            lineshape=lineshape_table(device=device),
        )

    host = run("cpu")
    card = run("cuda").cpu()

    assert float((host - card).abs().max() / host.abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_volume_matches_the_whole_one():
    """Streaming cuts the voxel axis, which the bound pool's buffers follow."""
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
        bound_fraction=torch.linspace(0.0, 0.25, voxels),
        bound_exchange_hz=torch.linspace(5.0, 80.0, voxels),
    )
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    arguments = (prepared, packed.buffers, STATES, packed.output_count, 1)
    table = lineshape_table()

    whole = _run_packed(*arguments, lineshape=table)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed(*arguments, lineshape=table)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_forward_mode_matches_the_whole_one():
    """Streaming cuts the voxel axis, which the bound pool's seeds follow."""
    from torchsim.sequence._accelerators import _run_packed_jvp, offload

    voxels = 3000
    prepared = _prepared(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-2000.0, 2000.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.25, voxels),
        bound_exchange_hz=torch.linspace(5.0, 200.0, voxels),
        t1_bound_ms=torch.linspace(50.0, 900.0, voxels),
    )
    events = _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)
    seed = tuple(torch.ones_like(value) for value in prepared)
    arguments = (
        prepared,
        events,
        seed,
        tuple(torch.zeros_like(events[0]) for _ in range(3)),
        STATES,
        1,
        1,
    )
    table = lineshape_table()

    whole = _run_packed_jvp(*arguments, lineshape=table)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed_jvp(*arguments, lineshape=table)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 1e-5


def _spread(voxels: int) -> dict:
    """A tissue whose every bound-pool term varies across the voxels."""
    return dict(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-2000.0, 2000.0, voxels),
        bound_fraction=torch.linspace(0.02, 0.25, voxels),
        bound_exchange_hz=torch.linspace(5.0, 200.0, voxels),
        t1_bound_ms=torch.linspace(50.0, 900.0, voxels),
    )


def _agree(host, card, tolerance: float = 1e-4) -> None:
    """Compare gradient tuples, skipping what float32 cannot resolve.

    These span many orders of magnitude, and an entry far below the largest is
    under the rounding of the sums that produced it.
    """
    floor = 1e-6 * max(float(value.abs().max()) for value in host)
    compared = 0
    for index, (want, got) in enumerate(zip(host, card, strict=True)):
        scale = float(want.abs().max())
        if scale <= floor:
            continue
        assert float((want - got).abs().max()) / scale < tolerance, index
        compared += 1
    assert compared > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_adjoint_matches_the_cpu_kernel():
    """The reverse sweep is the half a forward parity check cannot see.

    A card running the single-pool adjoint against a two-pool forward agrees on
    the signal and disagrees on every gradient, including the ones that have
    nothing to do with the bound pool.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp

    voxels = 6
    events = _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)
    seed = torch.full((voxels, 1), 1.0 + 1.0j, dtype=torch.complex64)

    def run(device):
        return _run_packed_vjp(
            _prepared(device, **_spread(voxels)),
            tuple(value.to(device) for value in events),
            seed.to(device),
            state_count=STATES,
            output_count=1,
            threads=1,
            lineshape=lineshape_table(device=device),
        )

    _agree(run("cpu"), tuple(value.cpu() for value in run("cuda")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_second_order_pass_matches_the_cpu_kernel():
    """Forward-over-reverse, where a direction rides through the adjoint."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    voxels = 6
    events = _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)
    seed = torch.full((voxels, 1), 1.0 + 1.0j, dtype=torch.complex64)

    def run(device):
        prepared = _prepared(device, **_spread(voxels))
        directions = (
            *(torch.ones_like(value) for value in prepared),
            *(
                torch.zeros_like(events[index]).to(device)
                for index in (0, 2, 3)
            ),
        )
        return _run_packed_vjp_jvp(
            prepared,
            tuple(value.to(device) for value in events),
            directions,
            seed.to(device),
            state_count=STATES,
            output_count=1,
            threads=1,
            lineshape=lineshape_table(device=device),
        )

    host = run("cpu")
    card = run("cuda")
    for side, plane in enumerate(("curvature", "adjoint")):
        _agree(host[side], tuple(value.cpu() for value in card[side]))
        assert plane


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_adjoint_matches_the_whole_one():
    """Streaming cuts the voxel axis; a tissue gradient follows it and an event
    gradient collects a contribution from every chunk.

    The forward-over-reverse pass is what streams -- an adjoint asked for on
    its own is that pass given no direction to follow -- so it is the one the
    chunked buffers have to carry the second pool through.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp, offload

    voxels = 3000
    prepared = _prepared(**_spread(voxels))
    events = _saturation_events(MILLISECOND_PULSE, PULSE_OFFSET_HZ, delay_s=0.4)
    seed = torch.full((voxels, 1), 1.0 + 1.0j, dtype=torch.complex64)
    directions = (
        *(torch.ones_like(value) for value in prepared),
        *(torch.zeros_like(events[index]) for index in (0, 2, 3)),
    )
    table = lineshape_table()

    def run():
        return _run_packed_vjp_jvp(
            prepared,
            events,
            directions,
            seed,
            state_count=STATES,
            output_count=1,
            threads=1,
            lineshape=table,
        )

    whole = run()
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = run()

    for side in range(2):
        _agree(whole[side], streamed[side])


def _volume(voxels: int, **leaves) -> TissueProperties:
    """A tissue spread over a volume too large to hold on the card at once."""
    return TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b0_hz=torch.linspace(-300.0, 300.0, voxels),
        **leaves,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_streamed_public_simulation_matches_the_whole_one():
    """The path a user takes: a volume simulated through the public API with
    the devices given less memory than it needs, so it arrives in chunks.
    """
    from torchsim.sequence._accelerators import offload

    voxels = 3000
    tissue = _volume(
        voxels,
        bound_fraction=torch.linspace(0.02, 0.25, voxels),
        bound_exchange_hz=torch.linspace(5.0, 80.0, voxels),
        t1_bound_ms=torch.linspace(50.0, 900.0, voxels),
    )

    def run():
        return FSE().simulate(_description(), tissue, nstates=STATES).signal

    whole = run()
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = run()

    assert float(whole.abs().max()) > 0.0
    assert float((whole - streamed).abs().max() / whole.abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_gradient_taken_after_a_streamed_forward_matches_the_whole_one():
    """Fitting a bound fraction over a volume is the reason the adjoint exists.

    An offload plan reaches the forward, which arrives in chunks; the adjoint
    replays the trajectory from the inputs rather than from anything the
    forward left, so the two halves have to agree across that boundary.
    """
    from torchsim.sequence._accelerators import offload

    voxels = 3000

    def gradient():
        leaves = {
            name: value.clone().requires_grad_(True)
            for name, value in (
                ("bound_fraction", torch.linspace(0.02, 0.25, voxels)),
                ("bound_exchange_hz", torch.linspace(5.0, 80.0, voxels)),
                ("t1_bound_ms", torch.linspace(50.0, 900.0, voxels)),
            )
        }
        signal = FSE().simulate(
            _description(), _volume(voxels, **leaves), nstates=STATES
        ).signal
        signal.abs().square().sum().backward()
        return tuple(leaves[name].grad for name in leaves)

    whole = gradient()
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = gradient()

    _agree(whole, streamed, tolerance=1e-3)
