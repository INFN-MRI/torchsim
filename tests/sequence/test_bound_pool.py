"""A bound proton pool on the longitudinal axis, through the fused kernels.

Every tissue holds a share of its protons bound to macromolecules. That pool
has essentially no T2, so it never shows up in a signal directly; it reaches
one by exchanging magnetization with the free water, and by absorbing RF power
that the free water at the same offset does not. What the kernels carry is
therefore asymmetric -- ``F+`` and ``F-`` stay single-pool and only the ``Z``
step and the RF operator change -- and these tests pin each half of that.

The derivative kernels carry one pool, so asking them for the Jacobian of a
bound-pool run is refused rather than answered with the single-pool one.
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
from torchsim.sequence._description import RfDefinition, RfShape
from torchsim.sequence._lineshape import lineshape_table
from torchsim.sequence._parameters import TISSUE_NAMES
from torchsim.sequence._simulation import _prepare_tissue

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
    gated = _signal(bound_fraction=0.0, exchange_rate_hz=RATE_HZ, t1_bound_ms=400.0)

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
            bound_fraction=fraction, exchange_rate_hz=RATE_HZ
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
            bound_fraction=fraction, exchange_rate_hz=RATE_HZ
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
        exchange_rate_hz=RATE_HZ,
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
        DELAYS_S, bound_fraction=FRACTION, exchange_rate_hz=0.0, t1_bound_ms=50.0
    )
    exchanging = _inversion_recovery(
        DELAYS_S, bound_fraction=FRACTION, exchange_rate_hz=200.0, t1_bound_ms=50.0
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
        exchange_rate_hz=RATE_HZ,
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


def _saturate_then_read(saturation, offset_hz, delay_s=0.3, **properties):
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

    tissue = TissueProperties(t1_ms=1000.0, t2_ms=80.0, **properties)
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    events = (
        torch.tensor([[0.0, delay_s, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([1, 0, 1, 2], dtype=torch.int32),
        torch.tensor(
            [[0.5 * torch.pi, 0.0, 0.5 * torch.pi, 0.0]], dtype=torch.float32
        ),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.tensor(
            [_EXCITATION | _SPOIL_AFTER, 0, _EXCITATION, _RECORD],
            dtype=torch.uint8,
        ),
        torch.tensor([-1, -1, -1, 0], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([saturation, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([offset_hz, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    return float(
        _run_packed(
            prepared, events, STATES, 1, 1, lineshape=lineshape_table()
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
    pools = dict(bound_fraction=FRACTION, exchange_rate_hz=200.0)
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
    pools = dict(bound_fraction=FRACTION, exchange_rate_hz=200.0)
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
    pools = dict(bound_fraction=FRACTION, exchange_rate_hz=200.0)
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


# --- what the derivative kernels will not do yet ---


def test_a_derivative_of_a_bound_pool_run_is_refused():
    """The derivative kernels carry one pool.

    Answering a two-pool forward with the single-pool Jacobian would be a wrong
    number rather than a missing one, so it is refused instead.
    """
    t2 = torch.tensor([80.0], requires_grad=True)
    signal = FSE().simulate(
        _description(),
        TissueProperties(
            t1_ms=torch.tensor([1000.0]),
            t2_ms=t2,
            bound_fraction=FRACTION,
            exchange_rate_hz=RATE_HZ,
        ),
        nstates=STATES,
    ).signal

    with pytest.raises(NotImplementedError, match="carry one pool"):
        signal.abs().square().sum().backward()


def test_the_single_pool_gradient_still_reaches_every_property():
    """The bound pool's properties are arguments the single-pool machine does
    not take, so their gradients come back at zero -- and no other gradient is
    disturbed by their being in the list.
    """
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (
            ("t1_ms", 1000.0), ("t2_ms", 80.0), ("bound_fraction", 0.0),
            ("exchange_rate_hz", 0.0), ("t1_bound_ms", 1000.0),
        )
    }
    signal = FSE().simulate(
        _description(), TissueProperties(**leaves), nstates=STATES
    ).signal
    signal.abs().square().sum().backward()

    assert float(leaves["t1_ms"].grad.abs().max()) > 0.0
    assert float(leaves["t2_ms"].grad.abs().max()) > 0.0
    for name in ("bound_fraction", "exchange_rate_hz", "t1_bound_ms"):
        assert float(leaves[name].grad.abs().max()) == 0.0


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
        exchange_rate_hz=torch.linspace(5.0, 80.0, 6),
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
        exchange_rate_hz=torch.linspace(5.0, 80.0, voxels),
    )
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    arguments = (prepared, packed.buffers, STATES, packed.output_count, 1)
    table = lineshape_table()

    whole = _run_packed(*arguments, lineshape=table)
    with offload(["cuda"], budget_bytes=1 << 20):
        streamed = _run_packed(*arguments, lineshape=table)

    assert float((whole - streamed).abs().max() / whole.abs().max()) < 1e-5
