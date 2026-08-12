"""Detection of sequences whose states never leave a real subspace.

Each case pairs the predicate against the signal it describes, so the rule
cannot drift away from the behaviour it is meant to predict.
"""

import pytest
import torch

from torchsim import FSE, fse_description
from torchsim.sequence._accelerators import _pack_events, real_subspace_axis
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

ECHO_SPACING_S = 5e-3
ECHOES = 10


def _flip() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.deg2rad(80.0 + 80.0 * torch.rand(3, ECHOES, generator=generator))


def _tissue(b0_hz: float, b1_phase_rad: float) -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        b0_hz=torch.tensor([b0_hz, b0_hz]),
        b1_phase_rad=torch.tensor([b1_phase_rad, b1_phase_rad]),
    )


def _axis(phases, excitation, b0_hz=0.0, b1_phase_rad=0.0):
    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=phases,
        excitation_phase_rad=excitation,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )
    prepared, _, _ = _prepare_tissue(_tissue(b0_hz, b1_phase_rad), "cpu")
    signal = FSE().simulate(description, _tissue(b0_hz, b1_phase_rad)).signal
    return real_subspace_axis(events, prepared), signal


@pytest.mark.parametrize("phase", [0.0, torch.pi / 4, torch.pi / 2])
def test_uniform_rf_phase_stays_imaginary(phase):
    axis, signal = _axis(phase, phase)
    assert axis == 1
    # The predicate claims it; the signal must show it, at every echo.
    assert signal.real.abs().max() < 1e-12 * signal.imag.abs().max().clamp_min(1e-30)


def test_transmit_phase_breaks_the_subspace():
    axis, signal = _axis(torch.pi / 2, torch.pi / 2, b1_phase_rad=0.3)
    assert axis is None
    assert signal.real.abs().max() > 0.1 * signal.imag.abs().max()


def test_alternating_refocusing_phase_breaks_the_subspace():
    phases = torch.full((ECHOES,), torch.pi / 2)
    phases[::2] = 0.0
    axis, signal = _axis(phases, torch.pi / 2)
    assert axis is None
    assert signal.real.abs().max() > 0.1 * signal.imag.abs().max()


def test_off_resonance_is_rejected_despite_real_looking_echoes():
    """The echoes refocus off-resonance; the states in between do not."""
    axis, signal = _axis(torch.pi / 2, torch.pi / 2, b0_hz=20.0)
    assert axis is None
    # The recorded samples alone would wrongly suggest the subspace holds.
    assert signal.real.abs().max() < 1e-5 * signal.imag.abs().max()


def test_quarter_turn_excitation_stays_real():
    """CPMG: excitation a quarter turn from the refocusing pulses."""
    axis, signal = _axis(torch.pi / 2, 0.0)
    assert axis == 0
    assert signal.imag.abs().max() < 1e-6 * signal.real.abs().max()


def _buffers(packed):
    return (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )


@pytest.mark.parametrize("phase", [0.0, torch.pi / 4, torch.pi / 2])
def test_real_kernel_reproduces_the_complex_one(phase):
    """The real kernel is an optimization, not an approximation."""
    from torchsim.sequence._accelerators import _run_packed

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=phase,
        excitation_phase_rad=phase,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    complex_signal = _run_packed(prepared, events, 10, packed.output_count, 1)
    real_signal = _run_packed(
        prepared, events, 10, packed.output_count, 1, real_axis=1
    )
    scale = complex_signal.abs().max()
    assert ((complex_signal - real_signal).abs().max() / scale) < 1e-6


def test_real_jvp_reproduces_the_complex_one():
    """Forward mode along T2 stays inside the subspace, so the kernels agree."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    event_seed = (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    arguments = (prepared, events, t2_seed, event_seed, 10, packed.output_count, 1)
    expected = _run_packed_jvp(*arguments)
    actual = _run_packed_jvp(*arguments, real_axis=1)
    scale = expected.abs().max()
    assert ((expected - actual).abs().max() / scale) < 1e-6


def test_real_second_order_kernel_reproduces_the_complex_one():
    """Forward-over-reverse agrees on every gradient the subspace contains."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    tangents = (
        *t2_seed,
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    torch.manual_seed(0)
    seed = torch.randn(
        (events[2].shape[0], prepared[0].numel(), packed.output_count),
        dtype=torch.complex64,
    )
    arguments = dict(
        state_count=10, output_count=packed.output_count, threads=1
    )
    expected, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, **arguments
    )
    actual, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, real_axis=1, **arguments
    )

    # t1, t2, m0, b1, inversion efficiency, duration, flip.
    for index in (0, 1, 2, 3, 6, 7, 8):
        scale = expected[index].abs().max()
        if scale == 0:
            assert actual[index].abs().max() == 0
            continue
        assert ((expected[index] - actual[index]).abs().max() / scale) < 1e-5

    # b1_phase, b0 and RF phase leave the subspace and are not produced.
    for index in (4, 5, 9):
        assert actual[index].abs().max() == 0


def _packed(trains):
    generator = torch.Generator().manual_seed(0)
    flip = torch.deg2rad(
        80.0 + 80.0 * torch.rand(trains, ECHOES, generator=generator)
    )
    description = fse_description(
        flip,
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    return _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )


# The real kernels process a fixed-width block of trains at a time, so a train
# count that does not fill the last block leaves lanes that carry a repeat of
# the block's first train. These counts straddle that boundary.
@pytest.mark.parametrize("trains", [1, 7, 8, 9, 17])
def test_partial_train_blocks_match_the_complex_kernel(trains):
    from torchsim.sequence._accelerators import _run_packed_jvp, _run_packed_vjp_jvp

    packed = _packed(trains)
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    event_seed = (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    arguments = (prepared, events, t2_seed, event_seed, 10, packed.output_count, 1)
    expected = _run_packed_jvp(*arguments)
    actual = _run_packed_jvp(*arguments, real_axis=1)
    scale = expected.abs().max()
    assert ((expected - actual).abs().max() / scale) < 1e-6

    torch.manual_seed(0)
    seed = torch.randn(
        (trains, prepared[0].numel(), packed.output_count), dtype=torch.complex64
    )
    tangents = (*t2_seed, *event_seed)
    keywords = dict(state_count=10, output_count=packed.output_count, threads=1)
    reference, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, **keywords
    )
    result, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, real_axis=1, **keywords
    )
    # t1, t2, m0, b1 and inversion efficiency sum across the whole block, so a
    # repeated lane would show up here as a gradient counted more than once.
    for index in (0, 1, 2, 3, 6, 7, 8):
        magnitude = reference[index].abs().max()
        if magnitude == 0:
            assert result[index].abs().max() == 0
            continue
        assert ((reference[index] - result[index]).abs().max() / magnitude) < 1e-4
