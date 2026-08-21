"""Tests for sequence-description objects and builders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim.sequence import (
    AdcRole,
    EventType,
    RfDefinition,
    RfMode,
    RfShape,
    RfUse,
    SequenceEvent,
    decompress_shape,
    fse_description,
    mpnrage_description,
    mprage_description,
    mrf_description,
    spgr_description,
)


def test_event_constructors_keep_seqdesc_field_order() -> None:
    event = SequenceEvent.rf(10.0, 4, RfUse.REFOCUSING, 2.0, 0.5)

    assert event.type is EventType.RF
    assert event.rf_definition_id == 4
    assert event.rf_use is RfUse.REFOCUSING
    assert event.rf_amplitude_hz == 2.0
    assert event.rf_phase_rad == 0.5


def test_unit_definition_maps_amplitude_to_flip_angle() -> None:
    flip = torch.tensor(1.2, requires_grad=True)
    description = fse_description(flip[None], 5e-3)

    actual, phase = description.rf_definitions[0].flip_angle(flip)

    torch.testing.assert_close(actual, flip)
    torch.testing.assert_close(phase, torch.zeros_like(phase))
    actual.backward()
    torch.testing.assert_close(flip.grad, torch.ones_like(flip))


def test_builders_emit_monotonic_state_machines() -> None:
    descriptions = (
        fse_description(torch.tensor([1.0, 1.1]), 5e-3),
        mrf_description(torch.tensor([0.1, 0.2]), 10e-3, inversion_time_s=20e-3),
        spgr_description(torch.tensor([0.1, 0.2]), 10e-3, 4e-3),
    )

    for description in descriptions:
        timestamps = torch.stack(
            [torch.as_tensor(event.timestamp_us) for event in description.events]
        )
        assert torch.all(timestamps[1:] >= timestamps[:-1])
        assert len(description.adc_events) == 2


def test_decompress_shape_matches_pulseq_delta_rle() -> None:
    packed = np.asarray([1.0, 1.0, 2.0, -0.5], dtype=np.float32)

    actual = decompress_shape(packed, 5)

    np.testing.assert_allclose(actual, [1.0, 2.0, 3.0, 4.0, 3.5])


def test_spgr_rejects_echo_after_repetition() -> None:
    with pytest.raises(ValueError, match="TE"):
        spgr_description(torch.tensor([0.1]), 5e-3, 6e-3)


def test_magnetization_prepared_builders_label_acquired_readouts() -> None:
    mpnrage = mpnrage_description(5, torch.tensor(0.1), 5e-3)
    mprage = mprage_description(2, 2, torch.tensor(0.1), 5e-3, 20e-3)

    assert len(mpnrage.adc_events) == 5
    assert all(event.adc_role is AdcRole.SINGLE for event in mpnrage.adc_events)
    assert len(mprage.adc_events) == 5
    assert sum(
        event.adc_role is not AdcRole.NON_ACQUIRED for event in mprage.adc_events
    ) == 1


# --- what the waveform says about the pulse ---


def _shape(samples: np.ndarray) -> RfShape:
    return RfShape(samples.size, samples.astype(np.float32))


def _sinc(samples: int = 32, lobes: float = 2.0) -> np.ndarray:
    return np.sinc(np.linspace(-lobes, lobes, samples))


def _definition(magnitude, phase=None, *, bandwidth_hz: float = 0.0) -> RfDefinition:
    return RfDefinition(
        id=7,
        bandwidth_hz=bandwidth_hz,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=0.0,
        total_b1sq_power=0.0,
        magnitude=magnitude,
        phase=phase,
    )


def test_a_rectangle_off_resonance_free_is_the_hard_pulse_it_already_was() -> None:
    """Every builder emits one of these, and it must keep taking the
    instant-rotation path rather than being tabulated off-grid.
    """
    description = fse_description(torch.tensor([1.2]), 5e-3)

    assert description.rf_definitions[0].rf_mode() is RfMode.INSTANT


def test_a_shaped_pulse_asks_for_a_profile() -> None:
    assert _definition(_shape(_sinc())).rf_mode() is RfMode.PROFILED


def test_a_rectangle_that_selects_a_slice_asks_for_a_profile() -> None:
    """Bandwidth is what makes the rotation depend on where the voxel sits, so
    a flat envelope is no longer the whole story.
    """
    flat = _shape(np.ones(8))

    assert _definition(flat).rf_mode() is RfMode.INSTANT
    assert _definition(flat, bandwidth_hz=1200.0).rf_mode() is RfMode.PROFILED


def test_a_waveform_per_channel_asks_for_a_rotation_per_voxel() -> None:
    sinc = _sinc()
    definition = _definition((_shape(sinc), _shape(0.5 * sinc)))

    assert definition.channel_count == 2
    assert definition.rf_mode() is RfMode.DYNAMIC
    assert definition.complex_envelope().shape == (2, sinc.size)


def test_the_derived_quantities_read_the_combined_envelope() -> None:
    """Two channels driving half a shape each are the one-channel pulse, so the
    flip a voxel of unit sensitivity everywhere takes is the same flip.
    """
    sinc = _sinc()
    single = _definition(_shape(sinc))
    split = _definition((_shape(0.5 * sinc), _shape(0.5 * sinc)))

    np.testing.assert_allclose(
        split.combined_envelope(), single.combined_envelope(), atol=1e-6
    )
    assert split.integral() == pytest.approx(single.integral())
    assert split.saturation() == pytest.approx(single.saturation())


def test_a_channel_count_has_one_reading() -> None:
    sinc = _shape(_sinc())

    with pytest.raises(ValueError, match="phase per channel"):
        _definition((sinc, sinc), sinc)
    with pytest.raises(ValueError, match="magnitude per channel"):
        _definition(sinc, (sinc,))
    with pytest.raises(ValueError, match="drives no channel"):
        _definition(())
