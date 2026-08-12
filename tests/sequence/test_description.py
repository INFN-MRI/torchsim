"""Tests for sequence-description objects and builders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim.sequence import (
    AdcRole,
    EventType,
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
