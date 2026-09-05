"""A pulse built from the envelope a scanner plays.

The contract is one number: an event's amplitude, written in radians, is the
flip that event turns. That holds only if the envelope is scaled so its
integral is ``1 / (2 pi)`` seconds, and a pulse handed over peak-normalized and
left that way turns a fraction of a degree instead. So the assertions here are
absolute -- against the hard pulse the same flip describes -- rather than
relative to another run of the same shaped pulse, which two nothings would
satisfy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchsim import (
    rf_definition,
)
from torchsim.sequence import (
    EpgEngine,
    RfMode,
    exact_slice_profile,
    fse_description,
)
from torchsim.sequence._simulation import TissueProperties

ECHOES = 6
STATES = 10


def _windowed_sinc(samples: int = 512, cycles: float = 2.0) -> np.ndarray:
    """A Hamming-windowed sinc, in whatever amplitude units it comes in.

    Deliberately not normalized: what this test is about is that the scaling a
    caller happens to hand over does not reach the answer.
    """
    grid = np.linspace(-cycles, cycles, samples)
    envelope = np.sinc(grid) * np.hamming(samples)
    return (137.0 * envelope).astype(np.complex128)


def _describe(definition=None):
    description = fse_description(
        torch.deg2rad(torch.full((ECHOES,), 150.0)),
        echo_spacing_s=5e-3,
        phases_rad=math.pi / 2,
        excitation_phase_rad=math.pi / 2,
    )
    if definition is None:
        return description
    from dataclasses import replace

    return replace(description, rf_definitions={definition.id: definition})


def _tissue():
    return TissueProperties(
        t1_ms=torch.tensor([830.0, 1400.0]), t2_ms=torch.tensor([80.0, 110.0])
    )


def _signal(description, profile=None):
    return (
        EpgEngine()
        .simulate(description, _tissue(), across_slice=profile, nstates=STATES)
        .signal
    )


@pytest.mark.parametrize("samples", [2, 8, 64, 512])
def test_a_flat_envelope_is_the_hard_pulse_it_describes(samples: int) -> None:
    """A rectangle selecting nothing turns exactly what an ideal pulse turns.

    Whatever it is sampled at: the scale the envelope arrives in and the number
    of points it arrives on are both the caller's business and neither reaches
    the flip.
    """
    flat = rf_definition(np.full(samples, 7.5, dtype=np.complex128), dwell_s=1e-6)
    hard = _signal(_describe())
    shaped = _signal(_describe(flat))

    assert (shaped - hard).abs().max() < 1e-6 * hard.abs().max()


@pytest.mark.parametrize("flip_rad", [0.25, 1.0, math.pi / 2, math.pi])
def test_an_amplitude_in_radians_is_the_flip_it_turns(flip_rad: float) -> None:
    """The whole of what the scaling is for."""
    pulse = rf_definition(_windowed_sinc(), dwell_s=2e-6)
    turned, _phase = pulse.flip_angle(flip_rad)

    assert abs(float(np.real(turned)) - flip_rad) < 1e-5


def test_the_slice_centre_turns_what_a_hard_pulse_turns() -> None:
    """A selective pulse is only selective away from the middle of its slice.

    At the centre it performs the rotation its nominal flip names, so a
    simulation sampling the centre alone has to land on the hard-pulse answer
    -- which is an absolute anchor rather than a comparison against itself.
    """
    selective = rf_definition(_windowed_sinc(), dwell_s=2e-6, bandwidth_hz=2000.0)
    hard = _signal(_describe())
    centre = _signal(_describe(selective), exact_slice_profile(torch.tensor([0.0])))

    assert (centre - hard).abs().max() < 1e-3 * hard.abs().max()


def test_integrating_across_the_slice_costs_signal() -> None:
    """Away from the centre the pulse turns less, so the average is smaller.

    This is the effect a slice profile exists to carry, and it is stated as a
    fraction of the hard-pulse answer so that a shaped pulse quietly turning
    nothing could not pass it.
    """
    selective = rf_definition(_windowed_sinc(), dwell_s=2e-6, bandwidth_hz=2000.0)
    hard = _signal(_describe()).abs().max()
    across = _signal(_describe(selective), exact_slice_profile(21, extent=1.2))

    assert across.abs().max() < 0.9 * hard
    assert across.abs().max() > 0.01 * hard


def test_a_pulse_given_per_channel_is_scaled_against_the_whole_drive() -> None:
    """The flip is what the channels turn together, not what one of them does.

    A drive split evenly over two channels is the same pulse and is scaled to
    the same flip. It also asks for the per-voxel rotation, since what a voxel
    sees then depends on the transmit field it sits in rather than on one
    complex number the whole slice shares.
    """
    envelope = _windowed_sinc()
    one = rf_definition(envelope, dwell_s=2e-6, bandwidth_hz=2000.0)
    two = rf_definition(
        np.stack([0.5 * envelope, 0.5 * envelope]), dwell_s=2e-6, bandwidth_hz=2000.0
    )

    assert one.channel_count == 1
    assert two.channel_count == 2
    assert two.rf_mode() is RfMode.DYNAMIC
    turned, _phase = two.flip_angle(math.pi / 2)
    assert abs(float(np.real(turned)) - math.pi / 2) < 1e-5


@pytest.mark.parametrize(
    "envelope, dwell",
    [
        (np.zeros(8, dtype=np.complex128), 1e-6),
        (np.ones(0, dtype=np.complex128), 1e-6),
        (np.ones(8, dtype=np.complex128), 0.0),
    ],
)
def test_an_envelope_with_no_flip_in_it_is_refused(envelope, dwell) -> None:
    """Silently turning nothing is the failure this is written against."""
    with pytest.raises(ValueError):
        rf_definition(envelope, dwell_s=dwell)


@pytest.mark.parametrize(
    "dwell_s, raster_s", [(1e-6, 1e-6), (4e-6, 1e-6), (1e-6, 2e-7)]
)
def test_the_dwell_need_not_be_the_raster(dwell_s: float, raster_s: float) -> None:
    """Sample times are carried in raster units, so the two are independent.

    A designer hands over a pulse on its own dwell and a run reads it back
    against the raster it uses. Getting that conversion wrong scales the
    integral, and the flip with it, so this is the same anchor as above read
    through a pulse that is not on the raster.
    """
    pulse = rf_definition(_windowed_sinc(), dwell_s=dwell_s, rf_raster_time_s=raster_s)
    turned, _phase = pulse.flip_angle(math.pi / 2, rf_raster_time_s=raster_s)

    assert abs(float(np.real(turned)) - math.pi / 2) < 1e-5


def test_a_pulse_carries_at_most_the_bands_a_description_has_room_for() -> None:
    """Eight, which is what the stream reserves."""
    envelope = _windowed_sinc()
    eight = rf_definition(
        envelope, dwell_s=1e-6, band_frequency_offsets_hz=tuple(range(8))
    )

    assert eight.num_bands == 8
    with pytest.raises(ValueError):
        rf_definition(envelope, dwell_s=1e-6, band_frequency_offsets_hz=tuple(range(9)))
