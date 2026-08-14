"""Tabulating the rotation a shaped pulse actually performs.

The state machine's pulse is instantaneous and its slice profile is a flip
scaling, which is a Bloch response pretending to be proportional to the pulse
driving it. This table is the response itself, sampled over slice position and
effective flip angle, and read back between the samples.

The checks that matter are that the integration is the pulse (at the slice
centre it must reduce to the instantaneous rotation, which pins the envelope
normalization), that the interpolation is faithful between the knots, and that
the stored slope is the derivative of the same integration rather than of
something near it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim.sequence._description import RfDefinition, RfShape
from torchsim.sequence._transition import transition_table

SAMPLES = 256
RASTER = 1e-5
# Four cycles of gradient winding across the pulse.
BANDWIDTH = 4.0 / (SAMPLES * RASTER)


def _envelope(phase_modulated: bool = False) -> np.ndarray:
    grid = np.linspace(-2.0, 2.0, SAMPLES)
    shape = np.sinc(grid) * (0.54 + 0.46 * np.cos(np.pi * grid / 2.0))
    shape = shape / np.abs(shape).max()
    if phase_modulated:
        shape = shape * np.exp(1j * 0.8 * grid**2)
    return shape


def _definition(phase_modulated: bool = False) -> RfDefinition:
    """The pulse as a description stores it: a magnitude and a phase.

    A sinc's negative lobes live in the phase, so both shapes are written even
    for a real envelope -- otherwise the definition is a different pulse from
    the one ``_envelope`` describes.
    """
    envelope = _envelope(phase_modulated)
    magnitude = RfShape(
        num_uncompressed=SAMPLES, samples=np.abs(envelope).astype(np.float32)
    )
    phase = RfShape(
        num_uncompressed=SAMPLES,
        samples=(np.angle(envelope) / (2.0 * np.pi)).astype(np.float32),
    )
    return RfDefinition(
        id=0,
        bandwidth_hz=BANDWIDTH,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=BANDWIDTH,
        total_b1sq_power=1.0,
        magnitude=magnitude,
        phase=phase,
    )


def _integrate(theta: float, position: float, phase_modulated: bool = False):
    """The same pulse, integrated straight, in double precision."""
    weight = _envelope(phase_modulated)
    weight = weight / weight.sum()
    a, b = 1.0 + 0j, 0.0 + 0j
    turn_z = 2.0 * np.pi * BANDWIDTH * position * RASTER
    for sample in weight:
        drive = theta * sample
        angle = np.sqrt(drive.real**2 + drive.imag**2 + turn_z**2)
        half = 0.5 * angle
        scale = np.sin(half) / angle if angle > 1e-9 else 0.5
        step_a = np.cos(half) - 1j * turn_z * scale
        step_b = -1j * (drive.real - 1j * drive.imag) * scale
        a, b = step_a * a - step_b * np.conj(b), step_b * np.conj(a) + step_a * b
    return a, b


@pytest.fixture(scope="module")
def table():
    return transition_table(
        _definition(),
        torch.linspace(-1.0, 1.0, 9),
        bins=64,
        rf_raster_time_s=RASTER,
    )


def test_the_pulse_leaves_a_rotation(table) -> None:
    """|a|^2 + |b|^2 is one for a rotation and nothing else."""
    length = table.a.abs() ** 2 + table.b.abs() ** 2
    assert (length - 1.0).abs().max() < 1e-6


def test_the_slice_centre_is_the_instantaneous_pulse(table) -> None:
    """On resonance every sample turns about the same axis, so they add.

    This is what pins the envelope normalization: the grid axis is a flip
    angle only if a nominal theta at the slice centre turns through theta.
    """
    centre = table.points // 2
    for bin_index in (0, 13, 31, 63):
        theta = table.theta_max * bin_index / (table.bins - 1)
        assert abs(complex(table.a[centre, bin_index]) - np.cos(theta / 2)) < 1e-5
        assert abs(complex(table.b[centre, bin_index]) + 1j * np.sin(theta / 2)) < 1e-5


def test_the_knots_are_the_integration(table) -> None:
    positions = torch.linspace(-1.0, 1.0, 9).tolist()
    worst = 0.0
    for index, position in enumerate(positions):
        for bin_index in (0, 7, 31, 63):
            theta = table.theta_max * bin_index / (table.bins - 1)
            expected = _integrate(theta, position)
            worst = max(
                worst,
                abs(expected[0] - complex(table.a[index, bin_index])),
                abs(expected[1] - complex(table.b[index, bin_index])),
            )
    assert worst < 1e-6


def test_reading_between_the_knots_is_still_the_integration(table) -> None:
    """The cubic has to carry the curve, not just touch it at the samples."""
    positions = torch.linspace(-1.0, 1.0, 9).tolist()
    worst = 0.0
    for index, position in enumerate(positions):
        for degrees in (13.0, 77.0, 154.0, 231.0, 305.0):
            theta = np.deg2rad(degrees)
            expected = _integrate(theta, position)
            got = table.at(
                torch.tensor(index), torch.tensor(theta, dtype=torch.float32)
            )
            worst = max(
                worst,
                abs(expected[0] - complex(got[0])),
                abs(expected[1] - complex(got[1])),
            )
    assert worst < 1e-5


def test_every_knot_and_slope_is_a_number(table) -> None:
    """The zero-flip knot at the slice centre sits where nothing is turning.

    The rotation angle is exactly zero there, and a square root is continuous
    at zero but its derivative is not, so a table that reaches the angle that
    way carries NaN in its slope -- in one corner, where a spot check does not
    look.
    """
    for name in ("a", "b", "slope_a", "slope_b"):
        assert torch.isfinite(getattr(table, name)).all(), name
    assert torch.isfinite(table.packed()).all()


def test_the_stored_slope_is_the_derivative_of_the_integration(table) -> None:
    positions = torch.linspace(-1.0, 1.0, 9).tolist()
    step = 1e-4
    worst = 0.0
    for index, position in enumerate(positions):
        for bin_index in (0, 1, 7, 31, 63):
            theta = table.theta_max * bin_index / (table.bins - 1)
            up = _integrate(theta + step, position)
            down = _integrate(theta - step, position)
            for side, stored in enumerate(
                (table.slope_a[index, bin_index], table.slope_b[index, bin_index])
            ):
                numeric = (up[side] - down[side]) / (2.0 * step)
                worst = max(worst, abs(numeric - complex(stored)))
    assert worst < 1e-5


def test_a_flip_past_the_grid_reads_the_last_knot(table) -> None:
    """Clamped rather than extrapolated: a cubic run off its end diverges."""
    beyond = table.at(
        torch.tensor(4), torch.tensor(2.0 * table.theta_max, dtype=torch.float32)
    )
    edge = table.at(
        torch.tensor(4), torch.tensor(table.theta_max, dtype=torch.float32)
    )
    assert abs(complex(beyond[0]) - complex(edge[0])) < 1e-6
    assert abs(complex(beyond[1]) - complex(edge[1])) < 1e-6


def test_a_phase_modulated_pulse_turns_about_a_moving_axis() -> None:
    """The envelope is complex, so each sample has an axis of its own."""
    modulated = transition_table(
        _definition(phase_modulated=True),
        torch.linspace(-1.0, 1.0, 5),
        bins=64,
        rf_raster_time_s=RASTER,
    )
    positions = torch.linspace(-1.0, 1.0, 5).tolist()
    worst = 0.0
    for index, position in enumerate(positions):
        for bin_index in (17, 48):
            theta = modulated.theta_max * bin_index / (modulated.bins - 1)
            expected = _integrate(theta, position, phase_modulated=True)
            worst = max(
                worst,
                abs(expected[0] - complex(modulated.a[index, bin_index])),
                abs(expected[1] - complex(modulated.b[index, bin_index])),
            )
    assert worst < 1e-5

    plain = transition_table(
        _definition(), torch.linspace(-1.0, 1.0, 5), bins=64, rf_raster_time_s=RASTER
    )
    assert not torch.allclose(modulated.b, plain.b, atol=1e-3)


def _excited(a, b):
    """Where a spin starting along +z ends up, as (F+, F-, Z).

    That is the third column of the rotation, so it needs no state machine --
    and it is the quantity the old flip-scaling model gets wrong.
    """
    from torchsim.epg import spinor_rf_pulse_op

    operator = spinor_rf_pulse_op(
        torch.as_tensor(a, dtype=torch.complex64),
        torch.as_tensor(b, dtype=torch.complex64),
    )
    return torch.stack([row[2].reshape(()) for row in operator])


@pytest.mark.parametrize("b1", [0.6, 0.8, 1.0, 1.2, 1.4])
def test_the_table_follows_the_transmit_where_a_flip_scaling_cannot(table, b1) -> None:
    """The point of the stage, stated as a number.

    A slice profile is a Bloch response, and the flip-scaling model treats it
    as proportional to the pulse driving it: the profile is fitted once at
    nominal amplitude and then multiplied by B1. That is exact only at B1 = 1.
    Reading the table at ``flip * b1`` is the response itself, at any B1.
    """
    nominal = np.deg2rad(180.0)
    positions = torch.linspace(-1.0, 1.0, 9)
    inside = positions.abs() <= 0.5

    # The profile the old model fits: the on-axis flip at nominal amplitude.
    fitted = []
    for position in positions.tolist():
        pair = _integrate(nominal, position)
        fitted.append(2.0 * np.arccos(np.clip(abs(pair[0]), -1.0, 1.0)))
    fitted = torch.tensor(fitted, dtype=torch.float32)
    fitted = fitted / fitted[len(positions) // 2]

    table_gap = scaling_gap = 0.0
    for index, position in enumerate(positions.tolist()):
        if not inside[index]:
            continue
        truth = _excited(*_integrate(nominal * b1, position))

        read = table.at(
            torch.tensor(index), torch.tensor(nominal * b1, dtype=torch.float32)
        )
        table_gap = max(table_gap, float((_excited(*read) - truth).abs().max()))

        scaled = nominal * b1 * float(fitted[index])
        model = _excited(np.cos(scaled / 2), -1j * np.sin(scaled / 2))
        scaling_gap = max(scaling_gap, float((model - truth).abs().max()))

    assert table_gap < 1e-4
    if b1 != 1.0:
        # The model it replaces is wrong by orders of magnitude more.
        assert scaling_gap > 100.0 * table_gap


def test_a_table_needs_two_knots_to_interpolate_between() -> None:
    with pytest.raises(ValueError, match="at least two knots"):
        transition_table(_definition(), torch.zeros(3), bins=1)


def test_a_pulse_with_no_samples_is_refused() -> None:
    empty = RfDefinition(
        id=3,
        bandwidth_hz=BANDWIDTH,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=BANDWIDTH,
        total_b1sq_power=1.0,
        magnitude=RfShape(num_uncompressed=0, samples=np.zeros(0, dtype=np.float32)),
    )
    with pytest.raises(ValueError, match="empty envelope"):
        transition_table(empty, torch.zeros(3))
