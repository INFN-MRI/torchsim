"""Tabulating how well a bound pool absorbs an off-resonance pulse.

The super-Lorentzian is an integral over orientations that no kernel can
evaluate per voxel per pulse, so it is sampled over the offset and read back
between the samples. What the tests have to hold is that the integration is
the lineshape, that the interpolation is faithful between the knots, that the
stored slope is the derivative of the same integration, and that the region
near resonance -- where the integral diverges and is filled instead -- is
filled by a curve that respects the one property the lineshape certainly has:
it is even in the offset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim.sequence._lineshape import BOUND_T2_S, lineshape_table


def _integral(offset_hz, bound_t2_s: float = BOUND_T2_S, quadrature: int = 20000):
    """The same integral, taken straight, in double precision numpy."""
    offset = np.atleast_1d(np.asarray(offset_hz, dtype=np.float64))
    grid = np.linspace(0.0, 1.0, quadrature)
    spacing = grid[1] - grid[0]
    denominator = 3.0 * grid**2 - 1.0
    open_pole = np.abs(denominator) > 1e-12
    guarded = np.where(open_pole, denominator, 1.0)
    amplitude = np.sqrt(2.0 / np.pi) * bound_t2_s / np.abs(guarded)
    exponent = -2.0 * (
        2.0 * np.pi * offset[:, None] * bound_t2_s / guarded[None, :]
    ) ** 2
    integrand = np.where(open_pole[None, :], amplitude[None, :] * np.exp(exponent), 0.0)
    return spacing * integrand.sum(axis=-1)


@pytest.fixture(scope="module")
def table():
    return lineshape_table(bins=128)


def test_past_the_cutoff_the_table_is_the_integral(table) -> None:
    """Where the model is not divergent, the table is the model."""
    offsets = np.linspace(1.05e3, 32e3, 401)
    expected = _integral(offsets)
    got = table.at(torch.as_tensor(offsets, dtype=torch.float32)).numpy()

    assert np.abs(got - expected).max() < 1e-9


def test_reading_between_the_knots_is_still_the_integral(table) -> None:
    """The cubic has to carry the curve, not just touch it at the samples.

    Probed at offsets deliberately off the 260 Hz knot spacing.
    """
    offsets = np.array([1137.0, 2611.0, 4903.0, 9677.0, 18211.0, 29399.0])
    expected = _integral(offsets)
    got = table.at(torch.as_tensor(offsets, dtype=torch.float32)).numpy()

    assert np.abs(got - expected).max() < 1e-9


def test_the_stored_slope_is_the_derivative_of_the_integration(table) -> None:
    step = 1.0
    for index in (6, 12, 31, 96):
        # The knots are what carries the slope, so the difference is taken
        # where the knot actually sits rather than near it.
        offset = index * table.step
        numeric = (_integral(offset + step) - _integral(offset - step))[0] / (2 * step)
        assert abs(numeric - float(table.slopes[index])) < 1e-14


def test_the_lineshape_is_even_so_its_slope_at_resonance_is_zero(table) -> None:
    """The one property the fill has to respect.

    The integrand depends on the offset only through its square, so the
    lineshape is even and its derivative at zero vanishes. A fill that misses
    that puts a kink at resonance, which the second-order pass differentiates.
    """
    assert float(table.slopes[0]) == 0.0
    positive = table.at(torch.tensor([250.0, 500.0, 750.0]))
    negative = table.at(torch.tensor([-250.0, -500.0, -750.0]))
    assert torch.equal(positive, negative)


def test_the_fill_joins_the_integral_at_a_knot(table) -> None:
    """The two curves meet at a knot, in value and in slope.

    Snapped there deliberately: with the cutoff between knots, one segment
    would interpolate a filled knot against an integrated one and pass through
    neither curve.
    """
    edge = table.cutoff_hz
    index = int(round(edge / table.step))
    assert abs(edge - index * table.step) < 1e-6

    assert abs(float(table.values[index]) - float(_integral(edge)[0])) < 1e-11
    numeric = (_integral(edge + 1.0) - _integral(edge - 1.0))[0] / 2.0
    assert abs(numeric - float(table.slopes[index])) < 1e-13

    # and the fill reaches the same place from below
    curvature = float(table.slopes[index]) / (2.0 * edge)
    level = float(table.values[index]) - curvature * edge * edge
    assert abs(float(table.values[index - 1]) - (
        level + curvature * (edge - table.step) ** 2
    )) < 1e-11


def test_the_lineshape_falls_away_from_resonance(table) -> None:
    """Monotone, which is what makes an off-resonance pulse saturate less."""
    offsets = torch.linspace(0.0, 33e3, 400)
    read = table.at(offsets)
    assert bool((read[1:] <= read[:-1] + 1e-12).all())
    assert float(read[0]) > 1000.0 * float(read[-1])


def test_every_knot_and_slope_is_a_number(table) -> None:
    """The orientation integral has a pole, and the fill has a division."""
    assert torch.isfinite(table.values).all()
    assert torch.isfinite(table.slopes).all()
    assert torch.isfinite(table.packed()).all()
    assert bool((table.values > 0.0).all())


def test_the_pole_does_not_reach_the_table(table) -> None:
    """The integrand has a pole, and the quadrature walks past it.

    At the orientation where ``3u^2 - 1`` vanishes the integrand is a pole
    times a Gaussian that closes faster, so its limit is zero away from
    resonance -- but a grid point landing near it evaluates a very large
    amplitude against a very small exponential. The invariant is that the
    answer does not depend on how close the walk happens to pass: these
    quadratures approach the pole to within 5e-7 and 3e-4 respectively.

    An exact landing cannot happen on a uniform grid, ``1/sqrt(3)`` being
    irrational, so the mask on the vanishing denominator is defence rather
    than a path the build takes.
    """
    probe = torch.tensor([2.0e3, 8.0e3, 25.0e3])
    reference = lineshape_table(bins=128, quadrature=20000).at(probe)
    for quadrature in (1352, 4327):
        built = lineshape_table(bins=128, quadrature=quadrature)
        assert torch.isfinite(built.values).all(), quadrature
        assert torch.isfinite(built.slopes).all(), quadrature
        assert float((built.at(probe) / reference - 1.0).abs().max()) < 5e-3


def test_a_wider_pulse_offset_needs_a_wider_table(table) -> None:
    """Reading past the end clamps, so the caller is meant to widen instead."""
    beyond = table.at(torch.tensor(2.0 * table.offset_max_hz))
    edge = table.at(torch.tensor(table.offset_max_hz))
    assert abs(float(beyond) - float(edge)) < 1e-12

    wider = lineshape_table(bins=128, offset_max_hz=60e3)
    assert float(wider.at(torch.tensor(50e3))) < float(table.at(torch.tensor(30e3)))


def test_a_shorter_bound_t2_broadens_the_lineshape() -> None:
    """T2b sets the width, so a shorter one absorbs further off resonance."""
    narrow = lineshape_table(bins=128, bound_t2_s=20e-6)
    broad = lineshape_table(bins=128, bound_t2_s=6e-6)
    far = torch.tensor(20e3)

    assert float(broad.at(far)) > float(narrow.at(far))
    assert float(broad.values[0]) < float(narrow.values[0])


def test_the_table_agrees_with_the_package_lineshape_past_the_cutoff(table) -> None:
    """The convention check, at the resolution that function actually holds.

    ``epg.super_lorentzian_lineshape`` evaluates on a 128-point grid and then
    takes the *nearest* one, so it is a step function of the offset with 516 Hz
    treads across which the lineshape moves by 10 to 20 percent. It pins the
    model -- the same integrand, the same T2b -- rather than the digits.
    """
    from torchsim.epg._rf_pulse import super_lorentzian_lineshape

    offsets = np.array([2.0e3, 5.0e3, 10.0e3, 20.0e3])
    expected = np.asarray(super_lorentzian_lineshape(offsets), dtype=np.float64)
    got = table.at(torch.as_tensor(offsets, dtype=torch.float32)).numpy()

    assert np.abs(got / expected - 1.0).max() < 0.25


def test_a_table_needs_two_knots_and_a_cutoff_inside_it() -> None:
    with pytest.raises(ValueError, match="at least two knots"):
        lineshape_table(bins=1)
    with pytest.raises(ValueError, match="cutoff must sit inside"):
        lineshape_table(bins=16, cutoff_hz=50e3, offset_max_hz=33e3)
