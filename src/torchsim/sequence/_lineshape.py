"""How well a bound pool absorbs an off-resonance pulse, worked out ahead of time.

RF saturates the semisolid pool at a rate set by the pulse's power and by the
pool's absorption lineshape at the offset the pulse is played at. That
lineshape is the super-Lorentzian: an average of Lorentzians over the
orientations of a randomly oriented solid,

    G(df) = int_0^1 sqrt(2/pi) T2b / |3u^2 - 1|
            * exp(-2 (2 pi df T2b / (3u^2 - 1))^2) du

which no kernel can integrate per voxel per pulse. It is tabulated here
instead, over the offset alone, because of what it depends on:

* **It is even in the offset.** The integrand depends on ``df`` only through
  its square, so the table covers ``|df|`` and half of it is never stored,
  and the slope at resonance is zero.
* **The bound pool's T2 is a model constant**, so the table stays
  one-dimensional. Making it a tissue property would make it two.

The offset a voxel sees is the pulse's own frequency offset less the voxel's
off-resonance, so the read is an event buffer against a voxel buffer -- the
same shape as ``theta = flip * b1`` driving the transition table, and read
back through the same cubic Hermite.

**Near resonance the integral diverges** -- the ``1 / |3u^2 - 1|`` singularity
stops being suppressed -- and the divergence is an artefact of the model
rather than physics, so the region inside ``cutoff_hz`` is filled instead.

The fill is a cubic spline through the knots in
``[cutoff, 2 * cutoff)``, mirrored about resonance and extrapolated inward. Mirroring is what keeps it even, the
interpolant through symmetric data being unique and therefore its own
reflection, so the slope at resonance still comes out at zero.

The fill is fitted further out, where the lineshape is flatter, and arrives
at the cutoff about 13% shallower than the integral does. That does not reach
the table: a knot carries one slope, the cutoff knot carries the integral's,
and the Hermite matches it from both sides -- so what the kernels read is C1
across the join exactly as it is across any other knot. The two conventions
differ only in the values they give inside the cutoff, by a couple of percent.
"""

from __future__ import annotations

__all__ = ["LineshapeTable", "lineshape_reaching", "lineshape_table"]

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline

# T2 of the semisolid compartment, in seconds. The conventional value for
# white matter, and the default the package's lineshape carries.
BOUND_T2_S = 12e-6

# How far off resonance a table reaches, and over how many knots, before a
# sequence asks for more. 33 kHz is where the package's own lineshape stops,
# and 128 knots across it carry the shape to five decades below its value at
# resonance.
OFFSET_MAX_HZ = 33e3
OFFSET_BINS = 128


@dataclass(frozen=True)
class LineshapeTable:
    """The absorption lineshape, sampled over the offset a pulse is played at.

    ``values`` and ``slopes`` hold the lineshape and its derivative in the
    offset at every knot, shaped ``(bins,)``. The offset axis runs from zero to
    ``offset_max_hz`` in evenly spaced knots and covers ``|df|`` only, the
    lineshape being even.
    """

    values: torch.Tensor
    slopes: torch.Tensor
    offset_max_hz: float
    cutoff_hz: float

    @property
    def bins(self) -> int:
        """How many offset knots the table carries."""
        return int(self.values.shape[0])

    @property
    def step(self) -> float:
        """Offset between neighbouring knots, in Hz."""
        return self.offset_max_hz / (self.bins - 1)

    def packed(self, device: torch.device | str | None = None) -> torch.Tensor:
        """The table laid out as the kernels index it.

        ``(bins, 2)``: the value then its slope, so the two knots a Hermite
        read needs are four contiguous floats.
        """
        return (
            torch.stack((self.values, self.slopes), dim=-1)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )

    def at(self, offset_hz: torch.Tensor) -> torch.Tensor:
        """The lineshape at an offset between the knots, by cubic Hermite.

        The offset is taken in magnitude, the lineshape being even, and
        clamped to the grid. Cubic rather than linear because the second-order
        pass differentiates this twice.
        """
        step = self.step
        scaled = (offset_hz.abs() / step).clamp(0.0, self.bins - 1.0)
        lower = scaled.floor().clamp(max=self.bins - 2.0)
        u = scaled - lower
        index = lower.to(torch.int64)

        u2 = u * u
        u3 = u2 * u
        near = self.values[index]
        far = self.values[index + 1]
        near_slope = self.slopes[index]
        far_slope = self.slopes[index + 1]
        return (
            (2.0 * u3 - 3.0 * u2 + 1.0) * near
            + (u3 - 2.0 * u2 + u) * step * near_slope
            + (-2.0 * u3 + 3.0 * u2) * far
            + (u3 - u2) * step * far_slope
        )


def _absorption(
    offset_hz: np.ndarray, bound_t2_s: float, quadrature: int
) -> tuple[np.ndarray, np.ndarray]:
    """The super-Lorentzian integral and its derivative in the offset.

    Evaluated by the trapezoid the model is conventionally written with. At
    the orientation where ``3u^2 - 1`` vanishes the integrand is a pole times
    a Gaussian that closes faster than the pole opens, so its limit is zero
    away from resonance -- but computed as written it is ``inf * 0``, which is
    why the vanishing denominator is masked out rather than left to the
    arithmetic.

    The derivative is taken under the integral rather than by differencing it,
    so it is the derivative of this quadrature and not of a nearby one.
    """
    grid = np.linspace(0.0, 1.0, quadrature, dtype=np.float64)
    spacing = 1.0 / (quadrature - 1)
    denominator = 3.0 * grid * grid - 1.0
    open_pole = np.abs(denominator) > 1e-12
    guarded = np.where(open_pole, denominator, 1.0)

    amplitude = np.sqrt(2.0 / np.pi) * bound_t2_s / np.abs(guarded)
    rate = 2.0 * np.pi * bound_t2_s / guarded
    scaled = offset_hz[:, None] * rate[None, :]
    weighted = np.where(
        open_pole[None, :], amplitude[None, :] * np.exp(-2.0 * scaled**2), 0.0
    )
    value = spacing * weighted.sum(axis=-1)
    slope = spacing * (weighted * (-4.0 * scaled * rate[None, :])).sum(axis=-1)
    return value, slope


@functools.lru_cache(maxsize=8)
def lineshape_table(
    *,
    bound_t2_s: float = BOUND_T2_S,
    offset_max_hz: float = OFFSET_MAX_HZ,
    bins: int = OFFSET_BINS,
    cutoff_hz: float = 1e3,
    quadrature: int = 20000,
    device: torch.device | str | None = None,
) -> LineshapeTable:
    """Integrate the super-Lorentzian into a table over the offset.

    Parameters
    ----------
    bound_t2_s
        T2 of the semisolid compartment. 12 us for white matter.
    offset_max_hz
        Largest offset the table covers. A read past it takes the last knot;
        :func:`lineshape_reaching` is what sizes a table so that a sequence
        never gets there.
    bins
        Knots along the offset axis. 128 over 33 kHz carries the lineshape to
        about 2e-10, which is five decades below its value at resonance.
    cutoff_hz
        Below this the integral diverges and is replaced by the fill
        described in the module docstring, interpolated from the knots
        between the cutoff and twice it. Snapped to the nearest knot, so that
        the fill and the integral meet at one; the table reports the snapped
        value.
    quadrature
        Points in the orientation integral. The integrand has a pole, so this
        converges slowly; it is paid once, on the host.

    Notes
    -----
    Memoized on its arguments. The integral costs far more than a simulation
    does and depends on nothing a caller varies between runs, and the table it
    returns is read-only, so every simulation of a given bound pool shares one.

    Returns
    -------
        The table, with slopes taken by differentiating the same integration
        rather than by differencing it.

    Raises
    ------
        ValueError: if ``bins`` is below two, the cutoff does not sit inside
            the offset range, or too few knots fall in the band the fill is
            interpolated from.
    """
    if bins < 2:
        raise ValueError(f"a Hermite table needs at least two knots, got {bins}")
    if not 0.0 < cutoff_hz < offset_max_hz:
        raise ValueError(
            f"the cutoff must sit inside the table, got {cutoff_hz} Hz in a "
            f"table reaching {offset_max_hz} Hz"
        )

    knots = np.linspace(0.0, offset_max_hz, bins, dtype=np.float64)
    values, slopes = _absorption(knots, bound_t2_s, quadrature)

    # The cutoff is snapped to a knot so that the fill and the integral meet
    # at one, and no single segment interpolates between a filled knot and an
    # integrated one. The snapped value is what the table reports.
    step = offset_max_hz / (bins - 1)
    edge_index = max(1, min(bins - 2, int(round(cutoff_hz / step))))
    edge = float(knots[edge_index])

    support = knots[(knots >= edge) & (knots < 2.0 * edge)]
    if support.size < 2:
        raise ValueError(
            f"the fill is interpolated from the knots between {edge:.0f} and "
            f"{2 * edge:.0f} Hz, and this table has {support.size} of them; "
            f"use more bins or a smaller cutoff"
        )
    mirrored = np.concatenate((-support[::-1], support))
    sampled, _ = _absorption(support, bound_t2_s, quadrature)
    curve = InterpolatedUnivariateSpline(
        mirrored,
        np.concatenate((sampled[::-1], sampled)),
        k=min(3, mirrored.size - 1),
    )

    inside = knots < edge
    filled = knots[inside]
    values[inside] = curve(filled)
    slopes[inside] = curve.derivative()(filled)

    return LineshapeTable(
        values=torch.tensor(values, dtype=torch.float32, device=device),
        slopes=torch.tensor(slopes, dtype=torch.float32, device=device),
        offset_max_hz=float(offset_max_hz),
        cutoff_hz=float(edge),
    )


def lineshape_reaching(
    offset_hz: float, *, device: torch.device | str | None = None
) -> LineshapeTable:
    """A table that reaches this far off resonance.

    The knot spacing is held at the default's, so a sequence played further
    out gets more knots rather than coarser ones -- and the extent is rounded
    up to a whole number of default tables, so runs asking for similar ranges
    share one instead of each integrating its own.

    Parameters
    ----------
    offset_hz
        The largest ``|pulse frequency - voxel off-resonance|`` the run can
        read the lineshape at.

    Returns
    -------
        The table, memoized with every other of its size.
    """
    blocks = max(1, math.ceil(abs(offset_hz) / OFFSET_MAX_HZ))
    return lineshape_table(
        offset_max_hz=blocks * OFFSET_MAX_HZ,
        bins=blocks * (OFFSET_BINS - 1) + 1,
        device=device,
    )
