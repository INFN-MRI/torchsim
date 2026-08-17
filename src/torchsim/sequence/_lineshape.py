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
  its square, so the table covers ``|df|`` and half of it is never stored.
  Evenness also fixes the slope at resonance at exactly zero, which is what
  chooses the fill below.
* **The bound pool's T2 is a model constant**, so the table stays
  one-dimensional. Making it a tissue property would make it two.

The offset a voxel sees is the pulse's own frequency offset less the voxel's
off-resonance, so the read is an event buffer against a voxel buffer -- the
same shape as ``theta = flip * b1`` driving the transition table, and read
back through the same cubic Hermite.

**Near resonance the integral diverges** -- the ``1 / |3u^2 - 1|`` singularity
stops being suppressed -- and the divergence is an artefact of the model
rather than physics, so the region inside ``cutoff_hz`` is filled instead.
The fill is the lowest-order curve that is even in the offset and joins the
integral with a continuous value and slope: ``a + b df^2``. Note that
:func:`torchsim.epg.super_lorentzian_lineshape` fills the same region with an
unconstrained cubic spline, which does not respect evenness and lands about
15% lower at resonance; the two agree once past the cutoff.
"""

from __future__ import annotations

__all__ = ["LineshapeTable", "lineshape_table"]

from dataclasses import dataclass

import torch

# T2 of the semisolid compartment, in seconds. The conventional value for
# white matter, and the default the package's lineshape carries.
BOUND_T2_S = 12e-6


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
    offset_hz: torch.Tensor, bound_t2_s: float, quadrature: int
) -> torch.Tensor:
    """The super-Lorentzian integral, and its derivative through the same grid.

    Evaluated by the trapezoid the model is conventionally written with. At
    the orientation where ``3u^2 - 1`` vanishes the integrand is a pole times
    a Gaussian that closes faster than the pole opens, so its limit is zero
    away from resonance -- but computed as written it is ``inf * 0``, which is
    why the vanishing denominator is masked out rather than left to the
    arithmetic.
    """
    grid = torch.linspace(
        0.0, 1.0, quadrature, dtype=torch.float64, device=offset_hz.device
    )
    spacing = 1.0 / (quadrature - 1)
    denominator = 3.0 * grid * grid - 1.0
    open_pole = denominator.abs() > 1e-12
    guarded = torch.where(open_pole, denominator, torch.ones_like(denominator))

    amplitude = (2.0 / torch.pi) ** 0.5 * bound_t2_s / guarded.abs()
    exponent = (
        -2.0
        * (
            2.0
            * torch.pi
            * offset_hz[:, None]
            * bound_t2_s
            / guarded[None, :]
        )
        ** 2
    )
    integrand = torch.where(
        open_pole[None, :], amplitude[None, :] * torch.exp(exponent), 0.0
    )
    return spacing * integrand.sum(dim=-1)


def lineshape_table(
    *,
    bound_t2_s: float = BOUND_T2_S,
    offset_max_hz: float = 33e3,
    bins: int = 128,
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
        Largest offset the table covers. A pulse driven past this is refused
        rather than read at the last knot.
    bins
        Knots along the offset axis. 128 over 33 kHz carries the lineshape to
        about 2e-10, which is five decades below its value at resonance.
    cutoff_hz
        Below this the integral diverges and is replaced by the even fill
        described in the module docstring. Snapped to the nearest knot, so
        that the fill and the integral meet at one; the table reports the
        snapped value.
    quadrature
        Points in the orientation integral. The integrand has a pole, so this
        converges slowly; it is paid once, on the host.

    Returns:
        The table, with slopes taken by differentiating the same integration
        rather than by differencing it.

    Raises:
        ValueError: if ``bins`` is below two, or the cutoff does not sit
            inside the offset range.
    """
    if bins < 2:
        raise ValueError(f"a Hermite table needs at least two knots, got {bins}")
    if not 0.0 < cutoff_hz < offset_max_hz:
        raise ValueError(
            f"the cutoff must sit inside the table, got {cutoff_hz} Hz in a "
            f"table reaching {offset_max_hz} Hz"
        )

    knots = torch.linspace(
        0.0, offset_max_hz, bins, dtype=torch.float64, device=device
    )

    def integrate(offset: torch.Tensor) -> torch.Tensor:
        return _absorption(offset, bound_t2_s, quadrature)

    values, slopes = torch.func.jvp(
        integrate, (knots,), (torch.ones_like(knots),)
    )

    # The fill: a + b * df^2, matched to the integral's value and slope at the
    # cutoff. Even by construction, so the slope at resonance is exactly zero.
    #
    # The cutoff is snapped to a knot so that the two curves meet at one, and
    # no single segment interpolates between a filled knot and an integrated
    # one. The snapped value is what the table reports.
    step = offset_max_hz / (bins - 1)
    edge_index = max(1, min(bins - 2, int(round(cutoff_hz / step))))
    edge = knots[edge_index]
    at_edge, slope_at_edge = torch.func.jvp(
        integrate,
        (edge.reshape(1),),
        (torch.ones(1, dtype=torch.float64, device=device),),
    )
    curvature = slope_at_edge[0] / (2.0 * edge)
    level = at_edge[0] - curvature * edge * edge

    inside = knots < edge
    values = torch.where(inside, level + curvature * knots * knots, values)
    slopes = torch.where(inside, 2.0 * curvature * knots, slopes)

    return LineshapeTable(
        values=values.to(torch.float32).contiguous(),
        slopes=slopes.to(torch.float32).contiguous(),
        offset_max_hz=float(offset_max_hz),
        cutoff_hz=float(edge),
    )
