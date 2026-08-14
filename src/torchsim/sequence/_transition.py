"""The rotation a shaped pulse performs, worked out ahead of time.

The state machine treats a pulse as instantaneous. A real one has a duration,
and under a slice-select gradient the spins it acts on are off-resonance by an
amount that grows with distance from the slice centre, so what it does to a
spin is a rotation about an axis that is neither transverse nor the same
everywhere. A flip angle and a phase cannot name that rotation; the
Cayley-Klein pair ``(a, b)`` can, and :func:`torchsim.epg.spinor_rf_pulse_op`
turns that pair into the operator.

Working the pair out during the simulation would mean integrating the pulse
once per voxel per event. It is tabulated here instead, and the table is small
because of what the rotation actually depends on:

* **Flip angle and transmit scaling enter only through their product.** During
  the pulse the effective field is ``(w1(t) * s, 0, gamma * G * z)`` and only
  the transverse part carries ``s``, so a nominal flip of 180 degrees at
  ``B1 = 0.7`` is the same rotation as 126 degrees at ``B1 = 1``. One axis, not
  two.
* **The RF phase factors out.** Turning the whole pulse by ``phi`` turns the
  rotation axis with it, which is ``b -> b * exp(-1j * phi)``. So the table is
  built for zero phase and the phase applied to the result, exactly as the
  instantaneous operator applies it.

That leaves a table over slice position and effective flip angle. Sampled with
its own slope and read back by cubic Hermite interpolation, 64 bins carry it to
about 1e-8, which is far below what float32 states resolve.
"""

from __future__ import annotations

__all__ = ["TransitionTable", "transition_table"]

from dataclasses import dataclass

import torch

from ._description import RfDefinition


@dataclass(frozen=True)
class TransitionTable:
    """A pulse's rotation, sampled over slice position and effective flip.

    ``a`` and ``b`` hold the Cayley-Klein pair at every knot, shaped
    ``(points, bins)``; ``slope_a`` and ``slope_b`` hold its derivative in the
    flip angle there. The flip axis runs from zero to ``theta_max`` radians in
    ``bins`` evenly spaced knots.
    """

    a: torch.Tensor
    b: torch.Tensor
    slope_a: torch.Tensor
    slope_b: torch.Tensor
    theta_max: float

    @property
    def points(self) -> int:
        """How many slice positions the table carries."""
        return int(self.a.shape[0])

    @property
    def bins(self) -> int:
        """How many flip-angle knots each position carries."""
        return int(self.a.shape[1])

    def at(
        self, position: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair at a flip angle between the knots, by cubic Hermite.

        ``position`` indexes the slice axis and ``theta`` is the effective flip
        in radians, clamped to the grid. Cubic rather than linear because the
        second-order pass differentiates this twice, and because storing the
        slope makes the cubic cost the same two loads.

        Returns the pair; the caller applies the RF phase to ``b``.
        """
        step = self.theta_max / (self.bins - 1)
        scaled = (theta / step).clamp(0.0, self.bins - 1.0)
        lower = scaled.floor().clamp(max=self.bins - 2.0)
        u = scaled - lower
        index = lower.to(torch.int64)

        # Hermite basis on the unit interval.
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        def blend(values: torch.Tensor, slopes: torch.Tensor) -> torch.Tensor:
            near = values[position, index]
            far = values[position, index + 1]
            near_slope = slopes[position, index]
            far_slope = slopes[position, index + 1]
            return (
                h00 * near
                + h10 * step * near_slope
                + h01 * far
                + h11 * step * far_slope
            )

        return blend(self.a, self.slope_a), blend(self.b, self.slope_b)


def transition_table(
    definition: RfDefinition,
    positions: torch.Tensor,
    *,
    bins: int = 64,
    theta_max: float = 2.0 * torch.pi,
    rf_raster_time_s: float = 1e-6,
    device: torch.device | str | None = None,
) -> TransitionTable:
    """Integrate a pulse into the rotation it performs, over a grid.

    Parameters
    ----------
    definition
        The pulse. Its complex envelope is played sample by sample at
        ``rf_raster_time_s``, and its ``bandwidth_hz`` sets how fast
        off-resonance grows away from the slice centre.
    positions
        Where across the slice to sample, in units of the slice thickness, so
        that the passband is ``[-0.5, 0.5]``.
    bins
        Knots along the flip axis. 64 carries a sinc to about 1e-8.
    theta_max
        Largest effective flip the table covers, in radians. A pulse driven
        past this reads the last knot.

    Returns:
        The table, with slopes taken by forward-mode differentiation of the
        same integration rather than by differencing it.

    Raises:
        ValueError: if the pulse has no samples, or ``bins`` is below two.
    """
    if bins < 2:
        raise ValueError(f"a Hermite table needs at least two knots, got {bins}")

    # Integrated in double and stored in single: the product runs once per
    # raster sample, so rounding accumulates over thousands of steps to reach a
    # table everything downstream is measured against.
    envelope = torch.as_tensor(
        definition.complex_envelope(), dtype=torch.complex128, device=device
    )
    if envelope.numel() == 0:
        raise ValueError(f"RF definition {definition.id} has an empty envelope")
    # Divided by its own integral, which is how RfDefinition.flip_angle reads a
    # pulse: the magnitude of the integral sets the flip and its angle sets the
    # axis. Normalizing by the complex sum puts both right -- an on-resonance
    # spin turns through exactly theta, about the axis at zero -- so the grid
    # axis is the same flip angle the packed events carry.
    area = envelope.sum()
    if area.abs() <= 1e-6 * envelope.abs().sum():
        raise ValueError(
            f"RF definition {definition.id} integrates to nothing, so it has no "
            f"flip angle to tabulate against"
        )
    weight = envelope / area

    positions = torch.as_tensor(positions, dtype=torch.float64, device=device)
    theta = torch.linspace(0.0, theta_max, bins, dtype=torch.float64, device=device)

    # Off-resonance a spin at each position accrues per sample. The pulse
    # selects one slice thickness across its bandwidth, so position measured in
    # thicknesses times bandwidth is the offset in Hz.
    offset = (
        2.0
        * torch.pi
        * float(definition.bandwidth_hz)
        * positions
        * rf_raster_time_s
    )

    def integrate(flip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair the whole pulse leaves, at every (position, flip).

        One SU(2) element per raster sample, composed in the order the
        scanner plays them. Each is the rotation about the effective field a
        spin sees while that sample lasts: the RF in the transverse plane, the
        gradient along z.
        """
        turn_z = offset[:, None].expand(positions.numel(), flip.numel())
        a = torch.ones_like(turn_z, dtype=torch.complex128)
        b = torch.zeros_like(a)
        for sample in weight:
            drive = flip[None, :] * sample
            turn_x = drive.real.expand_as(turn_z)
            turn_y = drive.imag.expand_as(turn_z)
            angle = torch.sqrt(turn_x**2 + turn_y**2 + turn_z**2)
            half = 0.5 * angle
            # sin(angle / 2) / angle, which is 1/2 at the origin rather than
            # the zero-over-zero that dividing would give.
            scale = torch.where(
                angle > 1e-9,
                torch.sin(half) / torch.where(angle > 1e-9, angle, torch.ones_like(angle)),
                0.5 - angle**2 / 48.0,
            )
            step_a = torch.cos(half) - 1j * turn_z * scale
            step_b = -1j * (turn_x - 1j * turn_y) * scale
            a, b = step_a * a - step_b * b.conj(), step_b * a.conj() + step_a * b
        return a, b

    values, slopes = torch.func.jvp(
        integrate, (theta,), (torch.ones_like(theta),)
    )
    return TransitionTable(
        a=values[0].to(torch.complex64).contiguous(),
        b=values[1].to(torch.complex64).contiguous(),
        slope_a=slopes[0].to(torch.complex64).contiguous(),
        slope_b=slopes[1].to(torch.complex64).contiguous(),
        theta_max=float(theta_max),
    )
