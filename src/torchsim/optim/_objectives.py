"""Built-in differentiable acquisition objectives."""

from __future__ import annotations

__all__ = ["FSET2Precision"]

import torch

from .._functional import fse_sim
from ._sequence import SequenceParameters


class FSET2Precision(torch.nn.Module):
    """A-optimal T2-precision objective for an FSE refocusing train.

    Parameters
    ----------
    t1_ms, t2_ms : torch.Tensor
        Tissue design points in milliseconds.
    echo_spacing_ms : float
        Echo spacing in milliseconds.
    phases_deg : float, optional
        Refocusing-pulse phase.
    smoothness_weight : float, optional
        Weight of the first-difference (slope) penalty.
    curvature_weight : float, optional
        Weight of the second-difference (curvature) penalty. This is the term
        that suppresses the alternating high/low trains which maximize raw T2
        information but cannot be played on a scanner.
    rf_power_weight : float, optional
        Weight of the mean squared flip angle, a proxy for SAR.
    parameter_name : str, optional
        Key used when a :class:`SequenceOptimizer` receives a dictionary.

    Notes
    -----
    The data term is the *relative* Cramer-Rao bound, ``mean((sigma_T2 /
    T2)**2)``, averaged over the tissue design points. Being dimensionless, it
    weights short- and long-T2 species comparably. The objective returns its
    logarithm, which makes the data term scale-free: its gradient is relative,
    so the penalty weights below keep their meaning independently of the noise
    level, the echo-train length, and the units of the signal.

    The penalties act on flip angles normalized by 180 degrees, so all three
    are of order one for a plausible train and their weights are directly
    comparable.
    """

    def __init__(
        self,
        t1_ms: torch.Tensor,
        t2_ms: torch.Tensor,
        echo_spacing_ms: float,
        *,
        phases_deg: float = 90.0,
        smoothness_weight: float = 0.5,
        curvature_weight: float = 60.0,
        rf_power_weight: float = 0.05,
        parameter_name: str = "flip_deg",
    ) -> None:
        super().__init__()
        t1_ms = torch.as_tensor(t1_ms)
        t2_ms = torch.as_tensor(t2_ms, device=t1_ms.device, dtype=t1_ms.dtype)
        self.register_buffer("t1_ms", t1_ms)
        self.register_buffer("t2_ms", t2_ms)
        self.echo_spacing_ms = float(echo_spacing_ms)
        self.phases_deg = float(phases_deg)
        self.smoothness_weight = float(smoothness_weight)
        self.curvature_weight = float(curvature_weight)
        self.rf_power_weight = float(rf_power_weight)
        self.parameter_name = parameter_name

    def forward(self, parameters: SequenceParameters) -> torch.Tensor:
        """Evaluate T2 precision and RF-train penalties."""
        flip_deg = (
            parameters[self.parameter_name]
            if isinstance(parameters, dict)
            else parameters
        )
        _, derivative = fse_sim(
            flip=flip_deg,
            phases=torch.full_like(flip_deg, self.phases_deg),
            ESP=self.echo_spacing_ms,
            T1=self.t1_ms,
            T2=self.t2_ms,
            diff="T2",
            device=flip_deg.device,
        )
        information = derivative.abs().square().sum(dim=-1).clamp_min(1e-12)
        # Cramer-Rao variance is 1 / information; dividing by T2**2 turns it
        # into the squared relative error, which is dimensionless.
        precision = (1.0 / (information * self.t2_ms.square())).mean()

        flip = flip_deg / 180.0
        zero = flip.new_zeros(())
        smoothness = (
            (flip[1:] - flip[:-1]).square().mean() if flip.numel() > 1 else zero
        )
        curvature = (
            (flip[2:] - 2.0 * flip[1:-1] + flip[:-2]).square().mean()
            if flip.numel() > 2
            else zero
        )
        return (
            precision.log()
            + self.smoothness_weight * smoothness
            + self.curvature_weight * curvature
            + self.rf_power_weight * flip.square().mean()
        )
