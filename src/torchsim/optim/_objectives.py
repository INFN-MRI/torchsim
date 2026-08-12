"""Built-in differentiable acquisition objectives."""

from __future__ import annotations

__all__ = ["FSET2Precision"]

import torch

from ..sequence._simulation import TissueProperties
from ._fast_fse import FseT2Plan
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
        # One plan per (echo train length, device); the structure is fixed.
        self._plans: dict[tuple[int, str], FseT2Plan] = {}

    def _t2_jacobian(self, flip_deg: torch.Tensor) -> torch.Tensor:
        """``d signal / d T2`` for the current refocusing train.

        Uses the direct kernel path, whose plan is reused across iterations; the
        generic model stack serves anything the plan does not cover.
        """
        echoes = flip_deg.shape[-1]
        key = (echoes, str(flip_deg.device))
        plan = self._plans.get(key)
        if plan is None:
            plan = FseT2Plan(
                echoes,
                self.echo_spacing_ms * 1e-3,
                phases_rad=torch.pi * self.phases_deg / 180.0,
                device=flip_deg.device,
            )
            self._plans[key] = plan
        tissue = TissueProperties(t1_ms=self.t1_ms, t2_ms=self.t2_ms)
        return plan.t2_jacobian(torch.pi * flip_deg / 180.0, tissue)

    def forward(self, parameters: SequenceParameters) -> torch.Tensor:
        """Evaluate T2 precision and RF-train penalties."""
        flip_deg = (
            parameters[self.parameter_name]
            if isinstance(parameters, dict)
            else parameters
        )
        derivative = self._t2_jacobian(flip_deg)
        information = derivative.abs().square().sum(dim=-1).clamp_min(1e-12)
        # Cramer-Rao variance is 1 / information; dividing by T2**2 turns it
        # into the squared relative error, which is dimensionless.
        precision = (1.0 / (information * self.t2_ms.square())).mean()

        # Penalties run along the echo axis so that a batch of trains, shaped
        # (n_trains, echo_train_length), is penalized per train rather than
        # across trains.
        flip = flip_deg / 180.0
        echoes = flip.shape[-1]
        zero = flip.new_zeros(())
        smoothness = (
            (flip[..., 1:] - flip[..., :-1]).square().mean() if echoes > 1 else zero
        )
        curvature = (
            (flip[..., 2:] - 2.0 * flip[..., 1:-1] + flip[..., :-2]).square().mean()
            if echoes > 2
            else zero
        )
        return (
            precision.log()
            + self.smoothness_weight * smoothness
            + self.curvature_weight * curvature
            + self.rf_power_weight * flip.square().mean()
        )
