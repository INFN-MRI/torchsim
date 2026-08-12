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
        Weight of first-difference flip-train smoothness.
    rf_power_weight : float, optional
        Weight of mean squared flip angle.
    parameter_name : str, optional
        Key used when a :class:`SequenceOptimizer` receives a dictionary.
    """

    def __init__(
        self,
        t1_ms: torch.Tensor,
        t2_ms: torch.Tensor,
        echo_spacing_ms: float,
        *,
        phases_deg: float = 90.0,
        smoothness_weight: float = 2e-4,
        rf_power_weight: float = 2e-6,
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
        precision = (self.t2_ms.square() / information).mean()
        smoothness = (
            (flip_deg[1:] - flip_deg[:-1]).square().mean()
            if flip_deg.numel() > 1
            else flip_deg.new_zeros(())
        )
        return (
            precision
            + self.smoothness_weight * smoothness
            + self.rf_power_weight * flip_deg.square().mean()
        )
