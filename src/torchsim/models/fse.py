"""Fast spin-echo model."""

from __future__ import annotations

__all__ = ["FSEModel"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import EpgModel
from ..sequence import FSE, SequenceDescription, fse_description


class FSEModel(EpgModel):
    """Fast spin echo, as an extended phase graph over the fused kernels.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.models import FSEModel

        model = FSEModel()
        signal = model.simulate(
            T1=1000.0, T2=80.0, flip=180.0 * torch.ones(128), ESP=2.0, TR=5000.0
        )

    """

    properties = {"T1": "t1_ms", "T2": "t2_ms", "M0": None, "B1": "b1"}
    simulator = FSE()

    def describe(
        self,
        *,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
    ) -> SequenceDescription:
        """Return the refocused train, in the units a description carries.

        Parameters
        ----------
        flip:
            Refocusing flip angles in degrees, one per echo.
        ESP:
            Echo spacing in milliseconds.
        phases:
            Refocusing phases in degrees.
        exc_flip, exc_phase:
            The excitation, in degrees.
        TR:
            Repetition time in milliseconds, which sets how far the
            longitudinal magnetization recovers before the next train.
        """
        del TR  # read by evaluate, which applies the recovery it implies
        radians = torch.pi / 180.0
        return fse_description(
            radians * torch.as_tensor(flip),
            ESP * 1e-3,
            phases_rad=radians * torch.as_tensor(phases),
            excitation_flip_rad=radians * exc_flip,
            excitation_phase_rad=radians * exc_phase,
        )

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Simulate one train, then let it recover for what is left of the TR."""
        signal = super().evaluate(properties, **sequence)
        flip = torch.atleast_1d(torch.as_tensor(sequence["flip"]))
        left_ms = sequence.get("TR", 1e6) - sequence["ESP"] * flip.shape[-1]
        rate = 1e3 / properties["T1"]
        recovered = torch.exp(-rate * left_ms * 1e-3)
        # Both carry the tissue shape and the signal is (..., tissue, echo), so
        # one trailing axis lines them up and any leading train axis broadcasts.
        recovered = recovered[..., None]
        density = properties.get("M0", 1.0)
        if torch.is_tensor(density):
            density = density[..., None]
        return density * signal * (1 - recovered) / (1 - recovered * signal)
