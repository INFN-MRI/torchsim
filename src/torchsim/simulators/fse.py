"""Fast spin echo."""

from __future__ import annotations

__all__ = ["FSESimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..sequence._array import as_torch, matched
from ..model import REFOCUSED, AbstractSimulator, StateMachineModel


class FSESimulator(AbstractSimulator):
    """A refocused echo train, sampled at every echo.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import FSESimulator

        sequence = FSESimulator(flip=180.0 * torch.ones(128), ESP=2.0, TR=5000.0)
        signal = sequence.simulate(T1=1000.0, T2=80.0)

    """

    model = StateMachineModel(
        properties={"T1": "t1_ms", "T2": "t2_ms", "M0": None, "B1": "b1"},
        triggers=REFOCUSED,
    )
    states = 10

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
    ) -> list:
        """Return the train, placed at the echo times it is timed from.

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
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        turns = torch.deg2rad(matched(phases, angles))
        # Not tensorized: a scalar spacing keeps the precision the caller
        # gave it, and the echo times are what the whole train is placed by.
        spacing_s = ESP * 1e-3

        parts = [(0.0, self.triggers.excitation(
            torch.pi / 180.0 * exc_flip, torch.pi / 180.0 * exc_phase
        ))]
        for index in range(angles.shape[-1]):
            echo_s = (index + 1) * spacing_s
            parts.append((
                echo_s - 0.5 * spacing_s,
                self.triggers.refocusing(angles[..., index], turns[..., index]),
            ))
            parts.append((echo_s, self.triggers.readout(turns[..., index])))
        return parts

    def repetition_s(self, played_s: Any, **protocol: Any) -> Any:
        """Return the TR, which the train waits out before the next one."""
        del played_s
        return protocol.get("TR", 1e6) * 1e-3

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Simulate one train, then let it recover for what is left of the TR."""
        played = self.played(**sequence)
        signal = super().evaluate(properties, **sequence)
        echoes = torch.atleast_1d(played["flip"]).shape[-1]
        left_ms = played.get("TR", 1e6) - played["ESP"] * echoes
        recovered = torch.exp(-1e3 / properties["T1"] * left_ms * 1e-3)
        # Both carry the tissue shape and the signal is (..., tissue, echo), so
        # one trailing axis lines them up and any leading train axis broadcasts.
        recovered = recovered[..., None]
        density = properties.get("M0", 1.0)
        if torch.is_tensor(density):
            density = density[..., None]
        return density * signal * (1 - recovered) / (1 - recovered * signal)
