"""Magnetization-prepared multi-echo spoiled GRE."""

from __future__ import annotations

__all__ = ["MPnRAGESimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..sequence._array import as_torch, matched
from ..model import SPOILED, AbstractSimulator, StateMachineModel


class MPnRAGESimulator(AbstractSimulator):
    """An inversion followed by a spoiled gradient-echo train, every shot read.

    Examples
    --------
    .. exec::

        from torchsim.simulators import MPnRAGESimulator

        sequence = MPnRAGESimulator(nshots=128, flip=5.0, TR=10.0)
        signal = sequence.simulate(T1=1000.0, inv_efficiency=0.95)

    """

    model = StateMachineModel(
        properties={
            "T1": "t1_ms",
            "M0": "m0",
            "B1": "b1",
            "inv_efficiency": "inversion_efficiency",
        },
        # The train samples at the pulse and spoils after it, so no transverse
        # magnetization survives an interval and no T2 is asked for.
        fixed={"t2_ms": 100.0},
        triggers=SPOILED,
    )
    states = 1

    def layout(
        self,
        *,
        nshots: int,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike = 0.0,
        phases: float | npt.ArrayLike = 0.0,
    ) -> list:
        """Return the prepared train, every shot acquired.

        Parameters
        ----------
        nshots : int
            Readouts per inversion block.
        flip : float or array-like
            Excitation flip angle in degrees, scalar or one per shot.
        TR : float or array-like
            Repetition time in milliseconds.
        TI : float or array-like, optional
            Inversion time in milliseconds.
        phases : float or array-like, optional
            Excitation phases in degrees.
        """
        shots = int(as_torch(nshots).reshape(()).item())
        if shots < 1:
            raise ValueError("nshots must be positive")
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        if angles.numel() == 1:
            angles = angles.expand(shots)
        elif angles.numel() != shots:
            raise ValueError("flip must be scalar or contain nshots values")
        turns = torch.deg2rad(matched(phases, angles))
        spacing_s = matched(TR, angles) * 1e-3

        parts = [self.triggers.inversion(duration_s=TI * 1e-3)]
        for index in range(shots):
            parts.append(self.triggers.excitation(angles[index], turns[index]))
            parts.append(
                self.triggers.readout(turns[index], duration_s=spacing_s[index])
            )
        return parts

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the train the inversion recovers through."""
        return 1j * super().evaluate(properties, **sequence)
