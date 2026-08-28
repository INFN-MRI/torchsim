"""Unbalanced SSFP MR fingerprinting."""

from __future__ import annotations

__all__ = ["MRFSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..sequence._array import as_torch, matched
from ..model import UNBALANCED, Simulator, SpinPhysics


class MRFSimulator(Simulator):
    """An inversion, then a variable flip-angle train whose readouts wind on.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import MRFSimulator

        sequence = MRFSimulator(
            flip=torch.linspace(5.0, 60.0, 1000), TR=10.0, states=20
        )
        signal = sequence.simulate(T1=1000.0, T2=80.0, inv_efficiency=0.95)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "T2": "t2_ms",
            "M0": "m0",
            "B1": "b1",
            "inv_efficiency": "inversion_efficiency",
        },
        operators=UNBALANCED,
    )
    states = 10

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike = 0.0,
        phases: float | npt.ArrayLike = 0.0,
    ) -> list:
        """Return the prepared train, one excitation and one sample per TR.

        Parameters
        ----------
        flip : float or array-like
            Excitation flip angles in degrees, one per repetition.
        TR : float or array-like
            Repetition time in milliseconds, scalar or one per repetition.
        TI : float or array-like, optional
            Inversion time in milliseconds.
        phases : float or array-like, optional
            Excitation phases in degrees.
        """
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        turns = torch.deg2rad(matched(phases, angles))
        spacing_s = matched(TR, angles) * 1e-3

        parts = [self.operators.inversion(duration_s=TI * 1e-3)]
        for index in range(angles.numel()):
            parts.append(self.operators.excitation(angles[index], turns[index]))
            parts.append(
                self.operators.readout(turns[index], duration_s=spacing_s[index])
            )
        return parts

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the last playing of the train, which is the one measured."""
        signal = super().evaluate(properties, **sequence)
        played = self.played(**sequence)
        echoes = torch.atleast_1d(played["flip"]).shape[-1]
        return 1j * signal[..., -echoes:]
