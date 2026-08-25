"""Unbalanced SSFP MR fingerprinting."""

from __future__ import annotations

__all__ = ["MRFSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import UNBALANCED, AbstractSimulator, StateMachineModel


class MRFSimulator(AbstractSimulator):
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

    model = StateMachineModel(
        properties={
            "T1": "t1_ms",
            "T2": "t2_ms",
            "M0": "m0",
            "B1": "b1",
            "inv_efficiency": "inversion_efficiency",
        },
        triggers=UNBALANCED,
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
        flip:
            Excitation flip angles in degrees, one per repetition.
        TR:
            Repetition time in milliseconds, scalar or one per repetition.
        TI:
            Inversion time in milliseconds.
        phases:
            Excitation phases in degrees.
        """
        angles = torch.deg2rad(torch.atleast_1d(torch.as_tensor(flip)))
        turns = torch.deg2rad(torch.as_tensor(phases, dtype=angles.dtype))
        turns = turns.expand_as(angles) if turns.numel() == 1 else turns
        spacing_s = torch.as_tensor(TR, dtype=angles.dtype) * 1e-3
        spacing_s = (
            spacing_s.expand_as(angles) if spacing_s.numel() == 1 else spacing_s
        )

        parts = [self.triggers.inversion(duration_s=torch.as_tensor(TI) * 1e-3)]
        for index in range(angles.numel()):
            parts.append(self.triggers.excitation(angles[index], turns[index]))
            parts.append(
                self.triggers.readout(turns[index], duration_s=spacing_s[index])
            )
        return parts

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the last playing of the train, which is the one measured."""
        signal = super().evaluate(properties, **sequence)
        played = {**self.protocol, **sequence}
        echoes = torch.atleast_1d(torch.as_tensor(played["flip"])).shape[-1]
        return 1j * signal[..., -echoes:]
