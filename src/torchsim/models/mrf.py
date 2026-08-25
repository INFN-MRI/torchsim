"""Unbalanced SSFP MR fingerprinting model."""

from __future__ import annotations

__all__ = ["MRFModel"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import EpgModel
from ..sequence import SSFPFID, SequenceDescription, mrf_description


class MRFModel(EpgModel):
    """Inversion-prepared unbalanced SSFP fingerprinting.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.models import MRFModel

        model = MRFModel()
        signal = model.simulate(
            T1=1000.0,
            T2=80.0,
            inv_efficiency=0.95,
            flip=torch.linspace(5.0, 60.0, 1000),
            TR=10.0,
            nstates=20,
        )

    """

    properties = {
        "T1": "t1_ms",
        "T2": "t2_ms",
        "M0": "m0",
        "B1": "b1",
        "inv_efficiency": "inversion_efficiency",
    }
    simulator = SSFPFID()
    states = 10

    def describe(
        self,
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike = 0.0,
    ) -> SequenceDescription:
        """Return the prepared fingerprint train, in a description's units.

        Parameters
        ----------
        flip:
            Excitation flip angles in degrees, one per repetition.
        TR:
            Repetition time in milliseconds, scalar or one per repetition.
        TI:
            Inversion time in milliseconds.
        """
        radians = torch.pi / 180.0
        return mrf_description(
            radians * torch.as_tensor(flip),
            torch.as_tensor(TR) * 1e-3,
            inversion_time_s=torch.as_tensor(TI) * 1e-3,
        )

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the last playing of the train, which is the one measured."""
        signal = super().evaluate(properties, **sequence)
        echoes = torch.atleast_1d(torch.as_tensor(sequence["flip"])).shape[-1]
        return 1j * signal[..., -echoes:]
