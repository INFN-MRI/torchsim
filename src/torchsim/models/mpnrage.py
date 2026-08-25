"""Magnetization-prepared multi-echo spoiled GRE model."""

from __future__ import annotations

__all__ = ["MPnRAGEModel"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import EpgModel
from ..sequence import SPGR, SequenceDescription, mpnrage_description


class MPnRAGEModel(EpgModel):
    """An inversion followed by a spoiled gradient-echo train, every shot read.

    Examples
    --------
    .. exec::

        from torchsim.models import MPnRAGEModel

        model = MPnRAGEModel()
        signal = model.simulate(
            T1=1000.0, inv_efficiency=0.95, nshots=128, flip=5.0, TR=10.0
        )

    """

    properties = {
        "T1": "t1_ms",
        "M0": "m0",
        "B1": "b1",
        "inv_efficiency": "inversion_efficiency",
    }
    # The train samples at the pulse and spoils after it, so no transverse
    # magnetization survives an interval and no T2 is asked for.
    fixed = {"t2_ms": 100.0}
    simulator = SPGR()
    states = 1

    def describe(
        self,
        *,
        nshots: int,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike = 0.0,
    ) -> SequenceDescription:
        """Return the prepared train, in the units a description carries.

        Parameters
        ----------
        nshots:
            Readouts per inversion block.
        flip:
            Excitation flip angle in degrees, scalar or one per shot.
        TR:
            Repetition time in milliseconds.
        TI:
            Inversion time in milliseconds.
        """
        radians = torch.pi / 180.0
        return mpnrage_description(
            int(torch.as_tensor(nshots).reshape(()).item()),
            radians * torch.as_tensor(flip),
            torch.as_tensor(TR) * 1e-3,
            inversion_time_s=torch.as_tensor(TI) * 1e-3,
        )

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the train the inversion recovers through."""
        return 1j * super().evaluate(properties, **sequence)
