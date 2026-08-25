"""Spoiled gradient-echo model."""

from __future__ import annotations

__all__ = ["SPGRModel"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import SignalModel
from ._contrast import across_contrasts


class SPGRModel(SignalModel):
    """The spoiled gradient-echo steady state, read at the echo time.

    Spoiling leaves no transverse magnetization to carry over, so the steady
    state is the Ernst expression in closed form and no state machine is run.

    Examples
    --------
    .. exec::

        from torchsim.models import SPGRModel

        model = SPGRModel()
        signal = model.simulate(T1=1000.0, T2star=30.0, flip=13.0, TR=10.0, TE=5.0)

    """

    properties = ("T1", "T2star", "M0", "B0", "chemshift")

    def evaluate(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the transverse magnetization a TE after the excitation.

        Parameters
        ----------
        properties:
            ``T1`` and ``T2star`` in milliseconds, ``B0`` and ``chemshift`` in
            Hz, and ``M0`` as a scaling.
        flip:
            Excitation flip angle in degrees.
        TR:
            Repetition time in milliseconds.
        TE:
            Echo time in milliseconds.
        """
        held = across_contrasts(properties, flip, TR, TE)
        angle = torch.pi / 180.0 * torch.as_tensor(flip)
        repetition_s = torch.as_tensor(TR) * 1e-3
        echo_s = torch.as_tensor(TE) * 1e-3

        density = held.get("M0", 1.0)
        # The off-resonance map follows the Freeman-Hill convention and this
        # expression the Ernst-Anderson one, which turn the opposite way.
        offset = 2 * torch.pi * (held.get("chemshift", 0.0) - held.get("B0", 0.0))

        recovery = torch.exp(-1e3 / held["T1"] * repetition_s)
        decay = torch.exp(-1e3 / held["T2star"] * echo_s)
        turn = torch.exp(1j * offset * echo_s)

        transverse = (
            density
            * (1 - recovery)
            * torch.sin(angle)
            / (1 - recovery * torch.cos(angle))
        )
        return transverse * decay * turn
