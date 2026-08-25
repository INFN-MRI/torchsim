"""Balanced steady-state free precession model."""

from __future__ import annotations

__all__ = ["bSSFPModel"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import SignalModel
from ._contrast import across_contrasts


class bSSFPModel(SignalModel):
    """The balanced SSFP steady state, read at the echo time.

    A balanced repetition returns the transverse magnetization to where it
    started, so the steady state is a closed form in the phase a voxel accrues
    across one TR and no state machine is run.

    Examples
    --------
    .. exec::

        from torchsim.models import bSSFPModel

        model = bSSFPModel()
        signal = model.simulate(T1=1000.0, T2=100.0, flip=60.0, TR=10.0, TE=5.0)

    """

    properties = ("T1", "T2", "M0", "B0", "chemshift")

    def evaluate(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike | None = None,
        phase_inc: float | npt.ArrayLike = 180.0,
    ) -> torch.Tensor:
        """Return the transverse magnetization a TE after the excitation.

        Parameters
        ----------
        properties:
            ``T1`` and ``T2`` in milliseconds, ``B0`` and ``chemshift`` in Hz,
            and ``M0`` as a scaling.
        flip:
            Excitation flip angle in degrees.
        TR:
            Repetition time in milliseconds.
        TE:
            Echo time in milliseconds, defaulting to the middle of the
            repetition, where a balanced sequence refocuses.
        phase_inc:
            Linear phase-cycling increment in degrees.
        """
        if TE is None:
            TE = torch.as_tensor(TR) / 2
        held = across_contrasts(properties, flip, TR, TE, phase_inc)
        radians = torch.pi / 180.0
        angle = radians * torch.as_tensor(flip)
        cycling = radians * torch.as_tensor(phase_inc)
        repetition_s = torch.as_tensor(TR) * 1e-3
        echo_s = torch.as_tensor(TE) * 1e-3

        density = held.get("M0", 1.0)
        # The off-resonance map follows the Freeman-Hill convention and this
        # expression the Ernst-Anderson one, which turn the opposite way.
        offset = 2 * torch.pi * (held.get("chemshift", 0.0) - held.get("B0", 0.0))

        recovery = torch.exp(-1e3 / held["T1"] * repetition_s)
        decay = torch.exp(-1e3 / held["T2"] * echo_s)
        turn = torch.exp(1j * offset * echo_s)

        # What a voxel accrues across one repetition, phase cycling included.
        accrued = offset * repetition_s + cycling
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        cos_t, sin_t = torch.cos(accrued), torch.sin(accrued)

        den = (1 - recovery * cos_a) * (1 - decay * cos_t) - (
            decay * (recovery - cos_a)
        ) * (decay - cos_t)
        along = -density * (1 - recovery) * decay * sin_a * sin_t / den
        across = density * (1 - recovery) * sin_a * (1 - decay * cos_t) / den

        return (along + 1j * across) * decay * turn
