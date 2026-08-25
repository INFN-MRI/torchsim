"""MP2RAGE, in closed form."""

from __future__ import annotations

__all__ = ["MP2RAGESimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import AbstractSimulator, StateMachineModel
from ..sequence._array import arrays, as_torch


class MP2RAGESimulator(AbstractSimulator):
    """Two gradient-echo blocks read at two inversion times, in closed form.

    The train is spoiled and sampled at the k-space centre of each block, so
    only the longitudinal magnetization carries between shots and the steady
    state it settles into has a closed form.

    Examples
    --------
    .. exec::

        from torchsim.simulators import MP2RAGESimulator

        sequence = MP2RAGESimulator(
            TI=(500.0, 1500.0),
            flip=5.0,
            TRspgr=5.0,
            TRmp2rage=3000.0,
            nshots=128,
        )
        signal = sequence.simulate(T1=(200.0, 1000.0), inv_efficiency=0.95)

    """

    model = StateMachineModel(
        properties={
            "T1": None,
            "M0": None,
            "inv_efficiency": None,
        },
    )

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float | npt.ArrayLike,
        TRmp2rage: float | npt.ArrayLike,
        nshots: int | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the two sampled magnetizations, along a trailing axis.

        Parameters
        ----------
        properties:
            ``T1`` in milliseconds, ``M0`` as a scaling, and the inversion
            efficiency.
        TI:
            The two inversion times in milliseconds, measured to the sampled
            shot of each block.
        flip:
            Excitation flip angle in degrees, one per block or one shared.
        TRspgr:
            Repetition time in milliseconds of one readout.
        TRmp2rage:
            Repetition time in milliseconds of the whole inversion block.
        nshots:
            Readouts per block, either the total -- halved for each block --
            or ``(before, after)`` the sampled shot.
        """
        radians = torch.pi / 180.0
        angle = _shared_or_two(radians * flip)
        before, after = _shots_either_side(nshots)
        inversion_s = TI.flatten() * 1e-3
        readout_s = TRspgr * 1e-3
        block_s = TRmp2rage * 1e-3

        efficiency = properties.get("inv_efficiency", 1.0)
        rate = 1e3 / properties["T1"]

        # The three waits the magnetization recovers through freely: before the
        # first block, between the two, and after the second.
        waits = (
            inversion_s[0] - before * readout_s,
            inversion_s[1] - inversion_s[0] - (after + before) * readout_s,
            block_s - inversion_s[1] - after * readout_s,
        )
        held = [torch.exp(-rate * wait) for wait in waits]
        shot = torch.exp(-rate * readout_s)
        turn = torch.cos(angle)
        shots = before + after

        # The steady state one whole block leaves behind, which is what the
        # inversion of the next block acts on.
        settled = (1 - held[0]) * _through_shots(1.0, turn[0], shot, shots)
        settled = settled * held[1] + (1 - held[1])
        settled = settled * _through_shots(1.0, turn[1], shot, shots)
        settled = settled * held[2] + (1 - held[2])
        settled = settled / (
            1
            + efficiency
            * held[0]
            * held[1]
            * held[2]
            * (turn[0] * shot * turn[1] * shot) ** shots
        )

        # The first block reads what the inversion left, driven down over the
        # readouts before its centre.
        driven = _through_shots(
            -efficiency * settled * held[0] + (1 - held[0]), turn[0], shot, before
        )
        first = torch.sin(angle[0]) * driven

        # The second reads what the rest of the first block, the wait between
        # them and its own leading readouts leave.
        driven = driven * _through_shots(1.0, turn[1], shot, after)
        driven = _through_shots(
            driven * held[1] + (1 - held[1]), turn[1], shot, before
        )
        second = torch.sin(angle[1]) * driven

        # Both blocks carry the voxel shape and the signal is (..., voxel,
        # block), so one trailing axis lines the density up with them.
        density = properties.get("M0", 1.0)
        if torch.is_tensor(density):
            density = density[..., None]
        return density * torch.stack((first, second), dim=-1)


# %% private module subroutines


def _through_shots(
    held: Any, turn: torch.Tensor, shot: torch.Tensor, count: Any
) -> torch.Tensor:
    """Return what ``count`` spoiled readouts leave of what they find.

    Each readout tips the longitudinal magnetization down by ``turn`` and lets
    it recover by ``shot``, which drives whatever it started from towards a
    steady state of its own.
    """
    survives = (turn * shot) ** count
    return held * survives + (1 - shot) * (1 - survives) / (1 - turn * shot)


def _shared_or_two(value: torch.Tensor) -> torch.Tensor:
    """Return one value per block, sharing a single one between the two."""
    flat = value.flatten()
    return flat.repeat(2) if flat.numel() == 1 else flat


def _shots_either_side(nshots: int | npt.ArrayLike) -> tuple[Any, Any]:
    """Split the readouts into those before and after the sampled one."""
    counts = as_torch(nshots).flatten()
    if counts.numel() == 1:
        half = counts[0] // 2
        return half, half
    return counts[0], counts[1]
