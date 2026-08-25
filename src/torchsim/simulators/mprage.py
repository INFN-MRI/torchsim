"""Magnetization-prepared rapid gradient echo."""

from __future__ import annotations

__all__ = ["MPRAGESimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..sequence._array import as_torch, matched
from ..model import SPOILED, AbstractSimulator, StateMachineModel
from ..sequence import AdcRole


class MPRAGESimulator(AbstractSimulator):
    """An inversion followed by a spoiled train, sampled at the k-space centre.

    Examples
    --------
    .. exec::

        from torchsim.simulators import MPRAGESimulator

        sequence = MPRAGESimulator(TI=500.0, flip=5.0, TRspgr=5.0, nshots=128)
        signal = sequence.simulate(T1=(200.0, 1000.0), inv_efficiency=0.95)

    """

    model = StateMachineModel(
        properties={
            "T1": "t1_ms",
            "M0": "m0",
            "inv_efficiency": "inversion_efficiency",
        },
        # The train samples at the pulse and spoils after it, so no transverse
        # magnetization survives an interval and no T2 is asked for.
        fixed={"t2_ms": 100.0},
        triggers=SPOILED,
    )
    states = 1
    record = "acquired"

    def __init__(self, **settings: Any) -> None:
        """Record only the shot that samples the k-space centre."""
        settings.setdefault("record", type(self).record)
        super().__init__(**settings)

    def layout(
        self,
        *,
        TI: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float | npt.ArrayLike,
        nshots: int | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
    ) -> list:
        """Return the prepared train, with one shot marked as acquired.

        Parameters
        ----------
        TI:
            Inversion time in milliseconds, measured to the sampled shot.
        flip:
            Excitation flip angle in degrees, scalar or one per shot.
        TRspgr:
            Repetition time in milliseconds of one readout.
        nshots:
            Readouts in the inversion block, either the total -- split as
            evenly as an odd centre allows -- or ``(before, after)`` the shot
            that samples the k-space centre.
        phases:
            Excitation phases in degrees.

        Raises
        ------
        ValueError
            If the inversion time falls before the train's first excitation.
        """
        before, after = _shots_either_side(nshots)
        shots = before + after + 1
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        if angles.numel() == 1:
            angles = angles.expand(shots)
        elif angles.numel() != shots:
            raise ValueError("flip must be scalar or contain one value per shot")
        turns = torch.deg2rad(matched(phases, angles))
        readout_s = matched(TRspgr, angles) * 1e-3
        inversion_s = TI * 1e-3
        if bool(inversion_s < readout_s[:before].sum()):
            raise ValueError("TI must not precede the first MPRAGE excitation")

        # The inversion time is measured to the sampled shot, so the wait after
        # the inversion is what is left once the shots before it have played.
        parts = [
            self.triggers.inversion(
                duration_s=inversion_s - readout_s[:before].sum()
            )
        ]
        for index in range(shots):
            acquired = index == before
            parts.append(self.triggers.excitation(angles[index], turns[index]))
            parts.append(
                self.triggers.readout(
                    turns[index],
                    role=AdcRole.SINGLE if acquired else AdcRole.NON_ACQUIRED,
                    is_echo=acquired,
                    duration_s=readout_s[index],
                )
            )
        return parts

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the one sample the train acquires."""
        return 1j * super().evaluate(properties, **sequence)[..., 0]


# %% private module subroutines


def _shots_either_side(nshots: int | npt.ArrayLike) -> tuple[int, int]:
    """Split the readouts into those before and after the sampled one."""
    counts = as_torch(nshots).flatten()
    if counts.numel() == 1:
        total = int(counts.item())
        if total < 1:
            raise ValueError("nshots must be positive")
        before = total // 2
        return before, total - before - 1
    if counts.numel() == 2:
        before, after = (int(value.item()) for value in counts)
        if before < 0 or after < 0:
            raise ValueError("nshots entries must be nonnegative")
        return before, after
    raise ValueError("nshots must be scalar or (before, after)")
