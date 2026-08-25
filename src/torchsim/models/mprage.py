"""Magnetization-prepared rapid gradient-echo model."""

from __future__ import annotations

__all__ = ["MPRAGEModel"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import EpgModel
from ..sequence import SequenceDescription, mprage_description


class MPRAGEModel(EpgModel):
    """An inversion followed by a spoiled train, sampled at the k-space centre.

    Examples
    --------
    .. exec::

        from torchsim.models import MPRAGEModel

        model = MPRAGEModel()
        signal = model.simulate(
            T1=(200.0, 1000.0),
            inv_efficiency=0.95,
            TI=500.0,
            flip=5.0,
            TRspgr=5.0,
            nshots=128,
        )

    """

    properties = {"T1": "t1_ms", "M0": "m0", "inv_efficiency": "inversion_efficiency"}
    # The train samples at the pulse and spoils after it, so no transverse
    # magnetization survives an interval and no T2 is asked for.
    fixed = {"t2_ms": 100.0}
    states = 1
    record = "acquired"

    def describe(
        self,
        *,
        TI: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float | npt.ArrayLike,
        nshots: int | npt.ArrayLike,
    ) -> SequenceDescription:
        """Return the prepared train, in the units a description carries.

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

        Raises
        ------
        ValueError
            If the inversion time falls before the train's first excitation.
        """
        before, after = _shots_either_side(nshots)
        radians = torch.pi / 180.0
        inversion_s = torch.as_tensor(TI) * 1e-3
        readout_s = torch.as_tensor(TRspgr) * 1e-3
        if bool(inversion_s < before * readout_s):
            raise ValueError("TI must not precede the first MPRAGE excitation")
        return mprage_description(
            before,
            after,
            radians * torch.as_tensor(flip),
            readout_s,
            inversion_s,
        )

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the one sample the train acquires."""
        return 1j * super().evaluate(properties, **sequence)[..., 0]


# %% private module subroutines


def _shots_either_side(nshots: int | npt.ArrayLike) -> tuple[int, int]:
    """Split the readouts into those before and after the sampled one."""
    counts = torch.as_tensor(nshots).flatten()
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
