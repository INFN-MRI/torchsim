"""MPnRAGE: inversion-prepared spoiled gradient echo, every shot read."""

from __future__ import annotations

__all__ = ["MPnRAGESimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import SPOILED, Simulator, SpinPhysics
from ..sequence._array import as_torch, matched


class MPnRAGESimulator(Simulator):
    """MPnRAGE: an inversion and a spoiled gradient-echo train, every shot read.

    One inversion, then ``nshots`` spoiled gradient-echo readouts, each sampled at
    the pulse. The signal is one complex sample per shot, along the
    inversion-recovery curve.

    Parameters
    ----------
    nshots : int
        Readouts per inversion block.
    flip : float or array-like
        Excitation flip angle, in degrees; scalar, or one per shot.
    TR : float or array-like
        Repetition time, in ms; scalar or one per shot.
    TI : float or array-like, default=0.0
        Delay from the inversion to the first excitation, in ms.
    phases : float or array-like, default=0.0
        Excitation phases, in degrees; scalar or one per shot.
    states : int, default=1
        Configuration states; only the longitudinal state is carried.

    Tissue properties
    -----------------
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.
    inv_efficiency : float or array-like, default=1.0
        Fraction of a perfect inversion.

    Examples
    --------
    .. exec::

        from blochsim.simulators import MPnRAGESimulator

        sequence = MPnRAGESimulator(nshots=128, flip=5.0, TR=10.0)
        signal = sequence.simulate(T1=1000.0, inv_efficiency=0.95)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "M0": "m0",
            "B1": "b1",
            "inv_efficiency": "inversion_efficiency",
        },
        # The train samples at the pulse and spoils after it, so no transverse
        # magnetization survives an interval and no T2 is asked for.
        fixed={"t2_ms": 100.0},
        operators=SPOILED,
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
        """Return the inversion and the train of excitations and readouts.

        Raises
        ------
        ValueError
            If ``nshots`` is not positive, or ``flip`` is neither scalar nor one
            per shot.
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

        parts = [self.operators.inversion(duration_s=TI * 1e-3)]
        for index in range(shots):
            parts.append(self.operators.excitation(angles[index], turns[index]))
            parts.append(
                self.operators.readout(turns[index], duration_s=spacing_s[index])
            )
        return parts

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the signal at every shot."""
        return 1j * super().evaluate(properties, **sequence)
