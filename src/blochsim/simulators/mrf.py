"""Unbalanced SSFP MR fingerprinting."""

from __future__ import annotations

__all__ = ["MRFSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import UNBALANCED, Simulator, SpinPhysics
from ..sequence._array import as_torch, matched


class MRFSimulator(Simulator):
    """MR fingerprinting with an unbalanced SSFP (FISP) train.

    An inversion, then one excitation and one readout per TR. The readout
    gradient is unbalanced: each TR shifts every configuration state by one
    order. The signal is one complex sample per flip angle, from the last
    playing of the train.

    Parameters
    ----------
    flip : float or array-like
        Flip angles, in degrees, one per TR; their number sets the train
        length.
    TR : float or array-like
        Repetition time, in ms; scalar or one per TR.
    TI : float or array-like, default=0.0
        Delay from the inversion to the first excitation, in ms.
    phases : float or array-like, default=0.0
        RF phases, in degrees; scalar or one per TR.
    states : int, default=10
        Configuration states kept by the EPG simulation.
    repetitions : int or "auto", default=1
        Playings of the train; the last is the signal.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.
    inv_efficiency : float or array-like, default=1.0
        Fraction of a perfect inversion.

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import MRFSimulator

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
        """Return the train: inversion, then an excitation and a readout per TR."""
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

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the signal of the recorded playing of the train."""
        signal = super().evaluate(properties, **sequence)
        played = self.played(**sequence)
        echoes = torch.atleast_1d(played["flip"]).shape[-1]
        return 1j * signal[..., -echoes:]
