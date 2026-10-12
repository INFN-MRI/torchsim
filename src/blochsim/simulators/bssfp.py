"""Balanced steady-state free precession, in closed form."""

from __future__ import annotations

__all__ = ["bSSFPSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from .._model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class bSSFPSimulator(Simulator):
    """Balanced steady-state free precession (bSSFP) signal at the echo time.

    Closed-form steady state of a balanced train as a function of the phase a
    voxel accrues across one TR, phase cycling included; no state machine is run.
    The signal is complex, with phase set by the off-resonance at ``TE``. Fitting
    T2 to a set of these, with T1 from :class:`SPGRSimulator`, is DESPOT2 [1]_.

    Parameters
    ----------
    flip : float or array-like
        Excitation flip angle, in degrees.
    TR : float or array-like
        Repetition time, in ms.
    TE : float or array-like, default=None
        Echo time, in ms; ``TR / 2`` when not given.
    phase_inc : float or array-like, default=180.0
        Phase-cycling increment of the excitation between repetitions, in
        degrees.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B0 : float or array-like, default=0.0
        Off-resonance, in Hz.
    chemshift : float or array-like, default=0.0
        Chemical shift of the species, in Hz; enters the phase with the opposite
        sign to ``B0``.

    References
    ----------
    .. [1] Deoni, S. C. L., Peters, T. M., Rutt, B. K., "High-resolution T1
       and T2 mapping of the brain in a clinically acceptable time with
       DESPOT1 and DESPOT2", Magnetic Resonance in Medicine 53.1 (2005),
       pp. 237-241. https://doi.org/10.1002/mrm.20314

    Examples
    --------
    .. exec::

        from blochsim.simulators import bSSFPSimulator

        sequence = bSSFPSimulator(flip=60.0, TR=10.0, TE=5.0)
        signal = sequence.simulate(T1=1000.0, T2=100.0)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2": None,
            "M0": None,
            "B0": None,
            "chemshift": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the closed-form signal."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TE: float | npt.ArrayLike | None = None,
        phase_inc: float | npt.ArrayLike = 180.0,
    ) -> torch.Tensor:
        """Return the transverse magnetization a TE after the excitation."""
        if TE is None:
            TE = TR / 2
        held = across_contrasts(properties, flip, TR, TE, phase_inc)
        radians = torch.pi / 180.0
        angle = radians * flip
        cycling = radians * phase_inc
        repetition_s = TR * 1e-3
        echo_s = TE * 1e-3

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
