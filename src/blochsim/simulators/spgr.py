"""Spoiled gradient echo, in closed form."""

from __future__ import annotations

__all__ = ["SPGRSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from .._model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class SPGRSimulator(Simulator):
    """Spoiled gradient echo (SPGR, FLASH) steady state, in closed form.

    Spoiling leaves no transverse magnetization to carry over, so the steady state
    is the Ernst expression [1]_,
    ``M0 (1 - E1) sin(a) / (1 - E1 cos(a)) exp(-TE / T2*)`` with
    ``E1 = exp(-TR / T1)``, times the phase ``exp(i 2 pi (chemshift - B0) TE)``
    accrued to the echo. The signal is one complex sample; fitting T1 to several
    flip angles is DESPOT1 [2]_.

    Parameters
    ----------
    flip : float or array-like
        Excitation flip angle, in degrees.
    TR : float or array-like
        Repetition time, in ms.
    TE : float or array-like
        Echo time, in ms.

    Tissue properties
    -----------------
    T1, T2star : float or array-like
        Longitudinal relaxation time and effective transverse relaxation time, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B0 : float or array-like, default=0.0
        Off-resonance, in Hz.
    chemshift : float or array-like, default=0.0
        Chemical shift, in Hz.

    References
    ----------
    .. [1] Ernst, R. R., Anderson, W. A., "Application of Fourier transform
       spectroscopy to magnetic resonance", Review of Scientific Instruments
       37.1 (1966), pp. 93-102. https://doi.org/10.1063/1.1719961

    .. [2] Deoni, S. C. L., Rutt, B. K., Peters, T. M., "Rapid combined T1 and
       T2 mapping using gradient recalled acquisition in the steady state",
       Magnetic Resonance in Medicine 49.3 (2003), pp. 515-526.
       https://doi.org/10.1002/mrm.10407

    Examples
    --------
    .. exec::

        from blochsim.simulators import SPGRSimulator

        sequence = SPGRSimulator(flip=13.0, TR=10.0, TE=5.0)
        signal = sequence.simulate(T1=1000.0, T2star=30.0)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2star": None,
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
        TE: float | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the transverse magnetization a TE after the excitation."""
        held = across_contrasts(properties, flip, TR, TE)
        angle = torch.pi / 180.0 * flip
        repetition_s = TR * 1e-3
        echo_s = TE * 1e-3

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
