"""Spoiled gradient echo, in closed form."""

from __future__ import annotations

__all__ = ["SPGRSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class SPGRSimulator(Simulator):
    """The spoiled gradient-echo steady state, read at the echo time.

    Spoiling leaves no transverse magnetization to carry over, so the steady
    state is the Ernst expression [1]_ in closed form and no state machine is
    run. Fitting T1 to a set of these at different flip angles is DESPOT1
    [2]_.

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

        from torchsim.simulators import SPGRSimulator

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

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
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
