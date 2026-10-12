"""The three relaxometry contrasts a quantitative protocol is built from."""

from __future__ import annotations

__all__ = [
    "DoubleAngleSimulator",
    "InversionRecoverySimulator",
    "MultiEchoSimulator",
]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class InversionRecoverySimulator(Simulator):
    """Longitudinal recovery from an inversion, read at a series of delays.

    The readout samples the longitudinal magnetization alone. When the repetition
    time is omitted, the magnetization is fully relaxed at each inversion.
    ``offset`` is a constant added to the recovery, such as a magnitude noise
    floor.

    Examples
    --------
    .. exec::

        from blochsim.simulators import InversionRecoverySimulator

        sequence = InversionRecoverySimulator(TI=(50.0, 400.0, 1100.0, 2500.0))
        signal = sequence.simulate(T1=(800.0, 1400.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "M0": None,
            "inv_efficiency": None,
            "offset": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the closed-form signal."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        TI: float | npt.ArrayLike,
        TR: float | npt.ArrayLike | None = None,
    ) -> torch.Tensor:
        """Return the longitudinal magnetization at each inversion time.

        Parameters
        ----------
        properties : mapping
            ``T1`` in ms, ``inv_efficiency`` as a fraction of a perfect inversion,
            ``offset`` added to the recovery, ``M0`` scaling.
        TI : float or array-like
            Inversion times in ms.
        TR : float or array-like, optional
            Repetition time in ms; fully relaxed at each inversion when not given.
        """
        held = across_contrasts(properties, TI, 0.0 if TR is None else TR)
        rate = 1e3 / held["T1"]
        efficiency = held.get("inv_efficiency", 1.0)

        # The excitation leaves nothing longitudinal behind, so what the next
        # inversion finds is what recovers in the time left after the readout.
        standing = 1.0 if TR is None else 1.0 - torch.exp(-rate * (TR - TI) * 1e-3)
        recovered = 1.0 - (1.0 + efficiency * standing) * torch.exp(-rate * TI * 1e-3)
        return held.get("M0", 1.0) * recovered + held.get("offset", 0.0)


class MultiEchoSimulator(Simulator):
    """Transverse decay, read at a series of echo times.

    The model is one exponential in the property ``T2``, which is T2 for a spin
    echo and T2* for a gradient echo.

    Examples
    --------
    .. exec::

        from blochsim.simulators import MultiEchoSimulator

        sequence = MultiEchoSimulator(TE=(10.0, 20.0, 40.0, 80.0))
        signal = sequence.simulate(T2=(40.0, 90.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T2": None,
            "M0": None,
            "offset": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the closed-form signal."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        TE: float | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the transverse magnetization at each echo time.

        Parameters
        ----------
        properties : mapping
            ``T2`` in ms (T2* for a gradient echo), ``offset`` added to the decay,
            ``M0`` scaling.
        TE : float or array-like
            Echo times in ms.
        """
        held = across_contrasts(properties, TE)
        decay = torch.exp(-1e3 / held["T2"] * TE * 1e-3)
        return held.get("M0", 1.0) * decay + held.get("offset", 0.0)


class DoubleAngleSimulator(Simulator):
    """Excitation at a series of flip angles, scaled by the transmit field.

    A voxel turns through the nominal angle times ``B1``, so a series of angles
    reads out ``B1``. Two angles suffice, and a ratio of two between them gives
    ``B1`` in closed form; any set of angles can be fitted.

    Examples
    --------
    .. exec::

        from blochsim.simulators import DoubleAngleSimulator

        sequence = DoubleAngleSimulator(flip=(60.0, 120.0))
        signal = sequence.simulate(B1=(0.8, 1.0, 1.2))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "B1": None,
            "M0": None,
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
    ) -> torch.Tensor:
        """Return the transverse magnetization each flip angle produces.

        Parameters
        ----------
        properties : mapping
            ``B1`` as a fraction of the nominal transmit field, ``M0`` scaling.
        flip : float or array-like
            Nominal flip angles in degrees.
        """
        held = across_contrasts(properties, flip)
        turned = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
        return held.get("M0", 1.0) * torch.sin(turned)
