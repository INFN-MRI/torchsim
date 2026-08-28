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

    A spin echo long enough after the inversion samples the longitudinal
    magnetization alone, so the contrast is the recovery curve itself. The
    repetition time bounds how much the magnetization has recovered by the
    time the next inversion arrives; leave it out for a fully relaxed one.

    ``offset`` is a constant added to the recovery, which is what a magnitude
    reconstruction's noise floor looks like to a fit and what stops the
    fitted T1 absorbing it.

    Examples
    --------
    .. exec::

        from torchsim.simulators import InversionRecoverySimulator

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

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
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
        properties:
            ``T1`` in milliseconds, ``M0`` as a scaling, ``inv_efficiency``
            as a fraction of a perfect inversion, and ``offset`` as a constant
            added to the recovery.
        TI:
            Inversion times in milliseconds.
        TR:
            Repetition time in milliseconds. Left out, the magnetization is
            fully relaxed when each inversion arrives.
        """
        held = across_contrasts(properties, TI, 0.0 if TR is None else TR)
        rate = 1e3 / held["T1"]
        efficiency = held.get("inv_efficiency", 1.0)

        # The excitation leaves nothing longitudinal behind, so what the next
        # inversion finds is what recovers in the time left after the readout.
        standing = (
            1.0
            if TR is None
            else 1.0 - torch.exp(-rate * (TR - TI) * 1e-3)
        )
        recovered = 1.0 - (1.0 + efficiency * standing) * torch.exp(
            -rate * TI * 1e-3
        )
        return held.get("M0", 1.0) * recovered + held.get("offset", 0.0)


class MultiEchoSimulator(Simulator):
    """Transverse decay, read at a series of echo times.

    A multi-echo spin echo decays with T2 and a multi-echo gradient echo with
    T2*; which one is being measured is a property of the sequence that
    produced the data, and the model is the same exponential either way.

    Examples
    --------
    .. exec::

        from torchsim.simulators import MultiEchoSimulator

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

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
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
        properties:
            ``T2`` in milliseconds -- or T2* for a gradient echo -- ``M0`` as
            a scaling, and ``offset`` as a constant added to the decay.
        TE:
            Echo times in milliseconds.
        """
        held = across_contrasts(properties, TE)
        decay = torch.exp(-1e3 / held["T2"] * TE * 1e-3)
        return held.get("M0", 1.0) * decay + held.get("offset", 0.0)


class DoubleAngleSimulator(Simulator):
    """Excitation at a series of flip angles, scaled by the transmit field.

    What a voxel actually turns through is the nominal angle times the local
    transmit efficiency, so a series of angles reads that efficiency out.
    Two angles are enough, and where the second is twice the first their
    ratio gives ``B1`` in closed form -- but nothing here requires that, so
    any set of angles can be fitted.

    Examples
    --------
    .. exec::

        from torchsim.simulators import DoubleAngleSimulator

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
    ) -> torch.Tensor:
        """Return the transverse magnetization each flip angle produces.

        Parameters
        ----------
        properties:
            ``B1`` as a fraction of the nominal transmit -- one where the
            pulse turns through exactly the angle it asks for -- and ``M0`` as
            a scaling.
        flip:
            Nominal flip angles in degrees.
        """
        held = across_contrasts(properties, flip)
        turned = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
        return held.get("M0", 1.0) * torch.sin(turned)
