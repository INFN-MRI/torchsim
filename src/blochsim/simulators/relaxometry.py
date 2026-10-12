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

from .._model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class InversionRecoverySimulator(Simulator):
    """Inversion-recovery T1 relaxometry, in closed form.

    An inversion, a delay ``TI`` and a readout of the longitudinal magnetization,
    repeated every ``TR``. The signal is
    ``M0 (1 - (1 + inv_efficiency (1 - exp(-(TR - TI) / T1))) exp(-TI / T1)) + offset``,
    one real sample per inversion time; with ``TR`` omitted the magnetization is
    fully relaxed at each inversion and the bracket reduces to
    ``1 - (1 + inv_efficiency) exp(-TI / T1)``. The excitation leaves nothing
    longitudinal behind.

    Parameters
    ----------
    TI : float or array-like
        Inversion times, in ms; the trailing axis of the signal.
    TR : float or array-like, default=None
        Repetition time, in ms. Fully relaxed at each inversion when not given.

    Tissue properties
    -----------------
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    inv_efficiency : float or array-like, default=1.0
        Fraction of a perfect inversion.
    offset : float or array-like, default=0.0
        Constant added to the recovery, such as a magnitude noise floor.

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
        """Return the longitudinal magnetization at each inversion time."""
        held = across_contrasts(properties, TI, 0.0 if TR is None else TR)
        rate = 1e3 / held["T1"]
        efficiency = held.get("inv_efficiency", 1.0)

        # The excitation leaves nothing longitudinal behind, so what the next
        # inversion finds is what recovers in the time left after the readout.
        standing = 1.0 if TR is None else 1.0 - torch.exp(-rate * (TR - TI) * 1e-3)
        recovered = 1.0 - (1.0 + efficiency * standing) * torch.exp(-rate * TI * 1e-3)
        return held.get("M0", 1.0) * recovered + held.get("offset", 0.0)


class MultiEchoSimulator(Simulator):
    """Multi-echo T2 relaxometry, in closed form.

    Monoexponential transverse decay ``M0 exp(-TE / T2) + offset``, one real
    sample per echo time. The property ``T2`` is T2 for a spin-echo train and T2*
    for a gradient-echo train.

    Parameters
    ----------
    TE : float or array-like
        Echo times, in ms; the trailing axis of the signal.

    Tissue properties
    -----------------
    T2 : float or array-like
        Transverse relaxation time, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    offset : float or array-like, default=0.0
        Constant added to the decay, such as a magnitude noise floor.

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
        """Return the transverse magnetization at each echo time."""
        held = across_contrasts(properties, TE)
        decay = torch.exp(-1e3 / held["T2"] * TE * 1e-3)
        return held.get("M0", 1.0) * decay + held.get("offset", 0.0)


class DoubleAngleSimulator(Simulator):
    """Double-angle B1 mapping, in closed form.

    A single excitation at each nominal flip angle, read at full relaxation. A
    voxel turns through the nominal angle times ``B1``, so the signal is
    ``M0 sin(B1 flip)``, one real sample per angle. Two angles in the ratio of
    two give ``B1`` in closed form; any set of angles can be fitted.

    Parameters
    ----------
    flip : float or array-like
        Nominal flip angles, in degrees; the trailing axis of the signal.

    Tissue properties
    -----------------
    B1 : float or array-like
        Transmit field, as a fraction of the nominal flip angle.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.

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
        """Return the transverse magnetization each flip angle produces."""
        held = across_contrasts(properties, flip)
        turned = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
        return held.get("M0", 1.0) * torch.sin(turned)
