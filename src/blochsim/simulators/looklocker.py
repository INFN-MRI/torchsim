"""Inversion recovery read by a continuous train of pulses, in closed form."""

from __future__ import annotations

__all__ = ["IRbSSFPSimulator", "LookLockerSimulator", "MOLLISimulator"]

from collections.abc import Mapping, Sequence
from typing import Any

import numpy.typing as npt
import torch

from .._model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


def _apparent(
    held: Mapping[str, Any],
    flip: torch.Tensor,
    TR: torch.Tensor,  # noqa: N803
    short_TR: bool,  # noqa: N803
) -> tuple[Any, Any, Any]:
    """Return the turned angle, the apparent rate R1* in 1/s and Mss / M0.

    The steady state under a spoiled train is Deichmann and Haase's [1]_
    ``(1 - E1) / (1 - E1 cos a)``; its short-TR limit is ``R1 / R1*``.

    References
    ----------
    .. [1] Deichmann, R., Haase, A., "Quantification of T1 values by
       SNAPSHOT-FLASH NMR imaging", Journal of Magnetic Resonance 96.3 (1992),
       pp. 608-612. https://doi.org/10.1016/0022-2364(92)90347-A
    """
    angle = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
    repetition_s = TR * 1e-3
    rate = 1e3 / held["T1"]
    apparent = rate - torch.log(torch.cos(angle)) / repetition_s
    if short_TR:
        steady = rate / apparent
    else:
        steady = (1 - torch.exp(-rate * repetition_s)) / (
            1 - torch.exp(-apparent * repetition_s)
        )
    return angle, apparent, steady


class LookLockerSimulator(Simulator):
    """Look-Locker inversion recovery under a spoiled train of small flip angles.

    The longitudinal magnetization recovers towards a steady state below
    equilibrium at the apparent rate ``R1* = R1 - ln(cos a) / TR`` [1]_. Each
    readout records the longitudinal magnetization just before its pulse, times
    ``sin a``. The signal is real, one sample per ``TI``.

    Parameters
    ----------
    flip : float or array-like
        Flip angle of the readout pulses, in degrees.
    TR : float or array-like
        Spacing of the readout pulses, in ms.
    TI : float or array-like
        Time of each readout after the inversion, in ms.
    short_TR : bool, default=False
        Take the steady state as ``M0 R1 / R1*``, its ``TR << T1`` limit.
    from_steady_state : bool, default=False
        Invert the steady state rather than the equilibrium magnetization.

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

    References
    ----------
    .. [1] Look, D. C., Locker, D. R., "Time saving in measurement of NMR and
       EPR relaxation times", Review of Scientific Instruments 41.2 (1970),
       pp. 250-251. https://doi.org/10.1063/1.1684482

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import LookLockerSimulator

        sequence = LookLockerSimulator(flip=6.0, TR=4.0, TI=4.0 * torch.arange(500))
        signal = sequence.simulate(T1=(800.0, 1400.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "M0": None,
            "B1": None,
            "inv_efficiency": None,
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
        TI: float | npt.ArrayLike,
        short_TR: bool = False,  # noqa: N803
        from_steady_state: bool = False,
    ) -> torch.Tensor:
        """Return the transverse magnetization each readout records."""
        held = across_contrasts(properties, flip, TR, TI)
        density = held.get("M0", 1.0)
        angle, apparent, steady = _apparent(held, flip, TR, short_TR)
        steady = density * steady
        start = -held.get("inv_efficiency", 1.0) * (
            steady if from_steady_state else density
        )
        longitudinal = steady - (steady - start) * torch.exp(-apparent * TI * 1e-3)
        return longitudinal * torch.sin(angle)


class MOLLISimulator(Simulator):
    """Modified Look-Locker inversion recovery (MOLLI): readout blocks separated by free recovery.

    MOLLI [1]_ reads the recovery after one inversion in blocks of Look-Locker
    readouts. The first block starts from the inversion; between blocks the pulses
    stop, and the longitudinal magnetization left by a block's last readout recovers
    towards ``M0`` for ``recovery`` ms before the next block starts from it. The
    signal is real, the blocks concatenated along the last axis.

    Parameters
    ----------
    flip : float or array-like
        Flip angle of the readout pulses, in degrees.
    TR : float or array-like
        Spacing of the readout pulses, in ms.
    readouts : sequence of int
        Readouts in each block.
    recovery : float or array-like
        Free recovery between blocks, in ms.
    short_TR : bool, default=False
        Take the steady state as ``M0 R1 / R1*``, its ``TR << T1`` limit.

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

    References
    ----------
    .. [1] Messroghli, D. R., Radjenovic, A., Kozerke, S., et al., "Modified
       Look-Locker inversion recovery (MOLLI) for high-resolution T1 mapping of
       the heart", Magnetic Resonance in Medicine 52.1 (2004), pp. 141-146.
       https://doi.org/10.1002/mrm.20110

    Examples
    --------
    .. exec::

        from blochsim.simulators import MOLLISimulator

        sequence = MOLLISimulator(flip=35.0, TR=2.5, readouts=(40, 40, 40), recovery=800.0)
        signal = sequence.simulate(T1=(800.0, 1400.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "M0": None,
            "B1": None,
            "inv_efficiency": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the closed-form signal."""
        return self._signal(properties, **self.played(**sequence))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        readouts: Sequence[int],
        recovery: float | npt.ArrayLike,
        short_TR: bool = False,  # noqa: N803
    ) -> torch.Tensor:
        """Return the transverse magnetization each readout records."""
        counts = [int(count) for count in readouts]
        positions = torch.cat([torch.arange(count) for count in counts])
        values = arrays({"flip": flip, "TR": TR, "recovery": recovery})
        flip, TR, recovery = values["flip"], values["TR"], values["recovery"]  # noqa: N806
        held = across_contrasts(properties, positions)
        density = held.get("M0", 1.0)
        angle, apparent, steady = _apparent(held, flip, TR, short_TR)
        steady = density * steady
        relaxed = torch.exp(-1e3 / held["T1"] * recovery * 1e-3)
        elapsed = (
            torch.arange(max(counts)).to(torch.as_tensor(apparent).dtype) * TR * 1e-3
        )

        start = -held.get("inv_efficiency", 1.0) * density
        blocks = []
        for count in counts:
            block = steady - (steady - start) * torch.exp(-apparent * elapsed[:count])
            blocks.append(block)
            start = density + (block[..., -1:] - density) * relaxed
        return torch.cat(blocks, dim=-1) * torch.sin(angle)


class IRbSSFPSimulator(Simulator):
    """Inversion-recovery balanced SSFP (IR-TrueFISP) signal, in closed form.

    After an alpha/2 preparation the magnetization approaches the bSSFP steady
    state along one exponential, at a rate mixing R1 and R2 by the flip angle
    [1]_. ``inv_efficiency`` scales the magnetization the recovery starts from.
    The signal is real, one sample per ``TI``.

    Parameters
    ----------
    flip : float or array-like
        Flip angle of the balanced train, in degrees.
    TI : float or array-like
        Time of each readout after the inversion, in ms: ``TR`` times the readout
        index.

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

    References
    ----------
    .. [1] Schmitt, P., Griswold, M. A., Jakob, P. M., et al., "Inversion
       recovery TrueFISP: quantification of T1, T2, and spin density",
       Magnetic Resonance in Medicine 51.4 (2004), pp. 661-667.
       https://doi.org/10.1002/mrm.20058

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import IRbSSFPSimulator

        sequence = IRbSSFPSimulator(flip=45.0, TI=4.5 * torch.arange(1000))
        signal = sequence.simulate(T1=1000.0, T2=(50.0, 100.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2": None,
            "M0": None,
            "B1": None,
            "inv_efficiency": None,
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
        TI: float | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the transverse magnetization at each time after the inversion."""
        held = across_contrasts(properties, flip, TI)
        angle = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
        density = held.get("M0", 1.0)
        ratio = held["T1"] / held["T2"]
        half = angle / 2
        start = held.get("inv_efficiency", 1.0) * density * torch.sin(half)
        apparent = 1e3 * (
            torch.cos(half) ** 2 / held["T1"] + torch.sin(half) ** 2 / held["T2"]
        )
        steady = (
            density * torch.sin(angle) / ((ratio + 1) - torch.cos(angle) * (ratio - 1))
        )
        return steady - (steady + start) * torch.exp(-apparent * TI * 1e-3)
