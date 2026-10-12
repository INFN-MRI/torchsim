"""Spin echo with partial longitudinal recovery, in closed form."""

from __future__ import annotations

__all__ = ["SpinEchoSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class SpinEchoSimulator(Simulator):
    """Spin echo with partial longitudinal recovery, in closed form.

    A 90-180 spin echo repeated every ``TR``, optionally behind an inversion. The
    refocusing pulse at ``TE / 2`` inverts the magnetization recovered since the
    excitation, so the next excitation finds
    ``1 - 2 exp(-(TR - TE / 2) / T1) + exp(-TR / T1)`` of equilibrium [1]_, and the
    echo decays with T2. An inversion time multiplies this by
    ``1 - (1 + inv_efficiency) exp(-TI / T1)``. The signal is one real sample per
    echo time.

    Parameters
    ----------
    TE : float or array-like
        Echo times, in ms.
    TR : float or array-like
        Repetition time, in ms.
    TI : float or array-like, optional
        Inversion times, in ms. No inversion when not given.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Longitudinal and transverse relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    inv_efficiency : float or array-like, default=1.0
        Fraction of a perfect inversion. Read only when ``TI`` is given.

    References
    ----------
    .. [1] Bernstein, M. A., King, K. F., Zhou, X. J., Handbook of MRI Pulse
       Sequences, Elsevier (2004), sections 14.2 and 14.3.
       https://doi.org/10.1016/B978-0-12-092861-3.X5000-6

    Examples
    --------
    .. exec::

        from blochsim.simulators import SpinEchoSimulator

        sequence = SpinEchoSimulator(TE=(10.0, 30.0, 60.0, 100.0), TR=2000.0)
        signal = sequence.simulate(T1=1000.0, T2=(50.0, 100.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2": None,
            "M0": None,
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
        TE: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike | None = None,
    ) -> torch.Tensor:
        """Return the echo amplitude."""
        timings = (TE, TR) if TI is None else (TE, TR, TI)
        held = across_contrasts(properties, *timings)
        rate = 1e3 / held["T1"]
        echo_s, repetition_s = TE * 1e-3, TR * 1e-3
        recovered = (
            1
            - 2 * torch.exp(-rate * (repetition_s - echo_s / 2))
            + torch.exp(-rate * repetition_s)
        )
        if TI is not None:
            efficiency = held.get("inv_efficiency", 1.0)
            recovered = recovered * (
                1 - (1 + efficiency) * torch.exp(-rate * TI * 1e-3)
            )
        decay = torch.exp(-1e3 / held["T2"] * echo_s)
        return held.get("M0", 1.0) * recovered * decay
