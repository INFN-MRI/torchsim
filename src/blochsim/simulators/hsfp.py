"""Hybrid-state free precession, in closed form."""

from __future__ import annotations

__all__ = ["HSFPSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class HSFPSimulator(Simulator):
    """Hybrid-state free precession (HSFP) of a balanced train, in closed form.

    For a balanced train with a slowly varying flip angle, the magnetization
    stays on the spin-locked axis and only its length evolves, relaxing at
    ``sin^2(a) / T2 + cos^2(a) / T1`` and driven by ``cos(a) / T1`` [1]_. The
    train is repeated, each pass starting from the end of the previous one times
    ``beta``. The signal is the magnetization along the spin-locked axis at each
    pulse, the transverse magnetization at the echo for a balanced train.

    Parameters
    ----------
    flip : array-like
        Flip-angle train, in degrees, one entry per pulse.
    TR : float or array-like
        Repetition time, in ms.
    beta : float, default=-1.0
        Factor applied to the magnetization between passes: -1 for an
        inversion, 1 for none.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.

    References
    ----------
    .. [1] Asslaender, J., Novikov, D. S., Lattanzi, R., Sodickson, D. K.,
       Cloos, M. A., "Hybrid-state free precession in nuclear magnetic
       resonance", Communications Physics 2 (2019), 73.
       https://doi.org/10.1038/s42005-019-0174-0

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import HSFPSimulator

        train = 60.0 * torch.sin(torch.linspace(0.0, torch.pi, 400)) ** 2
        sequence = HSFPSimulator(flip=train, TR=4.5)
        signal = sequence.simulate(T1=1000.0, T2=(50.0, 100.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2": None,
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
        flip: npt.ArrayLike,
        TR: float | npt.ArrayLike,
        beta: float = -1.0,
    ) -> torch.Tensor:
        """Return the magnetization along the spin-locked axis at each pulse."""
        held = across_contrasts(properties, flip)
        dtype = torch.promote_types(torch.as_tensor(held["T1"]).dtype, flip.dtype)
        angle = (torch.pi / 180.0) * flip.abs().to(torch.float64)
        repetition_s = torch.as_tensor(TR).to(torch.float64) * 1e-3
        r1 = 1e3 / torch.as_tensor(held["T1"]).to(torch.float64)
        r2 = 1e3 / torch.as_tensor(held["T2"]).to(torch.float64)

        # Everything below is a running sum along the train. Sums reach
        # exp(+R TR n), which single precision cannot hold over a long train.
        rates = torch.sin(angle) ** 2 * r2 + torch.cos(angle) ** 2 * r1
        through = torch.cumsum(rates * repetition_s, dim=-1)
        driven = (
            torch.cumsum(torch.cos(angle) * torch.exp(through), dim=-1) * repetition_s
        )
        before = torch.nn.functional.pad(through, (1, 0))[..., :-1]
        driven_before = torch.nn.functional.pad(driven, (1, 0))[..., :-1]

        decayed = torch.exp(-through[..., -1:])
        start = beta * r1 * decayed / (1 - beta * decayed) * driven[..., -1:]
        length = torch.exp(-before) * (start + r1 * driven_before)
        return (held.get("M0", 1.0) * length).to(dtype)
