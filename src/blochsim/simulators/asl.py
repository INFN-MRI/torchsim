"""Arterial spin labelling, by the general kinetic model."""

from __future__ import annotations

__all__ = ["ASLSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class ASLSimulator(Simulator):
    """Label-control difference of arterial spin labelling, over time.

    General kinetic model [1]_ for continuous, pseudo-continuous and pulsed
    labelling. Blood labelled for ``tau`` arrives after ``ATT``, is delivered at
    rate ``CBF``, and relaxes with ``T1_blood`` before arrival and with the
    apparent tissue T1, ``1 / (1 / T1 + f / lambda)``, after exchange. The default
    ``T1_blood``, ``partition`` and labelling efficiency are the 3 T consensus
    values [2]_.

    References
    ----------
    .. [1] Buxton, R. B., Frank, L. R., Wong, E. C., et al., "A general kinetic
       model for quantitative perfusion imaging with arterial spin labeling",
       Magnetic Resonance in Medicine 40.3 (1998), pp. 383-396.
       https://doi.org/10.1002/mrm.1910400308

    .. [2] Alsop, D. C., Detre, J. A., Golay, X., et al., "Recommended
       implementation of arterial spin-labeled perfusion MRI for clinical
       applications", Magnetic Resonance in Medicine 73.1 (2015),
       pp. 102-116. https://doi.org/10.1002/mrm.25197

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import ASLSimulator

        sequence = ASLSimulator(t=torch.linspace(0.0, 5000.0, 51), tau=1800.0)
        signal = sequence.simulate(CBF=(20.0, 60.0), ATT=1200.0, T1=1400.0)
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "CBF": None,
            "ATT": None,
            "T1": None,
            "M0": None,
            "T1_blood": None,
            "partition": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the closed-form signal."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        t: float | npt.ArrayLike,
        tau: float | npt.ArrayLike,
        labelling: str = "continuous",
        efficiency: float | npt.ArrayLike | None = None,
    ) -> torch.Tensor:
        """Return the magnetization difference at each time.

        Parameters
        ----------
        properties : mapping
            ``CBF`` in ml/100 g/min; ``ATT``, ``T1`` and ``T1_blood`` in ms
            (``T1_blood`` defaults to 1650); ``partition`` in ml/g (0.9); ``M0`` scaling.
        t : float or array-like
            Times from the start of labelling, in ms.
        tau : float or array-like
            Labelling duration in ms.
        labelling : {"continuous", "pulsed"}, optional
            Continuous (including pseudo-continuous) or pulsed labelling.
        efficiency : float or array-like, optional
            Labelling efficiency; defaults to 0.85 (continuous) or 0.98 (pulsed).

        Raises
        ------
        ValueError
            If ``labelling`` is neither ``"continuous"`` nor ``"pulsed"``.
        """
        if labelling not in ("continuous", "pulsed"):
            raise ValueError(
                f"labelling is 'continuous' or 'pulsed', got {labelling!r}"
            )
        pulsed = labelling == "pulsed"
        if efficiency is None:
            efficiency = 0.98 if pulsed else 0.85
        held = across_contrasts(properties, t)
        time_s = t * 1e-3
        duration_s = tau * 1e-3
        arrival_s = held["ATT"] * 1e-3
        partition = held.get("partition", 0.9)
        flow = held["CBF"] / 6000.0  # ml/100 g/min to ml/g/s
        blood_rate = 1e3 / held.get("T1_blood", 1650.0)
        apparent_s = 1 / (1e3 / held["T1"] + flow / partition)
        delivered = 2 * held.get("M0", 1.0) / partition * efficiency * flow

        before = time_s < arrival_s if not pulsed else time_s <= arrival_s
        ending = arrival_s + duration_s
        during = time_s <= ending if not pulsed else time_s < ending

        if pulsed:
            k = blood_rate - 1 / apparent_s
            decay = torch.exp(-time_s * blood_rate)
            upto = torch.where(during, time_s, ending)
            shape = torch.exp(k * time_s) * (
                torch.exp(-k * arrival_s) - torch.exp(-k * upto)
            )
            difference = delivered * decay * shape / k
        else:
            arrived = torch.exp(-arrival_s * blood_rate)
            filling = 1 - torch.exp(-(time_s - arrival_s) / apparent_s)
            washed = (1 - torch.exp(-duration_s / apparent_s)) * torch.exp(
                -(time_s - ending) / apparent_s
            )
            difference = (
                delivered * apparent_s * arrived * torch.where(during, filling, washed)
            )
        return torch.where(before, torch.zeros_like(difference), difference)
