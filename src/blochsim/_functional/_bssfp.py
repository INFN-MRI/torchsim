"""Balanced Steady State Free Precession simulator."""

__all__ = ["bssfp_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.bssfp import bSSFPSimulator
from ._run import evaluated


def bssfp_sim(
    flip: float | npt.ArrayLike,
    TE: float | npt.ArrayLike,
    TR: float,
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    phase_inc: float = 180.0,
    B0: float | npt.ArrayLike = 0.0,
    chemshift: float | npt.ArrayLike = 0.0,
    M0: float | npt.ArrayLike = 1.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate the balanced SSFP steady-state signal.

    Wraps :class:`~blochsim.simulators.bSSFPSimulator`.

    Parameters
    ----------
    flip : float or array-like
        Excitation flip angle in degrees.
    TE : float or array-like
        Echo time in ms.
    TR : float
        Repetition time in ms.
    T1 : float or array-like
        Longitudinal relaxation time in ms.
    T2 : float or array-like
        Transverse relaxation time in ms.
    diff : str or tuple of str, optional
        Parameters to differentiate with respect to; ``None`` returns the signal only.
    phase_inc : float, optional
        Linear phase-cycling increment in degrees.
    B0 : float or array-like, optional
        Off-resonance in Hz.
    chemshift : float or array-like, optional
        Chemical shift in Hz.
    M0 : float or array-like, optional
        Proton density scaling.
    device : str or torch.device, optional
        Device to run on; follows the inputs when ``None``.
    **values
        Further tissue, protocol or run arguments accepted by
        :meth:`~blochsim.model.Simulator.simulate`.

    Returns
    -------
    signal : array-like
        Signal of shape ``(..., len(flip))``.
    jacobian : array-like
        Derivatives with respect to ``diff``, of shape ``(..., len(diff), len(flip))``. Returned only
        when ``diff`` is given; a single name drops the ``len(diff)`` axis.
    """
    return evaluated(
        bSSFPSimulator(),
        diff,
        device,
        T1=T1,
        T2=T2,
        M0=M0,
        B0=B0,
        chemshift=chemshift,
        flip=flip,
        TR=TR,
        TE=TE,
        phase_inc=phase_inc,
        **values,
    )
