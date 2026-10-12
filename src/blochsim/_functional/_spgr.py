"""SPoiled Gradient Recalled echo simulator."""

__all__ = ["spgr_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.spgr import SPGRSimulator
from ._run import evaluated


def spgr_sim(
    flip: float | npt.ArrayLike,
    TE: float | npt.ArrayLike,
    TR: float,
    T1: float | npt.ArrayLike,
    T2star: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    B0: float | npt.ArrayLike = 0.0,
    chemshift: float | npt.ArrayLike = 0.0,
    M0: float | npt.ArrayLike = 1.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate the spoiled gradient-echo steady-state signal.

    Spoiled gradient-recalled echo (SPGR, FLASH, T1-FFE) in closed form: the
    Ernst steady state, attenuated by ``T2star`` decay and rotated by the
    off-resonance phase accrued up to the echo time. Wraps
    :class:`~blochsim.simulators.SPGRSimulator`.

    Parameters
    ----------
    flip : float or array-like
        Excitation flip angle, in degrees. An array adds a trailing contrast
        axis.
    TE : float or array-like
        Echo time, in ms.
    TR : float
        Repetition time, in ms.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    T2star : float or array-like
        Effective transverse relaxation time, in ms.
    diff : str or tuple of str, default=None
        Names of the properties to differentiate with respect to. ``None``
        returns the signal alone.
    B0 : float or array-like, default=0.0
        Off-resonance, in Hz.
    chemshift : float or array-like, default=0.0
        Chemical shift of the spins, in Hz, added to the off-resonance.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    device : str or torch.device, default=None
        Device to run on. ``None`` follows the inputs.
    **values
        Any further property or protocol argument of the simulator, such as
        ``B0`` or ``B1``, passed to :meth:`~blochsim.model.Simulator.simulate`.
        Arguments given as ``None`` are dropped.

    Returns
    -------
    signal : complex array-like
        Signal of shape ``(..., len(flip))``, where ``...`` is the broadcast shape of
        the tissue properties.
    jacobian : same array type as ``signal``
        Derivatives of the signal with respect to ``diff``, of shape
        ``(..., len(diff), len(flip))``. Returned only when ``diff`` is given; a single name
        omits the ``len(diff)`` axis.
    """
    return evaluated(
        SPGRSimulator(),
        diff,
        device,
        T1=T1,
        T2star=T2star,
        M0=M0,
        B0=B0,
        chemshift=chemshift,
        flip=flip,
        TR=TR,
        TE=TE,
        **values,
    )
