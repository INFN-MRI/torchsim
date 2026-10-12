"""MPRAGE simulator."""

__all__ = ["mprage_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.mprage import MPRAGESimulator
from ._run import evaluated


def mprage_sim(
    TI: float,
    flip: float,
    TRspgr: float,
    nshots: int | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    phases: float | npt.ArrayLike = 0.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate the sampled MPRAGE shot in closed form.

    Wraps :class:`~blochsim.simulators.MPRAGESimulator`.

    Parameters
    ----------
    TI : float
        Inversion time in ms, measured to the sampled shot.
    flip : float
        Excitation flip angle in degrees.
    TRspgr : float
        Repetition time of one readout, in ms.
    nshots : int or array-like
        Readouts in the block: a total including the sampled shot, split evenly
        around it, or ``(before, after)`` the sampled shot.
    T1 : float or array-like
        Longitudinal relaxation time in ms.
    diff : str or tuple of str, optional
        Parameters to differentiate with respect to; ``None`` returns the signal only.
    inv_efficiency : float or array-like, optional
        Inversion efficiency, a fraction of a perfect inversion.
    M0 : float or array-like, optional
        Proton density scaling.
    phases : float or array-like, optional
        Excitation phases in degrees.
    device : str or torch.device, optional
        Device to run on; follows the inputs when ``None``.
    **values
        Further tissue, protocol or run arguments accepted by
        :meth:`~blochsim.model.Simulator.simulate`.

    Returns
    -------
    signal : array-like
        Signal of shape ``(...)``.
    jacobian : array-like
        Derivatives with respect to ``diff``, of shape ``(..., len(diff))``. Returned only
        when ``diff`` is given; a single name drops the ``len(diff)`` axis.
    """
    return evaluated(
        MPRAGESimulator(),
        diff,
        device,
        T1=T1,
        M0=M0,
        inv_efficiency=inv_efficiency,
        TI=TI,
        flip=flip,
        TRspgr=TRspgr,
        nshots=nshots,
        phases=phases,
        **values,
    )
