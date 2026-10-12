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

    Magnetization-prepared rapid gradient echo: an inversion followed by a
    spoiled gradient-echo train, of which the shot at the k-space centre is
    sampled. Only the longitudinal magnetization carries between shots. The
    signal is that of the sampled shot, one value per voxel. Wraps
    :class:`~blochsim.simulators.MPRAGESimulator`.

    Parameters
    ----------
    TI : float
        Inversion time, in ms, measured from the inversion to the sampled
        shot.
    flip : float
        Excitation flip angle, in degrees.
    TRspgr : float
        Repetition time of one readout, in ms.
    nshots : int or array-like
        Readouts in the block: a total including the sampled shot, split
        evenly around it, or ``(before, after)`` the sampled shot.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    diff : str or tuple of str, default=None
        Names of the properties to differentiate with respect to. ``None``
        returns the signal alone.
    inv_efficiency : float or array-like, default=1.0
        Inversion efficiency, as a fraction of a perfect inversion.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    phases : float or array-like, default=0.0
        Excitation RF phases, in degrees.
    device : str or torch.device, default=None
        Device to run on. ``None`` follows the inputs.
    **values
        Any further property or protocol argument of the simulator, such as
        ``B0`` or ``B1``, passed to :meth:`~blochsim.model.Simulator.simulate`.
        Arguments given as ``None`` are dropped.

    Returns
    -------
    signal : complex array-like
        Signal of shape ``(...)``, where ``...`` is the broadcast shape of
        the tissue properties.
    jacobian : same array type as ``signal``
        Derivatives of the signal with respect to ``diff``, of shape
        ``(..., len(diff))``. Returned only when ``diff`` is given; a single name
        omits the ``len(diff)`` axis.

    Raises
    ------
    ValueError
        If ``TI`` falls before the first excitation.
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
