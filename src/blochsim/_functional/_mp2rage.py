"""MP2RAGE simulator."""

__all__ = ["mp2rage_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.mp2rage import MP2RAGESimulator
from ._run import evaluated


def mp2rage_sim(
    TI: npt.ArrayLike,
    flip: float | npt.ArrayLike,
    TRspgr: float,
    TRmp2rage: float,
    nshots: int | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate the two MP2RAGE blocks in closed form.

    Magnetization-prepared two rapid gradient echoes: one inversion followed
    by two spoiled gradient-echo blocks, each sampled at its k-space centre
    at its own inversion time. Only the longitudinal magnetization carries
    between shots, so the steady state has a closed form; the two returned
    values are the signals of the first and second block. Wraps
    :class:`~blochsim.simulators.MP2RAGESimulator`.

    Parameters
    ----------
    TI : array-like
        The two inversion times, in ms, each measured from the inversion to
        the sampled shot of its block.
    flip : float or array-like
        Excitation flip angle, in degrees; one value shared by both blocks, or
        one per block.
    TRspgr : float
        Repetition time of one readout, in ms.
    TRmp2rage : float
        Repetition time of the whole inversion block, in ms.
    nshots : int or array-like
        Readouts per block: a total, split evenly before and after the sampled
        shot, or ``(before, after)``.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    diff : str or tuple of str, default=None
        Names of the properties to differentiate with respect to. ``None``
        returns the signal alone.
    inv_efficiency : float or array-like, default=1.0
        Inversion efficiency, as a fraction of a perfect inversion.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    device : str or torch.device, default=None
        Device to run on. ``None`` follows the inputs.
    **values
        Any further property or protocol argument of the simulator, such as
        ``B0`` or ``B1``, passed to :meth:`~blochsim.Simulator.simulate`.
        Arguments given as ``None`` are dropped.

    Returns
    -------
    signal : array-like
        Signal of shape ``(..., 2)``, where ``...`` is the broadcast shape of
        the tissue properties.
    jacobian : same array type as ``signal``
        Derivatives of the signal with respect to ``diff``, of shape
        ``(..., len(diff), 2)``. Returned only when ``diff`` is given; a single name
        omits the ``len(diff)`` axis.

    Raises
    ------
    ValueError
        If an inversion time falls before its block's first excitation, or the
        second block does not fit inside ``TRmp2rage``.
    """
    return evaluated(
        MP2RAGESimulator(),
        diff,
        device,
        T1=T1,
        M0=M0,
        inv_efficiency=inv_efficiency,
        TI=TI,
        flip=flip,
        TRspgr=TRspgr,
        TRmp2rage=TRmp2rage,
        nshots=nshots,
        **values,
    )
