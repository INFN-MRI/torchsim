"""MP2RAGE simulator."""

__all__ = ["mp2rage_sim"]

import numpy.typing as npt
import torch

from ..models.mp2rage import MP2RAGEModel
from ._run import evaluated


def mp2rage_sim(
    TI: npt.ArrayLike,
    flip: float | npt.ArrayLike,
    TRspgr: float,
    TRmp2rage: float,
    nshots: int | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    diff: str | tuple[str] = None,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    device: str | torch.device = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    MP2RAGE simulator wrapper.

    Parameters
    ----------
    TI : npt.ArrayLike
        Inversion time (s) in milliseconds of shape ``(2,)``.
    flip : float | npt.ArrayLike
        Flip angle train in degrees of shape ``(2,)``.
        If scalar, assume same angle for both blocks.
    TRspgr : float
        Repetition time in milliseconds for each SPGR readout.
    TRmp2rage : float
        Repetition time in milliseconds for the whole inversion block.
    nshots : int | npt.ArrayLike
        Number of SPGR readout within the inversion block of shape ``(npre, npost)``
        If scalar, assume ``npre == npost == 0.5 * nshots``. Usually, this
        is the number of slice encoding lines ``(nshots = nz / Rz)``,
        i.e., the number of slices divided by the total acceleration factor along ``z``.
    T1 : float | npt.ArrayLike
        Longitudinal relaxation time in milliseconds.
    diff : str | tuple[str], optional
        Arguments to get the signal derivative with respect to.
        The default is ``None`` (no differentation).
    inv_efficiency : float | npt.ArrayLike, optional
        Inversion efficiency map, default is ``1.0``.
    M0 : float or array-like, optional
        Proton density scaling factor, default is ``1.0``.
    TI : float | npt.ArrayLike, optional
        Inversion time in milliseconds.
        The default is ``0.0``.
    device : str | torch.device, optional
        Computational device for simulation.
        The default is ``None`` (infer from input).

    Returns
    -------
    sig : npt.ArrayLike
        Signal evolution of shape ``(..., len(flip))``.
    jac : npt.ArrayLike
        Derivatives of signal wrt ``diff`` parameters,
        of shape ``(..., len(diff), len(flip))``.
        Not returned if ``diff`` is ``None``.

    """
    return evaluated(
        MP2RAGEModel(),
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
    )
