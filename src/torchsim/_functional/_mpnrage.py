"""MPnRAGE simulator."""

__all__ = ["mpnrage_sim"]

import numpy.typing as npt
import torch

from ..models.mpnrage import MPnRAGEModel


def mpnrage_sim(
    nshots: int,
    flip: npt.ArrayLike,
    TR: float | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    diff: str | tuple[str] = None,
    B1: float | npt.ArrayLike = 1.0,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    TI: float = 0.0,
    chunk_size: int = None,
    device: str | torch.device = None,
):
    """
    MPnRAGE simulator wrapper.

    Parameters
    ----------
    nshots : int
        Number of SPGR shots per inversion block.
    flip : float | npt.ArrayLike
        Flip angle train in degrees.
    TR : float | npt.ArrayLike
        Repetition time in milliseconds.
    T1 : float | npt.ArrayLike
        Longitudinal relaxation time in milliseconds.
    diff : str | tuple[str], optional
        Arguments to get the signal derivative with respect to.
        The default is ``None`` (no differentation).
    B1 : float | npt.ArrayLike, optional
        Flip angle scaling map, default is ``1.0``.
    inv_efficiency : float | npt.ArrayLike, optional
        Inversion efficiency map, default is ``1.0``.
    M0 : float or array-like, optional
        Proton density scaling factor, default is ``1.0``.
    TI : float | npt.ArrayLike, optional
        Inversion time in milliseconds.
        The default is ``0.0``.
    chunk_size : int, optional
        Number of atoms to be simulated in parallel.
        The default is ``None``.
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
    model = MPnRAGEModel(diff, chunk_size, device)
    model.set_properties(T1, M0, B1, inv_efficiency)
    model.set_sequence(nshots, flip, TR, TI)
    return model()
