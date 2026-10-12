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

    Wraps :class:`~blochsim.simulators.MP2RAGESimulator`.

    Parameters
    ----------
    TI : array-like
        The two inversion times in ms, each measured to the sampled shot of its block.
    flip : float or array-like
        Excitation flip angle in degrees, one per block or shared.
    TRspgr : float
        Repetition time of one readout, in ms.
    TRmp2rage : float
        Repetition time of the whole inversion block, in ms.
    nshots : int or array-like
        Readouts per block: a total, split evenly before and after the sampled
        shot, or ``(before, after)``.
    T1 : float or array-like
        Longitudinal relaxation time in ms.
    diff : str or tuple of str, optional
        Parameters to differentiate with respect to; ``None`` returns the signal only.
    inv_efficiency : float or array-like, optional
        Inversion efficiency, a fraction of a perfect inversion.
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
        Signal of shape ``(..., 2)``.
    jacobian : array-like
        Derivatives with respect to ``diff``, of shape ``(..., len(diff), 2)``. Returned only
        when ``diff`` is given; a single name drops the ``len(diff)`` axis.
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
