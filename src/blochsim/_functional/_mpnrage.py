"""MPnRAGE simulator."""

__all__ = ["mpnrage_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.mpnrage import MPnRAGESimulator
from ._run import evaluated


def mpnrage_sim(
    nshots: int,
    flip: npt.ArrayLike,
    TR: float | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    B1: float | npt.ArrayLike = 1.0,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    TI: float = 0.0,
    phases: float | npt.ArrayLike = 0.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate an MPnRAGE train, every shot read.

    Wraps :class:`~blochsim.simulators.MPnRAGESimulator`.

    Parameters
    ----------
    nshots : int
        Readouts per inversion block.
    flip : float or array-like
        Excitation flip angle in degrees, scalar or one per shot.
    TR : float or array-like
        Repetition time in ms.
    T1 : float or array-like
        Longitudinal relaxation time in ms.
    diff : str or tuple of str, optional
        Parameters to differentiate with respect to; ``None`` returns the signal only.
    B1 : float or array-like, optional
        Transmit field scaling, a fraction of the nominal flip angle.
    inv_efficiency : float or array-like, optional
        Inversion efficiency, a fraction of a perfect inversion.
    M0 : float or array-like, optional
        Proton density scaling.
    TI : float, optional
        Inversion time in ms.
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
        Signal of shape ``(..., nshots)``.
    jacobian : array-like
        Derivatives with respect to ``diff``, of shape ``(..., len(diff), nshots)``. Returned only
        when ``diff`` is given; a single name drops the ``len(diff)`` axis.
    """
    return evaluated(
        MPnRAGESimulator(),
        diff,
        device,
        T1=T1,
        M0=M0,
        B1=B1,
        inv_efficiency=inv_efficiency,
        nshots=nshots,
        flip=flip,
        TR=TR,
        TI=TI,
        phases=phases,
        **values,
    )
