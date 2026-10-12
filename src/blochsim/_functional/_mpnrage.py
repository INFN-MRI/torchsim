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

    Magnetization-prepared spoiled gradient-echo train, one inversion
    followed by ``nshots`` readouts, each sampled; the signal follows the
    longitudinal recovery after the inversion. Wraps :class:`~blochsim.simulators.MPnRAGESimulator`.

    Parameters
    ----------
    nshots : int
        Readouts per inversion block.
    flip : array-like
        Excitation flip angle, in degrees; scalar, or one per shot.
    TR : float or array-like
        Repetition time, in ms; scalar or one per shot.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    diff : str or tuple of str, default=None
        Names of the properties to differentiate with respect to. ``None``
        returns the signal alone.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.
    inv_efficiency : float or array-like, default=1.0
        Inversion efficiency, as a fraction of a perfect inversion.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    TI : float, default=0.0
        Delay from the inversion to the first excitation, in ms.
    phases : float or array-like, default=0.0
        Excitation RF phases, in degrees; scalar or one per shot.
    device : str or torch.device, default=None
        Device to run on. ``None`` follows the inputs.
    **values
        Any further property or protocol argument of the simulator, such as
        ``B0`` or ``B1``, passed to :meth:`~blochsim.model.Simulator.simulate`.
        Arguments given as ``None`` are dropped.

    Returns
    -------
    signal : complex array-like
        Signal of shape ``(..., nshots)``, where ``...`` is the broadcast shape of
        the tissue properties.
    jacobian : same array type as ``signal``
        Derivatives of the signal with respect to ``diff``, of shape
        ``(..., len(diff), nshots)``. Returned only when ``diff`` is given; a single name
        omits the ``len(diff)`` axis.

    Raises
    ------
    ValueError
        If ``nshots`` is not positive, or ``flip`` is neither scalar nor of
        length ``nshots``.
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
