"""MR Fingerprinting simulator."""

__all__ = ["mrf_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.mrf import MRFSimulator
from ._run import evaluated


def mrf_sim(
    flip: npt.ArrayLike,
    TR: float | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    B1: float | npt.ArrayLike = 1.0,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    TI: float = 0.0,
    phases: float | npt.ArrayLike = 0.0,
    states: int = 10,
    nreps: int = 1,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate an unbalanced SSFP MR fingerprinting train.

    Wraps :class:`~blochsim.simulators.MRFSimulator`.

    Parameters
    ----------
    flip : float or array-like
        Excitation flip angles in degrees, one per repetition.
    TR : float or array-like
        Repetition time in ms.
    T1 : float or array-like
        Longitudinal relaxation time in ms.
    T2 : float or array-like
        Transverse relaxation time in ms.
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
    states : int, optional
        Configuration orders retained (EPG).
    nreps : int, optional
        Playings of the train; the last is recorded.
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
        MRFSimulator(),
        diff,
        device,
        T1=T1,
        T2=T2,
        M0=M0,
        B1=B1,
        inv_efficiency=inv_efficiency,
        flip=flip,
        TR=TR,
        TI=TI,
        phases=phases,
        states=states,
        **{"repetitions": nreps, **values},
    )
