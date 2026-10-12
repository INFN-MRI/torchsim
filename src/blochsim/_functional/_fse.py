"""Fast Spin Echo simulator."""

__all__ = ["fse_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.fse import FSESimulator
from ._run import evaluated


def fse_sim(
    flip: npt.ArrayLike,
    ESP: float,
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    phases: float | npt.ArrayLike = 0.0,
    TR: float | npt.ArrayLike = 1e6,
    exc_flip: float = 90.0,
    exc_phase: float = 90.0,
    diff: str | tuple[str, ...] | None = None,
    B1: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    states: int = 10,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate a fast spin echo train.

    Wraps :class:`~blochsim.simulators.FSESimulator`.

    Parameters
    ----------
    flip : float or array-like
        Refocusing flip angles in degrees, one per echo.
    ESP : float
        Echo spacing in ms.
    T1 : float or array-like
        Longitudinal relaxation time in ms.
    T2 : float or array-like
        Transverse relaxation time in ms.
    phases : float or array-like, optional
        Refocusing phases in degrees.
    TR : float or array-like, optional
        Repetition time in ms.
    exc_flip : float, optional
        Excitation flip angle in degrees.
    exc_phase : float, optional
        Excitation phase in degrees.
    diff : str or tuple of str, optional
        Parameters to differentiate with respect to; ``None`` returns the signal only.
    B1 : float or array-like, optional
        Transmit field scaling, a fraction of the nominal flip angle.
    M0 : float or array-like, optional
        Proton density scaling.
    states : int, optional
        Configuration orders retained (EPG).
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
        FSESimulator(),
        diff,
        device,
        T1=T1,
        T2=T2,
        M0=M0,
        B1=B1,
        flip=flip,
        ESP=ESP,
        phases=phases,
        TR=TR,
        exc_flip=exc_flip,
        exc_phase=exc_phase,
        states=states,
        **values,
    )
