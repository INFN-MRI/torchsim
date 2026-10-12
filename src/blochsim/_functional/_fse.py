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

    Fast (turbo) spin echo: an excitation followed by a train of refocusing
    pulses, with one echo sampled midway between each pair, simulated with
    extended phase graphs. The train is played twice by default so that the
    recorded echoes start from the longitudinal magnetization left by the
    previous train. Wraps :class:`~blochsim.simulators.FSESimulator`.

    Parameters
    ----------
    flip : array-like
        Refocusing flip angles, in degrees, one per echo; their number is the
        echo train length.
    ESP : float
        Echo spacing, in ms.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    T2 : float or array-like
        Transverse relaxation time, in ms.
    phases : float or array-like, default=0.0
        Refocusing RF phases, in degrees; scalar or one per echo.
    TR : float or array-like, default=1e6
        Repetition time, in ms, setting the longitudinal recovery between
        trains. The default is effectively full recovery.
    exc_flip : float, default=90.0
        Excitation flip angle, in degrees.
    exc_phase : float, default=90.0
        Excitation RF phase, in degrees.
    diff : str or tuple of str, default=None
        Names of the properties to differentiate with respect to. ``None``
        returns the signal alone.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    states : int, default=10
        Configuration orders retained by the extended-phase-graph simulation.
    device : str or torch.device, default=None
        Device to run on. ``None`` follows the inputs.
    **values
        Any further property or protocol argument of the simulator, such as
        ``B0`` or ``B1``, passed to :meth:`~blochsim.model.Simulator.simulate`.
        Arguments given as ``None`` are dropped.

    Returns
    -------
    signal : complex array-like
        Signal of shape ``(..., len(flip))``, where ``...`` is the broadcast shape of
        the tissue properties.
    jacobian : same array type as ``signal``
        Derivatives of the signal with respect to ``diff``, of shape
        ``(..., len(diff), len(flip))``. Returned only when ``diff`` is given; a single name
        omits the ``len(diff)`` axis.
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
