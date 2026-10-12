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

    MR fingerprinting with a FISP train: an inversion, then one excitation
    and one readout per TR, with flip angles and phases varying along the
    train. The readout gradient is unbalanced, and the signal is one complex
    sample per flip angle from the last playing of the train, simulated with
    extended phase graphs. Wraps :class:`~blochsim.simulators.MRFSimulator`.

    Parameters
    ----------
    flip : array-like
        Excitation flip angles, in degrees, one per TR; their number is the
        train length.
    TR : float or array-like
        Repetition time, in ms; scalar or one per TR.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    T2 : float or array-like
        Transverse relaxation time, in ms.
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
        Excitation RF phases, in degrees; scalar or one per TR.
    states : int, default=10
        Configuration orders retained by the extended-phase-graph simulation.
    nreps : int, default=1
        Playings of the train; the last is the signal. Passed to the simulator
        as ``repetitions``.
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
