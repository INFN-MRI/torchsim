"""Balanced Steady State Free Precession simulator."""

__all__ = ["bssfp_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.bssfp import bSSFPSimulator
from ._run import evaluated


def bssfp_sim(
    flip: float | npt.ArrayLike,
    TE: float | npt.ArrayLike,
    TR: float,
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    phase_inc: float = 180.0,
    B0: float | npt.ArrayLike = 0.0,
    chemshift: float | npt.ArrayLike = 0.0,
    M0: float | npt.ArrayLike = 1.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate the balanced SSFP steady-state signal.

    Balanced steady-state free precession (bSSFP, TrueFISP, FIESTA) with
    alternating RF phase, evaluated in closed form at the echo time; no state
    machine is run. The signal depends on the phase accrued across one TR, so
    ``B0`` and ``chemshift`` produce the characteristic banding. Wraps
    :class:`~blochsim.simulators.bSSFPSimulator`.

    Parameters
    ----------
    flip : float or array-like
        Excitation flip angle, in degrees. An array adds a trailing contrast
        axis.
    TE : float or array-like
        Echo time, in ms, measured from the excitation.
    TR : float
        Repetition time, in ms.
    T1 : float or array-like
        Longitudinal relaxation time, in ms.
    T2 : float or array-like
        Transverse relaxation time, in ms.
    diff : str or tuple of str, default=None
        Names of the properties to differentiate with respect to. ``None``
        returns the signal alone.
    phase_inc : float, default=180.0
        RF phase increment between successive excitations, in degrees. The
        default is the alternating phase of the standard balanced sequence.
    B0 : float or array-like, default=0.0
        Off-resonance, in Hz.
    chemshift : float or array-like, default=0.0
        Chemical shift of the spins, in Hz, added to the off-resonance.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
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
        bSSFPSimulator(),
        diff,
        device,
        T1=T1,
        T2=T2,
        M0=M0,
        B0=B0,
        chemshift=chemshift,
        flip=flip,
        TR=TR,
        TE=TE,
        phase_inc=phase_inc,
        **values,
    )
