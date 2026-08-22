"""Read signal from EPG matrix."""

__all__ = ["get_signal", "get_demodulated_signal"]

from types import SimpleNamespace

import numpy.typing as npt

import torch


def get_signal(
    states: SimpleNamespace, order: int | npt.ArrayLike = 0
) -> float | complex:
    """
    Get signal from EPG matrix.

    Parameters
    ----------
    states : SimpleNamespace
        EPG states object.
    order : int | npt.ArrayLike, optional
        Dephasing order(s) contributing to signal.
        The default is 0.

    Returns
    -------
    float | complex
        MR signal.

    """
    # Get Fplus
    mxy = states.Fplus[order]

    # Sum over pools
    mxy = mxy.sum(axis=-1)

    # Average over locations
    mxy = mxy.mean(axis=-1)

    # Sum over orders
    return mxy.sum()


def get_demodulated_signal(
    states: SimpleNamespace, phi: float, order: int | npt.ArrayLike = 0
) -> float | complex:
    """
    Get signal from EPG matrix.

    Also demodulate RF phase.

    Parameters
    ----------
    states : SimpleNamespace
        EPG states object.
    phi : float
        RF phase to be demodulated.
    order : int | npt.ArrayLike, optional
        Dephasing order(s) contributing to signal.
        The default is 0.

    Returns
    -------
    float | complex
        MR signal.

    """
    # Get Fplus
    mxy = states.Fplus[order]

    # Demodulate
    mxy = mxy * torch.exp(-1j * phi)

    # Sum over pools
    mxy = mxy.sum(axis=-1)

    # Average over locations
    mxy = mxy.mean(axis=-1)

    # Sum over orders
    return mxy.sum()
