"""B1 rms estimation."""

__all__ = ["b1rms"]


import numpy as np
import numpy.typing as npt

gamma = 2 * np.pi * 42.58 * 1e6


def b1rms(duration: float, rf_envelope: npt.ArrayLike, rescale: bool = False) -> float:
    """Compute the root-mean-square B1 of an RF pulse.

    Parameters
    ----------
    duration : float
        Pulse duration in s.
    rf_envelope : array-like
        RF waveform envelope in T, uniformly sampled over ``duration``.
    rescale : bool, optional
        Scale the pulse to a 1 rad flip angle first. Defaults to ``False``.

    Returns
    -------
    float
        B1 rms in T.
    """
    dt = duration / len(rf_envelope)  # sampling time in [s]

    # rescale to 1 rad if desired
    if rescale:
        scale = gamma * sum(rf_envelope) * dt  # rad
        rf_envelope = rf_envelope / scale

    # get integral of B1**2 over time
    B1sqrd_tau = sum(rf_envelope**2 * dt)  # T**2 * s

    # get B1 rms
    b1rms = (B1sqrd_tau / duration) ** 0.5  # T

    return b1rms
