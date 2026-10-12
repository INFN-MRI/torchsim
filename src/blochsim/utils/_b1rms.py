"""B1 rms estimation."""

__all__ = ["b1rms"]


import numpy as np
import numpy.typing as npt

gamma = 2 * np.pi * 42.58 * 1e6


def b1rms(duration: float, rf_envelope: npt.ArrayLike, rescale: bool = False) -> float:
    """Root-mean-square B1 of an RF pulse, in T.

    The rms is taken over ``duration``, ``sqrt(sum(B1**2 * dt) / duration)``,
    with ``dt = duration / len(rf_envelope)``. The envelope must be real; pass the
    magnitude of a complex waveform. ``rescale`` uses the gyromagnetic ratio of
    1H, 42.58 MHz/T.

    Parameters
    ----------
    duration : float
        Pulse duration, in s.
    rf_envelope : numpy.ndarray or torch.Tensor
        RF waveform amplitude, in T, uniformly sampled over ``duration``;
        shape ``(samples,)``. A list is not accepted.
    rescale : bool, default=False
        Scale the pulse to a flip angle of 1 rad (``gamma * sum(B1) * dt = 1``)
        before taking the rms, giving the rms per radian of flip angle.

    Returns
    -------
    float
        B1 rms, in T; in T per rad of flip angle when ``rescale`` is true.
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
