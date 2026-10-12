"""Frequency-modulated balanced SSFP."""

from __future__ import annotations

__all__ = ["fmSSFPSimulator"]

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence import Readout, module
from ._pulses import excitation
from .flash import _per_shot


class fmSSFPSimulator(Simulator):
    """Frequency-modulated balanced SSFP (fmSSFP) train.

    The RF phase is swept through one passband per sweep: pulse ``n`` of a sweep
    is played ``pi + 2 pi n / nshots`` rad ahead in phase of the previous pulse,
    so each voxel is carried once through the ``1 / TR`` period of the bSSFP
    off-resonance profile [1]_. Each sample is demodulated by its pulse's phase,
    and ``sweeps`` sweeps are played before the recorded one, as in the radial
    acquisition of [2]_. The signal is one complex sample per shot of the recorded
    sweep. Pulses are instantaneous rotations unless ``pulse_duration`` is given,
    and are then windowed sincs.

    Parameters
    ----------
    flip : float or array-like
        Flip angle, in degrees; scalar or one per shot of a sweep.
    TR : float
        Repetition time, in ms, pulse centre to pulse centre.
    nshots : int
        Excitations in one sweep.
    sweeps : int, default=1
        Sweeps played before the recorded one.
    TE : float, default=None
        Echo time, in ms, from the pulse centre; ``TR / 2`` when not given.
    pulse_duration : float, default=0.0
        Pulse duration, in ms; zero is an instantaneous rotation. A whole number
        of ``dwell``.
    bandwidth_time : float, default=4.0
        Zero crossings of the Hamming-windowed sinc across a pulse.
    dwell : float, default=0.01
        Hold time of each pulse sample, in ms.
    states : int, default=1
        Configuration states kept; a balanced train winds nothing on, so order
        zero is the whole state.
    repetitions : int or "auto", default=1
        Playings of the whole train; the last is the signal.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.
    B0 : float or array-like, default=0.0
        Off-resonance, in Hz.

    Notes
    -----
    Corresponds to the sequence BART's ``epg`` plays as fmSSFP.

    References
    ----------
    .. [1] Foxall, D. L., "Frequency-modulated steady-state free precession
       imaging", Magnetic Resonance in Medicine 48.3 (2002), pp. 502-508.
       https://doi.org/10.1002/mrm.10225

    .. [2] Roeloffs, V., Rosenzweig, S., Holme, H. C. M., Uecker, M., Frahm,
       J., "Frequency-modulated SSFP with radial sampling and subspace
       reconstruction: A time-efficient alternative to phase-cycled bSSFP",
       Magnetic Resonance in Medicine 81.3 (2019), pp. 1566-1579.
       https://doi.org/10.1002/mrm.27505

    Examples
    --------
    .. exec::

        from blochsim.simulators import fmSSFPSimulator

        sequence = fmSSFPSimulator(flip=30.0, TR=5.0, nshots=100)
        signal = sequence.simulate(T1=1000.0, T2=100.0, B0=20.0)
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "T2": "t2_ms",
            "M0": "m0",
            "B1": "b1",
            "B0": "b0_hz",
        },
    )
    # A balanced train winds nothing on, so order zero is the whole state.
    states = 1

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        TR: float,
        nshots: int,
        sweeps: int = 1,
        TE: float | None = None,
        pulse_duration: float = 0.0,
        bandwidth_time: float = 4.0,
        dwell: float = 0.01,
    ) -> list:
        """Return the sweeps, one sample per shot of the last.

        Raises
        ------
        ValueError
            If the sample falls inside the pulse or after the next one, if ``sweeps``
            is negative, if ``flip`` is neither scalar nor one per shot, or if a pulse
            is not a whole number of samples.
        """
        pulse_s, dwell_s = 1e-3 * pulse_duration, 1e-3 * dwell
        repetition_s = 1e-3 * TR
        echo_s = repetition_s / 2 if TE is None else 1e-3 * TE
        if echo_s < pulse_s / 2 or pulse_s / 2 + echo_s > repetition_s:
            raise ValueError(
                "TE is measured from the centre of the pulse, so it is at least "
                "half the pulse and ends before the next one begins"
            )
        if int(sweeps) < 0:
            raise ValueError(f"sweeps must not be negative, not {sweeps!r}")
        angles = _per_shot(flip, nshots)
        pulse = excitation(pulse_s, dwell_s, bandwidth_time=bandwidth_time)

        played = (int(sweeps) + 1) * int(nshots)
        within = torch.arange(played, dtype=torch.float64) % int(nshots)
        increments = torch.pi + 2 * torch.pi * within / int(nshots)
        turns = torch.remainder(torch.cumsum(increments, 0), 2 * torch.pi).tolist()
        recorded = played - int(nshots)

        parts = []
        for shot, turn in enumerate(turns):
            angle = angles[shot % int(nshots)]
            if shot < recorded:
                parts.append(module(pulse(angle, turn), duration_s=repetition_s))
            else:
                parts.append(
                    module(
                        pulse(angle, turn),
                        (pulse_s / 2 + echo_s, Readout(turn)),
                        duration_s=repetition_s,
                    )
                )
        return parts
