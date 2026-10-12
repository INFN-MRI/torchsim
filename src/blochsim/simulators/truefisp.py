"""A balanced SSFP train played with pulses of finite duration."""

from __future__ import annotations

__all__ = ["TrueFISPSimulator"]

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence import Readout, module
from ._pulses import excitation, half_alpha_parts, inversion_parts
from .flash import _per_shot


class TrueFISPSimulator(Simulator):
    """Balanced SSFP (TrueFISP, FIESTA) train from equilibrium, optionally inverted first.

    Excitations alternate in phase between 0 and 180 degrees and each sample is
    demodulated by its pulse's phase; the signal is one complex sample per shot.
    An alpha/2 pulse of opposite phase before the train damps the approach to the
    steady state [1]_; after an inversion the train is the inversion-recovery
    TrueFISP [2]_. Each pulse is a Hamming-windowed sinc played sample by sample,
    or an instantaneous rotation when ``pulse_duration`` is zero. Gradients are
    balanced over each TR, so order zero is the whole state.

    Parameters
    ----------
    flip : float or array-like
        Flip angle, in degrees; scalar, or one per shot.
    TR : float
        Repetition time, in ms, pulse centre to pulse centre.
    nshots : int
        Excitations in the train.
    TE : float, default=None
        Echo time, in ms, from the pulse centre. ``TR / 2`` when not given.
    pulse_duration : float, default=1.0
        Pulse duration, in ms; zero is an instantaneous rotation.
    bandwidth_time : float, default=4.0
        Time-bandwidth product of the pulse: zero crossings of the sinc across
        its duration.
    half_alpha : bool, default=True
        Prepare the train with an alpha/2 pulse.
    preparation : float, default=None
        Time, in ms, from the start of the alpha/2 pulse to the start of the first
        pulse. ``TR / 2`` when not given. Zero is an instantaneous alpha/2
        rotation at the start of the train.
    inversion : {None, "ideal", "adiabatic"}, default=None
        Preparation before the alpha/2 pulse: none, an instantaneous inversion
        scaled by ``inv_efficiency``, or a sampled hyperbolic secant.
    inversion_duration : float, default=10.0
        Duration of the adiabatic inversion, in ms.
    inversion_phase : float, default=0.0
        Phase of the adiabatic inversion, in degrees, relative to the first pulse.
    spoiler : float, default=0.0
        Time between inversion and preparation, in ms, ending in a spoiler.
    dwell : float, default=0.01
        Hold time of each pulse sample, in ms.
    states : int, default=1
        Configuration states.

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
    inv_efficiency : float or array-like, default=1.0
        Fraction of a perfect inversion; read when ``inversion="ideal"``.

    Notes
    -----
    Corresponds to the sequences BART's ``sim`` plays as ``BSSFP`` and
    ``IR-BSSFP``, and with instantaneous pulses to the one its ``epg`` plays as
    bSSFP.

    References
    ----------
    .. [1] Deimling, M., Heid, O., "Magnetization prepared True FISP
       imaging", Proceedings of the International Society for Magnetic
       Resonance in Medicine 2 (1994), p. 495.

    .. [2] Schmitt, P., Griswold, M. A., Jakob, P. M., et al., "Inversion
       recovery TrueFISP: quantification of T1, T2, and spin density",
       Magnetic Resonance in Medicine 51.4 (2004), pp. 661-667.
       https://doi.org/10.1002/mrm.20058

    Examples
    --------
    .. exec::

        from blochsim.simulators import TrueFISPSimulator

        sequence = TrueFISPSimulator(
            flip=45.0, TR=4.5, nshots=200, inversion="adiabatic"
        )
        signal = sequence.simulate(T1=1000.0, T2=100.0)
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
        TE: float | None = None,
        pulse_duration: float = 1.0,
        bandwidth_time: float = 4.0,
        half_alpha: bool = True,
        preparation: float | None = None,
        inversion: str | None = None,
        inversion_duration: float = 10.0,
        inversion_phase: float = 0.0,
        spoiler: float = 0.0,
        dwell: float = 0.01,
    ) -> list:
        """Return the train, one sample per shot.

        Raises
        ------
        ValueError
            If the sample falls inside the pulse or after the next one, if the
            preparation is shorter than its pulse, if ``flip`` is neither scalar
            nor one per shot, or if a pulse is not a whole number of samples.
        """
        pulse_s, dwell_s = 1e-3 * pulse_duration, 1e-3 * dwell
        repetition_s = 1e-3 * TR
        echo_s = repetition_s / 2 if TE is None else 1e-3 * TE
        if echo_s < pulse_s / 2 or pulse_s / 2 + echo_s > repetition_s:
            raise ValueError(
                "TE is measured from the centre of the pulse, so it is at least "
                "half the pulse and ends before the next one begins"
            )
        angles = _per_shot(flip, nshots)
        pulse = excitation(pulse_s, dwell_s, bandwidth_time=bandwidth_time)
        parts = inversion_parts(
            inversion,
            duration_s=1e-3 * inversion_duration,
            phase_rad=torch.deg2rad(torch.as_tensor(inversion_phase)),
            spoiler_s=1e-3 * spoiler,
            dwell_s=dwell_s,
        )
        if half_alpha:
            spacing_s = repetition_s / 2 if preparation is None else 1e-3 * preparation
            parts += half_alpha_parts(
                angles[0],
                spacing_s=spacing_s,
                duration_s=pulse_s,
                dwell_s=dwell_s,
                bandwidth_time=bandwidth_time,
            )
        for shot, angle in enumerate(angles.unbind(0)):
            turn = torch.pi * (shot % 2)
            parts.append(
                module(
                    pulse(angle, turn),
                    (pulse_s / 2 + echo_s, Readout(turn)),
                    duration_s=repetition_s,
                )
            )
        return parts
