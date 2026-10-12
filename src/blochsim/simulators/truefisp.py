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
    """Balanced SSFP train from equilibrium, optionally inverted first.

    Pulses alternate in phase and each sample is demodulated by its pulse's phase.
    An alpha/2 pulse of opposite phase before the train damps the approach to the
    steady state [1]_; after an inversion the train is the inversion-recovery
    TrueFISP [2]_. Each pulse is a windowed sinc played sample by sample, or an
    instantaneous rotation when ``pulse_duration`` is zero.

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

        Parameters
        ----------
        flip : float or array-like
            Flip angle in degrees, scalar or one per shot.
        TR : float
            Repetition time in ms, pulse centre to pulse centre.
        nshots : int
            Excitations in the train.
        TE : float, optional
            Echo time in ms, from the pulse centre; defaults to ``TR / 2``.
        pulse_duration : float, optional
            Pulse duration in ms; zero is an instantaneous rotation.
        bandwidth_time : float, optional
            Zero crossings of the Hamming-windowed sinc across a pulse.
        half_alpha : bool, optional
            Prepare the train with an alpha/2 pulse.
        preparation : float, optional
            Time in ms from the start of the alpha/2 pulse to the start of the first
            pulse; defaults to ``TR / 2``. Zero is an instantaneous alpha/2 rotation
            at the start of the train.
        inversion : {None, "ideal", "adiabatic"}, optional
            Preceding the preparation: none, an instantaneous inversion scaled by
            ``inv_efficiency``, or a sampled hyperbolic secant.
        inversion_duration : float, optional
            Duration of the adiabatic inversion in ms.
        inversion_phase : float, optional
            Phase of the adiabatic inversion in degrees, relative to the first pulse.
        spoiler : float, optional
            Time in ms between inversion and preparation, ending in a spoiler.
        dwell : float, optional
            Hold time of each pulse sample in ms.

        Raises
        ------
        ValueError
            If the sample falls inside the pulse or after the next one, if the
            preparation is shorter than its pulse, if ``flip`` is neither scalar nor
            one per shot, or if a pulse is not a whole number of samples.
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
