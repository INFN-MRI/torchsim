"""Fast spin echo."""

from __future__ import annotations

__all__ = ["FSESimulator", "HyperechoSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from .._model import REFOCUSED, Simulator, SpinPhysics
from ..sequence._array import as_torch, matched


class FSESimulator(Simulator):
    """Fast (turbo) spin echo (FSE, RARE) train, sampled at every echo.

    An excitation followed by one refocusing pulse per echo, spaced ``ESP``, with an
    echo midway between consecutive pulses. The extended phase graph carries the
    stimulated and indirect echoes of refocusing angles below 180 degrees. The signal
    is one complex sample per echo, demodulated by the refocusing phase, from the
    second playing of the train. Trains with several values of ``TR`` are simulated
    separately and stacked along the leading axis.

    Parameters
    ----------
    flip : float or array-like
        Refocusing flip angles, in degrees, one per echo; their number sets the
        echo train length.
    ESP : float or array-like
        Echo spacing, in ms.
    phases : float or array-like, default=0.0
        Refocusing phases, in degrees; scalar or one per echo.
    exc_flip : float, default=90.0
        Excitation flip angle, in degrees.
    exc_phase : float, default=90.0
        Excitation phase, in degrees.
    TR : float or array-like, default=1e6
        Repetition time, in ms; sets the longitudinal recovery between trains.
    states : int, default=10
        Configuration states kept by the EPG simulation.
    repetitions : int or "auto", default=2
        Playings of the train. The first leaves the longitudinal magnetization the
        TR recovers to, and the second, which is recorded, starts from it; that is
        the steady state after a 90 degree excitation. Another excitation needs
        ``"auto"``.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import FSESimulator

        sequence = FSESimulator(flip=180.0 * torch.ones(128), ESP=2.0, TR=5000.0)
        signal = sequence.simulate(T1=1000.0, T2=80.0)

    """

    model = SpinPhysics(
        properties={"T1": "t1_ms", "T2": "t2_ms", "M0": "m0", "B1": "b1"},
        operators=REFOCUSED,
    )
    states = 10
    # The first playing leaves the longitudinal magnetization the TR recovers
    # to, and the second, which is recorded, starts from it. A 90 degree
    # excitation leaves no longitudinal memory of the train before, so two
    # playings are the steady state; another excitation needs repetitions="auto".
    repetitions = 2

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
    ) -> list:
        """Return the train, placed at the echo times it is timed from."""
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        turns = torch.deg2rad(matched(phases, angles))
        return self._train(angles, turns, turns, ESP, exc_flip, exc_phase)

    def _train(
        self,
        angles: torch.Tensor,
        turns: torch.Tensor,
        demodulation: torch.Tensor,
        ESP: Any,
        exc_flip: float,
        exc_phase: float,
    ) -> list:
        """The excitation, then each refocusing pulse and the echo it forms."""
        # Not tensorized: a scalar spacing keeps the precision the caller
        # gave it, and the echo times are what the whole train is placed by.
        spacing_s = ESP * 1e-3

        parts = [
            (
                0.0,
                self.operators.excitation(
                    torch.pi / 180.0 * exc_flip, torch.pi / 180.0 * exc_phase
                ),
            )
        ]
        for index in range(angles.shape[-1]):
            echo_s = (index + 1) * spacing_s
            parts.append(
                (
                    echo_s - 0.5 * spacing_s,
                    self.operators.refocusing(angles[..., index], turns[..., index]),
                )
            )
            parts.append((echo_s, self.operators.readout(demodulation[..., index])))
        return parts

    def repetition_s(self, played_s: Any, **protocol: Any) -> Any:
        """Return ``TR`` in seconds, 1e6 ms when not given."""
        del played_s
        return protocol.get("TR", 1e6) * 1e-3

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the signal at every echo.

        Trains with different ``TR`` are simulated separately and stacked along the
        leading axis of ``TR`` and of the per-echo arguments batched with it.
        """
        played = self.played(**sequence)
        repetition = played.get("TR")
        playings = sequence.get("repetitions", self.repetitions)
        if not torch.is_tensor(repetition) or repetition.numel() < 2 or playings == 1:
            return super().evaluate(properties, **sequence)
        trains = repetition.shape[0]
        batched = {
            name: value
            for name, value in played.items()
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == trains
        }
        return torch.stack(
            [
                super(FSESimulator, self).evaluate(
                    properties,
                    **{
                        **sequence,
                        **{name: value[train] for name, value in batched.items()},
                        "TR": repetition.reshape(trains, -1)[train, 0],
                    },
                )
                for train in range(trains)
            ]
        )


class HyperechoSimulator(FSESimulator):
    """Hyperecho train: refocusing pulses mirrored about a central pulse.

    The pulses after the central one replay those before it in reverse order, each
    with flip angle ``-alpha`` about phase ``-phi`` (``alpha`` at ``180 - phi``
    degrees), so the last echo is the hyperecho [1]_: apart from relaxation, the
    full spin echo of the excitation. Every echo is demodulated at zero phase, the
    axis of the central pulse. The signal is one complex sample per echo, ``2 n + 1``
    for ``n`` flip angles.

    Parameters
    ----------
    flip : float or array-like
        Flip angles of the refocusing pulses before the central one, in degrees,
        one per echo.
    ESP : float or array-like
        Echo spacing, in ms.
    phases : float or array-like, default=0.0
        Phases of the refocusing pulses before the central one, in degrees.
    refocusing : float, default=180.0
        Flip angle of the central pulse, in degrees, played at zero phase.
    exc_flip : float, default=90.0
        Excitation flip angle, in degrees.
    exc_phase : float, default=90.0
        Excitation phase, in degrees.
    TR : float or array-like, default=1e6
        Repetition time, in ms.
    states : int, default=None
        Configuration states kept. When not given, sized from the whole winding,
        since the hyperecho is brought back from the highest order the first half
        reached.
    repetitions : int or "auto", default=2
        Playings of the train; the last is the signal.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.

    Notes
    -----
    With one flip angle at zero phase throughout, this is the sequence BART's
    ``epg`` plays as Hyperecho.

    References
    ----------
    .. [1] Hennig, J., Scheffler, K., "Hyperechoes", Magnetic Resonance in
       Medicine 46.1 (2001), pp. 6-12. https://doi.org/10.1002/mrm.1153

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import HyperechoSimulator

        sequence = HyperechoSimulator(flip=60.0 * torch.ones(5), ESP=10.0)
        signal = sequence.simulate(T1=1000.0, T2=80.0)
        print(signal.shape)

    """

    # The hyperecho is brought back from the highest order the first half
    # reached, so the orders are sized from the whole winding.
    states = None

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        refocusing: float = 180.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
    ) -> list:
        """Return the train, placed at the echo times it is timed from."""
        before = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        turns = torch.deg2rad(matched(phases, before))
        central = torch.full_like(before[..., :1], torch.pi / 180.0 * refocusing)
        angles = torch.cat([before, central, before.flip(-1)], dim=-1)
        turns = torch.cat(
            [turns, torch.zeros_like(central), torch.pi - turns.flip(-1)], dim=-1
        )
        return self._train(
            angles, turns, torch.zeros_like(turns), ESP, exc_flip, exc_phase
        )
