"""The Z-spectrum: Lorentzian lines in closed form, and pulsed saturation simulated."""

from __future__ import annotations

__all__ = ["CESTSimulator", "LorentzianSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence import Delay, Excitation, Readout, SampledPulse, Spoil
from ..sequence._array import arrays, as_torch
from ._contrast import across_contrasts
from ._pulses import sample_count
from .gradient_echo import GAMMA_BAR_HZ_PER_T


class LorentzianSimulator(Simulator):
    """Z-spectrum as equilibrium minus one Lorentzian line per pool.

    Each pool saturates the water signal along a Lorentzian line centred on its
    own offset [1]_, and the lines add:

    ``M0 (1 - sum_i A_i (W_i / 2)^2 / ((W_i / 2)^2 + (offset - shift_i)^2))``

    Pool ``i``, counted from one, exposes ``pool{i}_amplitude``, ``pool{i}_width``
    (full width at half maximum) and ``pool{i}_shift``.

    Parameters
    ----------
    pools : int, optional
        Number of Lorentzian lines.
    **values
        Properties or sequence arguments to fix, as for any
        :class:`~blochsim.model.Simulator`.

    References
    ----------
    .. [1] Zaiss, M., Schmitt, B., Bachert, P., "Quantitative separation of
       CEST effect from magnetization transfer and spillover effects by
       Lorentzian-line-fit analysis of z-spectra", Journal of Magnetic
       Resonance 211.2 (2011), pp. 149-155.
       https://doi.org/10.1016/j.jmr.2011.05.001

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import LorentzianSimulator

        sequence = LorentzianSimulator(2, offsets=torch.linspace(-5.0, 5.0, 41))
        signal = sequence.simulate(
            pool1_amplitude=0.9, pool1_width=1.5, pool1_shift=0.0,
            pool2_amplitude=0.1, pool2_width=2.0, pool2_shift=3.5,
        )
        print(signal.shape)

    """

    properties = ("M0",)
    pools: int = 1

    def __init__(self, pools: int = 1, **values: Any) -> None:
        self.pools = int(pools)
        if self.pools < 1:
            raise ValueError(f"a Z-spectrum needs at least one pool, got {pools}")
        self.properties = (
            "M0",
            *(
                f"pool{index}_{what}"
                for index in range(1, self.pools + 1)
                for what in ("amplitude", "width", "shift")
            ),
        )
        super().__init__(**values)

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the closed-form signal."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        offsets: npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the saturated water signal at each offset.

        Parameters
        ----------
        properties : mapping
            ``pool{i}_amplitude`` as a fraction of ``M0``; ``pool{i}_width`` and
            ``pool{i}_shift`` in the unit of ``offsets``; ``M0`` scaling.
        offsets : array-like
            Saturation offsets from water, conventionally in ppm.
        """
        held = across_contrasts(properties, offsets)
        remaining = 1.0
        for index in range(1, self.pools + 1):
            amplitude = held[f"pool{index}_amplitude"]
            half = held[f"pool{index}_width"] / 2
            distance = offsets - held[f"pool{index}_shift"]
            remaining = remaining - amplitude * half**2 / (half**2 + distance**2)
        return held.get("M0", 1.0) * remaining


class CESTSimulator(Simulator):
    """Z-spectrum simulated through a train of rectangular saturation pulses.

    Each offset starts from equilibrium. Free water and the exchanging pools
    (``poolB_*`` to ``poolE_*``, the Bloch-McConnell model [1]_) evolve through
    the pulses and a spoiled delay, and the longitudinal magnetization at the read
    is the point of the spectrum. Each pool has a ``fraction`` of ``M0``, an
    exchange rate, relaxation times and a chemical shift in Hz. Each pulse starts
    at the phase the previous one ended at; the delays between pulses accrue none.

    Notes
    -----
    Corresponds to the sequence BART's ``sim`` plays as ``CEST``.

    References
    ----------
    .. [1] Zaiss, M., Bachert, P., "Chemical exchange saturation transfer
       (CEST) and MR Z-spectroscopy in vivo: a review of theoretical
       approaches and methods", Physics in Medicine and Biology 58.22 (2013),
       pp. R221-R269. https://doi.org/10.1088/0031-9155/58/22/R221

    Examples
    --------
    .. exec::

        import torch
        from blochsim.simulators import CESTSimulator

        sequence = CESTSimulator(offsets=torch.linspace(-5.0, 5.0, 21))
        spectrum = sequence.simulate(
            T1=1300.0, T2=75.0,
            poolB_fraction=0.001, poolB_exchange=1000.0, poolB_T1=1300.0,
            poolB_T2=10.0, poolB_shift=3.5 * 127.7,
        )
        print(spectrum.shape)

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
    # The pulses are played in the rotating frame and the delay before the
    # read spoils, so no order but zero ever holds anything.
    states = 1

    def layout(
        self,
        *,
        offsets: npt.ArrayLike,
        field_strength: float = 3.0,
        B1sat: float = 1.0,
        npulses: int = 1,
        pulse_duration: float = 100.0,
        interpulse_delay: float = 0.0,
        recovery: float = 6.5,
        dwell: float = 0.01,
    ) -> list:
        """Return one saturation train and read per offset.

        Parameters
        ----------
        offsets : array-like
            Saturation offsets from the free water, in ppm.
        field_strength : float, optional
            Main field in T; converts ppm to Hz.
        B1sat : float, optional
            Saturation pulse amplitude in microtesla.
        npulses : int, optional
            Saturation pulses per offset.
        pulse_duration : float, optional
            Duration of each pulse in ms.
        interpulse_delay : float, optional
            Delay between pulses in ms.
        recovery : float, optional
            Spoiled delay from the last pulse to the read, in ms.
        dwell : float, optional
            Hold time of each pulse sample in ms.

        Raises
        ------
        ValueError
            If a pulse is not a whole number of samples.
        """
        dwell_s = 1e-3 * dwell
        count = sample_count(1e-3 * pulse_duration, dwell_s)
        nutation = 2 * torch.pi * GAMMA_BAR_HZ_PER_T * 1e-6 * as_torch(B1sat)
        waveform = nutation * torch.ones(count, dtype=torch.float64)
        offset_hz = (
            GAMMA_BAR_HZ_PER_T
            * 1e-6
            * field_strength
            * torch.atleast_1d(as_torch(offsets)).to(torch.float64)
        )
        parts: list = []
        pulse_s = count * dwell_s
        for pulse in range(int(npulses)):
            # Each pulse takes up the phase the one before it ended at.
            carried = -2 * torch.pi * offset_hz * pulse * pulse_s
            parts.append(
                SampledPulse(waveform, dwell_s, phase_rad=carried, offset_hz=offset_hz)
            )
            if pulse < npulses - 1 and interpulse_delay > 0:
                parts.append(Delay(1e-3 * interpulse_delay))
        parts += [Spoil(), Delay(1e-3 * recovery), Excitation(torch.pi / 2), Readout()]
        return parts

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the longitudinal magnetization, offsets last."""
        # One train per offset, each read once: (offsets, voxels) or, for a
        # single voxel, (offsets,).
        signal = super().evaluate(properties, **sequence)[..., 0].movedim(0, -1)
        # The read is a pi/2 pulse, which the transmit field scales and turns
        # like every other, so it lays Mz along -i exp(i phase) sin(b1 pi/2).
        tissue = self.model.tissue(properties)
        b1 = torch.as_tensor(tissue.b1, device=signal.device)
        phase = torch.as_tensor(tissue.b1_phase_rad, device=signal.device)
        read = -1j * torch.sin(b1 * torch.pi / 2) * torch.exp(1j * phase)
        batch = self._batch(properties)
        if batch:
            read = torch.broadcast_to(read, batch).reshape(-1, 1)
        return (signal / read.to(signal.dtype)).real
