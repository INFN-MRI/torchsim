"""Fast spin-echo model."""

from __future__ import annotations

__all__ = ["FSEModel"]

import numpy.typing as npt
import torch

from ..base import AbstractModel, autocast
from ..sequence import FSE, TissueProperties, fse_description


class FSEModel(AbstractModel):
    """
    Fast Spin Echo (FSE) Model.

    This class models fast spin echo (FSE) signals based on
    tissue properties, pulse sequence parameters, and experimental conditions. It
    uses Extended Phase Graph (EPG) formalism to compute the magnetization evolution
    over time.

    Methods
    -------
    set_properties(T1, T2, M0=1.0, B1=1.0):
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(flip, ESP, phases=0.0, TR=1e6, exc_flip=90.0, exc_phase=90.0, nstates=10):
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, T2, flip, ESP, phases, TR=1e6, exc_flip=90.0, exc_phase=90.0, M0=1.0, B1=1.0, nstates=10):
        Computes the FSE signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.models import FSEModel

        model = FSEModel()
        model.set_properties(T1=1000, T2=80, M0=1.0, B1=1.0)
        model.set_sequence(flip=180.0 * torch.ones(128), ESP=2.0, TR=5000.0)
        signal = model()

    """

    vectorized_engine = True

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
    ):
        """
        Set tissue and system-specific properties for the MRF model.

        Parameters
        ----------
        T1 : float | npt.ArrayLike
            Longitudinal relaxation time in milliseconds.
        T2 : float | npt.ArrayLike
            Transverse relaxation time in milliseconds.
        M0 : float or array-like, optional
            Proton density scaling factor, default is ``1.0``.
        B1 : float | npt.ArrayLike, optional
            Flip angle scaling map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.T2 = T2
        self.properties.M0 = M0
        self.properties.B1 = B1

    @autocast
    def set_sequence(
        self,
        flip: float | npt.ArrayLike,
        ESP: float,
        phases: float | npt.ArrayLike = 0.0,
        TR: float | npt.ArrayLike = 1e6,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        nstates: int = 10,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        flip : float | npt.ArrayLike
            Refocusing flip angle train in degrees.
        ESP : float
            Echo spacing in milliseconds.
        phases : float | npt.ArrayLike, optional
            Refocusing flip angle phases in degrees.
            The default is ``90.0``.
        TR : float | npt.ArrayLike, optional
            Repetition time in milliseconds.
            The default is ``1e6``.
        exc_flip : float, optional
            Excitation flip angle train in degrees.
            The default is ``90.0``.
        exc_phase : float, optional
            Excitation flip angle phase in degrees.
            The default is ``90.0``.
        nstates : int, optional
            Number of EPG states to be retained.
            The default is ``10``.

        """
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.ESP = ESP * 1e-3  # ms -> s
        if phases.numel() == 1:
            phases = phases * torch.ones_like(flip)
        self.sequence.phases = torch.pi * phases / 180.0
        self.sequence.exc_flip = torch.pi * exc_flip / 180.0
        self.sequence.exc_phase = torch.pi * exc_phase / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.nstates = nstates

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
    ):
        description = fse_description(
            flip,
            ESP,
            phases_rad=phases,
            excitation_flip_rad=exc_flip,
            excitation_phase_rad=exc_phase,
        )
        signal = FSE().simulate(
            description,
            TissueProperties(T1, T2, b1=B1),
            nstates=nstates,
        ).signal
        # Get elapsed time and time left before next TR
        echo_train_length = torch.atleast_1d(torch.as_tensor(flip)).shape[-1]
        elapsed_time = ESP * echo_train_length
        dt = TR - elapsed_time

        # Calculate relaxation until TR
        R1 = 1e3 / T1
        ETR = torch.exp(-R1 * dt)  # (nTR,)
        # Both carry the tissue shape, and ``signal`` is (..., tissue, echo):
        # one trailing axis lines them up with the echoes, and any leading train
        # axis broadcasts on its own.
        ETR = ETR[..., None]
        M0 = M0[..., None]

        # Apply modulation
        signal = M0 * signal * (1 - ETR) / (1 - ETR * signal)  # (etl, nTR)

        return signal
