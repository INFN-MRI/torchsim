"""Unbalanced SSFP MR fingerprinting model."""

from __future__ import annotations

__all__ = ["MRFModel"]

import numpy.typing as npt
import torch

from ..base import AbstractModel, autocast
from ..sequence import SSFPFID, TissueProperties, mrf_description


class MRFModel(AbstractModel):
    """
    SSFP Magnetic Resonance Fingerprinting (MRF) Model.

    This class models steady-state free precession (SSFP) MRF signals based on
    tissue properties, pulse sequence parameters, and experimental conditions. It
    uses Extended Phase Graph (EPG) formalism to compute the magnetization evolution
    over time.

    Methods
    -------
    set_properties(T1, T2, M0=1.0, B1=1.0, inv_efficiency=1.0):
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(flip, TR, TI=0.0, slice_prof=1.0, nstates=10, nreps=1):
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, T2, flip, TR, TI=0.0, M0=1.0, B1=1.0, inv_efficiency=1.0, slice_prof=1.0, nstates=10, nreps=1):
        Computes the MRF signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.models import MRFModel

        model = MRFModel()
        model.set_properties(T1=1000, T2=80, M0=1.0, B1=1.0, inv_efficiency=0.95)
        model.set_sequence(flip=torch.linspace(5.0, 60.0, 1000), TR=10.0, nstates=20, nreps=1)
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
        inv_efficiency: float | npt.ArrayLike = 1.0,
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
        inv_efficiency : float | npt.ArrayLike, optional
            Inversion efficiency map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.T2 = T2
        self.properties.M0 = M0
        self.properties.B1 = B1
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float = 0.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
        nreps: int = 1,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        flip : float | npt.ArrayLike
            Flip angle train in degrees.
        TR : float | npt.ArrayLike
            Repetition time in milliseconds.
        TI : float, optional
            Inversion time in milliseconds.
            The default is ``0.0``.
        slice_prof : float | npt.ArrayLike, optional
            Flip angle scaling along slice profile.
            The default is ``1.0``.
        nstates : int, optional
            Number of EPG states to be retained.
            The default is ``10``.
        nreps : int, optional
            Number of simulation repetitions.
            The default is ``1``.

        """
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.TI = TI * 1e-3  # ms -> s
        self.sequence.slice_prof = slice_prof
        self.sequence.nstates = nstates
        self.sequence.nreps = nreps

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        T2: float | npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float = 0.0,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        nstates: int = 10,
        nreps: int = 1,
    ):
        description = mrf_description(flip, TR, inversion_time_s=TI)
        signal = SSFPFID().simulate(
            description,
            TissueProperties(
                T1,
                T2,
                m0=M0,
                b1=B1,
                inversion_efficiency=inv_efficiency,
            ),
            repetitions=nreps,
            nstates=nstates,
            slice_profile=slice_prof,
        ).signal
        return 1j * signal[..., -len(flip) :]
