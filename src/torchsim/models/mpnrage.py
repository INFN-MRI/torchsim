"""MPnRAGE sub-routines."""

__all__ = ["MPnRAGEModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch

from .. import epg


class MPnRAGEModel(AbstractModel):
    """
    Magnetization Prepared (n) RApid Gradient Echo (MPnRAGE) Model.

    This class models Magnetization Prepared RApid Gradient Echo with n volumes per segment
    (MPnRAGE) signals based on tissue properties, pulse sequence parameters,
    and experimental conditions. It uses Extended Phase Graph (EPG) formalism
    to compute the magnetization evolution over time.

    Methods
    -------
    set_properties(T1, M0=1.0, B1=1.0, inv_efficiency=1.0)
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(nshots, flip, TR, TI=0.0, slice_prof=1.0)
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, flip, TR, TI=0.0, M0=1.0, B1=1.0, inv_efficiency=1.0,
            slice_prof=1.0)
        Computes the MPnRAGE signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        from torchsim.models import MPnRAGEModel

        model = MPnRAGEModel()
        model.set_properties(T1=1000, inv_efficiency=0.95)
        model.set_sequence(nshots=128, flip=5.0, TR=10.0)
        signal = model()

    """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
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
        M0 : float or array-like, optional
            Proton density scaling factor, default is ``1.0``.
        B1 : float | npt.ArrayLike, optional
            Flip angle scaling map, default is ``1.0``.
        inv_efficiency : float | npt.ArrayLike, optional
            Inversion efficiency map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.M0 = M0
        self.properties.B1 = B1
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        nshots: int,
        flip: float,
        TR: float,
        MPRAGE_TR: float | None = None,
        num_inversions: int = 1,
        TI: float = 0.0,
        slice_prof: float | npt.ArrayLike = 1.0,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        nshots : int
            Number of SPGR shots per inversion block.
        flip : float
            Flip angle train in degrees.
        TR : float
            Repetition time in milliseconds.
        TI : float, optional
            Inversion time in milliseconds.
            The default is ``0.0``.
        MPRAGE_TR : float default is None
            Repetition time in milliseconds for the whole inversion block.
        num_inversions : int, optional
            Number of inversion pulses, default is ``1``.
        slice_prof : float | npt.ArrayLike, optional
            Flip angle scaling along slice profile.
            The default is ``1.0``.

        """
        self.sequence.nshots = nshots
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.TI = TI * 1e-3  # ms -> s
        self.sequence.slice_prof = slice_prof
        self.sequence.num_inversions = num_inversions
        self.sequence.MPRAGE_TR = MPRAGE_TR * 1e-3 if MPRAGE_TR is not None else None

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        nshots: int,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float = 0.0,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
        slice_prof: float | npt.ArrayLike = 1.0,
        num_inversions: int = 1,
        MPRAGE_TR: float = None,
    ):
        # Prepare relaxation parameters
        R1 = 1e3 / T1

        # Prepare EPG states matrix
        states = epg.states_matrix(
            device=R1.device,
            nlocs=slice_prof.numel(),
            nstates=1,
        )
        # Prepare excitation pulse
        RF = epg.rf_pulse_op(flip, slice_prof, B1)

        # Prepare relaxation operator for preparation pulse
        E1inv, rE1inv = epg.longitudinal_relaxation_op(R1, TI)

        # Prepare relaxation operator for sequence loop
        E1, rE1 = epg.longitudinal_relaxation_op(R1, TR)
        if MPRAGE_TR is not None:
            mprageE1, mpragerE1 = epg.longitudinal_relaxation_op(R1, MPRAGE_TR)
        # Initialize signal
        signal = []
        for i in range(num_inversions):
            # Apply inversion
            states = epg.adiabatic_inversion(states, inv_efficiency)
            states = epg.longitudinal_relaxation(states, E1inv, rE1inv)
            states = epg.spoil(states)

            # Scan loop
            for p in range(nshots):

                # Apply RF pulse
                states = epg.rf_pulse(states, RF)

                # Record signal
                signal.append(epg.get_signal(states))

                # Evolve
                states = epg.longitudinal_relaxation(states, E1, rE1)
                states = epg.spoil(states)
            if MPRAGE_TR is not None:
                epg.longitudinal_relaxation(states, mprageE1, mpragerE1)

        return M0 * 1j * torch.stack(signal)
