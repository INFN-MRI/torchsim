"""Magnetization-prepared multi-echo spoiled GRE model."""

from __future__ import annotations

__all__ = ["MPnRAGEModel"]

import numpy.typing as npt
import torch

from ..base import AbstractModel, autocast
from ..sequence import SPGR, TissueProperties, mpnrage_description


class MPnRAGEModel(AbstractModel):
    """
    Magnetization Prepared (n) RApid Gradient Echo (MPnRAGE) Model.

    This class models Magnetization Prepared RApid Gradient Echo with n volumes per segment
    (MPnRAGE) signals based on tissue properties, pulse sequence parameters,
    and experimental conditions. It uses Extended Phase Graph (EPG) formalism
    to compute the magnetization evolution over time.

    Methods
    -------
    set_properties(T1, M0=1.0, B1=1.0, inv_efficiency=1.0):
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(nshots, flip, TR, TI=0.0):
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, flip, TR, TI=0.0, M0=1.0, B1=1.0, inv_efficiency=1.0):
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

    vectorized_engine = True

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        B1: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ) -> None:
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
        TI: float = 0.0,
    ) -> None:
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

        """
        self.sequence.nshots = int(nshots.reshape(()).item())
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TR = TR * 1e-3  # ms -> s
        self.sequence.TI = TI * 1e-3  # ms -> s

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
    ) -> torch.Tensor:
        description = mpnrage_description(
            nshots,
            flip,
            TR,
            inversion_time_s=TI,
        )
        signal = SPGR().simulate(
            description,
            TissueProperties(
                T1,
                T1,
                m0=M0,
                b1=B1,
                inversion_efficiency=inv_efficiency,
            ),
            nstates=1,
        ).signal
        return 1j * signal
