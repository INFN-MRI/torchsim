"""Magnetization-prepared rapid gradient-echo model."""

from __future__ import annotations

__all__ = ["MPRAGEModel"]

import numpy.typing as npt
import torch

from ..base import AbstractModel, autocast
from ..sequence import SPGR, TissueProperties, mprage_description


class MPRAGEModel(AbstractModel):
    """
    Magnetization Prepared RApid Gradient Echo (MPnRAGE) Model.

    This class models Magnetization Prepared RApid Gradient Echo (MPRAGE) signals
    based on tissue properties, pulse sequence parameters, and experimental conditions.
    It uses Extended Phase Graph (EPG) formalism to compute the magnetization evolution over time.

    Assume that signal is sampled at center of k-space only.

    Methods
    -------
    set_properties(T1, M0=1.0, inv_efficiency=1.0)
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(nshots, flip, TR, TI=0.0)
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, TI, flip, TRspgr, nshots, M0=1.0, inv_efficiency=1.0)
        Computes the MPRAGE signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        from torchsim.models import MPRAGEModel

        model = MPRAGEModel()
        model.set_properties(T1=(200, 1000), inv_efficiency=0.95)
        model.set_sequence(TI=500.0, flip=5.0, TRspgr=5.0, nshots=128)
        signal = model()

    """

    vectorized_engine = True

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
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
        inv_efficiency : float | npt.ArrayLike, optional
            Inversion efficiency map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.M0 = M0
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        TI: float,
        flip: float,
        TRspgr: float,
        nshots: int | npt.ArrayLike,
    ) -> None:
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        TI : float
            Inversion time in milliseconds of shape ``(2,)``.
        flip : float | npt.ArrayLike
            Flip angle train in degrees.
        TRspgr : float
            Repetition time in milliseconds for each SPGR readout.
        TRmprage : float
            Repetition time in milliseconds for the whole inversion block.
        nshots : int | npt.ArrayLike
            Number of SPGR readout within the inversion block of shape ``(npre, npost)``
            If scalar, assume ``npre == npost == 0.5 * nshots``. Usually, this
            is the number of slice encoding lines ``(nshots = nz / Rz)``,
            i.e., the number of slices divided by the total acceleration factor along ``z``.

        """
        self.sequence.TI = TI * 1e-3  # ms -> s
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TRspgr = TRspgr * 1e-3  # ms -> s
        nshots = nshots.flatten()
        if nshots.numel() == 1:
            shot_count = int(nshots.item())
            if shot_count < 1:
                raise ValueError("nshots must be positive")
            nshots_before = shot_count // 2
            nshots_after = shot_count - nshots_before - 1
        elif nshots.numel() == 2:
            nshots_before, nshots_after = (int(value.item()) for value in nshots)
            if nshots_before < 0 or nshots_after < 0:
                raise ValueError("nshots entries must be nonnegative")
        else:
            raise ValueError("nshots must be scalar or (before, after)")
        if bool(self.sequence.TI < nshots_before * self.sequence.TRspgr):
            raise ValueError("TI must not precede the first MPRAGE excitation")
        self.sequence.nshots_before = nshots_before
        self.sequence.nshots_after = nshots_after

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float,
        nshots_before: int,
        nshots_after: int,
        M0: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ) -> torch.Tensor:
        description = mprage_description(
            nshots_before,
            nshots_after,
            flip,
            TRspgr,
            TI,
        )
        signal = SPGR().simulate(
            description,
            TissueProperties(
                T1,
                T1,
                m0=M0,
                inversion_efficiency=inv_efficiency,
            ),
            record="acquired",
            nstates=1,
        ).signal
        return 1j * signal[..., 0]
