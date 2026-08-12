"""MP2RAGE sub-routines."""

__all__ = ["MP2RAGEModel"]

from ..base import AbstractModel
from ..base import autocast

import numpy.typing as npt
import torch


class MP2RAGEModel(AbstractModel):
    """
    Magnetization Prepared (2) RApid Gradient Echo (MPnRAGE) Model.

    This class models Magnetization Prepared RApid Gradient Echo with 2 volumes per segment
    (MP2RAGE) signals based on tissue properties, pulse sequence parameters,
    and experimental conditions. It uses Extended Phase Graph (EPG) formalism
    to compute the magnetization evolution over time.

    Assume that signal is sampled at center of k-space only.

    Methods
    -------
    set_properties(T1, M0=1.0, inv_efficiency=1.0):
        Sets tissue relaxation properties and experimental conditions.

    set_sequence(nshots, flip, TR, TI=0.0):
        Configures the pulse sequence parameters for the simulation.

    _engine(T1, TI, flip, TRspgr, TRmp2rage, nshots, M0=1.0, inv_efficiency=1.0):
        Computes the MP2RAGE signal for given tissue properties and sequence parameters.

    Examples
    --------
    .. exec::

        from torchsim.models import MP2RAGEModel

        model = MP2RAGEModel()
        model.set_properties(T1=(200, 1000), inv_efficiency=0.95)
        model.set_sequence(TI=(500.0, 1500.0), flip=5.0, TRspgr=5.0, TRmp2rage=3000.0, nshots=128)
        signal = model()

    """

    @autocast
    def set_properties(
        self,
        T1: float | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
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
        inv_efficiency : float | npt.ArrayLike, optional
            Inversion efficiency map, default is ``1.0``.

        """
        self.properties.T1 = T1
        self.properties.M0 = M0
        self.properties.inv_efficiency = inv_efficiency

    @autocast
    def set_sequence(
        self,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float,
        TRmp2rage: float,
        nshots: int | npt.ArrayLike,
    ):
        """
        Set sequence parameters for the SPGR model.

        Parameters
        ----------
        TI : npt.ArrayLike
            Inversion time (s) in milliseconds of shape ``(2,)``.
        flip : float | npt.ArrayLike
            Flip angle train in degrees of shape ``(2,)``.
            If scalar, assume same angle for both blocks.
        TRspgr : float
            Repetition time in milliseconds for each SPGR readout.
        TRmp2rage : float
            Repetition time in milliseconds for the whole inversion block.
        nshots : int | npt.ArrayLike
            Number of SPGR readout within the inversion block of shape ``(npre, npost)``
            If scalar, assume ``npre == npost == 0.5 * nshots``. Usually, this
            is the number of slice encoding lines ``(nshots = nz / Rz)``,
            i.e., the number of slices divided by the total acceleration factor along ``z``.

        """
        self.sequence.nshots = nshots
        self.sequence.TI = TI * 1e-3  # ms -> s
        if flip.numel() == 1:
            flip = torch.repeat_interleave(flip, 2)
        self.sequence.flip = torch.pi * flip / 180.0
        self.sequence.TRspgr = TRspgr * 1e-3  # ms -> s
        self.sequence.TRmp2rage = TRmp2rage * 1e-3  # ms -> s
        if nshots.numel() == 1:
            nshots = torch.repeat_interleave(nshots // 2, 2)
        self.sequence.nshots = nshots

    @staticmethod
    def _engine(
        T1: float | npt.ArrayLike,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float,
        TRmp2rage: float,
        nshots: int | npt.ArrayLike,
        M0: float | npt.ArrayLike = 1.0,
        inv_efficiency: float | npt.ArrayLike = 1.0,
    ):
        R1 = 1e3 / T1

        # Calculate number of shots before and after DC sampling
        nshots_bef = nshots[0]
        nshots_aft = nshots[1]
        nslices = torch.sum(nshots)

        time_bef = nshots_bef * TRspgr
        time_aft = nshots_aft * TRspgr

        # calculate timing
        TD = []
        TD.append(TI[0] - time_bef)
        TD.append(TI[1] - TI[0] - (time_aft + time_bef))
        TD.append(TRmp2rage - TI[1] - time_aft)

        # Calculate longitudinal relaxation operators
        E1 = torch.exp(-R1 * TRspgr)  # within SPGR shot relaxation
        ETD = [torch.exp(-R1 * time) for time in TD]  # between SPGR blocks relaxation

        # Compute RF rotation
        ca = torch.cos(flip)
        sa = torch.sin(flip)

        # compute steady state longitudinal magnetization
        MZsteadystate = 1 / (
            1
            + inv_efficiency * (ETD[0] * ETD[1] * ETD[2] * (ca * E1).prod() ** nslices)
        )
        MZsteadystatenumerator = 1 - ETD[0]
        MZsteadystatenumerator *= (ca[0] * E1) ** nslices + (1 - E1) * (
            1 - (ca[0] * E1) ** nslices
        ) / (1 - ca[0] * E1)
        MZsteadystatenumerator = MZsteadystatenumerator * ETD[1] + (1 - ETD[1])
        MZsteadystatenumerator *= (ca[1] * E1) ** nslices + (1 - E1) * (
            1 - (ca[1] * E1) ** nslices
        ) / (1 - ca[1] * E1)
        MZsteadystatenumerator = MZsteadystatenumerator * ETD[2] + (1 - ETD[2])
        MZsteadystate = MZsteadystate * MZsteadystatenumerator

        # Initialize signal
        signal = []

        # Compute signal for first volume
        temp = (-inv_efficiency * MZsteadystate * ETD[0] + (1 - ETD[0])) * (
            ca[0] * E1
        ) ** (nshots_bef) + (1 - E1) * (1 - (ca[0] * E1) ** (nshots_bef)) / (
            1 - (ca[0] * E1)
        )
        signal.append(sa[0] * temp)

        # signal for second volume
        temp *= (ca[1] * E1) ** (nshots_aft) + (1 - E1) * (
            1 - (ca[1] * E1) ** (nshots_aft)
        ) / (1 - (ca[1] * E1))
        temp = (temp * ETD[1] + (1 - ETD[1])) * (ca[1] * E1) ** (nshots_bef) + (
            1 - E1
        ) * (1 - (ca[1] * E1) ** (nshots_bef)) / (1 - (ca[1] * E1))
        signal.append(sa[1] * temp)

        return M0 * torch.stack(signal)
