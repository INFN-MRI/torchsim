"""The sequences that ship with BlochSim, as :class:`~blochsim.model.Simulator` subclasses.

Each is a pulse sequence named as in the literature: gradient-echo, spin-echo and
balanced SSFP trains, inversion-prepared (MPRAGE, MP2RAGE, MPnRAGE, Look-Locker)
and fingerprinting sequences, and the closed-form relaxometry contrasts. The
sequence's protocol (flip angles, timings, counts) is given to the constructor
and tissue properties to :meth:`~blochsim.model.Simulator.simulate`; each class
documents both. Closed-form classes evaluate their expression directly, the
others play their event train through the state machine.
"""

from __future__ import annotations

__all__ = [
    "ASLSimulator",
    "CESTSimulator",
    "DiffusionSimulator",
    "DoubleAngleSimulator",
    "FLASHSimulator",
    "FSESimulator",
    "HSFPSimulator",
    "HyperechoSimulator",
    "IRMultiGradientEchoSimulator",
    "IRbSSFPSimulator",
    "InversionRecoverySimulator",
    "LookLockerSimulator",
    "LorentzianSimulator",
    "MOLLISimulator",
    "MP2RAGESimulator",
    "MPRAGESimulator",
    "MPnRAGESimulator",
    "MRFSimulator",
    "MultiEchoSimulator",
    "MultiGradientEchoSimulator",
    "SPGRSimulator",
    "SpinEchoSimulator",
    "StimulatedEchoSimulator",
    "TrueFISPSimulator",
    "bSSFPSimulator",
    "fmSSFPSimulator",
]

from .asl import ASLSimulator
from .bssfp import bSSFPSimulator
from .cest import CESTSimulator, LorentzianSimulator
from .diffusion import DiffusionSimulator
from .flash import FLASHSimulator
from .fmssfp import fmSSFPSimulator
from .fse import FSESimulator, HyperechoSimulator
from .gradient_echo import IRMultiGradientEchoSimulator, MultiGradientEchoSimulator
from .hsfp import HSFPSimulator
from .looklocker import IRbSSFPSimulator, LookLockerSimulator, MOLLISimulator
from .mp2rage import MP2RAGESimulator
from .mpnrage import MPnRAGESimulator
from .mprage import MPRAGESimulator
from .mrf import MRFSimulator
from .relaxometry import (
    DoubleAngleSimulator,
    InversionRecoverySimulator,
    MultiEchoSimulator,
)
from .spgr import SPGRSimulator
from .spin_echo import SpinEchoSimulator
from .stimulated_echo import StimulatedEchoSimulator
from .truefisp import TrueFISPSimulator
