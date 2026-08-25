"""The sequences that ship with TorchSim, as simulators.

Each names its protocol arguments at construction and its tissue properties
at the call, so parameter inference, sequence optimization and a
reconstruction pipeline take all of them the same way.
"""

from __future__ import annotations

__all__ = [
    "FSESimulator",
    "MP2RAGESimulator",
    "MPRAGESimulator",
    "MPnRAGESimulator",
    "MRFSimulator",
    "SPGRSimulator",
    "bSSFPSimulator",
]

from .bssfp import bSSFPSimulator
from .fse import FSESimulator
from .mp2rage import MP2RAGESimulator
from .mpnrage import MPnRAGESimulator
from .mprage import MPRAGESimulator
from .mrf import MRFSimulator
from .spgr import SPGRSimulator
