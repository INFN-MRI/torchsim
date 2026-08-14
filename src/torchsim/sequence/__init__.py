"""Sequence descriptions and differentiable state-machine simulation."""

from __future__ import annotations

__all__ = [
    "AdcRole",
    "ExactSliceProfile",
    "EventType",
    "RfDefinition",
    "RfShape",
    "RfUse",
    "SequenceDescription",
    "SequenceEvent",
    "ShimDefinition",
    "BSSFP",
    "FSE",
    "SPGR",
    "SSFPFID",
    "EpgSimulator",
    "SSFPEcho",
    "SimulationResult",
    "SubspaceBasis",
    "TissueProperties",
    "calibrate",
    "decompress_shape",
    "execution",
    "distribute",
    "exact_slice_profile",
    "fse_description",
    "make_simulator",
    "mpnrage_description",
    "mprage_description",
    "mrf_description",
    "offload",
    "simulate_subspace",
    "spgr_description",
]

from ._accelerators import distribute, execution, offload
from ._calibration import calibrate
from ._builders import (
    fse_description,
    mpnrage_description,
    mprage_description,
    mrf_description,
    spgr_description,
)
from ._description import (
    AdcRole,
    EventType,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceEvent,
    ShimDefinition,
    decompress_shape,
)
from ._transition import ExactSliceProfile, exact_slice_profile
from ._simulation import (
    BSSFP,
    FSE,
    SPGR,
    SSFPEcho,
    SSFPFID,
    EpgSimulator,
    SimulationResult,
    SubspaceBasis,
    TissueProperties,
    make_simulator,
    simulate_subspace,
)
