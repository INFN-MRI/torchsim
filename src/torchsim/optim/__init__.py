"""Designing a sequence: an acquisition, a cost, and the parameters under it."""

from __future__ import annotations

__all__ = [
    "Acquisition",
    "Bounded",
    "SequenceDesign",
    "SequenceOptimization",
    "crlb",
]

from ._design import (
    Acquisition,
    Bounded,
    SequenceDesign,
    SequenceOptimization,
    crlb,
)
