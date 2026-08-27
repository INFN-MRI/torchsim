"""Designing a sequence: an acquisition, a cost, and the parameters under it."""

from __future__ import annotations

__all__ = [
    "Bounded",
    "SequenceDesign",
    "SequenceOptimization",
    "crlb",
]

from ._design import (
    Bounded,
    SequenceDesign,
    SequenceOptimization,
    crlb,
)
