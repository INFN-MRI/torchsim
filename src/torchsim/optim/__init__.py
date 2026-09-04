"""Designing a sequence: an acquisition, a cost, and the parameters under it."""

from __future__ import annotations

__all__ = [
    "Bounded",
    "SequenceDesign",
    "crlb",
]

from ._design import (
    Bounded,
    SequenceDesign,
    SequenceOptimization,  # noqa: F401
    crlb,
)
