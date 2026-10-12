"""Authoring signal models: :class:`SpinPhysics` and :class:`Simulator`."""

from __future__ import annotations

__all__ = [
    "BALANCED",
    "REFOCUSED",
    "SPOILED",
    "UNBALANCED",
    "EventOperators",
    "Simulator",
    "SpinPhysics",
]

from ._state_machine import (
    BALANCED,
    REFOCUSED,
    SPOILED,
    UNBALANCED,
    EventOperators,
    Simulator,
    SpinPhysics,
)
