"""Signal-model authoring: the :class:`Simulator` base class and its event handlers.

:class:`SpinPhysics`, :class:`EventOperators` and the presets :data:`BALANCED`,
:data:`UNBALANCED`, :data:`SPOILED` and :data:`REFOCUSED` are what the shipped
simulators are written with; a user subclass sets handler attributes instead.
"""

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
