"""Authoring signal models.

A model is written in two pieces. :class:`StateMachineModel` says what a voxel
holds and what each kind of event does to it; :class:`AbstractSimulator` says
what order the events are played in and is what everything downstream
consumes. :class:`SignalModel` is the root both rest on, and is all a closed
form needs.
"""

from __future__ import annotations

__all__ = [
    "BALANCED",
    "REFOCUSED",
    "SPOILED",
    "UNBALANCED",
    "AbstractSimulator",
    "SignalModel",
    "StateMachineModel",
    "Triggers",
]

from ._signal import SignalModel
from ._state_machine import (
    BALANCED,
    REFOCUSED,
    SPOILED,
    UNBALANCED,
    AbstractSimulator,
    StateMachineModel,
    Triggers,
)
