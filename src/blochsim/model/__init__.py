"""Authoring signal models from a sequence description and the physics it is played with.

A signal model is a :class:`Simulator` subclass: it names the tissue
properties it exposes and how each kind of event acts on the spins through a
:class:`SpinPhysics`, and it returns the events of one repetition from its
protocol parameters. :class:`EventOperators` collects the operator factories;
:data:`BALANCED`, :data:`UNBALANCED`, :data:`SPOILED` and :data:`REFOCUSED`
are the presets for the standard sequence families. The shipped sequences are
in :mod:`blochsim.simulators`.
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
