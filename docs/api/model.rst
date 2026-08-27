Signal models
=============

.. currentmodule:: torchsim.model

A sequence is written in two pieces. A :class:`StateMachineModel` says what a
voxel holds -- which tissue properties are exposed, and so which physics the
kernels carry -- and what each kind of event does to it. An
:class:`AbstractSimulator` says what order the events are played in. Which
kernel runs, how the work is cut across memory and devices, and how derivatives
are taken all follow from those two.

:class:`SignalModel` is the interface everything downstream reads: the
estimators, the model-based operator and the sequence optimizer take one and
never ask which of the two kinds it is.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SignalModel
   StateMachineModel
   AbstractSimulator
   Triggers

Trigger sets
------------

A :class:`Triggers` instance names the transition each kind of event applies.
Four are supplied, and they are what a state-machine model picks its physics
from: ``SPOILED`` and ``UNBALANCED`` for gradient-spoiled and gradient-echo
trains, ``BALANCED`` for a fully refocused steady state, and ``REFOCUSED`` for
a spin-echo train. Import them from :mod:`torchsim.model`.
