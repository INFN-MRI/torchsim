Signal models
=============

.. currentmodule:: torchsim.model

**There are two base classes and you write exactly one of them.**

:class:`SignalModel` is the interface, and the only thing anything downstream
ever sees: the estimators, the model-based operator and the sequence optimizer
all take a ``SignalModel`` and never ask which kind it is. Whichever of the two
you write, that is what you have made.

Write a :class:`Simulator` when the signal has to be *played* -- a train of
pulses whose magnetization state carries from one event to the next, which is
almost every quantitative sequence. You set :attr:`~Simulator.model` to a
:class:`SpinPhysics` and implement :meth:`~Simulator.layout`, which returns the
operators of one repetition in the order they are played. The extended
phase-graph engine, the derivative, the device placement and the memory
policy all follow from that and none of them is yours to write.

Subclass :class:`SignalModel` directly and implement
:meth:`~SignalModel.evaluate` only when the signal has a closed form -- a
mono-exponential decay, an inversion-recovery curve -- so there is nothing to
play and no state to carry. There is no ``SpinPhysics`` and no ``layout`` in
that case, because there are no events for either to describe.

The two pieces a :class:`Simulator` is written from are separate so that
either can change without the other: :class:`SpinPhysics` says what a voxel
holds -- which tissue properties are exposed, and so which terms the kernels
carry -- and what each kind of event does to it, while
:meth:`~Simulator.layout` says only what order the events come in. Giving an
MRF timing a selective excitation, or a refocused train a readout that spoils
rather than winds, is then an assignment rather than a new model.

Either kind fixes its arguments the same way: a constructor takes the keywords
:meth:`~SignalModel.simulate` takes, :meth:`~SignalModel.bind` adds more to a
copy, and a call overrides either.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SignalModel
   Simulator
   SpinPhysics
   EventOperators

Operator sets
-------------

An :class:`EventOperators` instance names the operator each kind of event
plays. Four are supplied, and they are what a :class:`SpinPhysics` picks from:
``SPOILED`` and ``UNBALANCED`` for gradient-spoiled and gradient-echo trains,
``BALANCED`` for a fully refocused steady state, and ``REFOCUSED`` for a
spin-echo train. Import them from :mod:`torchsim.model`.

These are the roles a sequence is written in terms of. What the events they
emit are tagged with -- :class:`~torchsim.RfUse` for a Pulseq file,
:class:`~torchsim.EventAction` for the kernels -- is on :doc:`sequence`, and
does not line up one to one with the roles here.
