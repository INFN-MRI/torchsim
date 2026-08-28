Running a simulation
====================

.. currentmodule:: torchsim

The differentiable state machine a sequence description is run on, what the
transmit chain does to the pulses it asked for, and where the per-voxel work
is placed.

Simulating a volume and mapping one are both per-voxel, and both outgrow a card
long before they outgrow a host. What to do about that is one policy -- stay on
the host when a launch would not repay itself, cross whole when the problem
fits, stream through in chunks when it does not, and spread across as many
cards as there are -- and it is stated once, around the call.

The engine
----------

:class:`EpgEngine` runs a :class:`SequenceDescription` directly, which is what
you reach for when a builder or an interpreter handed you a description rather
than a simulator. A :class:`~torchsim.model.Simulator` wraps it and is how a
sequence is usually written; see :doc:`model`.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   EpgEngine
   SimulationResult

Transmit calibration
--------------------

What the transmit chain does to the pulse the description asked for: the
amplitude a flip angle needs, and the profile a finite-bandwidth pulse leaves
across the slice.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   calibrate
   exact_slice_profile
   ExactSliceProfile

Where the work runs
-------------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   execution
   offload
   distribute

Utilities
---------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   utils.b1rms
