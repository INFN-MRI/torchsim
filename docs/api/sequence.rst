Sequences
=========

.. currentmodule:: torchsim

Device-agnostic description of an acquisition, shared by the interpreter, the
estimators and the sequence optimizer. A description is a list of events with
the timing and the tissue interaction each one carries; nothing in it names a
vendor or a device.

Description
-----------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceDescription
   SequenceEvent
   TissueProperties
   SimulationResult

Event vocabulary
~~~~~~~~~~~~~~~~

What an event may be, and what it does when it is reached.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   EventType
   EventAction
   AdcRole
   RfDefinition
   RfMode
   RfShape
   RfUse
   ShimDefinition
   decompress_shape

Operators
---------

The modules a description is assembled from, and the registry that reaches one
by name. Writing a new operator is writing a Python function that returns
events; it reaches the fused kernels with no change to them.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Operator
   compose
   module
   Excitation
   Refocusing
   Inversion
   Saturation
   Readout
   Delay
   Dephase
   Spoil
   bSSFPReadout
   SSFPFidReadout
   SPGRReadout
   FSEReadout
   ideal_rf_definition
   register_operator
   operator
   operator_names

Builders
--------

Whole sequences, assembled from the operators above.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   fse_description
   spgr_description
   mprage_description
   mpnrage_description
   mrf_description

Engine
------

The differentiable state machine a sequence description is run on.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   EpgEngine

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

Subspace
--------

The low-rank temporal basis a train spans, fitted to simulated signals and used
by both the estimators and a subspace reconstruction.

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Subspace
   SubspaceBasis
   simulate_subspace
