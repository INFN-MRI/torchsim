Sequences
=========

.. currentmodule:: torchsim

Device-agnostic description of an acquisition, shared by the interpreter, the
estimators and the sequence optimizer. A description is a list of events with
the timing and the tissue interaction each one carries; nothing in it names a
vendor or a device.

This page is the description and what it is assembled from. Running one is on
:doc:`execution`, and the temporal basis a run spans is on :doc:`recon`.

Description
-----------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceDescription
   SequenceEvent
   TissueProperties

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
