API References
==============

Signal model authoring
----------------------
What a sequence of your own is written from: a state-machine model saying what
each kind of event does to the spins, and a simulator saying what order they
are played in.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.model.SignalModel
   torchsim.model.StateMachineModel
   torchsim.model.AbstractSimulator
   torchsim.model.Triggers

Simulators
----------
The sequences that ship with TorchSim. Each names its protocol at construction
and its tissue at the call.

Closed form
~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.simulators.bSSFPSimulator
   torchsim.simulators.SPGRSimulator
   torchsim.simulators.MP2RAGESimulator

State machine
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.simulators.FSESimulator
   torchsim.simulators.MPRAGESimulator
   torchsim.simulators.MPnRAGESimulator
   torchsim.simulators.MRFSimulator
   
Functional
----------
Functional wrappers for signal models.

Analytical
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   torchsim.bssfp_sim
   torchsim.spgr_sim
    
Iterative
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   torchsim.fse_sim
   torchsim.mprage_sim
   torchsim.mp2rage_sim
   torchsim.mpnrage_sim
   torchsim.mrf_sim
    
Sequence Description
--------------------
Device-agnostic description of an acquisition, shared by the interpreter,
the estimators, and the sequence optimizer.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.SequenceDescription
   torchsim.SequenceEvent
   torchsim.TissueProperties
   torchsim.SimulationResult

Operators
~~~~~~~~~
The modules a description is assembled from, and the registry that reaches one
by name.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.Operator
   torchsim.compose
   torchsim.module
   torchsim.Excitation
   torchsim.Refocusing
   torchsim.Inversion
   torchsim.Saturation
   torchsim.Readout
   torchsim.Delay
   torchsim.Dephase
   torchsim.Spoil
   torchsim.bSSFPReadout
   torchsim.SSFPFidReadout
   torchsim.SPGRReadout
   torchsim.FSEReadout
   torchsim.ideal_rf_definition
   torchsim.register_operator
   torchsim.operator
   torchsim.operator_names

Builders
~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.fse_description
   torchsim.spgr_description
   torchsim.mprage_description
   torchsim.mpnrage_description
   torchsim.mrf_description

Engine
~~~~~~
The differentiable state machine a sequence description is run on.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.EpgEngine

Subspace
~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.Subspace
   torchsim.simulate_subspace
   torchsim.SubspaceBasis

Parameter mapping
-----------------
Estimating tissue properties from a measured volume. A
:class:`~torchsim.ParameterMapping` states the problem over an
:class:`~torchsim.Acquisition` -- what is unknown and over what range, what is
measured separately, how noisy -- and any :class:`~torchsim.Estimator` fills it
in, returning one named map per unknown.

Both shipped methods honour :func:`~torchsim.execution`, so a volume larger
than a card is streamed through it and a second card halves the work.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.ParameterMapping
   torchsim.Estimator
   torchsim.DictionaryMatcher
   torchsim.DictionaryMatch
   torchsim.PERK

Sequence design
---------------
Choosing a sequence's parameters by minimizing a cost you write. An
:class:`~torchsim.Acquisition` is a simulator with the tissue it is designed
for already in place; the cost is a plain function of what it records; a
:class:`~torchsim.SequenceDesign` holds the parameters and runs the loop.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.Acquisition
   torchsim.Bounded
   torchsim.SequenceDesign
   torchsim.SequenceOptimization
   torchsim.crlb

Miscellaneous
-------------
Other simulation utilities.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.utils.b1rms


