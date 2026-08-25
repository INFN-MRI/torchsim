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

   torchsim.simulate_subspace
   torchsim.SubspaceBasis

Parameter Estimation
--------------------
Quantitative estimators built on TorchSim signal models.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.DictionaryMatcher
   torchsim.DictionaryMatch
   torchsim.PERK

Sequence Optimization
---------------------
Differentiable acquisition design.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.SequenceOptimizer
   torchsim.SequenceOptimization
   torchsim.FSET2Precision

Miscellaneous
-------------
Other simulation utilities.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.utils.b1rms


