API References
==============

Signal model authoring
----------------------
Base classes for writing a signal model of your own.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.model.SignalModel
   torchsim.model.EpgModel

Signal Models
-------------
Pre-defined signal models.

Analytical
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   torchsim.models.bSSFPModel
   torchsim.models.SPGRModel
   
Iterative
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   torchsim.models.FSEModel
   torchsim.models.MPRAGEModel
   torchsim.models.MP2RAGEModel
   torchsim.models.MPnRAGEModel
   torchsim.models.MRFModel
   
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
   torchsim.excitation
   torchsim.refocusing
   torchsim.inversion
   torchsim.saturation
   torchsim.readout
   torchsim.delay
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


