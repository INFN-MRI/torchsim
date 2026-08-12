API References
==============

Base
----
Base classes and routines for MR simulator implementation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.base.AbstractModel
   torchsim.base.autocast
   
Parameter configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   
   torchsim.base.prepare_environmental_parameters
   torchsim.base.prepare_single_pool
   torchsim.base.prepare_two_pool_bm
   torchsim.base.prepare_two_pool_mt
   torchsim.base.prepare_three_pool

Extended Phase Graphs
---------------------
Subroutines for Extended Phase Graphs based simulators.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.epg.states_matrix
   torchsim.epg.get_signal
   torchsim.epg.get_demodulated_signal

RF Pulses
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.epg.rf_pulse_op
   torchsim.epg.phased_rf_pulse_op
   torchsim.epg.multidrive_rf_pulse_op
   torchsim.epg.phased_multidrive_rf_pulse_op
   torchsim.epg.rf_pulse
   
   torchsim.epg.initialize_mt_sat
   torchsim.epg.mt_sat_op
   torchsim.epg.multidrive_mt_sat_op
   torchsim.epg.mt_sat
    
Relaxation and Exchange
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.epg.longitudinal_relaxation_op
   torchsim.epg.longitudinal_relaxation
   torchsim.epg.longitudinal_relaxation_exchange_op
   torchsim.epg.longitudinal_relaxation_exchange
   torchsim.epg.transverse_relaxation_op
   torchsim.epg.transverse_relaxation
   torchsim.epg.transverse_relaxation_exchange_op
   torchsim.epg.transverse_relaxation_exchange
   
Gradient Dephasing
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.epg.shift
   torchsim.epg.spoil  
   
Magnetization Prep
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   
   torchsim.epg.adiabatic_inversion

Flow and Diffusion
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.epg.diffusion_op
   torchsim.epg.diffusion
   torchsim.epg.flow_op
   torchsim.epg.flow   
   
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

Interpreters
~~~~~~~~~~~~
Differentiable EPG interpreters driven by a sequence description.

.. autosummary::
   :toctree: generated
   :nosignatures:

   torchsim.EpgSimulator
   torchsim.FSE
   torchsim.SPGR
   torchsim.BSSFP
   torchsim.SSFPFID
   torchsim.SSFPEcho
   torchsim.make_simulator

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
   torchsim.utils.slice_prof


