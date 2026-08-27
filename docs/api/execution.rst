Execution and utilities
=======================

.. currentmodule:: torchsim

Simulating a volume and mapping one are both per-voxel, and both outgrow a card
long before they outgrow a host. What to do about that is one policy -- stay on
the host when a launch would not repay itself, cross whole when the problem
fits, stream through in chunks when it does not, and spread across as many
cards as there are -- and it is stated once, around the call.

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
