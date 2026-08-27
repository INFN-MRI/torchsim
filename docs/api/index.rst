API References
==============

TorchSim is organized around one idea: a **signal model** is the only thing a
sequence has to supply, and everything else -- differentiation, execution
across devices, estimation, reconstruction, design -- is written once against
that interface.

The pages below follow the subpackages.

:doc:`model`
    What a sequence of your own is written from: a state-machine model saying
    what each kind of event does to the spins, and a simulator saying what
    order they are played in.

:doc:`simulators`
    The sequences that ship with TorchSim, as classes and as one-call
    functional wrappers.

:doc:`sequence`
    The device-agnostic description an acquisition is assembled from -- events,
    operators, builders -- and the differentiable engine it is run on.

:doc:`estimators`
    Estimating tissue properties from a measured volume: dictionary matching,
    lookup tables, nonlinear least squares, kernel regression.

:doc:`recon`
    Solving for parameter maps straight from k-space, with the signal model
    inside the forward operator.

:doc:`optim`
    Choosing a sequence's parameters by minimizing a cost you write.

:doc:`execution`
    Where a per-voxel workload runs, and the remaining utilities.

.. toctree::
   :hidden:
   :maxdepth: 1

   model
   simulators
   sequence
   estimators
   recon
   optim
   execution
