# API

The API reference follows the subpackages. A signal model is the interface
that estimation, reconstruction and design are written against.

{doc}`model`
: Signal models: the physics of each event type and the simulator
  that orders the events.

{doc}`simulators`
: The shipped sequences, as classes and as functional wrappers.

{doc}`sequence`
: Sequence descriptions: events, the operators that emit them, and
  builders.

{doc}`estimators`
: Tissue property estimation: dictionary matching, lookup tables,
  nonlinear least squares, kernel regression.

{doc}`recon`
: Model-based reconstruction of parameter maps from k-space, and the
  temporal basis.

{doc}`optim`
: Sequence parameter optimization against a user-defined cost.

{doc}`execution`
: Execution policy: where per-voxel work runs.

```{toctree}
:hidden:
:maxdepth: 1

model
simulators
sequence
estimators
recon
optim
execution
```
