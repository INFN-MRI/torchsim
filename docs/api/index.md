# API reference

The reference follows the subpackages. Conceptual background is in
{doc}`../explanations/index`; a signal model is the interface that estimation,
reconstruction and design are written against.

| Page | Module | What it documents |
| --- | --- | --- |
| {doc}`model` | `blochsim.model` | Signal models: the physics of each event type and the simulator that orders the events. |
| {doc}`simulators` | `blochsim.simulators` | The shipped sequences, as classes and as functional wrappers. |
| {doc}`sequence` | `blochsim.sequence` | Sequence descriptions: events, the operators that emit them, and builders. |
| {doc}`estimators` | `blochsim.estimators` | Tissue property estimation: dictionary matching, lookup tables, nonlinear least squares, kernel regression. |
| {doc}`recon` | `blochsim.recon` | Model-based reconstruction of parameter maps from k-space, and the temporal basis. |
| {doc}`optim` | `blochsim.optim` | Sequence parameter optimization against a user-defined cost. |
| {doc}`execution` | `blochsim` | Execution policy: where per-voxel work runs. |

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
