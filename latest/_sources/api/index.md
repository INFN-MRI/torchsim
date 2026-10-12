# API reference

The reference follows the subpackages. Conceptual background is in
{doc}`../explanations/index`; a signal model is the interface that estimation,
reconstruction and design are written against. The reconstruction itself, and
the per-voxel nonlinear least-squares fit, are bartorch's.

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`model` | `blochsim` | Simulator base class, handlers, model operator, temporal basis |
| {doc}`simulators` | `blochsim.simulators` | Shipped sequences, classes and functional wrappers |
| {doc}`sequence` | `blochsim` | Operators, sequence descriptions, RF definitions |
| {doc}`estimators` | `blochsim.estimators` | Dictionary matching, lookup tables, PERK |
| {doc}`optim` | `blochsim.optim` | Sequence parameter optimization |

```{toctree}
:hidden:
:maxdepth: 1

model
simulators
sequence
estimators
optim
```
