# API reference

The reference follows the subpackages. Conceptual background is in
{doc}`../explanations/index`; a signal model is the interface that estimation,
reconstruction and design are written against.

| Page | Module | Topic |
| --- | --- | --- |
| {doc}`model` | `blochsim.model` | Signal models, event physics, simulator. |
| {doc}`simulators` | `blochsim.simulators` | Shipped sequences, classes and functional wrappers. |
| {doc}`sequence` | `blochsim.sequence` | Sequence descriptions, events, operators, builders. |
| {doc}`estimators` | `blochsim.estimators` | Dictionary matching, lookup tables, nonlinear least squares, PERK. |
| {doc}`recon` | `blochsim.recon` | Model-based reconstruction, temporal basis. |
| {doc}`optim` | `blochsim.optim` | Sequence parameter optimization. |
| {doc}`execution` | `blochsim` | Execution policy. |

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
