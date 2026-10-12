# Explanations

What blochsim does, and how, one concept per page, in the order data flows
through it.

| Page | Topic |
| --- | --- |
| {doc}`signal-model` | Signal model: event handlers and layout. |
| {doc}`epg` | Extended phase graphs: relaxation, B1, B0, pools, diffusion, flow. |
| {doc}`description` | Pulseq and MRD sequence descriptions. |
| {doc}`derivatives` | Forward- and reverse-mode derivatives; Cramér-Rao bound. |
| {doc}`execution` | CPU and GPU execution, chunking, binding. |
| {doc}`parameter-estimation` | Dictionary matching, lookup tables, nonlinear least squares, PERK. |
| {doc}`model-based-reconstruction` | Linear subspace and nonlinear model-based reconstruction. |
| {doc}`sequence-design` | Protocol optimization against the Cramér-Rao bound. |

The derivations, format rules and kernels behind these pages are in
{doc}`../developer-guide/internals/index`.

```{toctree}
:hidden:
:maxdepth: 1

signal-model
epg
description
derivatives
execution
parameter-estimation
model-based-reconstruction
sequence-design
```
