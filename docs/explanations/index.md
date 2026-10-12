# Explanations

What blochsim does, and how, one concept per page, in the order data flows
through it.

{doc}`signal-model`
: A sequence written as handlers and a layout.

{doc}`epg`
: The physics in the kernels: configuration states, relaxation, B1 and B0,
  pools, diffusion and flow.

{doc}`description`
: What a Pulseq file or an MRD stream gives a simulator.

{doc}`derivatives`
: Forward mode for tissue properties, reverse mode for sequence parameters,
  and the Cramér-Rao bound built on them.

{doc}`execution`
: CPU threads or a GPU, chunking to fit memory, and structure resolved once.

{doc}`parameter-estimation`
: Dictionary matching, low-rank matching, lookup tables, nonlinear least
  squares and PERK.

{doc}`model-based-reconstruction`
: The signal model inside a reconstruction, linear in a subspace or
  nonlinear.

{doc}`sequence-design`
: Protocol parameters optimized against a Cramér-Rao bound.

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
