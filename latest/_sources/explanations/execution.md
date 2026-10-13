# Execution

```{admonition} TL;DR
:class: tldr

- A small call runs where its tensors are; a large one is cut into chunks of voxels and spread over every GPU.
- A card holds one chunk at a time, sized to its free memory, so a volume of any size runs; the answer comes back where the tensors are.
- A simulator resolves its sequence structure on the first call and rebinds only the numbers on later calls.
```

You choose where a simulation, a dictionary match or a model-based
reconstruction runs. Each simulator accepts scalar and tensor-valued inputs;
tensor-valued inputs are simulated in parallel, over multiple CPU threads or on
every GPU of the machine.

## Capabilities

::::{container} capabilities

- **Runs a call too small to repay a launch where its tensors are, and a call carrying a gradient through a simulator there too.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *work too small to repay a launch runs where its tensors are* (`test_execution.py`); *a gradient through a simulator runs where its tensors are* (`test_chunks.py`).
  :::
- **Runs the host kernels over a pool of `torch.get_num_threads()` threads, or `BLOCHSIM_NUM_THREADS`, with the same answer at any count.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_threads.hpp`. Tests: *fused vjp matches the reference*; *fused vjp is bitwise deterministic* (`test_vjp.py`).
  :::
- **Cuts a larger call into chunks of voxels and deals them to every GPU, whether the tensors are on the host or on a card.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *work that fits is shared evenly between the cards*; *a volume larger than the card is cut into chunks* (`test_execution.py`); *a device resident volume is chunked on its card and stays there* (`test_offload.py`).
  :::
- **Sizes a chunk to the smallest card's free memory less a reserve, so the cards hold a bounded share of the volume at a time.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *the reserve is left alone* (`test_execution.py`); *the device footprint stays inside the budget*; *the adjoint footprint stays inside the budget* (`test_offload.py`).
  :::
- **Cuts every physics a voxel carries with it: transmit shims, slice profiles, per-voxel rotations, semisolid, exchanging and tabulated pools, diffusion and flow.**

  :::{dropdown} Show code and tests
  Code: `blochsim/sequence/_accelerators.py`. Tests: *a chunked pass matches the whole volume*; *tabulated pools chunk like the whole volume*; *a chunk holds whole voxels when a table is read*; *a rotation per voxel is cut with its voxels* (`test_chunks.py`).
  :::
- **Runs on the host or on named devices whatever the size, under `execution="cpu"` or `execution="cuda:0"`.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *the host can be demanded whatever the size*; *a named device is used however small the work*; *a device resident call can be forced back to the host* (`test_execution.py`).
  :::
- **Returns the same signal, Jacobian and gradient on every route.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *every route gives the same answer* (`test_execution.py`); *a streamed real adjoint matches the cpu run* (`test_offload.py`).
  :::
- **Applies the same policy to dictionary matching, PERK and the model operator of a reconstruction.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.estimators.DictionaryMatcher`, {class}`~blochsim.estimators.PERK`, {class}`~blochsim.ModelOperator`. Tests: *a match is the same wherever it runs*; *a mapping is the same wherever it runs*; *a small budget really does split the volume* (`test_streaming.py`); *a policy changes where the work runs and nothing else* (`test_operator.py`).
  :::
- **Resolves the sequence structure once and rebinds the numbers on later calls (`resolve=True`, the default).**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.Simulator`. Tests: *a resolved simulator stops packing*; *a resolved simulator answers what the plain one does*; *a layout the map cannot follow is refused* (`test_binding.py`).
  :::

::::

## Division of responsibility

- **Multi-node execution.** Several cards in one process are supported; several processes or machines are not.
- **Thresholds measured on your machine.** The work below which a call runs where it is is a fixed constant, not measured where it runs.
- **Memory you keep elsewhere.** The free memory a card reports is what a chunk is sized against; memory another process claims after the call starts is not.

## Method

```{figure} /generated/figures/execution_policy.png
:width: 100%
:alt: A decision fan from the problem to running in place or chunks over every card.

The two routes `execution="auto"` chooses between, per call, from the work
in the problem and the free memory each card reports.
```

### Route selection

The policy is `"auto"` unless the `execution=` argument of a simulator, of an
estimator's `map` or of a {class}`~blochsim.ModelOperator` names another. Work
is counted in (voxel, train, event) triples and compared with a crossover;
below it the call runs where its tensors are. Past it, the voxels are cut into
chunks no wider than an even share per card, and no wider than the smallest
card's free memory less a 512 MiB reserve allows, which leaves room for other
processes on a reconstruction server.

### Chunks

Each card runs its chunks on a stream of its own. A chunk leaving the host is
staged through pinned memory, so its transfer overlaps another chunk's
arithmetic; the events, the slice and lineshape tables and the pool indices
cross to each card once. Per-voxel answers are written into place on the
device the inputs came from, and the event gradients, to which every voxel
contributes, are added up. A slice profile keeps a voxel's copies along the
slice in one chunk.

### Structure and values

The first call walks the layout and packs the event stream. Later calls
rebuild only the durations, flip angles, phases and sample times, from an
affine map of the arguments; where that map does not hold, the simulator packs
every call instead. A rebound call agrees with a fresh packing to float32
round-off, since the same product is formed in a different order. What each
declared tissue property costs is on {doc}`signal-model`; the packed buffers
and the kernels are on {doc}`../developer-guide/internals/kernels`.

## Examples

- {doc}`../generated/autoexamples/01-framework/01-first-simulation`: a simulation forced onto the host.
- {doc}`../generated/autoexamples/02-parameter-inference/01-dictionary-matching`: a dictionary match under `execution="auto"`.
- {doc}`../generated/autoexamples/03-sequence-optimization/01-echo-train-design`: a design loop over a resolved structure.
