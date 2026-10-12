# Execution

```{admonition} TL;DR
:class: tldr

- A call runs where its tensors are; `execution=` chooses the host, a GPU, several GPUs or streaming instead.
- A volume larger than a card's free memory is streamed through it in chunks sized to a memory budget.
- A simulator resolves its sequence structure on the first call and rebinds only the numbers on later calls.
```

You choose where a simulation, a dictionary match or a model-based
reconstruction runs. Each simulator accepts scalar and tensor-valued inputs;
tensor-valued inputs are simulated in parallel, over multiple CPU threads or on
a GPU.

## Capabilities

::::{container} capabilities

- **Runs a call where its tensors are when no `execution=` is given.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Test: *without a block a call is left where it is* (`test_execution.py`).
  :::
- **Runs the host kernels over a pool of `torch.get_num_threads()` threads, or `BLOCHSIM_NUM_THREADS`, with the same answer at any count.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_threads.hpp`. Tests: *fused vjp matches the reference*; *fused vjp is bitwise deterministic* (`test_vjp.py`).
  :::
- **Decides per call under `execution="auto"`: the host for small work, one card when it fits, otherwise the fewest cards that hold it, otherwise streaming.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *work too small to repay a launch stays on the host*; *work that fits goes across in one piece*; *a volume larger than the card is streamed* (`test_execution.py`).
  :::
- **Runs on the host or on named devices whatever the size, under `execution="cpu"` or `execution="cuda:0"`.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *the host can be demanded whatever the size*; *a named device is used however small the work*; *a device resident call can be forced back to the host* (`test_execution.py`).
  :::
- **Sizes streamed chunks to each card's free memory less a reserve.**

  :::{dropdown} Show code and tests
  Code: `blochsim/_execution.py`. Tests: *the budget can be set outright*; *the reserve is left alone* (`test_execution.py`); *the device footprint stays inside the budget*; *a streamed volume matches the cpu run* (`test_offload.py`).
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

- **Placement without a request.** With no `execution=`, nothing moves a tensor between host and card.
- **Multi-node execution.** Several cards in one process are supported; several processes or machines are not.
- **Thresholds measured on your machine.** The work below which the host is preferred is a fixed constant, not measured where it runs.

## Method

```{figure} /generated/figures/execution_policy.png
:width: 100%
:alt: A decision fan from the problem to the host, one card, several cards, or streaming.

The four routes `execution="auto"` chooses between, per call, from the work
in the problem and the free memory each card reports.
```

### Route selection

The policy is the `execution=` argument of a simulator, at construction or per
call, of an estimator's `map`, and of a {class}`~blochsim.ModelOperator`. Work is counted in (voxel,
train, event) triples and compared with a crossover; past it, the device
footprint is compared with each card's free memory less `reserve_bytes`
(512 MiB by default), which leaves room for other processes on a
reconstruction server.

### Streaming

Chunks of voxels are staged through pinned host memory, and the result returns
to the device the inputs came from. A smaller budget gives smaller chunks,
which fill the card less and pay more launch and transfer latency.

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
