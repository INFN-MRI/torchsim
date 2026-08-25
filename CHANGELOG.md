# Changelog

## Unreleased

### Added

- **`torchsim.ParameterMapping`**, with `Estimator` and `Subspace`. A mapping
  problem is stated the way a design problem is: an `Acquisition`, the
  properties that are unknown and the range to train each over, the ones
  measured separately, and the noise. `train(method)` simulates the training
  set and fits any `Estimator` to it; calling the mapping returns one **named**
  map per unknown, shaped like the volume, rather than columns whose order the
  caller has to remember.

  `PERK` and `DictionaryMatcher` are both `Estimator`s, so swapping one for the
  other is a word. `DictionaryMatcher` gains `fit`, which is also what stops a
  user assembling a dictionary by hand; its `dictionary` argument is now
  optional.

  `Subspace` is the compression both can use. It is fitted by SVD of the
  training signals and reports `retained` -- and that number *is* the
  approximation: one minus it equals the relative squared error of projecting
  those signals through the basis and back. At rank 16 out of 500 contrasts a
  dictionary match does about thirty times fewer operations.

- **Fused kernels for PERK**, forward and adjoint, in Triton on CUDA and in a
  new `torchsim._perk_cpu` extension on the host. The feature matrix is never
  written: a tile of features is formed and consumed into the output
  accumulator in registers. On a million voxels of 64 contrasts at a thousand
  features that is 3.3x on this card and 2.4-2.9x on this host, agreeing with
  the composed path to float32 round-off. The host kernel carries its own
  vectorized cosine, because `libm`'s does not vectorize and is the single
  largest term; `target_clones` emits one copy per instruction set so an AVX2
  machine gets AVX2 without the wheel requiring it.

  `PERK` stays differentiable with respect to its input, and keeps the fused
  kernel while doing it.

- **The estimators run under `execution()`**, which until now only reached
  simulation. Voxels are independent, so a volume too large for a card is
  streamed through it a chunk at a time with the transfers overlapping, and a
  machine with two cards uses both. Streaming a host-resident 488 MiB volume
  through this card runs 8.4x faster than the host does it. What the policy
  cannot carry is a gradient -- a streamed chunk goes through a pinned buffer
  that the next chunk overwrites -- so a call that wants one keeps the
  ordinary path.

- **`scripts/build_docs.sh`**, which builds the HTML documentation with every
  example executed. `docs/requirements.txt` now lists what that actually needs
  -- `sphinx-gallery`, and the `sigpy` and `torchio` the examples import -- and
  `docs/conf.py` drops sphinx-gallery's code-link pass on interpreters whose
  standard library has no `dbm`.

- **`torchsim.model`**, with `SignalModel`, `StateMachineModel`, `Triggers`
  and `AbstractSimulator`. A signal model is written in two pieces: a
  state-machine model saying what a voxel holds -- `properties` maps the name
  a caller uses to the tissue field it fills, so declaring `b0_hz` or a second
  pool is what asks the kernels to carry that physics -- and what each kind of
  event does to it, and a simulator saying what order the events are played
  in. Which kernel runs, how the work is cut across memory and devices, and
  how derivatives are taken all belong below both.

  `Triggers` is the assignable part: `BALANCED`, `UNBALANCED` and `SPOILED`
  say whether a readout is rewound, wound on, or spoiled. Swapping one is how
  a protocol changes what its events mean, and it is resolved when the
  simulator is constructed -- what a protocol produces is an ordinary
  description, and nothing consults a trigger during a run.

  `AbstractSimulator.from_description` takes a stream someone else assembled,
  which is the path a description arriving from a scanner takes.

- **`torchsim.simulators`**, replacing `torchsim.models`. Each shipped
  sequence names its protocol at construction (`MRFSimulator(flip=..., TR=...)`)
  and its tissue at the call (`.simulate(T1=..., T2=...)`), so parameter
  inference, sequence optimization and a reconstruction pipeline take all of
  them the same way. A protocol argument may still be overridden per call.

- **`torchsim.sequence` operators.** `Operator`, `compose`, `module` and the
  factories `excitation`, `refocusing`, `inversion`, `saturation`, `readout`
  and `delay`, with a name registry (`register_operator`, `operator`,
  `operator_names`). A preparation, a readout or a shaped pulse is written by
  composing these and reaches the fused kernels with no change to them. The
  five shipped builders are written over them.

- **`torchsim.sequence.ideal_rf_definition`**, the hard-pulse RF definition a
  description built by hand needs.

- **Sequence design: `Acquisition`, `Bounded`, `SequenceDesign` and `crlb`.**
  A design problem is stated in three pieces. An `Acquisition` is a simulator
  with the tissue it is being designed for already in place, answering the
  same two questions a simulator does -- `simulate` and `jacobian` -- with
  only the parameters under design left to give. The cost is a plain function
  taking those parameters by name and returning one number. A
  `SequenceDesign` holds the cost and the parameters, each with the limits it
  may move between, and `minimize()` runs the loop.

  Everything sequence-specific is in the cost, so the same object carries a
  quantitative design -- a Cramer-Rao bound on what the sequence estimates,
  read off the acquisition's Jacobian -- and an image-quality design, where
  the cost is a property of the point spread function the echo train
  produces and the tissue is never differentiated at all. `crlb` is the one
  statistic that ships, because every precision cost needs it and none of it
  is sequence-specific.

- **`AbstractSimulator.resolved()`**, which returns a copy holding the
  protocol's structure fixed so that each call rebinds only its values. A
  design loop plays the same sequence with different numbers every iteration
  and otherwise rebuilds the whole event stream every time, which on a small
  problem costs several times what the kernels do. An `Acquisition` asks for
  this by default. What the map cannot follow -- a layout that is not affine
  in what varies, or one event drawing on two of the values at once -- is
  simply not bound, and the ordinary path answers.

- **`Dephase()` and `Spoil()`**, and the composite readouts `bSSFPReadout`,
  `SSFPFidReadout`, `SPGRReadout` and `FSEReadout` built from them. An
  unbalanced gradient was previously spelled as a keyword on the sample; it is
  now an operator of its own, and each composite is held to the keyword it
  replaces with `torch.equal`.

- **Any array library, in and out.** Properties and sequence arguments may be
  NumPy, CuPy or torch, and the signal comes back in whichever the caller
  passed. The conversion is DLPack in both directions, so no element is
  copied: a NumPy array becomes a tensor over the same host memory and a CuPy
  array one over the same device memory. The first array a call carries
  decides; a torch caller is never handed something else.

  What the round trip cannot carry is the autograd graph. A cost
  differentiated with `backward()` must be built on torch inputs. Forward-mode
  derivatives are unaffected: `jacobian` takes them internally and hands back
  arrays in the caller's own library.

### Changed

- **`PERK.fit` walks its source twice rather than three times**, and once when
  `length_scale` is given. The covariance is accumulated as a raw second moment
  the mean is subtracted from afterwards, instead of needing the mean first.
  Under `fit_simulator` that is the difference between simulating the training
  set twice and simulating it three times.

- **`Acquisition` moved to `torchsim.model`**, where both `torchsim.optim` and
  `torchsim.estimators` can reach it without either depending on the other. It
  is still exported from `torchsim` and `torchsim.optim`.

- **`SubspaceBasis` carries a `Subspace`** rather than its own copy of the
  basis and singular values, which it still exposes under the same names.

- **A mapping's maps come back beside the volume**, not beside whatever device
  the method was fitted on.

- **`PERK.fit_simulator` takes a simulator**, with the properties being
  estimated named as `{name: value per training sample}` and any measured
  separately given as `known=`. It took a callable over a parameter matrix,
  which left the caller to keep the column order of that matrix in step with
  the sequence by hand. Handing it the simulator is what keeps the training
  signals and the ones the scanner will produce the same object.

- **Operators are named for the thing they are, not the act of making one:**
  `Excitation`, `Refocusing`, `Inversion`, `Saturation`, `Readout`, `Delay`,
  `Dephase`, `Spoil`. `bSSFPReadout` keeps the lowercase `b` the sequence is
  written with everywhere else. The slots on a `Triggers` table stay lowercase
  -- they are fields, and a field is not a constructor.

- **Models are called rather than configured.** `set_properties` /
  `set_sequence` / `__call__` are replaced by `simulate(**values)` and
  `jacobian(diff, **values)`, which take the property and sequence arguments
  together and tell them apart by what the model declares. The seven
  `*_sim` wrappers keep their signatures.

- **A model's signal keeps one dtype.** The old layer dropped a vanishing
  imaginary part, so the same quantity came back real from one call and
  complex from another depending on its value. It is now whatever the model
  computes, every time.

- **Only a differentiated property is broadcast to the voxel count.** A
  property left at its scalar default reaches the kernels as a scalar and its
  term is left out, which is what the feature gates are read from. Widening
  every declared property ahead of the simulation reported defaults as live
  maps and compiled the ungated kernel.

### Removed

- **`FSET2Precision`, `FseT2Plan` and `FseT2Optimizer`**, and the C++
  `optimize_fse_t2` behind the last of them. All three were one sequence and
  one cost: an A-optimal T2 objective for an FSE refocusing train, with its
  penalty weights, a packed event layout addressed by index, and an Adam loop
  compiled into the extension. A user wanting a different cost -- and for an
  anatomical sequence the interesting cost is not a Cramer-Rao bound at all --
  got nothing from any of it.

  Write the cost and hand it to `SequenceDesign`; `examples/04` designs for
  precision and `examples/07` for image quality. The reason the specialized
  path existed was that the generic one rebuilt the event stream every
  iteration, which `AbstractSimulator.resolved()` now does not.

- **`SequenceOptimizer`.** `SequenceDesign` replaces it. The cost is called
  with the designed parameters by keyword rather than with a dictionary, and
  each parameter carries its own limits through `Bounded` rather than through
  a parallel `bounds` argument.

- **`EpgSimulator` is now `EpgEngine`**, and its five subclasses -- `FSE`,
  `SPGR`, `SSFPFID`, `SSFPEcho`, `BSSFP` -- are gone with `make_simulator`.
  Once the crusher and the spoiler moved onto the events they are played
  around, the five overrode nothing but a name string no code read: running an
  SSFP-FID train under `BSSFP` gave a bit-identical answer. What they were
  reaching for is `Triggers`, where it decides something. `simulate_subspace`
  loses its simulator argument for the same reason.

- **`torchsim.models` and the seven `*Model` classes.** Use
  `torchsim.simulators` and the `*Simulator` classes, which take the protocol
  at construction. The seven `*_sim` functional wrappers are unchanged.

- **`torchsim.epg`.** The package held the state-machine operators the
  simulator's torch loop was written from. That loop is gone: every sequence
  now runs on the fused CPU and CUDA kernels, which reach the same operators
  through their own code. The operators survive as the tests' parity oracle
  and are no longer part of the public API.

  There is no replacement to import. Code that called them to run a sequence
  should describe the sequence and simulate it -- `fse_description(...)` and
  friends, then `FSE().simulate(...)` -- which is what the package was being
  used to assemble by hand.

- **`EpgSimulator.before_rf`, `after_rf`, `before_adc` and `after_adc`.** What
  a policy played around an event is carried by the event itself, as an
  `EventAction`, so a description says what it plays and both the builders and
  a Pulseq import can set it.

- **The `backend` argument to `simulate`.** There is one implementation, so
  there is nothing to select between. A tissue no kernel can take now raises.

- **`slice_profile=` as a tensor of flip-angle scalings**, and
  `torchsim.utils.slice_prof` with it. A slice profile is a Bloch response and
  is worked out from the RF definition; `slice_profile=` now says only where
  across the slice to sample it, through `exact_slice_profile(...)`.

- **`slice_prof` on `FSEModel`, `MRFModel`, `MPnRAGEModel`** and their
  functional wrappers, for the same reason. Give the RF definition its
  waveform instead.

- **`torchsim.base`**, with `AbstractModel`, `autocast`, and the
  `prepare_single_pool` / `prepare_two_pool_bm` / `prepare_two_pool_mt` /
  `prepare_three_pool` / `prepare_environmental_parameters` helpers. Write a
  model over `torchsim.model.SignalModel` or `EpgModel` instead; a tissue is
  built by naming the fields a model exposes.

- **`chunk_size` on the seven `*_sim` wrappers.** It selected a `torch.vmap`
  batch size on a route no model takes any more. Memory is `offload()` and
  devices are `distribute()`, both context managers around the call.
