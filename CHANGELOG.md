# Changelog

## Unreleased

### Added

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
