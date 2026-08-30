# Changelog

## Unreleased

### Fixed

- **A pass no longer leaves the worker pool slower than it found it.** The CPU
  kernels are multiversioned, so on a machine with AVX-512 the loader picks a
  clone that uses the upper halves of the vector registers. A pool worker
  returns from that clone and parks in a wait, executing nothing that would
  clear them, so every later kernel on that thread ran SSE-encoded arithmetic
  against a dirty register state and paid the transition penalty -- for the
  life of the process, on every thread but the one that returns to Python.

  One forward-mode pass was enough to make everything after it cost two and a
  half times as much: a forward pass over ten thousand tissues went from
  0.185 s to 0.52 s and stayed there. The pool now clears the state as each job
  ends. A two-property Jacobian falls from 3.09 s to 1.05 s -- it was poisoning
  its own threads as it ran -- and a gradient measured after one from 5.6 s to
  0.93 s.

- **A detection threshold is measured again rather than falling back.** What a
  subspace test costs is probed by running one, and the probe reached for a
  function the verdict no longer has; a probe that raises is caught on purpose,
  so the documented fallback quietly stood in for a measurement on every
  machine. It now times the half of the verdict a call actually pays -- the
  reductions over the tissue -- which is 26 us here against a 5 000-work-unit
  fallback, so the real kernels are reached at far smaller problems than the
  fallback allowed.

### Changed

- **`repetitions` plays a description into the state a scanner plays it in,
  and records the last playing rather than all of them.** A simulation starts
  from equilibrium, which is a transient a scanner plays at the beginning of an
  examination and never again: every later playing starts from what the one
  before it left. `repetitions=N` now settles the magnetization through N
  playings and records the Nth, on `EpgEngine.simulate`, on a simulator's
  constructor and per call; a sequence whose physics knows how far it has to
  settle sets its own class default.

  The settling playings cost their arithmetic and none of the signal, so a run
  holds one playing however many it took.

  It matters more than a refinement. A thousand-frame fingerprinting train
  simulated from equilibrium is 20% out on its first frame, 12% at frame 100
  and 51% at its worst against the train a scanner repeats; two playings settle
  it to float32, taking a ten-thousand-tissue dictionary from 0.50 s to 0.87 s.
  A spoiled train driven this way reproduces the Ernst equation to float32,
  which is the test it is held to.

  A caller who read every playing out of one call -- the only use of the old
  behaviour was to take the last of them -- now gets the last one directly, and
  a caller who wants the approach itself asks for each length in turn.


- **A description's pulses are packed a definition at a time.** Packing walked
  the event stream and then worked each pulse over on its own: a call to
  `RfDefinition.flip_angle` per pulse, a slice per pulse to place the angle it
  returned, an addition per pulse to fold in the phase its envelope carries,
  and a stack of as many scalars to finish. The occurrences of one definition
  are now gathered during the walk, turned through in one call, and written
  into the flip and phase buffers with one scatter each. A number among tensor
  amplitudes is widened rather than refused, which is what a refocused train
  is -- a fixed excitation among a schedule.

  Resolving a 500-echo refocused train falls from 3.65 s to 0.43 s and a
  500-repetition fingerprinting schedule from 2.38 s to 1.67 s, once per
  sequence shape. The buffers are bit for bit what the per-pulse path writes
  wherever nothing had to be widened.

- **A bulk off-resonance is applied to the samples, not carried by the states.**
  A static field offset turns the transverse states through a phase that grows
  with time and the gradients wind them through one that grows with area, so
  wherever the two grow together a state's configuration order stands for both
  and the turn belongs to the sample. Packing carries the signed time each
  sample has dephased through -- reset at an excitation, negated at a
  refocusing pulse -- and the run applies `exp(-2i.pi.f.tau)` to the recorded
  signal instead of handing the field to the kernels.

  What that buys is the real subspace. A train whose pulses share an axis used
  to lose the reduced kernels to any off-resonance at all; it no longer does,
  and the derivative along the field comes from the turn rather than from a
  kernel, so the reduced adjoint is available to a caller who asks for it. A
  500-repetition fingerprinting dictionary over ten thousand tissues at 50 Hz
  falls from 2.17 s to 0.18 s, its Jacobian from 5.88 s to 0.41 s and its
  gradient from 10.5 s to 0.53 s. The same holds on a card, where each of the
  four passes has a reduced kernel of its own.

  A sequence the analytic form does not fit keeps carrying the field through
  the states, decided once when the structure is packed: a balanced sequence,
  whose coherences cross a pulse unwound, and a train whose repetition time
  varies, whose stretches wind alike in unlike times.

- **The adjoint fills its lanes from the atom axis too.** The real subspace had
  a laned kernel for forward mode and a scalar one for the reverse pass, so a
  gradient reached the fast path and then ran an eighth as wide as it could.
  There is now `simulate_real_vjp_atom_lane_range`, built the way the
  forward-mode kernel is: the tissue gathered once per block of eight atoms,
  every event value a splat, the trajectory recorded lane-major, and an
  inactive lane of a partial block seeded with a zero cotangent so every
  gradient it computes stays zero and no sum over lanes needs a mask.

  A gradient over ten thousand tissues falls from 1.61 s to 0.92 s, 4.8x its
  own forward pass where it was 8.3x. Diffusion is left to the per-atom kernel,
  its damping factors varying with the configuration order as well as the atom;
  the gradient along the damping rate is still exact where no atom diffuses.

- **The real-subspace verdict is worked out once per sequence, not once per
  call.** The verdict has two halves and both were re-read on every
  `simulate`. The half that scans the event stream -- do all the pulses turn
  about one axis -- belongs to a structure a binding resolves once, so it is
  now remembered against the buffers it was read from, by identity and version,
  through weak references that a streamed chunk is free to outlive. The half
  that asks whether the tissue carries off-resonance, transmit phase or flow is
  answered from the feature set the caller's values already declare, and
  touches a buffer only for a term given as a full map.

  A 500-repetition fingerprinting verdict falls from 130 us to 3 us and, more
  to the point on a card, from one synchronizing round trip per call to none.
  A tissue that declares off-resonance as a map of zeros keeps the fast path,
  as it did.

- **A train with no refocusing pulse can reach the real kernels.** The
  real-subspace verdict asked for refocusing pulses and an excitation sharing
  their phase, which is one arrangement of the condition rather than the
  condition itself: what confines the states to an axis is every pulse turning
  about the same one. It now reads exactly that -- all RF phases equal modulo a
  half turn, an ideal inversion excepted since it turns nothing -- which admits
  the spoiled, constant-phase trains that fingerprinting is made of and still
  refuses RF spoiling, a quarter-turn excitation, off-resonance, transmit phase
  and flow.

  A ten-thousand-atom fingerprinting dictionary goes from 2.34 s to 0.18 s on
  four CPU cores, which is the difference between the complex kernels and the
  lane-vectorized real ones. The signal is unchanged to the last bit the
  comparison in `benchmarks/validate.py` can see.

- **A forward-mode pass fills its lanes from the axis the run is wide in.** A
  block of eight lanes carried eight trains of one atom, and a dictionary has
  one train per atom: seven lanes held repeats of the first and the kernel paid
  for arithmetic it discarded, which made the lane kernel slower than the
  scalar one it was chosen over. There is now a kernel that fills a block with
  eight *atoms* of one train -- the cheaper way round as well as the wider one,
  since a tissue property is contiguous in the atom index while an event value
  is one number every lane shares -- and the dispatch picks whichever axis has
  eight entries to give, atoms first, or the scalar kernel where neither does.

  A two-property Jacobian over ten thousand atoms falls from 16.9 s to 3.1 s,
  and a forward-mode pass is 8.3x its own forward pass per property where it
  was 45x. Nothing about a slice-profiled run changes: that is where the trains
  are, and the per-train kernel still serves it.

- **Packing a description computes each pulse's flip angle once per RF
  definition**, over all its occurrences at once, rather than once per event.
  Resolving a binding packs the description several times over -- once plainly
  and twice per differentiated argument, under forward-mode -- and the per-event
  arithmetic dominated all of them: 20 000 scalar tensor operations for a
  500-repetition train, now 4 000. First-call structure resolution halves.

### Added

- **`SSFPEchoReadout`**, the other sample an unbalanced repetition can take.
  An ADC placed after the winding gradient rather than before it reads the
  order the next pulse would refocus, which is the strongly T2-weighted half of
  a reversed-FISP pair. It is `Dephase`, `Readout`, `Delay` -- the same
  operators `SSFPFidReadout` composes, in the other order -- so nothing in the
  kernels changed to make it possible, and `tests/sequence/test_ssfp_readouts.py`
  pins both readouts against an extended phase graph written out in the test.

- **An Explanation section, ahead of the examples.** Two pages, each built
  around figures that are re-rendered from the working tree on every
  documentation build. *Extended phase graphs* is the physics: dephasing as a
  helix, configuration states as its Fourier coefficients, the RF and shift
  operators, relaxation reaching only order zero, the phase graph read against
  the echo amplitudes TorchSim computes for the same train, why a low-flip
  train is not a mono-exponential, how many orders a sequence has to carry,
  the order-weighted diffusion b-factors, and the two-pool extensions. *How
  TorchSim runs it* is the realization: events carrying their own action word,
  one fused kernel per voxel, the terms a declaration switches on, the real
  subspace, forward against reverse derivatives, the affine rebinding, the
  execution policy, and the closed forms the state machine is held to.

  `docs/explanation_figures.py` draws every one of them; `conf.py` calls it
  before reading a page, so a figure cannot outlive what it shows.

- **Issue, discussion and pull-request templates, and a security policy**,
  under `.github/`. The bug form asks for the reproduction, the environment
  and whether the CPU or the CUDA kernels were involved; the security policy
  says that data reaching the kernels is in scope and that wrong physics is a
  bug report.

- **`examples/02` is a synthetic-data pipeline rather than a tour of calls.**
  A subject is segmented into tissue classes with SimpleITK's multi-level
  Otsu threshold; each class is given an M0 and a T2 from its own median and a
  T1 from a table; **one voxel is simulated per class**, so four extended
  phase graph runs stand in for sixty-five thousand; every voxel of a class is
  handed its class's evolution; and the volume is weighted by SigPy birdcage
  sensitivities and pushed through a frame-wise mri-nufft transform and back.
  The undersampled coil-combined series and the fully sampled one it came from
  are the pair, exported with the voxelised M0, T1 and T2, the segmentation
  and the schedule.

  The example measures the thing the method rests on. At one spiral arm per
  frame -- twenty-one-fold undersampled, which is how fingerprinting is
  actually run -- a single frame is 49% wrong inside the brain while the time
  courses still agree with the truth above 0.86 for nine voxels in ten. It
  also names the assumption a hard segmentation makes: every voxel of a tissue
  carries the same curve, and a fuzzy segmentation would have to mix the
  *signals*, never the averaged relaxation times.

  Nothing was added to the library for it.

- **`torchsim.recon`: the signal model as an operator.** A physics-based
  reconstruction writes its forward operator as `P F C M` -- sampling, Fourier
  encoding, coil sensitivities, signal model -- and solves for the parameter
  maps against k-space directly. Only `M` changes with the sequence, and only
  `M` is here. The encoding is composed with, never reimplemented: anything
  exposing `A` and `A_adjoint` works, and `Subspace.modes` hands the temporal
  basis to a subspace operator in the layout it reads.

  `ModelOperator` turns any `Acquisition` into `M`. It carries a **complex
  amplitude** so the model stays real-valued and physical -- its two Jacobian
  columns are the model output itself, so it costs no extra pass -- and takes
  the same `bounds=` mapping `NonlinearLeastSquares` takes, kept by solving for
  a transformed variable. That matters more under an encoding operator than in
  a fit: the model is evaluated at every voxel to predict every k-space sample,
  so one unphysical voxel corrupts the whole residual. `A_jvp` and `A_vjp` are
  one forward and one reverse pass and never build the Jacobian; `physics()`
  hands the operator to deepinv. Everything runs under `execution()`.

  `GaussNewton` inverts the chain by repeated linearization. **The damping is
  a policy and the inner solve is a callable**, which is what makes the same
  loop both methods: `Schedule` is an iteratively regularized Gauss-Newton,
  `TrustRegion` is Levenberg-Marquardt, and a closure around a proximal solver
  from elsewhere is how a regularizer enters.

  **No linear solver is written here.** `iterative()` hands the linearized
  problem to deepinv, whose `least_squares` minimizes exactly what a
  Gauss-Newton step leaves, by conjugate gradients, LSQR, BiCGStab or MINRES.
  `direct` is the one exception and is not a general solver: where the
  operator is voxel-diagonal it is a batched `torch.linalg.lstsq` over the
  augmented damped system, which *is* the Levenberg-Marquardt step -- and
  because that system has full column rank for any positive damping, there is
  no singular case to detect and none to work around. The default follows the
  problem: `direct` where the model stands alone, `iterative()` where an
  encoding operator has left no independent voxels.

  `NonlinearLeastSquares` is an `Estimator` face on that loop and holds no
  algorithm of its own -- every knob a Levenberg-Marquardt has lives on the
  `loop=` it is given, so there is one place to set it. Its answer matches the
  loop's to the bit.

  `examples/08` reconstructs a T2 map from ninefold-undersampled radial
  k-space four ways on one BrainWeb slice -- the adjoint per echo, an
  iterative least squares per echo, an iterative subspace reconstruction, and
  the nonlinear model -- each timed end to end and each given the best of a
  short regularization sweep. The two routes that constrain the echoes against
  one another err by about 12.5% of T2 against 20% for either route that
  reconstructs each contrast on its own, and iterating per echo gains nothing
  over gridding: sixteen spokes leave that problem underdetermined, so there
  is nothing to converge to. The subspace and the nonlinear model land
  together, which is *not* the ordering the review reports -- on eight echoes
  of one exponential a rank-three basis leaves a nonlinear model almost
  nothing to add, and the example says so. Per conjugate-gradient step the
  model and the encoding cost about the same, so the model is not what a
  faster reconstruction would optimize.

- **`ParameterMapping.from_coefficients`**, which reads maps from coefficients
  that are already in the mapping's basis. A subspace reconstruction never
  forms the contrast images, so what it returns must not be projected a second
  time. One mapping then supplies the basis the reconstruction is given and
  consumes what it produces.

- **`Acquisition.to()` and `Acquisition.bound()`.** A simulator carries its
  protocol and an acquisition carries its tissue, and the two have to arrive on
  a card together -- properties moved on their own would be multiplied against
  echo times still on the host. `bound()` adds or replaces tissue, which is
  what a fit does with a property measured separately.

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

  `examples/03` maps a BrainWeb slice from a four-hundred-contrast
  fingerprinting train three ways -- exhaustive matching, matching a clustered
  dictionary, and kernel regression -- and sweeps the rank first, so what the
  compression costs is read off a table rather than asserted. Rank four leaves
  5e-4 of the energy outside the basis, which is already under what the noise
  puts in.

  The phantom is BrainWeb's fuzzy tissue memberships rather than a handful of
  labelled classes, so a third of the brain voxels are mixtures and the truth
  is a continuum. Proton density is mapped alongside the relaxation times, for
  every method and at no extra cost: both answer with relaxation times, a
  fingerprint at those times is a shape the measurement is some multiple of,
  and the multiple is one inner product per voxel.

- **Group matching for `DictionaryMatcher`**, after Cauley et al., Magn Reson
  Med 74:523 (2015). Bloch simulations over a parameter grid are not spread
  evenly through signal space -- neighbouring tissues make nearly parallel
  signals -- so a dictionary is already clustered before anything is done to
  it. Passing `groups=` clusters it and gives each group one representative
  signal; a voxel is matched against those first, and only the groups it could
  still be in are opened. Inside a group the atoms are written in that group's
  own truncated basis, so the inner products are short as well as few.

  On a 20000-atom, 500-contrast dictionary and twenty thousand voxels, 64
  groups run **24x** faster than direct matching, with 6.7 groups surviving per
  voxel and mean relative errors of 0.02% in T1 and 0.11% in T2. Too many
  groups costs more than it saves -- at 256 the per-group work outweighs the
  pruning -- and `Grouping.condition`, the condition number of the
  representative signals, is what says so before any data is matched.

  Compression comes first and is global: one temporal basis for the whole
  dictionary, which the signals are in too, whether projected on the way in or
  solved for there by a subspace reconstruction. Clustering then happens
  inside that basis, so a group is entered without leaving the space the
  measurement is in. The two savings multiply -- the basis shortens every
  inner product, the grouping cuts how many are taken. On the same dictionary
  at rank 16, compression alone runs 2.4x and compression with 32 groups runs
  **22.5x**, for 0.03% in T1 and 0.15% in T2.

  `prune` is the accuracy-for-time knob and its default is the one Cauley et
  al. tuned on 280 groups of 700 atoms. On a smaller dictionary or a coarser
  grid it can rule out the group holding the match; widening it recovers the
  voxel, and a test pins both halves of that.

- **`torchsim.NonlinearLeastSquares`**, Levenberg-Marquardt stepping every
  voxel at once. Where a dictionary spans a grid whose size is the product of
  the parameter ranges, a nonlinear fit pays for a third parameter with a third
  column of the Jacobian. Voxels do not take turns: each carries its own
  damping and accepts or rejects on its own, and one that has converged drops
  out so the rest close up. On twenty thousand voxels of a ten-echo decay this
  lands where `scipy.optimize.least_squares` lands to 6e-3 ms, at least
  twenty-six times faster than calling it per voxel.

  The Jacobian is the model's own, from `SignalModel.jacobian` -- forward mode,
  one pass per property, every voxel at once -- rather than a finite
  difference.

  **Inequality constraints** are given as `bounds={"T2": (1.0, 500.0)}` and are
  kept by fitting a transformed variable, so no iterate ever leaves the
  interval and a bound cannot end up sitting exactly on the answer. The
  transform also puts every parameter on the same scale whatever its units,
  which is what a single damping term assumes. A starting value *on* a bound
  is refused rather than nudged: the transformed variable is infinite there.

  **Equality constraints** belong in the model. A constraint that fixes one
  parameter in terms of the others removes a degree of freedom, so the way to
  impose it is to not have that freedom -- in a fat-water fit, make the fat
  fraction the only unknown and write water as `1 - f`. The constraint then
  holds at every iterate rather than being restored after each one.

- **Three relaxometry contrasts**: `InversionRecoverySimulator`,
  `MultiEchoSimulator` and `DoubleAngleSimulator`, the models an inversion
  recovery, a multi-echo decay and a double-angle transmit map are read
  through. All three are closed forms, so a fit reaches them at the cost of the
  expression. The inversion recovery carries the repetition time as well, which
  the usual fully-relaxed expression drops.

- **`torchsim.LookupTable`**, for a model with a single unknown. Its atoms lie
  on a curve rather than filling a space, so the nearest one is found by
  looking along it, and interpolating between the two nearest removes the grid
  spacing from the answer -- which is otherwise what a matched estimate is
  limited by. On a 50 ms T1 grid the interpolated estimate is out by 0.9 ms on
  average where a match could not beat 25 ms.

  A signal curve is invertible only where it is monotonic, so the table keeps
  the longest run over which its curve does not turn back, and reads a
  measurement past either end as the endpoint rather than as a NaN. The
  combination that makes a curve monotonic belongs to the sequence, not to the
  table, so it is given as `combine=` -- for MP2RAGE, the unified image.

- **MP2RAGE plays as its train as well as in closed form.**
  `MP2RAGESimulator` lays out every readout and flags the one reaching the
  k-space centre of each block, so `describe` writes the sequence out and
  `from_description` plays a stream a scanner assembled. Which shot carries the
  contrast is read rather than assumed, and with it the encoding order and the
  real timing -- an MP2RAGE contrast is a subset of its readouts, and `is_echo`
  is what separates that case from an echo-resolved train. The closed form is
  what a lookup table is built from, because it costs one expression per T1,
  and the two agree to float32 round-off wherever the closed form's
  assumptions hold.

  Both are held to the two-block signal equation over T1 in [0.05, 5] s, at
  three positions of the k-space centre, and the unified image they combine
  into is held to it separately -- that ratio spans ±0.5 and a T1 map
  interpolates along it, so an error there limits the map however well the
  individual readouts agree.

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

- **`PERK.fit` runs under `execution()` too.** Fitting is a reduction -- the
  covariance of a thousand features is eight megabytes however many samples
  built it -- so it stays in one place and the training set is fed to it a
  chunk at a time; the policy chooses the place. The random Fourier features
  are drawn on the host whatever that choice is, because Torch's generators do
  not agree between devices at the same seed and where a fit ran must not
  decide which estimator comes out of it.

  On a consumer card this measures out in favour of the host, because the
  covariance accumulates in float64 and consumer fp64 is a sixty-fourth of
  fp32; `execution()` discovers that rather than assuming either way.

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

### Fixed

- **A complex measurement of a real-valued model kept only its real part.**
  `Subspace.project` and `DictionaryMatcher` both narrowed the signal to the
  basis's or the dictionary's dtype. Normalizing put the size back, so
  noiseless data still matched and the mistake did not show; what it also
  scaled was the signal-to-noise ratio, by `Re(rho)`, leaving a voxel whose
  phase approached a quarter turn matching noise. Both now promote, and
  matching correlates the two parts against a real dictionary separately
  rather than paying for a complex one.

### Changed

- **The documentation has a User Guide and a Developer Guide** in place of
  *Getting Started* and *Miscellaneous*. The user guide installs PyTorch first
  -- the build depends on hardware pip cannot see -- with tabs for the isolated
  environment and for CPU, CUDA and Apple silicon, then maps the rest of the
  documentation and says where to ask a question, report a bug and report a
  vulnerability. The developer guide carries the compiler prerequisite, the
  editable install, the style rules, the test suite's markers and cost, the
  documentation build, and how a pull request is opened.

- **One `dev` extra, in place of `dev`, `test` and `doc`.** A contributor
  installs the whole toolchain with `pip install -e ".[dev]"`.

- **One citation format across the documentation.** Numbered `[N]_` markers in
  the prose, resolved by a `References` section holding
  `Surname, A. B., …, "Title", Journal vol.issue (Year), pp. X-Y.` and a DOI
  link. The gallery's two cited examples moved their references to the end of
  the page, and the methods that were named without a source now carry one:
  PERK (Nataraj et al.), the Ernst steady state and DESPOT1 (Ernst and
  Anderson; Deoni et al.), DESPOT2, MP2RAGE (Marques et al.) and the group
  matching `DictionaryMatcher` prunes with (Cauley et al.).

- **The two authoring classes are named for what they are.**
  `AbstractSimulator` is `Simulator` -- users subclass it and write `layout`,
  and "abstract" said nothing a docstring does not. `StateMachineModel` is
  `SpinPhysics`: it is a frozen record of which properties a voxel has and what
  each kind of event does to it, not a model in the `SignalModel` sense and
  not a class anyone subclasses.

- **`DictionaryMatch` is `MatchResult`**, one letter apart from
  `DictionaryMatcher` no longer, and named as `SimulationResult` and
  `Solution` are.

- **`Subspace` carries the dictionary it was fitted to.**
  `simulate_subspace` returns a `Subspace` with `dictionary` and `simulation`
  filled in; `SubspaceBasis`, which held a `Subspace` and forwarded two
  attributes to it, is gone.

- **`Grouping` is internal.** Nothing outside `DictionaryMatcher` constructs
  one; a `groups=` count is the whole public surface.

- **An estimator states its own mapping problem, in three steps.** `PERK`,
  `DictionaryMatcher`, `LookupTable` and `NonlinearLeastSquares` are each made
  from the acquisition they invert and their own settings, `fit` states the
  sampling and draws the training set, and `map` returns named maps:

  ```python
  fitter = PERK(acquisition, n_features=1000)
  fitter.fit({"T1": (200.0, 3000.0), "T2": (10.0, 300.0)}, noise_std=0.01)
  maps = fitter.map(volume)
  ```

  The sampling belongs to the fit rather than to the estimator, so one
  estimator can be fitted over a different range without being rebuilt, and no
  tissue name ever shares a keyword namespace with a method setting. Naming
  the properties as keywords is the same thing and is usually shorter.

  `ParameterMapping` is gone, and with it `ModelEstimator` and the `bind`
  hook: an estimator that fits the model rather than a sampling of it reads
  the acquisition it was made with. `fit(signals=..., parameters=...)` still
  takes arrays from elsewhere -- `map` then returns the parameter columns
  rather than named maps.

  `DictionaryMatcher`'s `dictionary` and `parameters` are keyword-only, the
  first positional argument now being the acquisition. `LookupTable.rank` is
  `points`, which is what it counts, leaving `rank` to mean the subspace
  everywhere. `PERK`'s `seed` is `feature_seed`, distinct from the `seed` that
  `fit` draws the training set with.

- **`PERK.fit_simulator` is `PERK(..., stream=True)`.** Where a training set
  is too large to hold, the flag says so once and the ordinary `fit` reads its
  chunks from the acquisition, accumulating covariances as it goes. Without
  it, `fit` simulates the dictionary whole -- which is what lets a `rank` be
  read off it -- fits the method, and drops it. Asking a streaming fit for a
  rank is refused rather than quietly ignored, since the basis it would need
  is exactly the thing streaming does not keep.

  The basis a rank produces stays on the estimator as `subspace`, on a
  `DictionaryMatcher` and a `PERK` alike, so a subspace reconstruction can be
  handed it and the coefficients it returns read back through
  `from_coefficients` without being projected twice.

- **`fit(subspace=...)` works in a basis fitted elsewhere**, rather than
  fitting one from the training set. A reconstruction and the estimator that
  reads its coefficients then hold the same basis by construction, instead of
  both being asked for the same rank and trusted to agree. It is the
  alternative to `rank`, not a companion to it, and giving both is refused.
  A borrowed basis streams: each chunk is projected as it is simulated, so
  `stream=True` is no longer shut out of working in a subspace.

- **`EventOperators` is what `Triggers` was called.** It holds one operator
  factory per role a sequence is written in terms of, and `SpinPhysics` takes
  it as `operators=`. It is not the vocabulary the events end up carrying:
  `RfUse` is the Pulseq tag and `EventAction` is the bit field the kernels
  read, and the three do not line up one to one -- the `saturation` slot plays
  a pulse tagged `RfUse.EXCITATION`.

- **`iterative` takes a solver object, and only a solver object.**
  Anything matching the `LeastSquares` protocol -- `A`, `AT`, `y`, `z`,
  `gamma`, `max_iter`, `tol` in, the damped step out -- is called directly. A
  reconstruction that brings its own conjugate gradients, or wraps a proximal
  solver to carry a regularizer, needs nothing from a dependency TorchSim does
  not have. Given nothing, it falls back to deepinv's `least_squares`, which
  satisfies the protocol unchanged; one of deepinv's others is that same
  function with its argument bound, which is the caller's own composition
  rather than a name TorchSim interprets.

- **The API pages follow one subject each.** *Sequences* is the description
  and what assembles it -- events, operators, builders. The engine and the
  transmit calibration that run one are on *Running a simulation* with the
  placement policy, and `Subspace` and `simulate_subspace` are on
  *Model-based reconstruction*, next to the operator that reads the basis.

- **`ModelEstimator` declares the hook `ParameterMapping.train` already
  called.** A method that fits the model rather than a sampling of it is
  handed the acquisition, the names being solved for and the basis in use
  before it is fitted. That was a `getattr(method, "bind", None)` no
  user-written estimator could have known about; it is a protocol now, and
  `NonlinearLeastSquares` satisfies it.

- **`PERK.fit` walks its source twice rather than three times**, and once when
  `length_scale` is given. The covariance is accumulated as a raw second moment
  the mean is subtracted from afterwards, instead of needing the mean first.
  Under a streaming fit that is the difference between simulating the training
  set twice and simulating it three times.

- **`Acquisition` moved to `torchsim.model`**, where both `torchsim.optim` and
  `torchsim.estimators` can reach it without either depending on the other. It
  is still exported from `torchsim` and `torchsim.optim`.

- **`SubspaceBasis` carries a `Subspace`** rather than its own copy of the
  basis and singular values, which it still exposes under the same names.

- **A mapping's maps come back beside the volume**, not beside whatever device
  the method was fitted on.

- **A streaming fit reads its signals from the simulator**, rather than from a
  callable over a parameter matrix that left the caller to keep the column
  order of that matrix in step with the sequence by hand. Taking the simulator
  is what keeps the training signals and the ones the scanner will produce the
  same object.

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
