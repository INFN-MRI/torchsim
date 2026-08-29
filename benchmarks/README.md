# Benchmarks

What a dictionary costs here, and what the same dictionary costs in the other
open-source simulators that can express it. Five packages, one task, one set of
metrics, and a cross-implementation check that says they compute the same
thing before any of the timings are read.

The point is not a league table. The packages compared solve overlapping but
different problems, and the differences the numbers show are mostly
differences of *interface* and *specialization* -- where the loop over tissues
lives, and whether a package can tell that a sequence's states stay real --
rather than differences of arithmetic.

## Getting set up

```sh
bash benchmarks/setup.sh          # or: setup.sh python / setup.sh julia
source benchmarks/.env/activate
```

That builds a virtual environment with PyTorch, TorchSim, epgpy and sycomore
in it, and a Julia with BlochSimulators.jl and KomaMRI.jl, all under
`benchmarks/.env`, which is the only thing to delete afterwards. Every piece is
optional: `run_all.py` skips a backend that is not installed and says so.

Two notes on the awkward ones. **Sycomore** builds from source under `pip`, and
needs the xsimd headers and pybind11's CMake package to do it; conda-forge has
a built package and that is what the script installs. **BlochSimulators.jl**
depends on CUDA.jl, whose extension will not precompile on a machine with no
driver -- expected, and nothing here uses it.

## Running it

```sh
python benchmarks/run_all.py                     # the whole sweep, into results/
python benchmarks/run_all.py --quick             # stop at 1000 tissues
python benchmarks/run_all.py --backends torchsim,blochsimulators
python benchmarks/validate.py                    # do they agree?
python benchmarks/summarize.py                   # the table, from results/
python benchmarks/make_figures.py                # the figures, into figures/
```

On a card:

```sh
python benchmarks/bench_torchsim.py --atoms 1000000 --device cuda --mode jacobian
```

## The task

One MR fingerprinting dictionary, which is the workload every extended
phase-graph package here can state:

- an inversion, then **500 repetitions**, one excitation and one sample each;
- flip angles ramping 5 -> 60 -> 2 degrees, the original fingerprinting
  schedule stretched to that length;
- **TR = 10 ms**, the sample taken immediately after each excitation, the
  states wound on by one order per repetition;
- **32 configuration orders** kept -- BlochSimulators requires a multiple of
  32, so that is what everything carries;
- tissues log-spaced over T1 = 200-3000 ms and T2 = 10-300 ms, from one of
  them to a hundred thousand.

`_common.py` and `julia/common.jl` state that task, in the units each side
takes, and every backend script reads it from there. The two statements are
independent, which is why `validate.py` compares the signals rather than
trusting them.

KomaMRI is the exception and cannot state this task at all: it carries
isochromats through real gradient waveforms, so an ideally spoiled repetition
is reproduced by *spreading spins through the spoiler* -- a bundle spanning one
dephasing cycle, averaged. That is a strictly larger model, and the benchmark
measures what it costs rather than pretending it is the same computation.

## What is measured

**Wall time**, best of three timed runs after one warm-up run. The warm-up
matters on both sides: TorchSim resolves the *structure* of a sequence once and
rebinds values on every call afterwards, and Julia compiles on first call.
Both are reported separately as `setup_seconds` rather than hidden inside the
timings.

**Peak resident set**, from `getrusage` (`Sys.maxrss` in Julia), in a process
of its own per measurement -- otherwise an import of PyTorch is charged to
whichever backend ran first. Both the peak and the baseline immediately after
import are recorded, since for a small problem the import *is* the footprint.

**Peak device memory**, from `torch.cuda.max_memory_allocated`, when a run is
placed on a card.

**The signal itself**, as a checksum in every record, so a change that makes a
run faster and the answer different is visible.

## Do they agree?

`validate.py` runs the same train in every package, against sycomore keeping
*every* order in double precision as the reference. Where 32 orders are enough
for the tissue -- everything at brain T2 -- the implementations agree to
**float32 round-off**, and what is left at long T2 is the truncation and not a
disagreement: it halves as the orders double. `results/validation.json` and
`results/validation.txt` hold the last run.

The KomaMRI comparison is the interesting one, because it is not a check of
arithmetic but of the two models against each other. An isochromat bundle
converges to the extended phase graph as spins are added --

```
spins    T1 1000 / T2 80 ms      T1 3000 / T2 300 ms
    4         NRMSE 1.1e-01            NRMSE 2.0e-01
   16               2.2e-02                  1.4e-01
   64               1.4e-04                  1.2e-02
  128               1.3e-04                  8.5e-04
```

-- so about **twice as many spins as orders** to reach the EPG answer at brain
T2, and more where the states barely decay. That is the cost of the isochromat
picture on a spoiled sequence stated as a number rather than as an argument,
and it is why KomaMRI sits two decades below the EPG packages in
`figures/throughput.png` while computing something strictly more general.

## The results

`results/table.md` is regenerated by `summarize.py` from the JSON records and
is the authoritative version; `figures/` holds the same thing drawn.

The measurements committed here were taken on an Intel Xeon at 2.10 GHz, 4
vCPU, 15 GB, no GPU; Python 3.11, PyTorch 2.13, NumPy 2.4, sycomore 1.3.2,
epgpy at 82eebbf, Julia 1.12.7, BlochSimulators 0.9.0, KomaMRICore 0.13.0.
TorchSim and BlochSimulators compute in float32, sycomore and epgpy in float64.
**These numbers belong to that machine**, which is a shared virtual one whose
clock drifted by up to a factor of two over the hours the sweep took -- the
same TorchSim measurement came back at 2.34 s and at 4.21 s in different
windows. Two things in the harness answer that, and both matter on a laptop
with thermal throttling too: `--rounds` passes over the whole matrix
interleaving the backends, so drift lands on all of them alike, and every point
keeps the fastest run it has ever managed rather than the most recent. The
comparisons quoted below were taken inside one interleaved window.

At ten thousand tissues, four threads, 32 orders:

| | forward | against TorchSim |
| --- | ---: | ---: |
| TorchSim | 0.184 s | -- |
| BlochSimulators.jl, real states | 0.262 s | 1.4x slower |
| BlochSimulators.jl, complex states | 0.990 s | 5.4x slower |
| sycomore | 24.09 s | 131x slower |
| epgpy | 36.37 s | 198x slower |

TorchSim's row was 2.344 s until the real-subspace verdict was widened to
cover a spoiled train that shares one RF phase; see the entry in
`CHANGELOG.md`. The two Julia rows and the two Python ones are unchanged, and
the whole table above was taken in one interleaved window after that change.

## Reading them

Four things the table does not say on its own.

**BlochSimulators.jl is the comparison that matters, and the two are now within
half a factor of each other.** It is the one other package built for exactly
this workload. Both take a real path when a train's pulses share one axis --
BlochSimulators reads that off the element type of its `RF_train`, TorchSim
decides it per run from the phases the description carries -- and both fall
back to complex arithmetic when they cannot. The real path is worth 3.8x in
BlochSimulators (0.262 s against 0.990 s) and 9.7x in TorchSim. That is the
arithmetic alone on both sides: TorchSim's forward real kernel is scalar, and
`TORCHSIM_REAL_SCALAR=1` leaves a forward pass at the same 0.185 s. The lane
kernels are on the differentiated paths, where the same states are carried
beside their duals.

**Where the loop lives is most of the difference among the Python packages.**
Sycomore's cost barely moves with the number of orders it carries -- the same
hundred tissues take 191 ms keeping every one of 501 orders and 133 ms pruned
to 2-8 -- because what is being paid for is two thousand Python calls per
tissue, not the EPG arithmetic underneath them. Epgpy, vectorized across
tissues, shows the same thing from the other side: its per-tissue cost falls by
an order of magnitude between one tissue and a hundred, then stops improving.

**The Jacobian reaches the fast path, and now so does the kernel behind it.**
Forcing the verdict rather than letting it be decided settles the first half:
`_run_packed_jvp` with the real axis asked for runs in the same time as the
call that decides for itself, so the decision was never the problem. The kernel
was. Lanes were filled from the *train* axis -- eight trains of one atom -- and
a dictionary has one train per atom, so seven lanes carried repeats. Filling
them from the atom axis instead, which is the axis a dictionary is wide in,
takes a forward-mode pass over ten thousand tissues from 16.9 s to 3.1 s.

At ten thousand tissues: BlochSimulators' finite differences 0.77 s, which is
2.5x its own forward pass and is what three passes should cost; TorchSim's dual
arithmetic 1.05 s for two properties and 0.56 s for one, which is 2.8x its own
forward pass per property; epgpy's analytic derivative 177 s, 4.9x its own.
TorchSim's is linear in the number of properties, as forward mode should be,
and exact where the finite differences are not.

**The reverse pass takes the same lanes.** A gradient through the same
dictionary takes 0.93 s, 5.0x its own forward pass; `TORCHSIM_REAL_SCALAR=1`
leaves it at 1.61 s. So the lanes are worth 1.8x on the adjoint against 2.8x on
the forward-mode pass -- the adjoint records a trajectory and walks it back,
and that traffic is the same however wide the arithmetic is.

**Timings used to depend on what the process had run before**, and finding out
why was worth more than any of the numbers above. A gradient measured after a
Jacobian cost 5.6 s where the same gradient measured first cost 1.6 s, and a
plain forward pass went from 0.185 s to 0.52 s -- while `torch.mm` in the same
process was untouched, and a single-threaded run was untouched. What a
forward-mode pass left behind was a dirty vector-register state on the pool's
worker threads: the kernels are multiversioned, the loader picks an AVX-512
clone, and a worker parks between jobs rather than executing anything that
would clear it, so every later kernel on that thread paid the transition
penalty for the life of the process. One `vzeroupper` per job settles it; see
`CHANGELOG.md`. It is why the Jacobian row above is a third of what it was --
the pass was poisoning its own threads as it ran.

**TorchSim pays a structure cost the others do not.** Resolving a
500-repetition train -- walking 1 500 events, packing them, learning the affine
rebinding -- takes 1.7 s, once per sequence *shape*; a 500-echo refocused train
takes 0.43 s. Both were several times that when the packing worked an event at
a time: a call to `RfDefinition.flip_angle` per pulse, a slice per pulse to
place what it returned, and a stack of fifteen hundred scalars to finish, all
of it under two nested `torch.func.jvp` interpreters. A definition's pulses are
now turned through in one call and placed in one scatter, which takes the
refocused train from 3.65 s to 0.43 s and the fingerprinting one from 2.38 s to
1.67 s.

What is left is the walk itself -- fifteen hundred events built and read in
Python, sixteen times over, once plainly and twice per differentiated argument.
A dictionary sweep, a design loop or a fit pays it once; a single curve pays it
for nothing, and is better served by sycomore or epgpy.

## What is still missing

**A GPU.** The machine these ran on had none, and it is the axis TorchSim is
built for: the CPU kernel is the fallback, the Triton kernel is the one that
matters at volume scale. BlochSimulators and KomaMRI are both CUDA-capable too,
so the comparison carries over -- and the first call on a card compiles, which
is what `setup_seconds` will show.

**Larger and longer.** Nothing here goes past a hundred thousand tissues or
five hundred repetitions, and a volume-scale run -- a million voxels, a
thousand repetitions -- is where the execution policy (streaming, sharding
across cards) starts to be the thing being measured rather than the kernel.

**A second and third task.** One 64-echo CPMG train, where TorchSim's
real-subspace specialization does apply; one two-pool run, which sycomore
cannot express at all and epgpy can. That pair separates "faster" from "does
more", which no single-task benchmark can.
