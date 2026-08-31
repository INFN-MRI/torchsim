# Benchmarks

What a dictionary costs here, and what the same dictionary costs in the other
open-source simulators that can express it. Five packages, one task, one set of
metrics, on a CPU and on a card, and a cross-implementation check that says
they compute the same thing before any of the timings are read.

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
in it, and a Julia with BlochSimulators.jl, KomaMRI.jl and CUDA.jl, all under
`benchmarks/.env`, which is the only thing to delete afterwards. Every piece is
optional: `run_all.py` skips a backend that is not installed and says so.

Three notes on the awkward ones.

**Which interpreter the environment is built on matters.** `PYTHON` says which,
and the default is whatever `python3` finds first. An interpreter that ships a
C++ runtime of its own -- a conda one does -- loads that runtime ahead of the
system's, and TorchSim's kernels are compiled against the system's. The
extension then fails to load with a missing `GLIBCXX` version, every fused
kernel is reported absent, and nothing here can run:

```sh
PYTHON=/usr/bin/python3.11 bash benchmarks/setup.sh python
```

**Sycomore** builds from source under `pip`, and needs the xsimd headers and
pybind11's CMake package to do it; conda-forge has a built package and that is
what the script installs. It leaves no distribution metadata a virtual
environment can read, so the version in a record reads `unknown`.

**The Julia side downloads a couple of gigabytes**, most of it the CUDA toolkit
that puts BlochSimulators and KomaMRI on a card. On a machine with no driver
the CUDA extensions fail to precompile, which is expected and leaves the CPU
benchmarks alone.

## Running it

```sh
python benchmarks/run_all.py                     # the CPU sweep, into results/
python benchmarks/run_all.py --device cuda       # the same sweep, on a card
python benchmarks/run_all.py --quick             # stop at 1000 tissues
python benchmarks/run_all.py --backends torchsim,blochsimulators
python benchmarks/validate.py                    # do they agree?
python benchmarks/summarize.py                   # the table, from results/
python benchmarks/make_figures.py                # the figures, into figures/
python benchmarks/anatomy.py [--device cuda]     # what TorchSim's runtime is made of
```

`--device` reaches every backend that has a card to be placed on. Sycomore and
epgpy have none and are skipped there; a record's tag carries the device, so
the two sweeps sit beside each other in `results/` and in one table.

The two Julia backends also answer a cross-implementation check, which takes
the tissues to run and writes the signal out:

```sh
julia -t4 --project=benchmarks/julia benchmarks/julia/bench_blochsimulators.jl \
  --tissues 200:10,600:40,1000:80,1500:120,3000:300,4000:2000 --dump /tmp/bs.csv
python benchmarks/validate.py --julia "BlochSimulators.jl"=/tmp/bs.csv
```

and, for the isochromat convergence, the same with `bench_koma.jl --spins N`
over the two tissues the table below carries.

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

**Wall time**, the fastest of the timed runs. What comes before them is a
warm-up given a budget of seconds rather than a count, because what has to be
warm differs by orders of magnitude: TorchSim resolves the *structure* of a
sequence once and rebinds values on every call afterwards, Julia compiles on
first call, and a card idles at a low clock and takes the better part of a
second of continuous work to reach its boost one. That last is the one that
bites -- a kernel of a few milliseconds timed over three runs after a single
warm-up reports the clock ramp and not the kernel, and on this card
BlochSimulators reads 32 ms for eight iterations and 4.4 ms thereafter. The
first call is reported separately as `setup_seconds` rather than hidden inside
the timings.

A card's timings are skewed rather than scattered -- a stable floor with a long
tail of runs that met a lower clock -- so a run placed on one takes fifteen
samples where a CPU takes three, and `--rounds` passes over the whole matrix
again later, when the machine is in a different state.

**Peak resident set**, from `getrusage` (`Sys.maxrss` in Julia), in a process
of its own per measurement -- otherwise an import of PyTorch is charged to
whichever backend ran first. Both the peak and the baseline immediately after
import are recorded, since for a small problem the import *is* the footprint.

**Peak device memory**, what the run's allocator took from the driver:
`torch.cuda.max_memory_reserved` on one side and CUDA.jl's pool on the other.
Neither counts the driver context underneath, which is a few hundred MiB for
either.

**The signal itself**, as a checksum in every record, so a change that makes a
run faster and the answer different is visible. It is also what says a run
placed on a card computed what the same run computes on a CPU.

## Do they agree?

`validate.py` runs the same train in every package, against sycomore keeping
*every* order in double precision as the reference. Where 32 orders are enough
for the tissue -- everything at brain T2 -- the implementations agree to
**float32 round-off**, and what is left at long T2 is the truncation and not a
disagreement: it halves as the orders double. TorchSim's two kernels are held
against each other as well, since they are separate implementations of one
recursion: a dictionary run on the card matches the same dictionary on the CPU
to **6e-06 relative**. `results/validation.json` and `results/validation.txt`
hold the last run.

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
and it is why KomaMRI sits more than two decades below the EPG packages in
`figures/throughput.png` while computing something strictly more general.

## The results

`results/table.md` is regenerated by `summarize.py` from the JSON records and
is the authoritative version; `figures/` holds the same thing drawn.

The measurements committed here were taken on an Intel Core i7-13700H -- six
performance cores, eight efficiency cores, 23 GB -- with an NVIDIA GeForce RTX
4060 Laptop GPU, 8 GB; Python 3.11.5, PyTorch 2.13 with CUDA 13.0, Triton
3.7.1, NumPy 2.4.6, sycomore 2.0.0, epgpy at 82eebbf, Julia 1.12.7,
BlochSimulators 0.9.0, KomaMRICore 0.13.0, CUDA.jl 5. TorchSim and
BlochSimulators compute in float32, sycomore and epgpy in float64. The CPU
sweep pins **four threads** for every backend that takes them, which is what
makes the packages comparable to each other rather than to the machine.

At ten thousand tissues, four CPU threads, 32 orders:

| | forward | against TorchSim |
| --- | ---: | ---: |
| TorchSim | 0.187 s | -- |
| BlochSimulators.jl, real states | 0.212 s | 1.1x slower |
| BlochSimulators.jl, complex states | 0.520 s | 2.8x slower |
| sycomore | 28.40 s | 152x slower |
| epgpy | 34.74 s | 186x slower |

At a hundred thousand tissues, where the card is full enough to be worth
asking:

| | CPU, four threads | RTX 4060 Laptop | what the card buys |
| --- | ---: | ---: | ---: |
| BlochSimulators.jl | 2.276 s | 0.037 s | 61x |
| TorchSim | 1.903 s | 0.196 s | 9.7x |
| TorchSim, Jacobian in T1 and T2 | 9.657 s | 1.085 s | 8.9x |
| BlochSimulators.jl, finite differences | 7.409 s | 0.119 s | 62x |

KomaMRI reaches a thousand tissues, and 64 isochromats each: 7.33 s on four
threads, 2.63 s on the card.

## Reading them

**On a CPU the two dictionary packages are level; on a card they are not.**
BlochSimulators.jl is the comparison that matters -- it is the one other
package built for exactly this workload -- and at ten thousand tissues on four
threads the two are within a tenth of each other. Both take a real path when a
train's pulses share one axis: BlochSimulators reads that off the element type
of its `RF_train`, TorchSim decides it per run from the phases the description
carries, and both fall back to complex arithmetic when they cannot. The real
path is worth 2.5x in BlochSimulators (0.212 s against 0.520 s) and 4.9x in
TorchSim, which `anatomy.py` measures on one event stream with only the verdict
changed.

On the card that parity goes. BlochSimulators simulates a hundred thousand
tissues in 37 ms against TorchSim's 196 ms, and takes its finite-difference
Jacobian in 119 ms against TorchSim's 1.09 s. The two lay the same recursion out
differently: BlochSimulators gives each voxel a whole warp and each thread
`states / 32` of the configuration orders, held in registers as an immutable
`SMatrix`, while TorchSim's Triton kernel takes a tile of voxels by all of
their orders per program. Whatever the gap is, it is in the kernel and not in
the dispatch around it: `anatomy.py --device cuda` puts a forced-real verdict at
106.3 ms against 107.2 ms for the automatic one, so the fast path is reached and
what is left is arithmetic.

**Below ten thousand tissues a card measures its own launch latency.** Every
GPU row under that size sits within a few milliseconds of every other, and
BlochSimulators' Jacobian at a thousand tissues comes out faster than its own
forward pass. Nothing is wrong with either; there is simply not enough work in
a thousand 500-step trains to fill an RTX 4060, and the throughput curves only
separate once there is.

**Where the loop lives is most of the difference among the Python packages.**
Sycomore's cost barely moves with the number of orders it carries -- the same
hundred tissues take 452 ms keeping every one of 353-501 orders and 278 ms
pruned to 11-158 -- because what is being paid for is two thousand Python calls
per tissue, not the EPG arithmetic underneath them. Epgpy, vectorized across
tissues, shows the same thing from the other side: its per-tissue cost falls by
an order of magnitude between one tissue and a hundred, then stops improving.

**A Jacobian costs a few times the forward pass it comes from, on every
package that has one.** At ten thousand tissues on four threads:
BlochSimulators' finite differences 0.847 s, which is 4.0x its own forward pass
and is what three passes should cost; TorchSim's dual arithmetic 1.088 s for
two properties and 0.551 s for one, which is 2.9x its own forward pass per
property; epgpy's analytic derivative 156 s, 4.5x its own. TorchSim's is linear
in the number of properties, as forward mode should be, and exact where the
finite differences are not. The reverse pass takes the same lanes: a gradient
through a 500-echo train costs 4.6x its forward pass on the real path, against
the Jacobian's 5.9x.

**TorchSim pays a structure cost the others do not.** Resolving a
500-repetition fingerprinting train -- walking 2 000 events, packing them,
learning the affine rebinding -- takes 1.9 s, once per sequence *shape*; a
500-echo refocused train takes 0.16 s. What it buys is the calls afterwards:
the fingerprinting train is 1.2 ms a call bound against 346 ms unresolved.

A dictionary sweep, a design loop or a fit pays that once; a single curve pays
it for nothing, and is better served by sycomore or epgpy.

## What is still missing

**Larger and longer.** A hundred thousand tissues and a Jacobian is 4.6 GB on
the card, which is where a modest one stops; past it a run spills into host
memory and what is measured is the spill. A volume-scale run -- a million
voxels, a thousand repetitions -- is where the execution policy (streaming,
sharding across cards) starts to be the thing being measured rather than the
kernel, and nothing here exercises it.

**One card, and a laptop one.** An RTX 4060 Laptop is power-limited and clocks
about as it pleases; the harness answers that with a warm-up budget, fifteen
samples and the fastest of them, but a desktop or datacentre card would
separate the kernels differently and neither ratio above should be read as
architectural.

**A second and third task.** One 64-echo CPMG train, where TorchSim's
real-subspace specialization does apply; one two-pool run, which sycomore
cannot express at all and epgpy can. That pair separates "faster" from "does
more", which no single-task benchmark can.
