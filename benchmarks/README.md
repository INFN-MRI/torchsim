# Benchmarks

What a dictionary costs here, and what the same dictionary costs in the other
open-source simulators that can express it. The point is not a league table:
the packages compared below solve overlapping but different problems, and the
differences the numbers show are mostly differences of *interface* -- where
the loop over tissues lives -- rather than of arithmetic.

## The task

One MR fingerprinting dictionary, which is the workload every extended
phase-graph package can state:

- an inversion, then **500 repetitions**, one excitation and one sample each;
- flip angles ramping 5 -> 60 -> 2 degrees, the original fingerprinting
  schedule stretched to that length;
- **TR = 10 ms**, the sample taken immediately after each excitation, the
  states wound on by one order per repetition;
- **20 configuration orders** kept;
- tissues log-spaced over T1 = 200-3000 ms and T2 = 10-300 ms, from one
  of them to a hundred thousand.

`_common.py` defines the schedule and the tissue grid, and every backend
script reads them from there, so the three implementations are asked for the
same numbers rather than for the same-looking numbers.

## What is measured

**Wall time**, best of three timed runs after one warm-up run. The warm-up
matters: TorchSim resolves the *structure* of a sequence once and rebinds
values on every call afterwards, and that first pass is reported separately as
`setup_seconds` rather than hidden inside the timings.

**Peak resident set**, from `getrusage`, in a process of its own per
measurement -- otherwise an import of PyTorch is charged to whichever backend
ran first. Both the peak and the baseline immediately after import are
recorded, since for a small problem the import *is* the footprint.

**Peak device memory**, from `torch.cuda.max_memory_allocated`, when a run is
placed on a card.

**The signal itself**, as a checksum in every record, so a change that makes a
run faster and the answer different is visible.

## Running it

```sh
pip install torchsim                                    # this package
conda install -c conda-forge sycomore                    # optional comparison
pip install git+https://github.com/py-baudin/epgpy       # optional comparison

python benchmarks/run_all.py            # the whole sweep, to results/
python benchmarks/run_all.py --quick    # stop at 1000 atoms
python benchmarks/validate.py           # do they agree?
python benchmarks/summarize.py          # the tables below, from results/

python benchmarks/bench_torchsim.py --atoms 100000 --device cuda --mode jacobian
```

## Do they agree?

`validate.py` runs the same train in all three, against sycomore keeping
*every* order in double precision as the reference:

```
 states   max |difference|    relative
      5          3.061e-02    1.75e-01
     10          1.761e-02    1.01e-01
     20          8.658e-03    4.95e-02
     40          4.390e-03    2.51e-02
     80          2.009e-03    1.15e-02

Per tissue, at 20 orders:
  T1 =    200 ms, T2 =     10 ms:  max 1.014e-06, NRMSE 1.247e-06
  T1 =    600 ms, T2 =     40 ms:  max 4.499e-07, NRMSE 1.309e-06
  T1 =   1000 ms, T2 =     80 ms:  max 2.696e-05, NRMSE 8.547e-05
  T1 =   1500 ms, T2 =    120 ms:  max 1.657e-04, NRMSE 6.685e-04
  T1 =   3000 ms, T2 =    300 ms:  max 1.685e-03, NRMSE 9.100e-03
  T1 =   4000 ms, T2 =   2000 ms:  max 8.658e-03, NRMSE 5.669e-02
```

Read down the second block rather than across the first. Where 20 orders are
enough for the tissue -- everything at brain T2 -- three independent
implementations agree to **around 1e-6 relative**, which is float32 round-off
and is as close as they can come. What is left at long T2 is the truncation,
not a disagreement: it halves as the orders double, and free water at
T2 = 2000 ms simply needs more of them. TorchSim against epgpy, both truncated
to 20 orders, tells the same story from the other side: 1e-6 where the
truncation does not bite, 2e-3 at free water, where the two libraries prune
the highest order slightly differently.

## Measured here

Intel Xeon at 2.10 GHz, 4 vCPU, 15 GB, no GPU; Python 3.11.15, PyTorch 2.13,
NumPy 2.4.6, sycomore 1.3.2, epgpy at 82eebbf. TorchSim computes in float32,
sycomore and epgpy in float64. **These numbers belong to this machine**; the
shape of them is what transfers.

### Forward, 500 repetitions

| backend | threads | atoms | best (s) | atoms/s | peak RSS (MiB) | over baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| torchsim | 4 | 1 | 0.0015 | 656 | 726 | 107 |
| torchsim | 4 | 100 | 0.0178 | 5 614 | 726 | 107 |
| torchsim | 4 | 1 000 | 0.1436 | 6 962 | 738 | 120 |
| torchsim | 4 | 10 000 | 1.3538 | 7 387 | 840 | 221 |
| torchsim | 4 | 100 000 | 13.216 | 7 566 | 1 874 | 1 255 |
| torchsim | 1 | 1 000 | 0.5255 | 1 903 | 743 | 124 |
| torchsim | 1 | 10 000 | 5.2154 | 1 918 | 841 | 221 |
| sycomore | 1 | 1 | 0.0014 | 719 | 33 | 0.5 |
| sycomore | 1 | 100 | 0.1464 | 683 | 35 | 3 |
| sycomore | 1 | 10 000 | 14.400 | 694 | 266 | 234 |
| epgpy | 1 | 1 | 0.0240 | 42 | 36 | 5 |
| epgpy | 1 | 100 | 0.1508 | 663 | 43 | 12 |
| epgpy | 1 | 10 000 | 19.559 | 511 | 762 | 730 |

At a dictionary of ten thousand atoms TorchSim is **10.6x** sycomore and
**14.4x** epgpy on four cores, and **2.8x** and **3.8x** on one -- so about
half the gain is threads and half is the fused kernel. Below a hundred atoms
none of that has appeared yet, and for a single tissue sycomore is the fastest
of the three.

Two things the table does not say on its own:

**Sycomore's cost is nearly independent of how many orders it carries.** The
same hundred atoms take 191 ms keeping every one of 501 orders, 142 ms pruned
to 11-158, and 133 ms pruned to 2-8. What is being paid for is two thousand
Python calls per atom, not the EPG arithmetic underneath them -- which is why
this is a comparison of interfaces. Epgpy, vectorized across tissues, shows
the same thing from the other direction: 24 ms for one atom, 151 ms for a
hundred.

**TorchSim pays a structure cost sycomore and epgpy do not.** Resolving a
500-repetition train -- walking 1 500 events, packing them, learning the
affine rebinding -- takes **5-7 s** on this machine, once per sequence
*shape*. A dictionary sweep, a design loop or a fit pays it once; a single
curve pays it for nothing, and is better served by either of the others.

### Derivatives

| backend | with respect to | atoms | best (s) | over its own forward |
| --- | --- | ---: | ---: | ---: |
| torchsim | T1 | 10 000 | 7.142 | 5.3x |
| torchsim | T1, T2 | 10 000 | 14.379 | 10.6x |
| epgpy | T1, T2 | 10 000 | 103.21 | 5.3x |

TorchSim's Jacobian is linear in the number of properties, as forward mode
should be: one property costs 5.3 forward passes, two cost 10.6. Each
directional derivative is therefore about **four to five times** a plain pass
rather than one -- dual arithmetic through every operator is not free, and a
three-point finite difference would be cheaper per parameter if it were as
accurate. What it buys is exactness and no step size to choose, and against
epgpy's analytic derivative -- the closest comparison there is -- it is
**7.2x** faster on four cores.

Peak memory over baseline at ten thousand atoms: TorchSim 458 MiB against
epgpy's 1 685 MiB for the same Jacobian.

### Memory, and where PyTorch sits in it

Importing PyTorch costs about **620 MiB** before any simulation runs, against
31 MiB for NumPy and 32 MiB for sycomore. For one curve that dominates
everything else in this document. It stops mattering somewhere around a
thousand atoms, and by a hundred thousand TorchSim's peak is 1.9 GB against
the 4.2 GB its own Jacobian takes -- both of which are the *output*, not the
state: a voxel at 20 orders is under half a kilobyte, so the states of a
million-voxel volume are a few hundred megabytes.

## What is missing, and how to add it

**A GPU.** The machine these ran on has none, and it is the axis TorchSim is
built for: the CPU kernel is the fallback, the Triton kernel is the one that
matters at volume scale. Every script takes `--device cuda` and records
`torch.cuda.max_memory_allocated`; the first call on a card compiles the
Triton specialization it needs, which is tens of seconds and is what
`setup_seconds` will show.

**KomaMRI.jl and BlochSimulators.jl.** Neither could be installed here -- the
Julia binaries are not reachable from this container -- so no Julia numbers
appear above, and none should be guessed. Adding them is worth doing properly:

- **BlochSimulators.jl** is the direct comparison. It has an EPG simulator,
  it is written to run a dictionary across CPU threads, processes or a card,
  and MR-STAT takes its derivatives by finite difference -- so the fair
  question is dictionary throughput at matched order counts, and the fair
  derivative comparison is TorchSim's forward mode against three of its
  forward passes per parameter.
- **KomaMRI.jl** answers a different question -- it carries isochromats
  through real gradient waveforms to k-space -- and a like-for-like run
  against an EPG package is only possible for the fingerprinting dictionary
  its `BlochDict` method targets. Comparing anything else compares the
  problems, not the packages.
- Both are compiled on first call. Time a Julia run with `BenchmarkTools.jl`
  after a warm-up, exactly as `timed()` here does, or the number is a compile.

**Larger and longer.** Nothing here goes past a hundred thousand atoms or five
hundred repetitions, and a volume-scale run -- a million voxels, a thousand
repetitions -- is where the execution policy (streaming, sharding across
cards) starts to be the thing being measured rather than the kernel.
