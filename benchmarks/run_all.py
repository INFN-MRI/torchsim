"""Drive the whole sweep, one subprocess per point, and write the table.

Each measurement runs in a process of its own so that the peak resident set it
reports is its own -- an import of PyTorch costs several hundred megabytes and
would otherwise be charged to whichever backend ran first.

A backend whose package is not installed is skipped with a line saying so, so
this runs usefully with any subset of them present. The Julia backends need a
``julia`` on the path (or ``JULIA`` in the environment pointing at one) and the
project in ``benchmarks/julia`` instantiated; ``benchmarks/setup.sh`` does all
of that.

Run as ``python benchmarks/run_all.py``; ``--quick`` cuts the sizes down,
``--backends`` picks a subset and ``--device cuda`` puts every backend that can
reach a card on one. Sycomore and epgpy have no device to be placed on and are
skipped there; a record's tag carries the device, so the two sweeps sit beside
each other in ``results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import summarize  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
JULIA_PROJECT = HERE / "julia"

# What each backend is asked for. The Julia EPG simulator reaches a hundred
# thousand atoms in seconds; KomaMRI carries `spins` isochromats per tissue and
# is stopped a decade earlier because of it.
SIZES = {
    "torchsim": (1, 10, 100, 1_000, 10_000, 100_000),
    "sycomore": (1, 10, 100, 1_000, 10_000),
    "epgpy": (1, 10, 100, 1_000, 10_000),
    "blochsimulators": (1, 10, 100, 1_000, 10_000, 100_000),
    "koma": (1, 10, 100, 1_000),
}
BACKENDS = tuple(SIZES)

# The three backends that have a card to be placed on. The sizes are the CPU
# ones: a hundred thousand tissues is where a dictionary and its Jacobian still
# fit in the memory of a modest card, and past that what is measured is
# whatever the run spills into rather than the kernel.
DEVICE_SIZES = {
    "torchsim": (1, 10, 100, 1_000, 10_000, 100_000),
    "blochsimulators": (1, 10, 100, 1_000, 10_000, 100_000),
    "koma": (1, 10, 100, 1_000),
}
ON_DEVICE = tuple(DEVICE_SIZES)


def available(backend: str, device: str) -> bool:
    """Whether this backend can run here at all, on this device."""
    if device != "cpu" and backend not in ON_DEVICE:
        return False
    if backend in ("blochsimulators", "koma"):
        return bool(julia())
    module = {"torchsim": "torchsim", "sycomore": "sycomore", "epgpy": "epgpy"}[backend]
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def julia() -> str | None:
    """The Julia to run, if there is one."""
    return os.environ.get("JULIA") or shutil.which("julia")


def run(command: list[str], tag: str) -> dict | None:
    """Run one measurement in its own process and read back what it recorded.

    Where a record for this tag is already on disk -- a previous round, or a
    previous run of the sweep -- the faster of the two is what stays. A shared
    machine's clock speed drifts over hours, and the fastest run a point has
    ever managed is the one least polluted by whatever else was running.
    """
    target = RESULTS / f"{tag}.json"
    previous = json.loads(target.read_text()) if target.exists() else None
    outcome = subprocess.run([*command, "--json", str(target)], check=False)
    if outcome.returncode != 0 or not target.exists():
        print(f"  ! {tag} failed", file=sys.stderr)
        return previous
    record = json.loads(target.read_text())
    if previous and previous["best"] < record["best"]:
        target.write_text(json.dumps(previous, indent=2) + "\n")
        return previous
    return record


def python_run(script: str, tag: str, **options: object) -> dict | None:
    """One Python backend, one point."""
    command = [sys.executable, str(HERE / script)]
    for name, value in options.items():
        command += [f"--{name.replace('_', '-')}", str(value)]
    return run(command, tag)


def julia_run(script: str, tag: str, threads: int, **options: object) -> dict | None:
    """One Julia backend, one point. Threads are a startup flag, not an argument."""
    command = [
        julia(),
        f"-t{threads}",
        f"--project={JULIA_PROJECT}",
        str(JULIA_PROJECT / script),
    ]
    for name, value in options.items():
        command += [f"--{name.replace('_', '-')}", str(value)]
    return run(command, tag)


def main() -> None:
    """Run every point, write the JSON records and the markdown table."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--quick", action="store_true", help="stop at 1000 atoms")
    cli.add_argument("--length", type=int, default=500)
    cli.add_argument(
        "--states",
        type=int,
        default=32,
        help="configuration orders; BlochSimulators requires a multiple of 32",
    )
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="passes over the whole matrix, interleaving the backends; each "
        "point keeps its fastest round",
    )
    cli.add_argument(
        "--sizes",
        default="",
        help="comma-separated tissue counts, in place of each backend's own list",
    )
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--spins", type=int, default=64, help="isochromats per tissue")
    cli.add_argument(
        "--forward-only",
        action="store_true",
        help="skip the Jacobian points, which are the slow ones",
    )
    cli.add_argument(
        "--device",
        default="cpu",
        help="where every backend that has a choice runs, e.g. cuda",
    )
    cli.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help=f"comma-separated subset of {', '.join(BACKENDS)}",
    )
    args = cli.parse_args()

    RESULTS.mkdir(exist_ok=True)
    limit = 1_000 if args.quick else 10**9
    chosen = [int(n) for n in args.sizes.split(",") if n] if args.sizes else None
    wanted = [name.strip() for name in args.backends.split(",") if name.strip()]
    device = args.device
    # A card's timings are skewed rather than scattered: a stable floor with a
    # long tail of runs that met a lower clock. The floor is what is reported,
    # and finding it takes more samples than a CPU's tight distribution does.
    repeats = args.repeats if device == "cpu" else max(args.repeats, 15)
    shared = {"length": args.length, "states": args.states, "repeats": repeats}
    # A tag names the point it measures: the backend, the mode, the size, where
    # it ran, and -- on a CPU, where it means something -- how many threads.
    suffix = f"-{device}-t{args.threads}" if device == "cpu" else f"-{device}"
    for round_ in range(args.rounds):
        if args.rounds > 1:
            print(f"\n--- round {round_ + 1} of {args.rounds}")
        for backend in wanted:
            if backend not in BACKENDS:
                raise SystemExit(f"unknown backend {backend!r}")
            if not available(backend, device):
                print(f"  - {backend} cannot run on {device} here, skipping")
                continue
            default = DEVICE_SIZES[backend] if device != "cpu" else SIZES[backend]
            sizes = [n for n in (chosen or default) if n <= limit]

            if backend == "torchsim":
                for atoms in sizes:
                    for mode in (
                        ("forward",) if args.forward_only else ("forward", "jacobian")
                    ):
                        python_run(
                            "bench_torchsim.py",
                            f"torchsim-{mode}-{atoms}{suffix}",
                            atoms=atoms,
                            mode=mode,
                            device=device,
                            threads=args.threads,
                            **shared,
                        )
                    if args.forward_only:
                        continue
                    python_run(
                        "bench_torchsim.py",
                        f"torchsim-jacobian1-{atoms}{suffix}",
                        atoms=atoms,
                        mode="jacobian",
                        diff="T1",
                        device=device,
                        threads=args.threads,
                        **shared,
                    )
                if device != "cpu":
                    continue
                # One thread against four says how much of the CPU number is
                # the pool; on a card there is no such pair to take.
                for atoms in [n for n in sizes if n >= 1_000]:
                    python_run(
                        "bench_torchsim.py",
                        f"torchsim-forward-{atoms}-{device}-t1",
                        atoms=atoms,
                        mode="forward",
                        device=device,
                        threads=1,
                        **shared,
                    )

            elif backend == "sycomore":
                for atoms in sizes:
                    python_run(
                        "bench_sycomore.py",
                        f"sycomore-forward-{atoms}",
                        atoms=atoms,
                        **shared,
                    )

            elif backend == "epgpy":
                for atoms in sizes:
                    python_run(
                        "bench_epgpy.py",
                        f"epgpy-forward-{atoms}",
                        atoms=atoms,
                        **shared,
                    )
                    python_run(
                        "bench_epgpy.py",
                        f"epgpy-jacobian-{atoms}",
                        atoms=atoms,
                        mode="jacobian",
                        **shared,
                    )

            elif backend == "blochsimulators":
                for atoms in sizes:
                    julia_run(
                        "bench_blochsimulators.jl",
                        f"blochsimulators-forward-{atoms}{suffix}",
                        args.threads,
                        atoms=atoms,
                        device=device,
                        **shared,
                    )
                for atoms in [n for n in sizes if n >= 1_000]:
                    if args.forward_only:
                        continue
                    julia_run(
                        "bench_blochsimulators.jl",
                        f"blochsimulators-jacobian-{atoms}{suffix}",
                        args.threads,
                        atoms=atoms,
                        mode="jacobian",
                        device=device,
                        **shared,
                    )
                    julia_run(
                        "bench_blochsimulators.jl",
                        f"blochsimulators-complex-{atoms}{suffix}",
                        args.threads,
                        atoms=atoms,
                        rf="complex",
                        device=device,
                        **shared,
                    )
                    if device != "cpu":
                        continue
                    julia_run(
                        "bench_blochsimulators.jl",
                        f"blochsimulators-forward-{atoms}-{device}-t1",
                        1,
                        atoms=atoms,
                        device=device,
                        **shared,
                    )

            elif backend == "koma":
                for atoms in sizes:
                    julia_run(
                        "bench_koma.jl",
                        f"koma-forward-{atoms}-s{args.spins}{suffix}",
                        args.threads,
                        atoms=atoms,
                        spins=args.spins,
                        device=device,
                        length=args.length,
                        repeats=min(repeats, 2) if device == "cpu" else repeats,
                    )

    # From the records on disk rather than from this run's, so a sweep over a
    # subset of the backends leaves the rest of the table where it was.
    summarize.main()


if __name__ == "__main__":
    main()
