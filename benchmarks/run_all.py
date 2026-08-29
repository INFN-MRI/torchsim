"""Drive the whole sweep, one subprocess per point, and write the table.

Each measurement runs in a process of its own so that the peak resident set it
reports is its own -- an import of PyTorch costs several hundred megabytes and
would otherwise be charged to whichever backend ran first.

Run as ``python benchmarks/run_all.py``; ``--quick`` cuts the sizes down.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TORCHSIM_SIZES = (1, 10, 100, 1_000, 10_000, 100_000)
SYCOMORE_SIZES = (1, 10, 100, 1_000, 10_000)


def run(script: str, tag: str, **options: object) -> dict | None:
    """Run one measurement in its own process and read back what it recorded."""
    target = RESULTS / f"{tag}.json"
    command = [sys.executable, str(HERE / script), "--json", str(target)]
    for name, value in options.items():
        command += [f"--{name.replace('_', '-')}", str(value)]
    outcome = subprocess.run(command, check=False)
    if outcome.returncode != 0 or not target.exists():
        print(f"  ! {tag} failed", file=sys.stderr)
        return None
    return json.loads(target.read_text())


def table(records: list[dict]) -> str:
    """The measurements as a markdown table."""
    header = (
        "| backend | mode | threads | atoms | best (s) | atoms/s | "
        "peak RSS (MiB) | over baseline (MiB) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        f"| {r['backend']} | {r['mode']} | {r['threads']} | {r['atoms']} | "
        f"{r['best']:.4f} | {r['atoms_per_second']:.0f} | {r['peak_rss_mib']:.0f} | "
        f"{r['peak_rss_mib'] - r['baseline_rss_mib']:.1f} |"
        for r in records
    ]
    return "\n".join([header, *rows])


def main() -> None:
    """Run every point, write the JSON records and the markdown table."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--quick", action="store_true", help="stop at 1000 atoms")
    cli.add_argument("--length", type=int, default=500)
    cli.add_argument("--states", type=int, default=20)
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--threads", type=int, default=0, help="TorchSim CPU threads")
    cli.add_argument("--device", default="cpu")
    args = cli.parse_args()

    RESULTS.mkdir(exist_ok=True)
    limit = 1_000 if args.quick else 10**9
    shared = {"length": args.length, "states": args.states, "repeats": args.repeats}
    records = []

    for atoms in (n for n in TORCHSIM_SIZES if n <= limit):
        for mode in ("forward", "jacobian"):
            tag = f"torchsim-{mode}-{atoms}-{args.device}-t{args.threads}"
            record = run(
                "bench_torchsim.py",
                tag,
                atoms=atoms,
                mode=mode,
                device=args.device,
                threads=args.threads,
                **shared,
            )
            if record:
                records.append(record)

    for atoms in (n for n in SYCOMORE_SIZES if n <= limit):
        record = run(
            "bench_sycomore.py",
            f"sycomore-forward-{atoms}",
            atoms=atoms,
            repeats=min(args.repeats, 2) if atoms >= 10_000 else args.repeats,
            length=args.length,
            states=args.states,
        )
        if record:
            records.append(record)

    written = RESULTS / "table.md"
    written.write_text(table(records) + "\n")
    print(f"\n{table(records)}\n\nwritten to {written}")


if __name__ == "__main__":
    main()
