"""Turn the JSON records in ``results/`` into the tables the README carries.

Run as ``python benchmarks/summarize.py`` after a sweep; it prints markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

HEADER = (
    "| backend | mode | threads | atoms | best (s) | atoms/s | "
    "peak RSS (MiB) | over baseline (MiB) |\n"
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def row(record: dict) -> str:
    """One measurement as a table row."""
    over = record["peak_rss_mib"] - record["baseline_rss_mib"]
    rate = record["atoms"] / record["best"]
    return (
        f"| {record['backend']} | {record['mode']} | {record['threads']} | "
        f"{record['atoms']} | {record['best']:.4f} | {rate:,.0f} | "
        f"{record['peak_rss_mib']:.0f} | {over:.1f} |"
    )


def main() -> None:
    """Print every record, grouped by backend and mode."""
    records = [
        json.loads(path.read_text())
        for path in sorted(RESULTS.glob("*.json"))
        if path.name != "validation.json"
    ]
    records.sort(key=lambda r: (r["backend"], r["mode"], r["threads"], r["atoms"]))
    print(HEADER)
    for record in records:
        print(row(record))
    print()
    for record in records:
        if record.get("note"):
            print(f"- {record['backend']} n={record['atoms']}: {record['note']}")


if __name__ == "__main__":
    main()
