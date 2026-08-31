"""Turn the JSON records in ``results/`` into the tables the README carries.

Run as ``python benchmarks/summarize.py`` after a sweep; it prints markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

HEADER = (
    "| backend | mode | device | threads | atoms | best (s) | atoms/s | "
    "peak RSS (MiB) | over baseline (MiB) | device (MiB) |\n"
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)


def row(record: dict) -> str:
    """One measurement as a table row."""
    over = record["peak_rss_mib"] - record["baseline_rss_mib"]
    rate = record["atoms"] / record["best"]
    card = record.get("peak_device_mib") or 0.0
    return (
        f"| {record['backend']} | {record['mode']} | {record['device']} | "
        f"{record['threads']} | {record['atoms']} | {record['best']:.4f} | "
        f"{rate:,.0f} | {record['peak_rss_mib']:.0f} | {over:.1f} | "
        f"{card:.0f} |"
    )


def main() -> None:
    """Write and print every record, grouped by backend and mode."""
    records = [
        json.loads(path.read_text())
        for path in sorted(RESULTS.glob("*.json"))
        if path.name != "validation.json"
    ]
    records.sort(
        key=lambda r: (r["backend"], r["mode"], r["device"], r["threads"], r["atoms"])
    )
    table = "\n".join([HEADER, *(row(record) for record in records)])
    notes = [
        f"- {record['backend']} n={record['atoms']}: {record['note']}"
        for record in records
        if record.get("note")
    ]
    written = RESULTS / "table.md"
    written.write_text("\n".join([table, "", *notes]) + "\n")
    print(table)
    print()
    print("\n".join(notes))
    print(f"\nwritten to {written}")


if __name__ == "__main__":
    main()
