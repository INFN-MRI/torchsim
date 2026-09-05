"""Draw the figures a paper carries, from whatever is in ``results/``.

Three of them, each answering one question:

``throughput.png``
    How many tissues a second, against how many were asked for at once. The
    crossover is the point: a batched kernel starts behind and ends ahead.
``derivatives.png``
    What a Jacobian costs, as a multiple of the same package's own forward
    pass, so the comparison survives the two packages having different
    forward speeds.
``memory.png``
    Peak resident set over the baseline the interpreter and its imports
    already cost.

Backends missing from ``results/`` are simply absent from the figures; there
is no placeholder and nothing is interpolated.

Run as ``python benchmarks/make_figures.py [--format pdf]``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Categorical slots in a fixed order: a backend keeps its colour whether or
# not the others are present.
COLORS = {
    "torchsim": "#2a78d6",
    "sycomore": "#eb6834",
    "epgpy": "#1baf7a",
    "BlochSimulators.jl": "#eda100",
    "KomaMRI.jl": "#e87ba4",
}
ORDER = list(COLORS)

INK = "#15181b"
INK_2 = "#4d555a"
RULE = "#dcdfdb"


def load() -> list[dict]:
    """Every measurement recorded so far."""
    return [
        json.loads(path.read_text())
        for path in sorted(RESULTS.glob("*.json"))
        if path.name != "validation.json"
    ]


def style(axes: plt.Axes) -> None:
    """The house style: recessive frame, recessive grid, ink text."""
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(RULE)
    axes.tick_params(colors=INK_2, labelsize=9, length=4)
    axes.grid(axis="y", color=RULE, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.xaxis.label.set_color(INK_2)
    axes.yaxis.label.set_color(INK_2)
    axes.title.set_color(INK)


def curves(
    records: list[dict], mode: str, value: Callable[[dict], float]
) -> dict[str, list[tuple[int, float]]]:
    """``{backend: [(atoms, value)]}`` for one mode, best point per size."""
    best: dict[tuple[str, int], float] = {}
    for record in records:
        if record["mode"] != mode:
            continue
        # A backend measured twice at one size keeps its fastest run.
        key = (label_of(record), record["atoms"])
        candidate = value(record)
        if key not in best or record["best"] < best[key][0]:
            best[key] = (record["best"], candidate)
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (backend, atoms), (_, candidate) in best.items():
        series[backend].append((atoms, candidate))
    for points in series.values():
        points.sort()
    return series


def label_of(record: dict) -> str:
    """The backend a record belongs to, and the device it ran on."""
    name = record["backend"]
    return f"{name}, GPU" if record.get("device", "cpu") != "cpu" else name


def base_of(label: str) -> str:
    """The backend a label names, whichever device it carries."""
    return label.removesuffix(", GPU")


def ordered(series: dict[str, list]) -> list[str]:
    """Backends in the fixed colour order, unknown ones last, CPU before GPU."""
    return sorted(
        series,
        key=lambda name: (
            ORDER.index(base_of(name)) if base_of(name) in ORDER else 99,
            name != base_of(name),
        ),
    )


def label_lines(
    axes: plt.Axes, ends: list[tuple[str, float, float]], gap: float
) -> None:
    """Write a name beside the end of each line, pushed apart where they collide.

    ``gap`` is the smallest vertical separation two labels may have, in decades
    where the axis is logarithmic and in its own units where it is not. Seven
    curves over six decades put several ends within a few pixels of each other,
    and a label that lands on top of another names neither.
    """
    logarithmic = axes.get_yscale() == "log"
    into = math.log10 if logarithmic else (lambda value: value)
    out_of = (lambda value: 10.0**value) if logarithmic else (lambda value: value)

    ends = sorted(ends, key=lambda end: end[2])
    placed: list[float] = []
    for _, _, y in ends:
        wanted = into(y)
        placed.append(wanted if not placed else max(wanted, placed[-1] + gap))
    for (name, x, _), at in zip(ends, placed, strict=True):
        axes.annotate(
            name,
            xy=(x, out_of(at)),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color=INK_2,
            va="center",
            annotation_clip=False,
        )


def spread(axes: plt.Axes, sizes: list[int]) -> None:
    """Room on the right for the direct labels, and no decade of empty axis."""
    axes.set_xlim(min(sizes) / 2.5, max(sizes) * 6)


def draw(axes: plt.Axes, name: str, points: list[tuple[int, float]]) -> None:
    """One backend's line: its own colour, dashed where it ran on a card."""
    axes.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=COLORS.get(base_of(name), INK_2),
        linewidth=2,
        linestyle="--" if name != base_of(name) else "-",
        marker="s" if name != base_of(name) else "o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=name,
    )


def throughput(records: list[dict], suffix: str) -> None:
    """Tissues a second against dictionary size, one line per backend."""
    threaded = [
        r for r in records if not (r["backend"] == "torchsim" and r["threads"] == 1)
    ]
    series = curves(threaded, "forward", lambda r: r["atoms"] / r["best"])
    figure, axes = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    style(axes)
    ends, sizes = [], []
    for name in ordered(series):
        points = series[name]
        sizes += [point[0] for point in points]
        draw(axes, name, points)
        ends.append((name, points[-1][0], points[-1][1]))
    axes.set_xscale("log")
    # Four orders of magnitude between the fastest and the slowest package, so
    # a linear axis would show one line and four flat ones.
    axes.set_yscale("log")
    axes.set_xlabel("tissues simulated in one call")
    axes.set_ylabel("tissues per second")
    axes.set_title("Dictionary throughput", fontsize=12, loc="left", pad=12)
    spread(axes, sizes)
    label_lines(axes, ends, gap=0.32)
    figure.tight_layout()
    figure.savefig(FIGURES / f"throughput.{suffix}", bbox_inches="tight")
    plt.close(figure)


def derivatives(records: list[dict], suffix: str) -> None:
    """A Jacobian as a multiple of the same backend's forward pass.

    One bar per backend, device and set of properties, taken at the largest
    dictionary that backend was asked for. The ratio is what the figure is
    about, and it barely moves with the size once a run is big enough to be
    bound by its arithmetic.
    """
    forward = {
        (label_of(r), r["atoms"]): r["best"] for r in records if r["mode"] == "forward"
    }
    widest: dict[tuple[str, str], tuple[int, float]] = {}
    for record in records:
        if not record["mode"].startswith("jacobian") or record["atoms"] < 1000:
            continue
        base = forward.get((label_of(record), record["atoms"]))
        if base is None:
            continue
        properties = "T1" if record["mode"].endswith("(T1)") else "T1, T2"
        key = (label_of(record), properties)
        if key not in widest or record["atoms"] > widest[key][0]:
            widest[key] = (record["atoms"], record["best"] / base)
    if not widest:
        return
    bars = sorted(
        (
            (name, properties, size, ratio)
            for (name, properties), (size, ratio) in widest.items()
        ),
        key=lambda bar: (
            ORDER.index(base_of(bar[0])) if base_of(bar[0]) in ORDER else 99,
            bar[0] != base_of(bar[0]),
            bar[1],
        ),
    )

    figure, axes = plt.subplots(figsize=(6.4, 0.52 * len(bars) + 1.4), dpi=200)
    style(axes)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=RULE, linewidth=0.8)
    labels = [
        f"{name}\n{properties}, {size:,} tissues" for name, properties, size, _ in bars
    ]
    values = [ratio for *_, ratio in bars]
    colors = [COLORS.get(base_of(name), INK_2) for name, *_ in bars]
    axes.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55)
    for index, value in enumerate(values[::-1]):
        axes.text(
            value + 0.06 * max(values),
            index,
            f"{value:.1f}x",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    axes.set_xlabel("cost, as a multiple of the same package's forward pass")
    axes.set_title("What a Jacobian costs", fontsize=12, loc="left", pad=12)
    axes.margins(x=0.12)
    figure.tight_layout()
    figure.savefig(FIGURES / f"derivatives.{suffix}", bbox_inches="tight")
    plt.close(figure)


def memory(records: list[dict], suffix: str) -> None:
    """Peak resident set over the baseline, against dictionary size.

    Host memory, so the runs placed on a card are left out: what they hold in
    host memory is a driver context and says nothing about the dictionary. What
    they hold on the card is ``peak_device_mib``, which the table carries.
    """
    threaded = [
        r
        for r in records
        if r.get("device", "cpu") == "cpu"
        and not (r["backend"] == "torchsim" and r["threads"] == 1)
    ]
    series = curves(
        threaded, "forward", lambda r: r["peak_rss_mib"] - r["baseline_rss_mib"]
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    style(axes)
    ends, sizes = [], []
    for name in ordered(series):
        points = [point for point in series[name] if point[1] > 0]
        if not points:
            continue
        sizes += [point[0] for point in points]
        draw(axes, name, points)
        ends.append((name, points[-1][0], points[-1][1]))
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("tissues simulated in one call")
    axes.set_ylabel("peak resident set over baseline (MiB)")
    axes.set_title("What the run costs in memory", fontsize=12, loc="left", pad=12)
    spread(axes, sizes)
    label_lines(axes, ends, gap=0.22)
    figure.tight_layout()
    figure.savefig(FIGURES / f"memory.{suffix}", bbox_inches="tight")
    plt.close(figure)


def convergence(suffix: str) -> None:
    """How many isochromats it takes to reproduce the extended phase graph.

    Reads what ``validate.py --julia "KomaMRI.jl, N spins"=...`` recorded: the
    error of an isochromat bundle against the untruncated EPG reference, per
    tissue, against the number of spins in the bundle.
    """
    written = RESULTS / "validation.json"
    if not written.exists():
        return
    record = json.loads(written.read_text())
    spins: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for label, rows in record.get("against", {}).items():
        found = re.search(r"(\d+)\s+spins", label)
        if not found:
            continue
        for row in rows:
            tissue = f"T1 {row['T1_ms']:.0f} ms, T2 {row['T2_ms']:.0f} ms"
            spins[tissue].append((int(found.group(1)), row["nrmse"]))
    if not spins:
        return
    for points in spins.values():
        points.sort()

    figure, axes = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    style(axes)
    palette = [COLORS["KomaMRI.jl"], COLORS["BlochSimulators.jl"], INK_2]
    sizes: list[int] = []
    for index, (tissue, points) in enumerate(sorted(spins.items())):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        sizes += x
        axes.plot(
            x,
            y,
            color=palette[index % len(palette)],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        axes.annotate(
            tissue,
            xy=(x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color=INK_2,
            va="center",
            annotation_clip=False,
        )
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    axes.set_xlabel("isochromats per tissue")
    axes.set_ylabel("error against the extended phase graph (NRMSE)")
    axes.set_title(
        "What the isochromat picture costs to match EPG",
        fontsize=12,
        loc="left",
        pad=12,
    )
    axes.set_xlim(min(sizes) / 1.4, max(sizes) * 3.4)
    figure.tight_layout()
    figure.savefig(FIGURES / f"convergence.{suffix}", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Draw every figure the results support."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    args = cli.parse_args()

    FIGURES.mkdir(exist_ok=True)
    records = load()
    if not records:
        raise SystemExit("no measurements in results/ -- run run_all.py first")
    throughput(records, args.format)
    derivatives(records, args.format)
    memory(records, args.format)
    convergence(args.format)
    print(f"wrote {FIGURES}/*.{args.format}")


if __name__ == "__main__":
    main()
