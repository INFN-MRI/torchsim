"""Figures for the explanation pages, drawn from the TorchSim in the tree.

Every figure is a function returning a Matplotlib figure, registered in
:data:`FIGURES` under the file stem the pages reference it by. The Sphinx
build calls :func:`render` before reading any page, so a figure on those pages
is never older than the code it illustrates.

Run it directly to write the images somewhere of your own::

    python docs/explanation_figures.py /tmp/figures
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsim.model import SPOILED, Simulator, SpinPhysics
from torchsim.sequence import Delay, EventAction, Excitation, SPGRReadout
from torchsim.simulators import FSESimulator, MRFSimulator, SPGRSimulator

#: Drawn at the width of the documentation column, so nothing is scaled on the
#: way in and type is the same size on every page.
PAGE_WIDTH = 8.6  # inches

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.constrained_layout.use": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

TRANSVERSE = "#1f6f8b"  # F states
LONGITUDINAL = "#c1553b"  # Z states
ACCENT = "#2a9d8f"
MUTED = "#8d99ae"
INK = "#22223b"

# White matter at 3 T, which every figure that needs a tissue is drawn on.
T1_MS, T2_MS = 830.0, 80.0


# ---------------------------------------------------------------------------
# The theory page.
# ---------------------------------------------------------------------------


def dephasing_helix():
    """Isochromats fanning out under a gradient, and the signal that leaves."""
    figure, axes = plt.subplots(1, 4, figsize=(PAGE_WIDTH, 2.7))
    position = np.linspace(-0.5, 0.5, 24)
    colors = plt.cm.viridis(position + 0.5)

    for axis, turns in zip(axes, (0.0, 0.25, 1.0, 3.0), strict=False):
        phase = 2 * np.pi * turns * position
        for angle, color in zip(phase, colors, strict=False):
            axis.plot([0, np.cos(angle)], [0, np.sin(angle)], color=color, lw=1.2)
        net = np.exp(1j * phase).mean()
        axis.arrow(
            0,
            0,
            net.real,
            net.imag,
            width=0.05,
            color=LONGITUDINAL,
            length_includes_head=True,
            zorder=5,
        )
        axis.set(
            xlim=(-1.2, 1.2),
            ylim=(-1.2, 1.2),
            xticks=[],
            yticks=[],
            title=f"$k = {turns:g}$" + r"$\times 2\pi$",
        )
        axis.set_aspect("equal")
        axis.set_xlabel(f"|signal| = {abs(net):.2f}", color=LONGITUDINAL)
        for spine in axis.spines.values():
            spine.set_visible(False)
    return figure


def configuration_states():
    """A dephased magnetization profile, and the few numbers that carry it."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8))
    orders = np.arange(-3, 4)
    amplitude = np.array([0.05, 0.18, 0.42, 0.30, 0.42, 0.18, 0.05])
    phase = np.array([0.9, -0.4, 0.2, 0.0, -0.2, 0.4, -0.9])
    coefficients = amplitude * np.exp(1j * phase)

    position = np.linspace(-0.5, 0.5, 400)
    profile = sum(
        c * np.exp(2j * np.pi * k * position)
        for k, c in zip(orders, coefficients, strict=False)
    )
    axes[0].plot(position, profile.real, color=TRANSVERSE, label=r"$M_x(r)$")
    axes[0].plot(position, profile.imag, color=LONGITUDINAL, lw=1.2, label=r"$M_y(r)$")
    axes[0].set(
        xlabel="position across the voxel",
        ylabel="transverse magnetization",
        title="what the spins are doing",
    )
    axes[0].legend(loc="upper right")

    axes[1].stem(orders, amplitude, basefmt=" ", linefmt=TRANSVERSE, markerfmt="o")
    axes[1].set(
        xlabel="dephasing order $k$",
        ylabel=r"$|\tilde F(k)|$",
        title="what the simulator stores",
        xticks=orders,
    )
    return figure


def state_ladder():
    """The three families of configuration state, on the integer k axis."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
    orders = range(5)
    rows = (
        (2, r"$\tilde F^{+}(k)$", TRANSVERSE),
        (1, r"$\tilde F^{-}(k)$", TRANSVERSE),
        (0, r"$\tilde Z(k)$", LONGITUDINAL),
    )
    for y, label, color in rows:
        for k in orders:
            axis.add_patch(
                plt.Rectangle(
                    (k - 0.22, y - 0.2),
                    0.44,
                    0.4,
                    facecolor=color,
                    alpha=0.18,
                    edgecolor=color,
                )
            )
            axis.text(k, y, f"{k}", ha="center", va="center", color=INK)
        axis.text(-0.7, y, label, ha="right", va="center", color=color)

    # the gradient, along the top row and back along the middle one
    axis.annotate(
        "",
        xy=(3.95, 2),
        xytext=(3.05, 2),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2),
    )
    axis.annotate(
        "",
        xy=(3.05, 1),
        xytext=(3.95, 1),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2),
    )
    axis.text(3.5, 2.55, "gradient: shift", color=ACCENT, ha="center")

    # the pulse, down one column
    axis.annotate(
        "",
        xy=(1.5, 0.2),
        xytext=(1.5, 1.8),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=2),
    )
    axis.text(1.5, 2.55, "RF: mixes within one $k$", color=MUTED, ha="center")

    # the redundancy at order zero
    axis.annotate(
        "",
        xy=(-0.35, 1.8),
        xytext=(-0.35, 1.2),
        arrowprops=dict(arrowstyle="<->", color=LONGITUDINAL, lw=2),
    )
    axis.text(
        0.0,
        -0.55,
        r"$\tilde F^{+}(0) = \tilde F^{-}(0)^{*}$",
        color=LONGITUDINAL,
        ha="center",
        va="center",
    )
    axis.set(xlim=(-1.6, 4.9), ylim=(-0.9, 2.9))
    axis.axis("off")
    return figure


def _rotation_matrix(flip_rad: float) -> np.ndarray:
    """The EPG transition matrix of a pulse about the x axis."""
    half = flip_rad / 2.0
    cos2, sin2 = np.cos(half) ** 2, np.sin(half) ** 2
    sin, cos = np.sin(flip_rad), np.cos(flip_rad)
    return np.array(
        [
            [cos2, sin2, sin],
            [sin2, cos2, -sin],
            [-0.5 * sin, 0.5 * sin, cos],
        ]
    )


def rf_operator():
    """What a pulse moves between the three families, at three flip angles."""
    figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.9))
    labels = [r"$\tilde F^{+}$", r"$\tilde F^{-}$", r"$\tilde Z$"]
    for axis, degrees in zip(axes, (30.0, 90.0, 180.0), strict=False):
        matrix = np.abs(_rotation_matrix(np.deg2rad(degrees)))
        axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.6 else INK,
                    fontsize=11,
                )
        axis.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=labels,
            yticklabels=labels,
            title=rf"$\alpha = {degrees:g}^\circ$",
        )
        axis.set_xlabel("from")
        if axis is axes[0]:
            axis.set_ylabel("to")
    return figure


def shift_operator():
    """One unbalanced gradient, order by order."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8), sharey=True)
    orders = np.arange(6)
    plus = np.array([0.35, 0.28, 0.12, 0.05, 0.0, 0.0])
    minus = np.array([0.35, 0.20, 0.08, 0.02, 0.0, 0.0])

    shifted_plus = np.concatenate(([minus[0]], plus[:-1]))
    shifted_minus = np.concatenate((minus[1:], [0.0]))

    for axis, (p, m), title in zip(
        axes,
        ((plus, minus), (shifted_plus, shifted_minus)),
        ("before", "after one shift"),
        strict=False,
    ):
        axis.bar(orders - 0.16, p, width=0.3, color=TRANSVERSE, label=r"$\tilde F^{+}$")
        axis.bar(
            orders + 0.16,
            m,
            width=0.3,
            color=TRANSVERSE,
            alpha=0.45,
            label=r"$\tilde F^{-}$",
        )
        axis.set(xlabel="dephasing order $k$", title=title, xticks=orders)
    axes[0].set_ylabel("population")
    axes[0].legend()
    axes[1].annotate(
        r"$\tilde F^{+}(0)$ comes from $\tilde F^{-}(0)^{*}$",
        xy=(0, shifted_plus[0]),
        xytext=(1.6, 0.30),
        arrowprops=dict(arrowstyle="->", color=LONGITUDINAL),
        color=LONGITUDINAL,
    )
    return figure


def relaxation():
    """Relaxation attenuates every order; recovery reaches only one."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8))
    interval_ms = np.linspace(0.0, 400.0, 200)
    axes[0].plot(
        interval_ms,
        np.exp(-interval_ms / T2_MS),
        color=TRANSVERSE,
        label=r"$E_2 = e^{-\Delta t/T_2}$ on every $\tilde F(k)$",
    )
    axes[0].plot(
        interval_ms,
        np.exp(-interval_ms / T1_MS),
        color=LONGITUDINAL,
        label=r"$E_1 = e^{-\Delta t/T_1}$ on every $\tilde Z(k)$",
    )
    axes[0].set(
        xlabel="interval [ms]",
        ylabel="surviving fraction",
        title="white matter at 3 T",
        ylim=(0, 1.05),
    )
    axes[0].legend(loc="upper right")

    orders = np.arange(5)
    before = np.array([0.30, 0.22, 0.10, 0.04, 0.01])
    factor = float(np.exp(-100.0 / T1_MS))
    after = before * factor
    after[0] += 1.0 - factor
    axes[1].bar(orders - 0.18, before, width=0.34, color=MUTED, label="before")
    axes[1].bar(orders + 0.18, after, width=0.34, color=LONGITUDINAL, label="after")
    axes[1].set(
        xlabel=r"dephasing order $k$ of $\tilde Z$",
        ylabel="population",
        title="100 ms of longitudinal relaxation",
        xticks=orders,
    )
    axes[1].legend()
    return figure


def phase_graph():
    """The pathways of a refocused train, and the echoes they add up to."""
    echoes, esp_ms, flip = 6, 5.0, 120.0
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(PAGE_WIDTH, 4.8),
        sharex=True,
        height_ratios=(2.0, 1.0),
    )

    # The graph is bookkeeping rather than physics: at every pulse a state at
    # order k continues, reflects, or is parked along z, and between pulses
    # whatever is transverse winds on by one.
    axis = axes[0]
    pulses = [(index + 0.5) * esp_ms for index in range(echoes)]
    ceiling = 3.01
    transverse = [(0.0, 0.0)]
    parked = []
    for pulse in pulses:
        continuing, still_parked = [], []
        for start, order in transverse:
            end = order + (pulse - start) / esp_ms
            axis.plot([start, pulse], [order, end], color=TRANSVERSE, lw=1.1)
            continuing.extend([(pulse, end), (pulse, -end)])
            still_parked.append((pulse, end))
        for start, order in parked:
            axis.plot(
                [start, pulse], [order, order], color=LONGITUDINAL, lw=0.9, ls=":"
            )
            continuing.append((pulse, order))
            still_parked.append((pulse, order))
        transverse = [pair for pair in continuing if abs(pair[1]) <= ceiling]
        parked = [pair for pair in still_parked if abs(pair[1]) <= ceiling]
        axis.axvline(pulse, color=MUTED, lw=1.0, ls="--")

    final = echoes * esp_ms
    for start, order in transverse:
        end = order + (final - start) / esp_ms
        if abs(end) <= ceiling:
            axis.plot([start, final], [order, end], color=TRANSVERSE, lw=1.1)
    for index in range(echoes):
        axis.plot((index + 1) * esp_ms, 0.0, "o", color=ACCENT, ms=8, zorder=5)
    axis.axhline(0.0, color=INK, lw=0.8)
    axis.set(
        ylabel="dephasing order $k$",
        ylim=(-3.3, 3.3),
        title=f"{echoes} pulses at {flip:g}$^\\circ$: every crossing of $k=0$ "
        "is an echo",
    )
    axis.text(pulses[0], 2.9, "RF", color=MUTED, ha="center")
    axis.legend(
        handles=[
            plt.Line2D([], [], color=TRANSVERSE, lw=1.4, label="transverse"),
            plt.Line2D(
                [], [], color=LONGITUDINAL, lw=1.2, ls=":", label="parked along $z$"
            ),
        ],
        loc="lower right",
        fontsize=10,
        ncols=2,
    )

    acquisition = FSESimulator(ESP=esp_ms, TR=3000.0, T1=T1_MS, T2=T2_MS)
    signal = acquisition.simulate(flip=torch.full((echoes,), flip)).abs().numpy()
    times = esp_ms * np.arange(1, echoes + 1)
    axes[1].stem(times, signal, basefmt=" ", linefmt=ACCENT, markerfmt="o")
    axes[1].set(
        xlabel="time [ms]",
        ylabel="|signal|",
        ylim=(0, 1.05 * signal.max()),
        title="what TorchSim records there",
    )
    return figure


def mono_exponential():
    """A train that is not a train of pure spin echoes."""
    echoes, esp_ms = 48, 5.0
    times = esp_ms * np.arange(1, echoes + 1)
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))

    acquisition = FSESimulator(ESP=esp_ms, TR=3000.0, T1=T1_MS, T2=T2_MS)
    trains = {}
    for degrees, color in ((180.0, MUTED), (120.0, TRANSVERSE), (60.0, LONGITUDINAL)):
        signal = acquisition.simulate(flip=torch.full((echoes,), degrees)).abs()
        trains[degrees] = signal.numpy()
        axes[0].plot(times, trains[degrees], color=color, label=f"{degrees:g}$^\\circ$")
    axes[0].plot(
        times,
        np.exp(-times / T2_MS),
        color="black",
        ls="--",
        label=r"$e^{-t/T_2}$",
    )
    axes[0].set(
        xlabel="echo time [ms]",
        ylabel="|signal|",
        title="the refocusing angle decides the decay",
    )
    axes[0].legend()

    # What a mono-exponential fit of each train returns, which is the number a
    # relaxometry experiment would report.
    fits = {}
    for degrees, signal in trains.items():
        slope, intercept = np.polyfit(times, np.log(signal), 1)
        fits[degrees] = (-1.0 / slope, np.exp(intercept))
    axes[1].bar(
        [f"{degrees:g}$^\\circ$" for degrees in trains],
        [fits[degrees][0] for degrees in trains],
        color=[MUTED, TRANSVERSE, LONGITUDINAL],
    )
    axes[1].axhline(T2_MS, color="black", ls="--", label=f"true $T_2$ = {T2_MS:g} ms")
    axes[1].set(
        xlabel="refocusing angle",
        ylabel="fitted $T_2$ [ms]",
        title=r"a mono-exponential fit of each train",
    )
    axes[1].legend()
    return figure


def truncation():
    """How many orders a sequence needs before the answer stops moving."""
    contrasts = 400
    repetition = torch.arange(contrasts, dtype=torch.float32)
    flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / contrasts) ** 2

    counts = [2, 4, 6, 8, 10, 14, 20, 30]
    reference = MRFSimulator(flip=flip, TR=10.0, TI=20.0, states=60).simulate(
        T1=T1_MS, T2=T2_MS
    )
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
    errors = []
    shown = {2: LONGITUDINAL, 6: ACCENT, 20: TRANSVERSE}
    for count in counts:
        signal = MRFSimulator(flip=flip, TR=10.0, TI=20.0, states=count).simulate(
            T1=T1_MS, T2=T2_MS
        )
        errors.append(float((signal - reference).abs().max() / reference.abs().max()))
        if count in shown:
            axes[0].plot(
                signal.abs().numpy(),
                lw=1.3,
                color=shown[count],
                label=f"{count} orders",
            )
    axes[0].set(
        xlabel="repetition",
        ylabel="|signal|",
        title="an unbalanced train, truncated",
    )
    axes[0].legend()

    axes[1].semilogy(counts, errors, "o-", color=TRANSVERSE)
    axes[1].set(
        xlabel="configuration orders carried",
        ylabel="largest error vs 60 orders",
        title="what truncation costs",
    )
    axes[1].grid(alpha=0.3)
    return figure


class _DiffusiveFSE(FSESimulator):
    """The shipped refocused train, on a tissue that diffuses."""

    model = replace(
        FSESimulator.model,
        properties={
            **FSESimulator.model.properties,
            "D": "diffusion_um2_per_ms",
        },
    )


def diffusion():
    """Higher orders spend longer wound up, and diffusion finds them there."""
    echoes, esp_ms = 32, 5.0
    times = esp_ms * np.arange(1, echoes + 1)
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))

    orders = np.arange(6)
    axes[0].bar(
        orders - 0.18,
        orders**2 + orders + 1 / 3,
        width=0.34,
        color=TRANSVERSE,
        label=r"transverse: $k^2 + k + 1/3$",
    )
    axes[0].bar(
        orders + 0.18,
        orders**2.0,
        width=0.34,
        color=LONGITUDINAL,
        label=r"longitudinal: $k^2$",
    )
    axes[0].set(
        xlabel="dephasing order $k$",
        ylabel="b-factor weight",
        xticks=orders,
        title="what an interval costs each order",
    )
    axes[0].legend()

    # A crusher pair winding 20 turns across a millimetre voxel, which is what
    # makes the effect large enough to see on this scale.
    acquisition = _DiffusiveFSE(
        ESP=esp_ms,
        TR=3000.0,
        T1=T1_MS,
        T2=T2_MS,
        crusher_dephasing_rad=20 * 2 * np.pi,
        voxel_size_m=1e-3,
        states=20,
    )
    flip = torch.full((echoes,), 120.0)
    still = acquisition.simulate(flip=flip, D=0.0).abs().numpy()
    for coefficient, color in ((1.0, ACCENT), (3.0, LONGITUDINAL)):
        moving = acquisition.simulate(flip=flip, D=coefficient).abs().numpy()
        axes[1].plot(
            times,
            moving / still,
            color=color,
            label=f"D = {coefficient:g} " + r"$\mu$m$^2$/ms",
        )
    axes[1].set(
        xlabel="echo time [ms]",
        ylabel="signal, relative to no diffusion",
        title=r"a 120$^\circ$ train between crushers",
    )
    axes[1].legend()
    return figure


class _MTFingerprinting(MRFSimulator):
    """The shipped fingerprinting train, on a tissue with a bound pool."""

    model = replace(
        MRFSimulator.model,
        properties={
            **MRFSimulator.model.properties,
            "bound_fraction": "bound_fraction",
            "bound_exchange": "bound_exchange_hz",
            "T1_bound": "t1_bound_ms",
        },
    )


def two_pool():
    """A pool no readout ever samples, moving the trajectory it does not appear in."""
    contrasts = 400
    repetition = torch.arange(contrasts, dtype=torch.float32)
    flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / contrasts) ** 2
    tissue = {"T1": T1_MS, "T2": T2_MS}
    pool = {"bound_fraction": 0.12, "bound_exchange": 30.0, "T1_bound": 1000.0}

    single = MRFSimulator(flip=flip, TR=10.0, TI=20.0, states=20)
    coupled = _MTFingerprinting(flip=flip, TR=10.0, TI=20.0, states=20)
    free = single.simulate(**tissue).abs().numpy()
    bound = coupled.simulate(**tissue, **pool).abs().numpy()
    _, sensitivity = coupled.jacobian("bound_fraction", **tissue, **pool)

    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
    axes[0].plot(free, color=TRANSVERSE, label="one pool")
    axes[0].plot(bound, color=LONGITUDINAL, label="12% bound")
    axes[0].set(
        xlabel="repetition",
        ylabel="|signal|",
        title="what the macromolecular pool does",
    )
    axes[0].legend()
    axes[1].plot(sensitivity.abs().numpy(), color=ACCENT)
    axes[1].set(
        xlabel="repetition",
        ylabel=r"$|\partial\,\mathrm{signal}\,/\,\partial f_\mathrm{b}|$",
        title="where the schedule makes it visible",
    )
    return figure


# ---------------------------------------------------------------------------
# The implementation page.
# ---------------------------------------------------------------------------


def _box(axis, x, y, width, height, title, subtitle, color, alpha=0.15):
    """A labelled block: a name, and a line or two of what it holds."""
    axis.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            lw=1.5,
            zorder=2,
        )
    )
    axis.text(
        x + width / 2,
        y + 0.68 * height,
        title,
        ha="center",
        va="center",
        color=INK,
        fontsize=11,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + 0.3 * height,
        subtitle,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=9,
        zorder=3,
    )


def _arrow(axis, start, end, color=MUTED):
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
    )


def pipeline():
    """From the sequence you write to the signal, and where each piece runs."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    stages = [
        ("operators", "Excitation, Readout,\nDephase, Delay", TRANSVERSE),
        ("description", "events, timestamps,\naction words", TRANSVERSE),
        ("packed buffers", "one array per\nevent field", ACCENT),
        ("fused kernel", "one program\nper voxel", LONGITUDINAL),
        ("signal", "and which echo\neach sample is", ACCENT),
    ]
    width, gap, height = 1.7, 0.5, 1.1
    for index, (title, subtitle, color) in enumerate(stages):
        x = index * (width + gap)
        _box(axis, x, 0.0, width, height, title, subtitle, color)
        if index:
            _arrow(axis, (x - gap + 0.05, height / 2), (x - 0.05, height / 2))

    boundary = 3 * (width + gap) - gap / 2
    axis.annotate(
        "",
        xy=(boundary, 1.45),
        xytext=(boundary, -0.25),
        arrowprops=dict(arrowstyle="-", color=MUTED, ls="--", lw=1.2),
    )
    axis.text(0.0, 1.35, "Python, once per structure", color=TRANSVERSE, fontsize=11)
    axis.text(
        boundary + 0.15,
        1.35,
        "C++ or Triton, once per run",
        color=LONGITUDINAL,
        fontsize=11,
    )
    axis.set(xlim=(-0.3, 5 * width + 4 * gap + 0.3), ylim=(-0.5, 1.75))
    axis.axis("off")
    return figure


def event_stream():
    """One repetition of a real description, as the kernels receive it."""
    echoes, esp_ms = 4, 5.0
    acquisition = FSESimulator(ESP=esp_ms, TR=60.0, T1=T1_MS, T2=T2_MS)
    description = acquisition.describe(
        **acquisition.played(flip=torch.full((echoes,), 120.0))
    )

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    kinds = {1: ("RF", TRANSVERSE, 2), 2: ("ADC", ACCENT, 1), 0: ("wait", MUTED, 0)}
    seen = set()
    for event in description.events:
        label, color, level = kinds[int(event.type)]
        time_ms = float(event.timestamp_us) * 1e-3
        axis.plot(
            [time_ms, time_ms],
            [level, level + 0.7],
            color=color,
            lw=3,
            label=label if label not in seen else None,
        )
        seen.add(label)
        action = EventAction(int(event.action))
        if action:
            axis.text(
                time_ms,
                level + 0.8,
                (action.name or "").replace("|", "\n"),
                ha="center",
                va="bottom",
                fontsize=9,
                color=LONGITUDINAL,
            )
    axis.set(
        xlabel="time [ms]",
        yticks=[0.35, 1.35, 2.35],
        yticklabels=["wait", "ADC", "RF"],
        ylim=(-0.2, 4.0),
        xlim=(-1.5, 26.0),
        title=f"a {echoes}-echo refocused train, as {len(description.events)} events",
    )
    axis.legend(loc="center right")
    return figure


def state_memory():
    """What one voxel costs while it is being simulated."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
    axis = axes[0]
    rows = (
        (2, r"$\tilde F^{+}$", TRANSVERSE),
        (1, r"$\tilde F^{-}$", TRANSVERSE),
        (0, r"$\tilde Z$", LONGITUDINAL),
    )
    for y, label, color in rows:
        for k in range(8):
            axis.add_patch(
                plt.Rectangle(
                    (k, y), 0.9, 0.8, facecolor=color, alpha=0.2, edgecolor=color
                )
            )
        axis.text(-0.3, y + 0.4, label, ha="right", va="center", color=color)
    for k in range(8):
        axis.text(
            k + 0.45, -0.35, f"{k}", ha="center", va="center", color=MUTED, fontsize=10
        )
    axis.text(4.0, -1.0, "dephasing order $k$", ha="center", color=MUTED)
    axis.set(xlim=(-1.4, 8.2), ylim=(-1.4, 3.1), title="one voxel's whole state")
    axis.axis("off")

    orders = np.arange(2, 65)
    axes[1].plot(orders, 3 * 2 * 4 * orders / 1024, color=TRANSVERSE)
    axes[1].set(
        xlabel="configuration orders carried",
        ylabel="state per voxel [kB]",
        title="what carrying them costs",
    )
    axes[1].grid(alpha=0.3)
    return figure


def fusion():
    """Why the state machine is one kernel and not a graph of operators."""
    figure, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 3.6))
    events = 12
    for axis in axes:
        axis.add_patch(
            plt.Rectangle(
                (-0.4, -1.5),
                events + 0.2,
                0.5,
                facecolor=MUTED,
                alpha=0.25,
                edgecolor=MUTED,
            )
        )
        axis.text(
            events / 2,
            -1.25,
            "memory",
            ha="center",
            va="center",
            color=INK,
            fontsize=10,
        )
        axis.set(xlim=(-0.8, events + 0.4), ylim=(-1.9, 1.3))
        axis.axis("off")

    for index in range(events):
        axes[0].add_patch(
            plt.Rectangle(
                (index, 0.2),
                0.7,
                0.7,
                facecolor=LONGITUDINAL,
                alpha=0.3,
                edgecolor=LONGITUDINAL,
            )
        )
        axes[0].annotate(
            "",
            xy=(index + 0.35, -0.95),
            xytext=(index + 0.35, 0.15),
            arrowprops=dict(arrowstyle="<->", color=LONGITUDINAL, lw=1.0),
        )
    axes[0].set_title(
        "an operator at a time: a launch per event, and the state crosses "
        "memory each way",
        fontsize=12,
    )

    axes[1].add_patch(
        plt.Rectangle(
            (0, 0.2),
            events - 0.3,
            0.7,
            facecolor=TRANSVERSE,
            alpha=0.25,
            edgecolor=TRANSVERSE,
        )
    )
    axes[1].text(
        (events - 0.3) / 2,
        0.55,
        "one program, every event, state in registers",
        ha="center",
        va="center",
        color=INK,
        fontsize=11,
    )
    for x in (0.3, events - 0.6):
        axes[1].annotate(
            "",
            xy=(x, -0.95),
            xytext=(x, 0.15),
            arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2),
        )
    axes[1].text(
        events / 2,
        -0.6,
        "read once, written once",
        ha="center",
        color=ACCENT,
        fontsize=10,
    )
    axes[1].set_title("fused: a launch per run", fontsize=12)
    return figure


def _fastest(call, repeats: int = 3) -> float:
    """The shortest wall-clock time of several runs, after a warm-up, in seconds."""
    call()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _interleaved(calls: dict, repeats: int = 4) -> dict:
    """The shortest time of each call, measured round-robin.

    A machine that is busy drifts over the length of a measurement, and taking
    each case to completion in turn would charge that drift to whichever case
    ran while it happened.
    """
    for call in calls.values():
        call()
    best = {label: float("inf") for label in calls}
    for _ in range(repeats):
        for label, call in calls.items():
            start = time.perf_counter()
            call()
            best[label] = min(best[label], time.perf_counter() - start)
    return best


def derivative_cost():
    """Which mode a derivative is taken in, and what each one costs."""
    contrasts = 500
    repetition = torch.arange(contrasts, dtype=torch.float32)
    flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / contrasts) ** 2
    voxels = 512
    tissue = {
        "T1": T1_MS * torch.ones(voxels),
        "T2": T2_MS * torch.ones(voxels),
        "M0": torch.ones(voxels),
        "B1": torch.ones(voxels),
    }
    acquisition = MRFSimulator(flip=flip, TR=10.0, TI=20.0, states=20)
    names = ("T1", "T2", "M0", "B1")

    forward = _fastest(lambda: acquisition.simulate(**tissue))
    costs = [
        _fastest(lambda count=count: acquisition.jacobian(names[:count], **tissue))
        for count in range(1, 5)
    ]

    design = flip.clone().requires_grad_(True)

    def adjoint():
        signal = acquisition.simulate(flip=design, T1=T1_MS, T2=T2_MS)
        signal.abs().square().sum().backward()
        design.grad = None

    reverse = _fastest(adjoint)
    scalar = _fastest(lambda: acquisition.simulate(flip=flip, T1=T1_MS, T2=T2_MS))

    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
    axes[0].plot(
        range(1, 5),
        [cost / forward for cost in costs],
        "o-",
        color=TRANSVERSE,
    )
    axes[0].set(
        xlabel="tissue properties differentiated",
        xticks=range(1, 5),
        ylabel="cost, in forward passes",
        ylim=(0, None),
        title=f"forward mode, {voxels} voxels at once",
    )
    axes[0].grid(alpha=0.3)

    axes[1].bar(
        ["forward\npass", f"gradient of a cost\nw.r.t. {contrasts} flip angles"],
        [1.0, reverse / scalar],
        color=[MUTED, LONGITUDINAL],
    )
    axes[1].set(
        ylabel="cost, in forward passes",
        title="reverse mode, one voxel",
    )
    return figure


def real_subspace():
    """When the states never leave a line, half the arithmetic is not needed."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2))
    steps = np.arange(24)
    amplitude = np.exp(-steps / 9.0) * np.cos(steps * 0.6)
    axes[0].plot(
        amplitude,
        np.zeros_like(amplitude),
        "o-",
        color=TRANSVERSE,
        label="one refocusing phase,\nno off-resonance",
    )
    turned = amplitude * np.exp(1j * steps * 0.35)
    axes[0].plot(
        turned.real,
        turned.imag,
        "o-",
        color=LONGITUDINAL,
        ms=4,
        label="off-resonance, transmit phase,\nflow, or a shaped pulse",
    )
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set(
        xlabel=r"Re $\tilde F^{+}(0)$",
        ylabel=r"Im $\tilde F^{+}(0)$",
        title="where a recorded state goes",
    )
    axes[0].legend(loc="lower center", fontsize=9)

    echoes, voxels = 64, 8192
    flip = torch.full((echoes,), 120.0)
    t1, t2 = T1_MS * torch.ones(voxels), T2_MS * torch.ones(voxels)

    class _OffResonant(FSESimulator):
        model = replace(
            FSESimulator.model,
            properties={**FSESimulator.model.properties, "B0": "b0_hz"},
        )

    aligned = FSESimulator(ESP=5.0, TR=3000.0, T1=t1, T2=t2, exc_phase=0.0)
    quadrature = FSESimulator(ESP=5.0, TR=3000.0, T1=t1, T2=t2, exc_phase=90.0)
    off_resonant = _OffResonant(
        ESP=5.0,
        TR=3000.0,
        T1=t1,
        T2=t2,
        exc_phase=0.0,
        B0=torch.full((voxels,), 20.0),
    )
    times = _interleaved(
        {
            "excitation in\nphase": lambda: aligned.simulate(flip=flip),
            "excitation a\nquarter turn away": lambda: quadrature.simulate(flip=flip),
            "off-resonance\ndeclared": lambda: off_resonant.simulate(flip=flip),
        }
    )
    reference = max(times.values())
    axes[1].bar(
        list(times),
        [value / reference for value in times.values()],
        color=[TRANSVERSE, LONGITUDINAL, LONGITUDINAL],
    )
    axes[1].set(
        ylabel="run time, relative",
        title=f"{echoes} echoes, {voxels} voxels",
    )
    axes[1].tick_params(axis="x", labelsize=10)
    return figure


def declared_physics():
    """A run pays for the terms its tissue declares, and for nothing else."""
    contrasts, voxels = 300, 2048
    repetition = torch.arange(contrasts, dtype=torch.float32)
    flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / contrasts) ** 2
    ones = torch.ones(voxels)
    base = {"T1": T1_MS * ones, "T2": T2_MS * ones}

    def _extended(**fields):
        return replace(
            MRFSimulator.model,
            properties={**MRFSimulator.model.properties, **fields},
        )

    class _WithB0(MRFSimulator):
        model = _extended(B0="b0_hz")

    class _WithDiffusion(MRFSimulator):
        model = _extended(D="diffusion_um2_per_ms")

    plain = MRFSimulator(flip=flip, TR=10.0, TI=20.0, states=20)
    off_resonant = _WithB0(flip=flip, TR=10.0, TI=20.0, states=20)
    diffusive = _WithDiffusion(
        flip=flip,
        TR=10.0,
        TI=20.0,
        states=20,
        crusher_dephasing_rad=2 * np.pi,
        voxel_size_m=1e-3,
    )
    times = _interleaved(
        {
            "T1, T2": lambda: plain.simulate(**base),
            "+ off-resonance": lambda: off_resonant.simulate(
                **base, B0=torch.full((voxels,), 20.0)
            ),
            "+ diffusion": lambda: diffusive.simulate(**base, D=3.0 * ones),
        }
    )
    reference = times["T1, T2"]

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.8))
    axis.bar(
        list(times),
        [value / reference for value in times.values()],
        color=[MUTED, ACCENT, LONGITUDINAL],
    )
    axis.axhline(1.0, color=INK, lw=0.8)
    axis.set(
        ylabel="run time, relative",
        title=f"the same {contrasts}-contrast schedule over {voxels} voxels, "
        "on this machine",
    )
    return figure


def execution_policy():
    """Where the per-voxel work goes, decided per call."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
    _box(axis, 0.0, 1.15, 2.0, 1.0, "the problem", "voxels x events", MUTED)
    choices = [
        (2.9, 2.5, "stay on the host", "a launch would not repay itself", MUTED),
        (2.9, 1.25, "cross whole", "it fits on the card", ACCENT),
        (2.9, 0.0, "stream in chunks", "transfer overlaps arithmetic", TRANSVERSE),
        (2.9, -1.25, "spread across cards", "one shard of voxels each", LONGITUDINAL),
    ]
    for x, y, title, subtitle, color in choices:
        _box(axis, x, y, 3.8, 1.0, title, subtitle, color)
        _arrow(axis, (2.1, 1.6), (x - 0.1, y + 0.5))
    axis.text(
        7.0,
        1.65,
        "execution() and offload()\nname the policy;\nleft alone, it "
        "is decided\nper call",
        color=INK,
        va="center",
        fontsize=11,
    )
    axis.set(xlim=(-0.3, 10.6), ylim=(-1.5, 3.8))
    axis.axis("off")
    return figure


def binding():
    """What a second call with different numbers actually has to rebuild."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    fields = [
        ("duration", True),
        ("flip", True),
        ("phase", True),
        ("time", True),
        ("kind", False),
        ("action", False),
        ("output index", False),
        ("shim", False),
        ("saturation", False),
        ("frequency", False),
    ]
    for index, (name, rebuilt) in enumerate(fields):
        color = ACCENT if rebuilt else MUTED
        x = index * 0.95
        axis.add_patch(
            plt.Rectangle(
                (x, 0.0),
                0.8,
                0.8,
                facecolor=color,
                alpha=0.35 if rebuilt else 0.15,
                edgecolor=color,
            )
        )
        axis.text(
            x + 0.4,
            -0.15,
            name,
            ha="right",
            va="top",
            rotation=35,
            fontsize=10,
            color=INK,
        )
    axis.text(
        1.9,
        1.05,
        "rebuilt per call,\nwhole-tensor arithmetic",
        ha="center",
        color=ACCENT,
        fontsize=11,
    )
    axis.text(
        6.6,
        1.05,
        "structure: resolved once, then reused",
        ha="center",
        color=MUTED,
        fontsize=11,
    )
    axis.annotate(
        "",
        xy=(3.75, 0.95),
        xytext=(3.75, -1.15),
        arrowprops=dict(arrowstyle="-", color=INK, ls="--", lw=1.0),
    )
    axis.set(xlim=(-0.6, 9.8), ylim=(-1.5, 1.6))
    axis.axis("off")
    axis.set_title("the packed buffers a description turns into")
    return figure


class _SpoiledTrain(Simulator):
    """A spoiled gradient echo played out, rather than solved."""

    model = SpinPhysics(
        properties={"T1": "t1_ms", "T2": "t2_ms", "M0": "m0"}, operators=SPOILED
    )
    states = 4

    def layout(self, *, flip, TR, repeats):
        """Return ``repeats`` excitations, each with a sample and a spoiler."""
        angle = torch.deg2rad(torch.as_tensor(flip))
        parts = []
        for _ in range(int(repeats)):
            parts.append(Excitation(angle))
            parts.append(SPGRReadout(duration_s=TR * 1e-3))
            parts.append(Delay(0.0))
        return parts


def closed_form_agreement():
    """The state machine against the steady state it has to reproduce."""
    angles = np.arange(2.0, 61.0, 2.0)
    repeats, tr_ms = 300, 10.0
    played, exact = [], []
    for degrees in angles:
        train = _SpoiledTrain(flip=float(degrees), TR=tr_ms, repeats=repeats)
        signal = train.simulate(T1=T1_MS, T2=T2_MS, M0=1.0)
        played.append(float(signal[..., -1].abs()))
        closed = SPGRSimulator(flip=float(degrees), TR=tr_ms, TE=0.0)
        exact.append(float(closed.simulate(T1=T1_MS, T2star=T2_MS, M0=1.0).abs()))

    played, exact = np.array(played), np.array(exact)
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
    axes[0].plot(angles, exact, color=MUTED, lw=3, label="Ernst steady state")
    axes[0].plot(angles, played, color=TRANSVERSE, ls="--", label="state machine")
    axes[0].axvline(
        float(angles[np.argmax(exact)]),
        color=LONGITUDINAL,
        lw=1.0,
        ls=":",
    )
    axes[0].set(
        xlabel="flip angle [deg]",
        ylabel="steady-state |signal|",
        title=f"{repeats} spoiled repetitions at TR = {tr_ms:g} ms",
    )
    axes[0].legend()

    axes[1].semilogy(angles, np.abs(played - exact) / exact.max(), color=ACCENT)
    axes[1].set(
        xlabel="flip angle [deg]",
        ylabel="relative difference",
        title="what is left between them",
    )
    axes[1].grid(alpha=0.3)
    return figure


FIGURES = {
    # the theory page
    "dephasing_helix": dephasing_helix,
    "configuration_states": configuration_states,
    "state_ladder": state_ladder,
    "rf_operator": rf_operator,
    "shift_operator": shift_operator,
    "relaxation": relaxation,
    "phase_graph": phase_graph,
    "mono_exponential": mono_exponential,
    "truncation": truncation,
    "diffusion": diffusion,
    "two_pool": two_pool,
    # the implementation page
    "pipeline": pipeline,
    "event_stream": event_stream,
    "state_memory": state_memory,
    "fusion": fusion,
    "derivative_cost": derivative_cost,
    "real_subspace": real_subspace,
    "declared_physics": declared_physics,
    "execution_policy": execution_policy,
    "binding": binding,
    "closed_form_agreement": closed_form_agreement,
}


def render(directory: str | Path, only: str | None = None) -> list[Path]:
    """Draw every figure into ``directory`` and return what was written."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(STYLE):
        for stem, draw in FIGURES.items():
            if only is not None and only != stem:
                continue
            figure = draw()
            path = target / f"{stem}.png"
            figure.savefig(path, facecolor="white")
            plt.close(figure)
            written.append(path)
    return written


if __name__ == "__main__":
    where = sys.argv[1] if len(sys.argv) > 1 else "generated/figures"
    for path in render(where, sys.argv[2] if len(sys.argv) > 2 else None):
        print(path)
