"""What TorchSim's own runtime is made of, on one CPU core count.

Four experiments, each isolating one thing the dictionary benchmark cannot:

**What the structure costs.** Resolving a sequence walks its events, packs
them and learns the affine rebinding, once per sequence shape; a call
afterwards rebinds values onto what it found. Both are timed at one tissue, so
what is measured is the structure rather than the dictionary.

**Fixed cost against state cost.** Run the same schedule at several
configuration-order counts. The state arithmetic scales with the orders; the
per-event work that does not -- the two exponentials, the loads, the branches --
is the intercept. That intercept is the ceiling on what hoisting relaxation
factors out of the event loop could ever buy.

**The real subspace against the complex path.** Run one refocused train twice,
with the excitation in phase with the refocusing pulses and a quarter turn from
them. The event stream, the order count and the arithmetic content are
identical; only the subspace verdict differs, so the ratio is what that
specialization is worth. (The two are different sequences and produce different
signals -- what is being compared is the cost of the same amount of work.)

**Whether the fast path is reached at all.** The same pass with the verdict
forced to the real kernels, to the complex ones, and left to decide for
itself. Where deciding costs what forcing costs, the decision is right and any
distance left is the kernel's; where it costs what the complex kernel costs,
the verdict is not reaching it. This is the one to run first on a card, since
it separates a plumbing problem from an arithmetic one without needing to know
what either should cost.

Run as ``python benchmarks/anatomy.py [--device cuda]``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import torch

from torchsim.simulators import FSESimulator, MRFSimulator

WARMUP_SECONDS = 2.0


def best(
    run: Callable[[], Any], repeats: int, synchronize: Callable[[], None]
) -> float:
    """The fastest of ``repeats`` timed runs, after a warm-up of a few seconds.

    The warm-up is a budget rather than a count because a card idles at a low
    clock and takes the better part of a second of continuous work to reach its
    boost one; timed over three runs after a single warm-up, a kernel of a few
    milliseconds reports the ramp and not the kernel.
    """
    deadline = time.perf_counter() + WARMUP_SECONDS
    while True:
        run()
        synchronize()
        if time.perf_counter() >= deadline:
            break
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


def _reached(
    device: torch.device,
    args: argparse.Namespace,
    T1: torch.Tensor,
    T2: torch.Tensor,
    echo_train: torch.Tensor,
    synchronize: Callable[[], None],
) -> None:
    """Auto against forced, for the plain pass and the forward-mode one.

    Reaches past the public interface on purpose: what is being measured is
    which kernel a verdict selects, which is not something a caller can ask
    for.
    """
    from torchsim.sequence import _builders
    from torchsim.sequence._accelerators import (
        _pack_events,
        _run_packed,
        _run_packed_jvp,
        real_subspace_axis,
    )
    from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

    description = _builders.fse_description(
        echo_train.to(torch.float32) * (torch.pi / 180.0),
        5e-3,
        phases_rad=0.0,
        excitation_phase_rad=0.0,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=device,
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    tissue, _, _ = _prepare_tissue(TissueProperties(t1_ms=T1, t2_ms=T2), device)
    if real_subspace_axis(events, tissue) != 1:
        print("\nThe train did not earn the real subspace; nothing to compare.\n")
        return

    tangents = [torch.zeros_like(value) for value in tissue]
    tangents[1] = torch.ones_like(tissue[1])
    event_zeros = tuple(torch.zeros_like(events[index]) for index in (0, 2, 3))
    shape = (32, packed.output_count, 0)

    print("\nIs the fast path reached? Auto against a forced verdict")
    print(f"{'pass':>12s}  {'auto':>9s}  {'forced real':>11s}  {'forced complex':>14s}")
    for name, run in (
        ("forward", lambda axis: lambda: _run_packed(tissue, events, *shape, axis)),
        (
            "forward-mode",
            lambda axis: (
                lambda: _run_packed_jvp(
                    tissue, events, tuple(tangents), event_zeros, *shape, axis
                )
            ),
        ),
    ):
        times = [best(run(axis), args.repeats, synchronize) for axis in (None, 1, -1)]
        print(
            f"{name:>12s}  {times[0] * 1e3:8.1f} ms  {times[1] * 1e3:8.1f} ms  "
            f"{times[2] * 1e3:11.1f} ms"
        )


def _structure(
    device: torch.device,
    args: argparse.Namespace,
    synchronize: Callable[[], None],
) -> None:
    """What resolving a sequence's structure costs, and what it saves per call.

    Resolving walks the event stream, packs it, and learns the affine map from
    a protocol's values onto those buffers. It happens once per sequence
    *shape*; every call afterwards rebinds values onto what it found. One
    tissue is enough to measure it, since what is being timed is the structure
    and not the arithmetic over the dictionary.
    """
    T1 = torch.full((1,), 1000.0, device=device)
    T2 = torch.full((1,), 80.0, device=device)
    flip = torch.linspace(5.0, 60.0, args.echoes, device=device)
    train = torch.full((args.echoes,), 120.0, device=device)

    def place(sequence: Any) -> Any:
        return sequence.to(device) if device.type != "cpu" else sequence

    sequences = (
        (
            f"{args.echoes}-repetition fingerprinting",
            lambda: place(MRFSimulator(flip=flip, TR=10.0, states=32)),
            {},
        ),
        (
            f"{args.echoes}-echo refocused",
            lambda: place(FSESimulator(ESP=5.0, TR=3000.0, states=32)),
            {"flip": train},
        ),
    )

    print("What building a simulator costs, and what holding one saves")
    print(
        f"{'sequence':>32s}  {'build (s)':>11s}  {'held (ms)':>10s}  "
        f"{'rebuilt (ms)':>15s}"
    )
    for label, build, extra in sequences:
        start = time.perf_counter()
        held_simulator = build()
        held_simulator.simulate(T1=T1, T2=T2, **extra)
        synchronize()
        cost = time.perf_counter() - start
        bound = best(
            lambda s=held_simulator, e=extra: s.simulate(T1=T1, T2=T2, **e),
            args.repeats,
            synchronize,
        )
        loose = best(
            lambda b=build, e=extra: b().simulate(T1=T1, T2=T2, **e),
            args.repeats,
            synchronize,
        )
        print(f"{label:>32s}  {cost:11.2f}  {bound * 1e3:10.2f}  {loose * 1e3:15.2f}")
    print(
        "\n  A dictionary sweep, a fit or a design loop builds once and pays the\n"
        "  held column on every call after it. A single curve pays the build\n"
        "  for nothing.\n"
    )


def main() -> None:
    """Print every anatomy."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--atoms", type=int, default=2000)
    cli.add_argument("--echoes", type=int, default=500)
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--device", default="cpu")
    args = cli.parse_args()

    device = torch.device(args.device)
    synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    T1 = torch.linspace(200.0, 3000.0, args.atoms, device=device)
    T2 = torch.linspace(10.0, 300.0, args.atoms, device=device)

    print(f"{args.atoms} tissues, {args.echoes} repetitions, {device}\n")

    _structure(device, args, synchronize)

    print("Fixed per-event cost against per-state cost")
    print(f"{'orders':>7s}  {'best (ms)':>10s}  {'ms per order':>13s}")
    flip = torch.linspace(5.0, 60.0, args.echoes, device=device)
    points = []
    for orders in (8, 16, 32, 64):
        sequence = MRFSimulator(flip=flip, TR=10.0, states=orders)
        if device.type != "cpu":
            sequence = sequence.to(device)
        seconds = best(
            lambda s=sequence: s.simulate(T1=T1, T2=T2), args.repeats, synchronize
        )
        points.append((orders, seconds))
        print(f"{orders:7d}  {seconds * 1e3:10.1f}  {seconds * 1e3 / orders:13.2f}")

    # Two points are enough for the split, and the widest pair is the least
    # sensitive to the noise on either.
    (low_orders, low), (high_orders, high) = points[0], points[-1]
    slope = (high - low) / (high_orders - low_orders)
    intercept = low - slope * low_orders
    if slope <= 0:
        # The state arithmetic grows with the orders, so a run whose timing does
        # not is bound by something else -- a card given too few tissues to fill
        # it, most often -- and there is no intercept to read out of it.
        print(
            f"\n  The timing did not grow with the orders, so this run is not\n"
            f"  bound by the state arithmetic and the split cannot be read from\n"
            f"  it. Ask for more tissues than {args.atoms}.\n"
        )
    else:
        share = intercept / (intercept + slope * 32)
        print(
            f"\n  fixed {intercept * 1e3:.1f} ms, {slope * 1e3:.2f} ms per order:"
            f" the fixed part is {share:.0%} of a 32-order run.\n"
            "  That is the whole of what hoisting the relaxation factors can save.\n"
        )

    print("The real subspace against the complex path, same event stream")
    echo_train = torch.full((args.echoes,), 120.0, device=device)
    timings = {}
    for label, exc_phase in (("in phase, real", 0.0), ("quarter turn, complex", 90.0)):
        sequence = FSESimulator(ESP=5.0, TR=3000.0, states=32, exc_phase=exc_phase)
        if device.type != "cpu":
            sequence = sequence.to(device)
        timings[label] = best(
            lambda s=sequence: s.simulate(flip=echo_train, T1=T1, T2=T2),
            args.repeats,
            synchronize,
        )
        print(f"  {label:24s} {timings[label] * 1e3:8.1f} ms")
    _reached(device, args, T1, T2, echo_train, synchronize)

    ratio = timings["quarter turn, complex"] / timings["in phase, real"]
    print(f"\n  the real path is {ratio:.1f}x the complex one on this machine.\n")

    print("What the Jacobian costs on each path, same event stream")
    for label, exc_phase in (("in phase, real", 0.0), ("quarter turn, complex", 90.0)):
        sequence = FSESimulator(ESP=5.0, TR=3000.0, states=32, exc_phase=exc_phase)
        if device.type != "cpu":
            sequence = sequence.to(device)
        forward = best(
            lambda s=sequence: s.simulate(flip=echo_train, T1=T1, T2=T2),
            args.repeats,
            synchronize,
        )
        jacobian = best(
            lambda s=sequence: s.jacobian(("T1", "T2"), flip=echo_train, T1=T1, T2=T2),
            args.repeats,
            synchronize,
        )

        def gradient(s: Any = sequence) -> None:
            t1 = T1.clone().requires_grad_(True)
            t2 = T2.clone().requires_grad_(True)
            s.simulate(flip=echo_train, T1=t1, T2=t2).abs().sum().backward()

        adjoint = best(gradient, args.repeats, synchronize)
        print(
            f"  {label:24s} forward {forward * 1e3:8.1f} ms   "
            f"jacobian {jacobian * 1e3:8.1f} ms   {jacobian / forward:5.1f}x   "
            f"gradient {adjoint * 1e3:8.1f} ms   {adjoint / forward:5.1f}x"
        )
    print(
        "\n  A pass that carries a derivative beside every quantity should cost a\n"
        "  few times the plain one. The multiple is the check that the dual and\n"
        "  adjoint kernels take the same path the plain ones take, rather than\n"
        "  the arithmetic being cheap where the plumbing is not. Run the whole\n"
        "  file again under TORCHSIM_REAL_SCALAR=1 to separate what the real\n"
        "  subspace is worth from what the lane kernels on top of it are worth:\n"
        "  a multiple that does not move is a pass with no laned kernel to take."
    )


if __name__ == "__main__":
    main()
