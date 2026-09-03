"""The fingerprinting dictionary, on TorchSim's fused state machine.

Run as ``python benchmarks/bench_torchsim.py --atoms 10000``. The structure of
the sequence is resolved once before anything is timed, which is what a
dictionary sweep or a design loop does too: the schedule is the same at every
call and only the tissue changes.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    Measurement,
    flip_train,
    machine,
    parser,
    peak_rss_mib,
    report,
    timed,
    tissue_grid,
)


def main() -> None:
    """Time one dictionary, and say what it cost."""
    cli = parser("torchsim")
    cli.add_argument(
        "--diff",
        default="T1,T2",
        help="properties the Jacobian is taken with respect to",
    )
    args = cli.parse_args()
    diff = tuple(name for name in args.diff.split(",") if name)
    if args.threads:
        os.environ["TORCHSIM_NUM_THREADS"] = str(args.threads)

    import time

    import torch

    from torchsim.simulators import MRFSimulator

    if args.threads:
        torch.set_num_threads(args.threads)
    baseline = peak_rss_mib()

    device = torch.device(args.device)
    flip = torch.tensor(flip_train(args.length), dtype=torch.float32, device=device)
    t1, t2 = tissue_grid(args.atoms)
    T1 = torch.tensor(t1, dtype=torch.float32, device=device)
    T2 = torch.tensor(t2, dtype=torch.float32, device=device)

    sequence = MRFSimulator(flip=flip, TR=10.0, states=args.states)
    if device.type != "cpu":
        sequence = sequence.to(device)

    # The structure is resolved once; the timed calls rebind values onto it.
    start = time.perf_counter()
    if args.mode == "jacobian":
        sequence.jacobian(diff, T1=T1[:1], T2=T2[:1])
    else:
        sequence.simulate(T1=T1[:1], T2=T2[:1])
    setup = time.perf_counter() - start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def forward() -> torch.Tensor:
        return sequence.simulate(T1=T1, T2=T2)

    def jacobian() -> torch.Tensor:
        return sequence.jacobian(diff, T1=T1, T2=T2)[1]

    run = jacobian if args.mode == "jacobian" else forward
    synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    seconds, signal = timed(run, repeats=args.repeats, synchronize=synchronize)

    measurement = Measurement(
        backend="torchsim",
        mode=args.mode if args.mode == "forward" else f"jacobian({args.diff})",
        atoms=args.atoms,
        length=args.length,
        states=args.states,
        device=str(device),
        threads=args.threads or torch.get_num_threads(),
        repeats=args.repeats,
        seconds=seconds,
        setup_seconds=setup,
        baseline_rss_mib=baseline,
        peak_rss_mib=peak_rss_mib(),
        # What the caching allocator took from the driver, which is what the
        # Julia backends report from their own pool. Neither counts the driver
        # context underneath.
        peak_device_mib=(
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda"
            else 0.0
        ),
        checksum=complex(signal.reshape(-1).to(torch.complex64).sum().cpu()),
        versions={"torchsim": version("torchsim"), "torch": torch.__version__},
        machine=machine(),
    )
    report(measurement, args.json)


if __name__ == "__main__":
    main()
