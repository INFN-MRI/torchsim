"""What a job leaves behind for the next one on the same thread.

The kernels are multiversioned, so a pass may run a clone built for a wider
instruction set than the code around it. On Intel parts that leaves the vector
registers in a state every later SSE-encoded instruction pays for, and a pool
worker parks between jobs rather than executing anything that would clear it --
so without a barrier one forward-mode pass makes every later kernel in the
process slower, for as long as the process lives.

Timed, because there is nothing else to look at: the arithmetic, the buffers
and the kernel selected are identical either side of it. The comparison is a
ratio of two measurements taken moments apart, which is what makes it readable
on a shared machine: whatever slows one arm slows the other.

Only a build that multiversions has a wide clone to leave a state behind. Where
the kernels are compiled once -- clang, MSVC, musl, anything not x86 -- the two
arms run the same code and the ratio measures how quiet the machine was, which
is not a property of TorchSim.
"""

from __future__ import annotations

import time

import pytest
import torch

import torchsim._epg_cpu as kernels
from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed,
    _run_packed_jvp,
)
from torchsim.sequence._builders import mrf_description
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

ATOMS = 4000
REPETITIONS = 120
STATES = 32
# Well clear of the 2.6x a dirty register state costs, and of the noise a
# loaded machine puts on either arm.
TOLERATED = 1.8


def _problem(atoms: int):
    flip = torch.linspace(5.0, 60.0, REPETITIONS) * (torch.pi / 180.0)
    packed = _pack_events(
        mrf_description(flip, torch.full((REPETITIONS,), 10e-3)),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(200.0, 3000.0, atoms),
            t2_ms=torch.linspace(10.0, 300.0, atoms),
        ),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return prepared, packed.buffers, packed.output_count


def _fastest(run, repeats: int = 5) -> float:
    run()
    return min(
        (lambda start: (run(), time.perf_counter() - start)[1])(time.perf_counter())
        for _ in range(repeats)
    )


@pytest.mark.skipif(
    not getattr(kernels, "multiversioned", 0),
    reason="the kernels are compiled once here, so no clone leaves a state behind",
)
def test_a_forward_mode_pass_leaves_the_pool_as_it_found_it():
    tissue, events, outputs = _problem(ATOMS)
    forward = lambda: _run_packed(tissue, events, STATES, outputs, 4)  # noqa: E731
    before = _fastest(forward)

    # The complex forward-mode kernel, which is the one whose widest clone the
    # loader picks on a machine that has one.
    seed_tissue, seed_events, seed_outputs = _problem(512)
    seeds = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(seed_tissue)
    )
    still = tuple(torch.zeros_like(seed_events[index]) for index in (0, 2, 3))
    _run_packed_jvp(
        seed_tissue,
        seed_events,
        seeds,
        still,
        STATES,
        seed_outputs,
        4,
        real_axis=-1,
    )

    after = _fastest(forward)
    assert after < TOLERATED * before, (
        f"a forward pass costs {after / before:.1f}x as much after a "
        f"forward-mode pass ({before * 1e3:.0f} ms then {after * 1e3:.0f} ms)"
    )
