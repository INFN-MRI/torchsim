"""The three-pool operator table, checked in Triton's CPU interpreter.

`TRITON_INTERPRET=1` runs a kernel in Python over host tensors, which reaches
the plumbing a compiled launch hides: a launcher whose positional list has
drifted from the kernel it calls, a trajectory plane without its buffer, a
replay disagreeing with the reverse sweep. Those are structural and show at
three voxels.

Run as ``python -m utils.interpreted <case>`` with ``TRITON_INTERPRET=1`` set
before the process starts -- Triton reads it at import. Each case exits
non-zero when a comparison drifts, so the pytest wrapper only has to run it.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
import triton
import triton.language as tl

# What a narrow row and a roots row are each held to against the arm that
# forms its operator per event. The roots row is looser because the table
# takes the series wherever a row's own spread allows it, which the per-event
# arm cannot do.
NARROW_TOLERANCE = 1e-6
WIDE_TOLERANCE = 1e-5


@triton.jit
def _acos(x: Any) -> Any:
    """``acos`` for the interpreter, which has no inverse trigonometry.

    ``libdevice`` is a CUDA extern and ``tl.math`` carries no acos, asin or
    atan2, so a kernel forming the three roots cannot run on the host at all.
    Newton on ``cos(theta) - x`` reaches 1.8e-6 over the range the callers
    clamp to, which is ample when both arms of a comparison use it.
    """
    theta = 1.5707963267948966 - x * (1.0 + 0.16666667 * x * x)
    for _ in tl.static_range(0, 12):
        sine = tl.sin(theta)
        theta = theta + (tl.cos(theta) - x) / tl.where(
            tl.abs(sine) > 1e-12, sine, 1e-12
        )
    return theta


class _Interpretable:
    acos = _acos


def install() -> None:
    """Point the kernels at an ``acos`` the interpreter can evaluate."""
    from torchsim.sequence import _epg_triton

    _epg_triton.libdevice = _Interpretable


def _tissue(voxels: int) -> tuple[torch.Tensor, ...]:
    from torchsim import TissueProperties
    from torchsim.sequence._simulation import _prepare_tissue

    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(600.0, 1400.0, voxels),
            t2_ms=torch.linspace(40.0, 120.0, voxels),
            bound_fraction=torch.linspace(0.02, 0.2, voxels),
            bound_exchange_hz=torch.linspace(5.0, 60.0, voxels),
            pool_b_fraction=torch.linspace(0.05, 0.4, voxels),
            pool_b_exchange_hz=torch.linspace(1.0, 80.0, voxels),
            t2_pool_b_ms=torch.linspace(10.0, 90.0, voxels),
            pool_b_shift_hz=torch.linspace(-500.0, 500.0, voxels),
        ),
        "cpu",
    )
    return tuple(value.to(torch.float32).contiguous() for value in prepared)


def _events(description: Any) -> tuple[tuple[torch.Tensor, ...], int]:
    from torchsim.sequence._accelerators import _pack_events

    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration, packed.kind, packed.flip, packed.phase, packed.action,
        packed.output_index, packed.shim_index, packed.saturation,
        packed.rf_frequency_hz,
    )
    return events, int(packed.output_index.max()) + 1


def _both(run: Any, force_narrow: bool) -> tuple[Any, Any]:
    """The same launch with the table refused and with it forced.

    ``force_narrow`` builds a table for a train the launch-wide gate calls
    narrow, so every row takes the series and the two arms should agree to the
    bit.
    """
    from torchsim.sequence import _epg_triton

    original = _epg_triton._tabulate_three_pool
    built: list[bool] = []

    def patched(tissue, duration, *, pools, narrow, problems=None):
        if not built_wanted[0]:
            return None, None, None
        rows, table, lengths = original(
            tissue, duration, pools=pools,
            narrow=False if force_narrow else narrow,
            problems=problems,
        )
        built.append(table is not None)
        return rows, table, lengths

    built_wanted = [False]
    _epg_triton._tabulate_three_pool = patched
    try:
        without = run()
        built_wanted[0] = True
        built.clear()
        with_table = run()
        # A fast-path check that does not show the fast path was taken shows
        # nothing at all.
        assert built and all(built), "the operator table was never built"
    finally:
        _epg_triton._tabulate_three_pool = original
    return without, with_table


def _worst(without: tuple[Any, ...], with_table: tuple[Any, ...]) -> float:
    scale = max(
        float(value.abs().max()) for value in without if value.numel()
    )
    return max(
        float((a - b).abs().max()) / max(scale, 1e-30)
        for a, b in zip(without, with_table, strict=True)
        if a.numel()
    )


def _chunked(voxels: int) -> Any:
    """Force the adjoint to take more than one chunk, whatever the volume.

    The cotangent table is sized and indexed per chunk, so a single-chunk run
    cannot tell a chunk-local index from a global one.
    """
    from torchsim.sequence import _epg_triton

    cut: list[int] = []

    def narrow_wave(*arguments: Any, **keywords: Any) -> int:
        wave = max(1, voxels // 2)
        cut.append(wave)
        return wave

    _epg_triton._trajectory_wave = narrow_wave
    return cut


def _case(name: str) -> None:
    install()
    from torchsim.sequence import _epg_triton
    from torchsim.sequence._lineshape import lineshape_table
    from torchsim.sequence import _builders

    voxels, states = 3, 4
    if name == "narrow":
        echoes = 6
        description = _builders.fse_description(
            torch.full((echoes,), math.radians(150.0)), 8e-3
        )
        force_narrow, tolerance = True, NARROW_TOLERANCE
    elif name in ("wide", "chunked"):
        shots = 6
        description = _builders.mrf_description(
            torch.full((shots,), math.radians(50.0)),
            torch.tensor([(6 + (i % 3)) * 1e-3 for i in range(shots)]),
            inversion_time_s=1.0,
        )
        force_narrow, tolerance = False, WIDE_TOLERANCE
    else:
        raise SystemExit(f"unknown case {name!r}")

    cut = _chunked(voxels) if name == "chunked" else None
    tissue = _tissue(voxels)
    events, outputs = _events(description)
    options: dict[str, Any] = dict(
        lineshape=lineshape_table(), exchanging=True
    )
    lengths = torch.unique(events[0].reshape(-1)).numel()
    print(f"{name}: {events[0].numel()} events over {lengths} lengths")

    without, with_table = _both(
        lambda: _epg_triton.simulate(
            tissue, events, state_count=states, output_count=outputs, **options
        ),
        force_narrow,
    )
    forward = float(
        (with_table - without).abs().max() / without.abs().max()
    )
    print(f"  forward             {forward:.2e}")
    assert forward <= tolerance, f"forward drifted: {forward:.2e}"

    seed = (
        torch.rand(
            voxels, outputs, generator=torch.Generator().manual_seed(7)
        ) * 2.0 - 1.0
    ).to(torch.complex64)
    without, with_table = _both(
        lambda: _epg_triton.simulate_vjp(
            tissue, events, seed, state_count=states, output_count=outputs,
            **options,
        ),
        force_narrow,
    )
    adjoint = _worst(without, with_table)
    print(f"  first-order adjoint {adjoint:.2e}")
    assert adjoint <= tolerance, f"adjoint drifted: {adjoint:.2e}"
    if cut is not None:
        # A chunked case that ran in one chunk tests nothing it claims to.
        assert cut and max(cut) < voxels, f"never chunked: waves {cut}"
        print(f"  chunks              {-(-voxels // max(cut))}")


if __name__ == "__main__":
    _case(sys.argv[1] if len(sys.argv) > 1 else "narrow")
