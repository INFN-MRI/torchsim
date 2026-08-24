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


@triton.jit
def _poisoned_acos(x: Any) -> Any:
    """An ``acos`` whose result cannot be used without showing.

    A narrow launch is supposed to reach only the series. Nothing that passes
    through this can stay finite, so a finite answer is proof the roots were
    never read.
    """
    return x * float("nan")


class _Interpretable:
    acos = _acos


class _Poisoned:
    acos = _poisoned_acos


def install(poison: bool = False) -> None:
    """Point the kernels at an ``acos`` the interpreter can evaluate."""
    from torchsim.sequence import _epg_triton

    _epg_triton.libdevice = _Poisoned if poison else _Interpretable


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
    """Run the same launch with the table refused and with it forced.

    ``force_narrow`` builds a table for a train the launch-wide gate calls
    narrow, so every row takes the series and the two arms should agree to the
    bit.
    """
    from torchsim.sequence import _epg_triton

    original = _epg_triton._tabulate_three_pool
    built: list[bool] = []

    def patched(
        tissue, duration, *, pools, narrow, problems=None,
        tangents=None,
    ):
        if not built_wanted[0]:
            return None, None, None
        rows, table, lengths = original(
            tissue, duration, pools=pools,
            narrow=False if force_narrow else narrow,
            problems=problems, tangents=tangents,
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


def _unread(voxels: int, states: int) -> None:
    """Check that a narrow launch does not read the three roots.

    The `close` select sits in the consumers of `_three_pool_pieces_jvp`, and
    each of them takes the series outright when the caller has bounded the
    spread -- so the roots the pieces still form are unread. Poisoning `acos`
    is what says so rather than reading the branches and believing it.
    """
    install(poison=True)
    from torchsim.sequence import _epg_triton
    from torchsim.sequence._lineshape import lineshape_table
    from torchsim.sequence._parameters import narrow_three_pool
    from torchsim.sequence import _builders

    echoes = 6
    tissue = _tissue(voxels)
    events, outputs = _events(_builders.fse_description(
        torch.full((echoes,), math.radians(150.0)), 8e-3
    ))
    assert narrow_three_pool(tissue, events[0].reshape(-1), pools=3), (
        "this train has to be narrow for the check to mean anything"
    )
    options: dict[str, Any] = dict(
        lineshape=lineshape_table(), exchanging=True
    )
    signal = _epg_triton.simulate(
        tissue, events, state_count=states, output_count=outputs, **options
    )
    assert bool(torch.isfinite(torch.view_as_real(signal)).all()), (
        "the forward read the roots"
    )
    seed = (
        torch.rand(
            voxels, outputs, generator=torch.Generator().manual_seed(7)
        ) * 2.0 - 1.0
    ).to(torch.complex64)
    for index, gradient in enumerate(_epg_triton.simulate_vjp(
        tissue, events, seed, state_count=states, output_count=outputs,
        **options,
    )):
        assert not gradient.numel() or bool(torch.isfinite(gradient).all()), (
            f"gradient {index} read the roots"
        )
    print("  the roots are unread")


def _streamed(voxels: int, states: int) -> None:
    """The chunked adjoint launcher, against the whole-volume one.

    It passes a fixed positional list for every optional buffer the kernel
    takes, so it is the launcher a grown kernel signature misaligns first --
    and nothing else here reaches it.
    """
    install()
    from torchsim.sequence import _epg_triton
    from torchsim.sequence import _builders

    echoes = 6
    tissue = _tissue(voxels)
    events, outputs = _events(_builders.fse_description(
        torch.full((echoes,), math.radians(150.0)), 8e-3
    ))
    seed = (
        torch.rand(
            voxels, outputs, generator=torch.Generator().manual_seed(3)
        ) * 2.0 - 1.0
    ).to(torch.complex64)
    whole = _epg_triton.simulate_vjp(
        tissue, events, seed, state_count=states, output_count=outputs
    )
    buffers = _epg_triton.GradientBuffers(
        events, voxels, state_count=states, output_count=outputs
    )
    chunked = _epg_triton.simulate_vjp_into(
        tissue, events, seed, buffers, state_count=states,
        output_count=outputs, atom_count=voxels,
    )
    scale = max(float(v.abs().max()) for v in whole if v.numel())
    worst = max(
        float((a - b).abs().max()) / max(scale, 1e-30)
        for a, b in zip(whole, chunked, strict=False)
        if a.numel() and b.numel()
    )
    print(f"  streamed vs whole   {worst:.2e}")
    assert worst <= NARROW_TOLERANCE, f"streamed drifted: {worst:.2e}"


def _washed(voxels: int, states: int) -> None:
    """The pooled adjoint where the interval carries a washout.

    Every event sharing a length shares its attenuation too, since washout is
    ``1 - rate dt`` -- but the row has to be given that attenuation rather
    than one, because the gradients it pools are scaled by it.
    """
    install()
    from torchsim.sequence import _epg_triton
    from torchsim.sequence._lineshape import lineshape_table
    from torchsim.sequence._parameters import Geometry, TISSUE_NAMES
    from torchsim.sequence import _builders

    echoes = 6
    tissue = list(_tissue(voxels))
    # A velocity, so the washout the geometry declares is genuinely live.
    tissue[TISSUE_NAMES.index("velocity_m_per_s")] = torch.linspace(
        0.02, 0.09, voxels
    )
    tissue = tuple(tissue)
    events, outputs = _events(_builders.mrf_description(
        torch.full((echoes,), math.radians(50.0)),
        torch.tensor([(6 + (i % 2)) * 1e-3 for i in range(echoes)]),
        inversion_time_s=1.0,
    ))
    options: dict[str, Any] = dict(
        lineshape=lineshape_table(), exchanging=True,
        geometry=Geometry(flow_scale=0.0, washout_scale=12.0),
    )
    seed = (
        torch.rand(
            voxels, outputs, generator=torch.Generator().manual_seed(11)
        ) * 2.0 - 1.0
    ).to(torch.complex64)
    without, with_table = _both(
        lambda: _epg_triton.simulate_vjp(
            tissue, events, seed, state_count=states, output_count=outputs,
            **options,
        ),
        False,
    )
    worst = _worst(without, with_table)
    print(f"  washed adjoint      {worst:.2e}")
    # Held to the narrow tolerance deliberately: a row given an attenuation of
    # one rather than its own reads about 4e-6 here, which this catches and
    # the looser bound would not.
    assert worst <= NARROW_TOLERANCE, f"washed adjoint drifted: {worst:.2e}"


def _shimmed(voxels: int, states: int) -> None:
    """The pooled adjoint where the pulse reads a transmit row per shim.

    The three-pool row index and the shim row are both natural to call `row`,
    and the shim one is rebound between the operator read and the cotangent
    pooling. Nothing else here drives a transmit array, so this is the only
    case that would show one standing in for the other.
    """
    install()
    from torchsim.sequence import _epg_triton
    from torchsim.sequence._lineshape import lineshape_table
    from torchsim.sequence import _builders

    shims, echoes = 2, 6
    held = _tissue(voxels)
    tissue = list(held)
    # A transmit row per shim, each row a different field, so a pulse reading
    # the wrong row cannot agree with one reading the right one.
    tissue[3] = torch.cat([
        torch.linspace(0.7 + 0.3 * shim, 1.1 + 0.3 * shim, voxels)
        for shim in range(shims)
    ]).contiguous()
    tissue[4] = torch.cat([
        torch.linspace(-0.4 + 0.8 * shim, 0.4 + 0.8 * shim, voxels)
        for shim in range(shims)
    ]).contiguous()
    tissue = tuple(tissue)
    events, outputs = _events(_builders.mrf_description(
        torch.full((echoes,), math.radians(50.0)),
        torch.tensor([(6 + (i % 3)) * 1e-3 for i in range(echoes)]),
        inversion_time_s=1.0,
    ))
    events = list(events)
    events[6] = (
        torch.arange(events[6].numel(), dtype=torch.int32) % shims
    ).contiguous()
    events = tuple(events)
    assert _epg_triton._shim_count(tissue) == shims, (
        "the transmit array did not reach the kernel"
    )

    options: dict[str, Any] = dict(
        lineshape=lineshape_table(), exchanging=True
    )
    seed = (
        torch.rand(
            voxels, outputs, generator=torch.Generator().manual_seed(5)
        ) * 2.0 - 1.0
    ).to(torch.complex64)
    without, with_table = _both(
        lambda: _epg_triton.simulate_vjp(
            tissue, events, seed, state_count=states, output_count=outputs,
            **options,
        ),
        False,
    )
    worst = _worst(without, with_table)
    print(f"  shimmed adjoint     {worst:.2e}")
    assert worst <= WIDE_TOLERANCE, f"shimmed adjoint drifted: {worst:.2e}"

    without, with_table = _both(
        lambda: _epg_triton.simulate_vjp_jvp(
            tissue, events,
            (
                *(
                    torch.linspace(0.01, 0.03, value.numel()).reshape(value.shape)
                    if value.numel() else value.clone()
                    for value in tissue
                ),
                torch.linspace(0.5e-4, 2e-4, events[0].numel()).reshape(
                    events[0].shape
                ),
                torch.zeros_like(events[2]),
                torch.zeros_like(events[3]),
            ),
            seed, state_count=states, output_count=outputs, **options,
        )[1],
        False,
    )
    second = _worst(without, with_table)
    print(f"  shimmed second order {second:.2e}")
    assert second <= WIDE_TOLERANCE, f"shimmed second order: {second:.2e}"


def _case(name: str) -> None:
    if name == "shimmed":
        _shimmed(3, 4)
        return
    if name == "washed":
        _washed(3, 4)
        return
    if name == "streamed":
        _streamed(3, 4)
        return
    if name == "unread":
        _unread(3, 4)
        return
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
    directions = tuple(
        torch.linspace(0.01, 0.03, value.numel()).reshape(value.shape)
        if value.numel() else value.clone()
        for value in tissue
    )
    event_directions = (
        # Varying, so that events sharing a length carry different interval
        # directions: a pooled second-order adjoint that weighed the whole row
        # by one direction would agree with the per-event arm under a uniform
        # one and only differ under this.
        torch.linspace(0.5e-4, 2e-4, events[0].numel()).reshape(
            events[0].shape
        ),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    without, with_table = _both(
        lambda: _epg_triton.simulate_jvp(
            tissue, events, directions, event_directions,
            state_count=states, output_count=outputs, **options,
        ),
        force_narrow,
    )
    forward_mode = float(
        (with_table - without).abs().max() / without.abs().max()
    )
    print(f"  forward mode        {forward_mode:.2e}")
    assert forward_mode <= tolerance, f"forward mode drifted: {forward_mode:.2e}"

    # Real directions, so the tangent half of the table is genuinely read.
    seeded = (*directions, *event_directions)
    for half, label in ((0, "curvature"), (1, "gradient")):
        without, with_table = _both(
            lambda h=half: _epg_triton.simulate_vjp_jvp(
                tissue, events, seeded, seed,
                state_count=states, output_count=outputs, **options,
            )[h],
            force_narrow,
        )
        second = _worst(without, with_table)
        print(f"  second order {label:<10} {second:.2e}")
        assert second <= tolerance, f"second order {label}: {second:.2e}"

    if cut is not None:
        # A chunked case that ran in one chunk tests nothing it claims to.
        assert cut and max(cut) < voxels, f"never chunked: waves {cut}"
        print(f"  chunks              {-(-voxels // max(cut))}")


if __name__ == "__main__":
    _case(sys.argv[1] if len(sys.argv) > 1 else "narrow")
    # The wrapper reads this: a case that fell out early prints no such line.
    print("checked")
