"""Driving the kernels' RF operator from a tabulated rotation.

A pulse whose shape matters turns about an axis that is neither transverse nor
the same across the slice, which no flip-and-phase pair reaches. The kernels
read the rotation out of a table instead, indexed by the voxel's position along
the slice and by the flip angle it actually sees.

Two properties carry the stage. A table built from a pulse with no gradient
across it holds the instantaneous rotation, so it must reproduce the operator
it replaces -- that pins the whole lookup, the phase convention and the matrix
at once. And a sequence with no table must be untouched, bit for bit, because
the table path is a separate kernel rather than a branch inside the old one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import TissueProperties, fse_description
from torchsim.sequence._parameters import FLOAT_NAMES
from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed,
    offload,
    real_subspace_axis,
)
from torchsim.sequence._description import RfDefinition, RfShape
from torchsim.sequence._parameters import TISSUE_COUNT
from torchsim.sequence._simulation import _prepare_tissue
from torchsim.sequence._transition import transition_table
from utils.packed_reference import simulate_packed

ECHOES = 4
STATES = 8
RASTER = 1e-6
SAMPLES = 256
BANDWIDTH = 4.0 / (SAMPLES * RASTER)


def _shaped(bandwidth_hz: float, samples: int = SAMPLES) -> RfDefinition:
    """A windowed sinc, its negative lobes carried in the phase."""
    grid = np.linspace(-2.0, 2.0, samples)
    envelope = np.sinc(grid) * (0.54 + 0.46 * np.cos(np.pi * grid / 2.0))
    envelope = envelope / np.abs(envelope).max()
    return RfDefinition(
        id=0,
        bandwidth_hz=bandwidth_hz,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=bandwidth_hz,
        total_b1sq_power=1.0,
        magnitude=RfShape(
            num_uncompressed=samples, samples=np.abs(envelope).astype(np.float32)
        ),
        phase=RfShape(
            num_uncompressed=samples,
            samples=(np.angle(envelope) / (2.0 * np.pi)).astype(np.float32),
        ),
    )


def _instantaneous_table():
    """A pulse with no gradient across it: one rotation, every position."""
    flat = RfDefinition(
        id=0,
        bandwidth_hz=0.0,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=0.0,
        total_b1sq_power=1.0,
        magnitude=RfShape(num_uncompressed=8, samples=np.ones(8, dtype=np.float32)),
    )
    return transition_table(
        flat, torch.zeros(1), bins=1024, rf_raster_time_s=RASTER
    )


def _packed(voxels: int):
    """Prepared tissue and packed events for a short echo train."""
    description = fse_description(
        torch.deg2rad(torch.full((ECHOES,), 150.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )
    generator = torch.Generator().manual_seed(0)
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        b1=0.7 + 0.6 * torch.rand(voxels, generator=generator),
        b1_phase_rad=torch.linspace(-0.4, 0.4, voxels),
    )
    prepared, _, device = _prepare_tissue(tissue, torch.device("cpu"))
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    packed = _pack_events(
                description,
        repetitions=1,
        record="all",
        device=device,
        rf_raster_time_s=RASTER,
    )
    return prepared, packed.buffers, packed.output_count


def test_a_table_of_an_instantaneous_pulse_is_the_operator_it_replaces() -> None:
    """The anchor: same rotation, reached two entirely different ways."""
    tissue, events, outputs = _packed(5)
    arguments = dict(state_count=STATES, output_count=outputs, threads=1)
    expected = _run_packed(tissue, events, **arguments)
    actual = _run_packed(tissue, events, profile=_instantaneous_table(), **arguments)

    assert (expected - actual).abs().max() < 1e-5 * expected.abs().max()


def test_the_kernel_reads_the_table_the_oracle_reads() -> None:
    voxels, locations = 3, 3
    table = transition_table(
        _shaped(BANDWIDTH),
        torch.linspace(-0.5, 0.5, locations),
        bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    expected = simulate_packed(
        tissue,
        events,
        state_count=STATES,
        output_count=outputs,
        profile=table,
        locations=locations,
    )
    actual = _run_packed(
        tissue, events, state_count=STATES, output_count=outputs, threads=1,
        profile=table,
    )

    assert (expected - actual).abs().max() < 1e-5 * expected.abs().max()


def test_a_sequence_without_a_table_is_untouched() -> None:
    """The table path is another kernel, not a branch inside this one."""
    tissue, events, outputs = _packed(7)
    arguments = dict(state_count=STATES, output_count=outputs, threads=1)

    assert torch.equal(
        _run_packed(tissue, events, **arguments),
        _run_packed(tissue, events, profile=None, **arguments),
    )


def test_each_voxel_reads_its_own_place_along_the_slice() -> None:
    """Voxels run voxel-major over the profile, so the row wraps with them."""
    locations = 4
    table = transition_table(
        _shaped(BANDWIDTH),
        torch.linspace(-0.5, 0.5, locations),
        bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(2 * locations)
    signal = _run_packed(
        tissue, events, state_count=STATES, output_count=outputs, threads=1,
        profile=table,
    )
    # Two voxels differing only in their slice position give different signals,
    # and one wrap apart they see the same row.
    assert not torch.allclose(signal[0], signal[1], atol=1e-6)
    assert signal.shape[0] == 2 * locations


def test_the_real_subspace_declines_a_shaped_pulse() -> None:
    """A shaped pulse's ``a`` is complex, which is what leaves the subspace."""
    tissue, events, _ = _packed(4)
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, 2), bins=32,
        rf_raster_time_s=RASTER,
    )
    plain = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, 4), t2_ms=torch.linspace(40.0, 120.0, 4)
    )
    real_tissue, _, _ = _prepare_tissue(plain, torch.device("cpu"))
    real_tissue = tuple(v.to(torch.float32).contiguous() for v in real_tissue)

    assert real_subspace_axis(events, real_tissue) == 1
    assert real_subspace_axis(events, real_tissue, table) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_card_reads_the_table_the_host_reads() -> None:
    voxels, locations = 3, 3
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    arguments = dict(
        state_count=STATES, output_count=outputs, threads=1, profile=table
    )
    expected = _run_packed(tissue, events, **arguments)
    card = torch.device("cuda")
    actual = _run_packed(
        tuple(value.to(card) for value in tissue),
        tuple(value.to(card) for value in events),
        **arguments,
    )

    assert (expected - actual.cpu()).abs().max() < 1e-5 * expected.abs().max()


def test_a_pulse_past_the_end_of_the_table_is_refused() -> None:
    """Saturating at the last knot returns numbers for a wrong simulation.

    A table and a sequence built for different rasters give flips that differ
    by the ratio between them, which is how this goes wrong in practice.
    """
    tissue, events, outputs = _packed(3)
    narrow = transition_table(
        _shaped(BANDWIDTH), torch.zeros(1), bins=32, theta_max=0.25,
        rf_raster_time_s=RASTER,
    )
    with pytest.raises(ValueError, match="past the"):
        _run_packed(
            tissue, events, state_count=STATES, output_count=outputs,
            threads=1, profile=narrow,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_streamed_volume_reads_the_table_a_whole_one_does() -> None:
    """A chunk boundary must not fall inside a voxel's slice copies."""
    locations = 3
    voxels = 32
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    arguments = dict(
        state_count=STATES, output_count=outputs, threads=1, profile=table
    )
    whole = _run_packed(tissue, events, **arguments)
    # Small enough that the planner's own chunk (17 voxels here) is not a
    # multiple of the profile's width, so the rounding is doing the work.
    with offload(["cuda"], budget_bytes=1 << 12):
        streamed = _run_packed(tissue, events, **arguments)

    assert (whole - streamed).abs().max() < 1e-4 * whole.abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_chunk_holds_whole_voxels_when_a_table_is_read() -> None:
    """The kernel takes the row from the index within the chunk.

    So a chunk boundary inside a voxel's slice copies shifts every row after
    it, which returns a plausible signal for an entirely different slice.
    """
    from torchsim.sequence._accelerators import _Offload, _offload_plan

    _, events, outputs = _packed(96)
    plan = _Offload(
        devices=(torch.device("cuda"),), budget_bytes=1 << 12, lanes=1
    )
    bare, _, _ = _offload_plan(plan, "forward", events, 96, outputs, STATES, None)
    # A slice width the unaligned chunk is not already a multiple of, so the
    # alignment has something to do however the budget happens to divide. Taken
    # from the measured width rather than fixed, which would go vacuous the
    # next time a tissue property changes what a voxel costs.
    locations = next(width for width in range(2, bare) if bare % width)
    aligned, _, _ = _offload_plan(
        plan, "forward", events, 96, outputs, STATES, None, locations=locations
    )

    assert aligned % locations == 0
    assert aligned <= bare


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_streamed_adjoint_reads_the_table_a_whole_one_does() -> None:
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    locations, voxels = 3, 24
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    generator = torch.Generator().manual_seed(11)
    seed = torch.complex(
        torch.randn(voxels * locations, outputs, generator=generator),
        torch.randn(voxels * locations, outputs, generator=generator),
    )
    directions = tuple(
        0.05 * torch.randn(value.shape, generator=generator)
        for value in (*tissue, events[0], events[2], events[3])
    )
    arguments = (tissue, events, directions, seed, STATES, outputs, 1)
    whole = _run_packed_vjp_jvp(*arguments, profile=table)
    with offload(["cuda"], budget_bytes=1 << 12):
        streamed = _run_packed_vjp_jvp(*arguments, profile=table)

    names = FLOAT_NAMES
    _compare(whole[0], streamed[0], names, 1e-3)
    _compare(
        whole[1], streamed[1],
        tuple(f"adjoint {name}" for name in names), 1e-3,
    )


def test_forward_mode_follows_the_table() -> None:
    """The tangent rides the slope the table already stores."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    locations, voxels = 3, 3
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    generator = torch.Generator().manual_seed(2)
    tissue_dot = tuple(
        0.1 * torch.randn(value.shape, generator=generator) for value in tissue
    )
    event_dot = tuple(
        0.01 * torch.randn(events[index].shape, generator=generator)
        for index in (0, 2, 3)
    )

    def forward(*values):
        return simulate_packed(
            values[:TISSUE_COUNT],
            (
                values[TISSUE_COUNT], events[1], values[TISSUE_COUNT + 1],
                values[TISSUE_COUNT + 2], *events[4:],
            ),
            state_count=STATES,
            output_count=outputs,
            profile=table,
            locations=locations,
        )

    _, expected = torch.func.jvp(
        forward,
        (*tissue, events[0], events[2], events[3]),
        (*tissue_dot, *event_dot),
    )
    actual = _run_packed_jvp(
        tissue, events, tissue_dot, event_dot, STATES, outputs, 1, profile=table
    )

    assert (expected - actual).abs().max() < 1e-5 * expected.abs().max()


def _compare(expected, actual, names, tolerance=2e-4) -> None:
    checked = 0
    for name, want, got in zip(names, expected, actual, strict=True):
        if want is None:
            # A parameter this sequence never touches, which the oracle drops
            # from its graph and the kernel returns as zeros.
            assert float(got.abs().max()) == 0.0, name
            continue
        scale = float(want.abs().max())
        if scale < 1e-7:
            continue
        assert float((want - got).abs().max()) < tolerance * scale, name
        checked += 1
    assert checked >= 6


def test_the_adjoint_follows_the_table() -> None:
    """Every gradient the table stands in front of, against the oracle."""
    from torchsim.sequence._accelerators import _run_packed_vjp

    locations, voxels = 3, 3
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    generator = torch.Generator().manual_seed(4)
    seed = torch.complex(
        torch.randn(voxels * locations, outputs, generator=generator),
        torch.randn(voxels * locations, outputs, generator=generator),
    )

    wrt = (*tissue, events[0], events[2], events[3])
    tracked = tuple(value.clone().requires_grad_(True) for value in wrt)
    signal = simulate_packed(
        tracked[:TISSUE_COUNT],
        (
            tracked[TISSUE_COUNT], events[1], tracked[TISSUE_COUNT + 1],
            tracked[TISSUE_COUNT + 2], *events[4:],
        ),
        state_count=STATES,
        output_count=outputs,
        profile=table,
        locations=locations,
    )
    expected = torch.autograd.grad(
        signal, tracked, grad_outputs=seed, allow_unused=True
    )

    actual = _run_packed_vjp(
        tissue, events, seed, STATES, outputs, 1, profile=table
    )
    _compare(
        expected,
        actual,
        FLOAT_NAMES,
    )


def test_an_unprofiled_adjoint_is_untouched() -> None:
    """The table path is another kernel, not a branch inside this one."""
    from torchsim.sequence._accelerators import _run_packed_vjp

    tissue, events, outputs = _packed(6)
    generator = torch.Generator().manual_seed(5)
    seed = torch.complex(
        torch.randn(6, outputs, generator=generator),
        torch.randn(6, outputs, generator=generator),
    )
    plain = _run_packed_vjp(tissue, events, seed, STATES, outputs, 1)
    again = _run_packed_vjp(tissue, events, seed, STATES, outputs, 1, profile=None)

    for left, right in zip(plain, again):
        assert torch.equal(left, right)


def test_the_second_order_pass_follows_the_table() -> None:
    """Forward-over-reverse, which needs the Hermite's own curvature."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    locations, voxels = 3, 2
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    generator = torch.Generator().manual_seed(6)
    seed = torch.complex(
        torch.randn(voxels * locations, outputs, generator=generator),
        torch.randn(voxels * locations, outputs, generator=generator),
    )
    primals = (*tissue, events[0], events[2], events[3])
    directions = tuple(
        0.05 * torch.randn(value.shape, generator=generator) for value in primals
    )

    leaves = tuple(value.clone().requires_grad_(True) for value in primals)
    signal = simulate_packed(
        leaves[:TISSUE_COUNT],
        (
            leaves[TISSUE_COUNT], events[1], leaves[TISSUE_COUNT + 1],
            leaves[TISSUE_COUNT + 2], *events[4:],
        ),
        state_count=STATES,
        output_count=outputs,
        profile=table,
        locations=locations,
    )
    first = torch.autograd.grad(
        (signal.real * seed.real + signal.imag * seed.imag).sum(),
        leaves,
        create_graph=True,
        materialize_grads=True,
    )
    second = torch.autograd.grad(
        sum((grad * step).sum() for grad, step in zip(first, directions)),
        leaves,
        materialize_grads=True,
    )
    curvature, adjoint = _run_packed_vjp_jvp(
        tissue, events, directions, seed, STATES, outputs, 1, profile=table
    )

    names = FLOAT_NAMES
    _compare(tuple(value.detach() for value in first), adjoint, names)
    _compare(second, curvature, tuple(f"d{name}" for name in names), 1e-3)


def test_an_unprofiled_second_order_pass_is_untouched() -> None:
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    tissue, events, outputs = _packed(6)
    generator = torch.Generator().manual_seed(7)
    seed = torch.complex(
        torch.randn(6, outputs, generator=generator),
        torch.randn(6, outputs, generator=generator),
    )
    directions = tuple(
        0.1 * torch.randn(value.shape, generator=generator)
        for value in (*tissue, events[0], events[2], events[3])
    )
    arguments = (tissue, events, directions, seed, STATES, outputs, 1)
    plain = _run_packed_vjp_jvp(*arguments)
    again = _run_packed_vjp_jvp(*arguments, profile=None)

    for left, right in zip(plain, again):
        for first, second in zip(left, right):
            assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_card_follows_the_table_through_both_adjoint_passes() -> None:
    """One kernel serves both: the card reaches a first adjoint through it."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    locations, voxels = 3, 2
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    generator = torch.Generator().manual_seed(9)
    seed = torch.complex(
        torch.randn(voxels * locations, outputs, generator=generator),
        torch.randn(voxels * locations, outputs, generator=generator),
    )
    directions = tuple(
        0.05 * torch.randn(value.shape, generator=generator)
        for value in (*tissue, events[0], events[2], events[3])
    )
    host = _run_packed_vjp_jvp(
        tissue, events, directions, seed, STATES, outputs, 1, profile=table
    )
    card = torch.device("cuda")
    on_card = _run_packed_vjp_jvp(
        tuple(value.to(card) for value in tissue),
        tuple(value.to(card) for value in events),
        tuple(value.to(card) for value in directions),
        seed.to(card),
        STATES,
        outputs,
        1,
        profile=table,
    )

    names = FLOAT_NAMES
    _compare(host[0], tuple(v.cpu() for v in on_card[0]), names, 1e-3)
    _compare(
        host[1], tuple(v.cpu() for v in on_card[1]),
        tuple(f"adjoint {name}" for name in names), 1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_card_follows_the_table_in_forward_mode() -> None:
    from torchsim.sequence._accelerators import _run_packed_jvp

    locations, voxels = 3, 3
    table = transition_table(
        _shaped(BANDWIDTH), torch.linspace(-0.5, 0.5, locations), bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    generator = torch.Generator().manual_seed(8)
    tissue_dot = tuple(
        0.1 * torch.randn(value.shape, generator=generator) for value in tissue
    )
    event_dot = tuple(
        0.01 * torch.randn(events[index].shape, generator=generator)
        for index in (0, 2, 3)
    )
    arguments = dict(profile=table)
    expected = _run_packed_jvp(
        tissue, events, tissue_dot, event_dot, STATES, outputs, 1, **arguments
    )
    card = torch.device("cuda")
    actual = _run_packed_jvp(
        tuple(value.to(card) for value in tissue),
        tuple(value.to(card) for value in events),
        tuple(value.to(card) for value in tissue_dot),
        tuple(value.to(card) for value in event_dot),
        STATES, outputs, 1, **arguments,
    )

    assert (expected - actual.cpu()).abs().max() < 1e-4 * expected.abs().max()
