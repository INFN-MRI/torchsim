"""Streaming a host-resident volume through a memory-limited device.

Parameter mapping and model-based imaging scale with voxels, and the signal
alone outgrows a recon server's spare VRAM well before the voxel counts get
interesting. The contract here is that a volume stays on the host, moves
through the device in chunks, and comes back with the same numbers a CPU run
would have produced -- whatever chunk size the budget implies.

The budget is the part worth pinning down: a chunk that does not divide the
volume leaves a short final chunk, and the kernels stride the signal by the
width they are handed, so a buffer sized for the wide case is not a buffer for
the narrow one.
"""

import pytest
import torch

import torchsim._execution as _policy
import torchsim.sequence._accelerators as accelerators
from torchsim.sequence import EpgEngine, offload
from torchsim.sequence._accelerators import (
    _FLOAT_INPUTS,
    _bytes_per_voxel,
    _chunk_voxels,
    _Lane,
    _Offload,
    _pack_events,
    _run_packed,
)
from torchsim.sequence._builders import fse_description
from torchsim.sequence._parameters import NO_GEOMETRY, OUTSIDE_THE_SUBSPACE, Geometry
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)

STATES = 10
ECHOES = 20


def _volume(voxels, trains=1):
    generator = torch.Generator().manual_seed(0)
    shape = (trains, ECHOES) if trains > 1 else (ECHOES,)
    flip = torch.deg2rad(80.0 + 80.0 * torch.rand(shape, generator=generator))
    packed = _pack_events(
        fse_description(
            flip,
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
            excitation_phase_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    tissue = TissueProperties(
        t1_ms=torch.linspace(300.0, 2000.0, voxels),
        t2_ms=torch.linspace(20.0, 200.0, voxels),
        b0_hz=torch.zeros(voxels),
        b1_phase_rad=torch.zeros(voxels),
    )
    prepared, _, _ = _prepare_tissue(tissue, torch.device("cpu"))
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return events, prepared, packed.output_count


@pytest.mark.parametrize("budget", [1 << 20, 4 << 20, 512 << 20])
@pytest.mark.parametrize("real_axis", [1, -1])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_volume_matches_the_cpu_run(budget, real_axis, trains):
    """Every budget cuts the volume differently; none may change the answer."""
    events, prepared, outputs = _volume(5000, trains=trains)
    expected = _run_packed(prepared, events, STATES, outputs, 0, real_axis=real_axis)
    with offload(["cuda"], budget_bytes=budget, lanes=2):
        actual = _run_packed(prepared, events, STATES, outputs, 0, real_axis=real_axis)

    assert actual.device.type == "cpu"
    assert actual.shape == expected.shape
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def test_a_volume_that_does_not_divide_the_chunk_is_still_right():
    """The final chunk is shorter, and its buffers are packed for that width."""
    events, prepared, outputs = _volume(1001)
    expected = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)
    # A chunk of 1001 voxels cannot divide evenly into anything but itself.
    with offload(["cuda"], budget_bytes=1 << 19, lanes=3):
        actual = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)

    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def test_one_chunk_covering_the_volume_is_still_right():
    """A budget nobody reaches must not take a different path."""
    events, prepared, outputs = _volume(512)
    expected = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)
    with offload(["cuda"], budget_bytes=1 << 30, lanes=1):
        actual = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)

    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def _peak_over_baseline(call):
    """Device bytes this call adds, ignoring whatever else is already live.

    ``max_memory_allocated`` is a high-water mark over everything alive in the
    process, so tensors another test still holds would be charged here.
    """
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    call()
    return torch.cuda.max_memory_allocated() - baseline


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_the_device_footprint_stays_inside_the_budget(lanes):
    """The point of the whole thing: the card never sees the whole volume."""
    events, prepared, outputs = _volume(200_000)
    budget = 16 << 20
    resident = _peak_over_baseline(
        lambda: _run_offload(prepared, events, outputs, budget, lanes)
    )

    assert resident <= budget * 1.1
    assert resident < 200_000 * outputs * 8


def _run_offload(prepared, events, outputs, budget, lanes):
    with offload(["cuda"], budget_bytes=budget, lanes=lanes):
        _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)


def test_one_lane_by_default():
    """A second lane takes its share out of the same budget.

    Every chunk gets narrower as a result, and narrower chunks lose more than
    the overlap recovers unless the volume was barely splitting to begin with.
    Measured on the streamed adjoint, one lane runs a 19-chunk volume in 54 ms
    where two take 94 ms; the regime where two win is 2-4 chunks, by 5-23%.
    """
    with offload(["cuda"]):
        assert _policy._OFFLOAD.lanes == 1


def test_more_lanes_do_not_widen_the_footprint():
    """The budget covers the lanes together, dividing it rather than multiplying."""
    events, prepared, outputs = _volume(200_000)
    budget = 32 << 20
    peaks = [
        _peak_over_baseline(
            lambda lanes=lanes: _run_offload(prepared, events, outputs, budget, lanes)
        )
        for lanes in (1, 4)
    ]

    assert max(peaks) <= budget * 1.1


def test_a_smaller_budget_makes_a_smaller_chunk():
    def plan(budget, lanes):
        return _Offload((torch.device("cuda"),), budget, lanes)

    shape = (1, 200, 100, STATES, 1)  # trains, events, outputs, states, real axis
    wide = _chunk_voxels(plan(1 << 30, 1), "forward", *shape)
    narrow = _chunk_voxels(plan(1 << 20, 1), "forward", *shape)
    shared = _chunk_voxels(plan(1 << 30, 4), "forward", *shape)

    assert wide > narrow >= 1
    assert shared * 4 <= wide * 1.01


def test_a_budget_too_small_for_one_voxel_still_runs():
    """Rounding the chunk to zero would loop forever rather than fail."""
    events, prepared, outputs = _volume(64)
    expected = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)
    with offload(["cuda"], budget_bytes=1, lanes=1):
        actual = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)

    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def test_a_device_resident_volume_is_left_alone():
    """Already on the card, so there is nothing to stream."""
    events, prepared, outputs = _volume(512)
    device_events = tuple(value.cuda() for value in events)
    device_tissue = tuple(value.cuda() for value in prepared)
    with offload(["cuda"], budget_bytes=1 << 20):
        actual = _run_packed(
            device_tissue, device_events, STATES, outputs, 0, real_axis=1
        )

    assert actual.device.type == "cuda"


def test_an_empty_device_list_is_refused():
    with pytest.raises(ValueError, match="at least one device"):
        with offload([]):
            pass


def test_a_host_device_is_refused():
    """Offloading to the host is what not using this at all does."""
    with pytest.raises(ValueError, match="CUDA devices"):
        with offload(["cpu"]):
            pass


@pytest.mark.parametrize("budget, lanes", [(0, 2), (-1, 2), (1 << 20, 0)])
def test_a_nonsense_budget_is_refused(budget, lanes):
    with pytest.raises(ValueError):
        with offload(["cuda"], budget_bytes=budget, lanes=lanes):
            pass


def test_the_previous_setting_comes_back_after_a_failure():
    with pytest.raises(RuntimeError):
        with offload(["cuda"], budget_bytes=1 << 20):
            raise RuntimeError("boom")

    assert _policy._OFFLOAD is None


# The spoiler this volume declares and the voxel it winds across, without
# which a spin velocity would reach neither the dephasing nor the washout.
SPGR_CRUSHER_RAD = 8.0 * torch.pi
SPGR_VOXEL_M = 5e-4
SPGR_GEOMETRY = Geometry(
    flow_scale=SPGR_CRUSHER_RAD / SPGR_VOXEL_M, washout_scale=1.0 / SPGR_VOXEL_M
)


def _spgr_volume(voxels, trains=1, pulses=12):
    """A spoiled train over voxels, where off-resonance reaches the signal.

    An echo train refocuses b0 at the samples it records, so its b0 gradient is
    rounding noise around zero and comparing it says nothing. SPGR records at an
    echo time, so every gradient is live and worth checking.
    """
    from torchsim.sequence._builders import spgr_description

    generator = torch.Generator().manual_seed(0)
    packed = [
        _pack_events(
            spgr_description(
                torch.deg2rad(5.0 + 20.0 * torch.rand(pulses, generator=generator)),
                repetition_time_s=10e-3,
                echo_time_s=4e-3,
                phases_rad=torch.pi / 3,
                crusher_dephasing_rad=SPGR_CRUSHER_RAD,
                voxel_size_m=SPGR_VOXEL_M,
            ),
            repetitions=1,
            record="all",
            device=torch.device("cpu"),
            rf_raster_time_s=1e-6,
        )
        for _ in range(trains)
    ]

    def stack(name):
        values = [getattr(value, name) for value in packed]
        return (values[0] if trains == 1 else torch.stack(values)).contiguous()

    events = (
        stack("duration"),
        packed[0].kind,
        stack("flip"),
        stack("phase"),
        packed[0].action,
        packed[0].output_index,
        packed[0].shim_index,
        packed[0].saturation,
        packed[0].rf_frequency_hz,
    )
    tissue = TissueProperties(
        t1_ms=torch.linspace(300.0, 2000.0, voxels),
        t2_ms=torch.linspace(20.0, 200.0, voxels),
        b0_hz=torch.linspace(-150.0, 150.0, voxels),
        b1_phase_rad=torch.linspace(0.0, 0.4, voxels),
        velocity_m_per_s=torch.linspace(-0.02, 0.02, voxels),
    )
    prepared, _, _ = _prepare_tissue(tissue, torch.device("cpu"))
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return events, prepared, packed[0].output_count


def _seeds(events, prepared, index):
    tissue = tuple(
        torch.ones_like(value) if position == index else torch.zeros_like(value)
        for position, value in enumerate(prepared)
    )
    event = tuple(
        torch.zeros_like(value) for value in (events[0], events[2], events[3])
    )
    return tissue, event


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
@pytest.mark.parametrize("real_axis", [1, -1])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_forward_mode_matches_the_cpu_run(budget, real_axis, trains):
    from torchsim.sequence._accelerators import _run_packed_jvp

    events, prepared, outputs = _volume(4000, trains=trains)
    tissue_seed, event_seed = _seeds(events, prepared, 1)
    arguments = (prepared, events, tissue_seed, event_seed, STATES, outputs, 0)
    expected = _run_packed_jvp(*arguments, real_axis=real_axis)
    with offload(["cuda"], budget_bytes=budget, lanes=2):
        actual = _run_packed_jvp(*arguments, real_axis=real_axis)

    assert actual.device.type == "cpu"
    assert actual.shape == expected.shape
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def test_an_event_seed_reaches_every_chunk():
    """Event seeds are shared by all voxels, so they are replicated not sliced."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    events, prepared, outputs = _volume(4000)
    tissue_seed = tuple(torch.zeros_like(value) for value in prepared)
    event_seed = (
        torch.zeros_like(events[0]),
        torch.ones_like(events[2]),
        torch.zeros_like(events[3]),
    )
    arguments = (prepared, events, tissue_seed, event_seed, STATES, outputs, 0)
    expected = _run_packed_jvp(*arguments, real_axis=1)
    with offload(["cuda"], budget_bytes=1 << 20, lanes=2):
        actual = _run_packed_jvp(*arguments, real_axis=1)

    assert expected.abs().max() > 0
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


def _adjoint(
    events, prepared, outputs, voxels, trains, real_axis, budget, geometry=NO_GEOMETRY
):
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    tissue_seed, event_seed = _seeds(events, prepared, 1)
    generator = torch.Generator().manual_seed(7)
    shape = (trains, voxels, outputs) if trains > 1 else (voxels, outputs)
    cotangent = torch.randn(shape, generator=generator, dtype=torch.complex64)
    arguments = (
        prepared,
        events,
        (*tissue_seed, *event_seed),
        cotangent,
        STATES,
        outputs,
        0,
    )
    if budget is None:
        return _run_packed_vjp_jvp(*arguments, real_axis=real_axis, geometry=geometry)
    with offload(["cuda"], budget_bytes=budget, lanes=2):
        return _run_packed_vjp_jvp(*arguments, real_axis=real_axis, geometry=geometry)


def _compare_gradients(expected, actual):
    worst = 0.0
    for side, other in zip(expected, actual, strict=True):
        for reference, result in zip(side, other, strict=True):
            assert reference.shape == result.shape
            scale = reference.abs().max()
            if scale == 0:
                assert result.abs().max() == 0
                continue
            worst = max(worst, ((reference - result).abs().max() / scale).item())
    return worst


@pytest.mark.parametrize("budget", [2 << 20, 512 << 20])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_real_adjoint_matches_the_cpu_run(budget, trains):
    """Tissue gradients are per voxel; event gradients sum over all of them."""
    voxels = 2000
    events, prepared, outputs = _volume(voxels, trains=trains)
    expected = _adjoint(events, prepared, outputs, voxels, trains, 1, None)
    actual = _adjoint(events, prepared, outputs, voxels, trains, 1, budget)

    assert _compare_gradients(expected, actual) < 1e-4


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_complex_adjoint_matches_the_cpu_run(budget, trains):
    """On SPGR, so b0 and the RF phase are live rather than refocused away."""
    voxels = 2000
    events, prepared, outputs = _spgr_volume(voxels, trains=trains)
    expected = _adjoint(
        events, prepared, outputs, voxels, trains, -1, None, SPGR_GEOMETRY
    )
    actual = _adjoint(
        events, prepared, outputs, voxels, trains, -1, budget, SPGR_GEOMETRY
    )

    # The gradients the real-subspace kernel cannot produce, checked here.
    for index in OUTSIDE_THE_SUBSPACE:
        assert expected[0][index].abs().max() > 0
    assert _compare_gradients(expected, actual) < 1e-4


def test_the_adjoint_footprint_stays_inside_the_budget():
    """The recorded trajectory is what would otherwise blow the card open."""
    voxels = 20_000
    events, prepared, outputs = _volume(voxels)
    budget = 32 << 20
    resident = _peak_over_baseline(
        lambda: _adjoint(events, prepared, outputs, voxels, 1, 1, budget)
    )

    assert resident <= budget * 1.1


def test_the_adjoint_claims_its_buffers_once():
    """Every buffer belongs to a lane, so nothing in the chunk loop allocates.

    A hundred chunks therefore cost exactly what one costs.
    """
    voxels = 20_000
    events, prepared, outputs = _volume(voxels)

    def allocations(budget):
        _adjoint(events, prepared, outputs, voxels, 1, 1, budget)
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        _adjoint(events, prepared, outputs, voxels, 1, 1, budget)
        return torch.cuda.memory_stats()["allocation.all.allocated"] - before

    plan = _Offload((torch.device("cuda"),), 4 << 20, 2)
    shape = (1, int(events[1].numel()), outputs, STATES, 1)
    assert voxels > 50 * _chunk_voxels(plan, "adjoint", *shape)
    assert allocations(4 << 20) == allocations(1 << 30)


# --- staging a chunk on the device ---


def test_a_lane_does_not_overwrite_a_chunk_still_in_flight():
    """A staged chunk is read asynchronously, long after the host wrote it.

    Nothing orders the host's next write against that read but the lane, so
    with a busy stream the first chunk has to survive the second being staged.
    """
    lane = _Lane(torch.device("cuda"), 4096, 1, 8, STATES, 1)
    first = torch.full((4096, 8), 1 + 1j, dtype=torch.complex64)
    second = torch.full((4096, 8), 2 + 2j, dtype=torch.complex64)
    ballast = torch.randn((1024, 1024), device="cuda")
    with torch.cuda.stream(lane.stream):
        for _ in range(40):
            ballast = ballast @ ballast
        landed = lane.send(first)
        snapshot = landed.clone()
        lane.send(second)
    torch.cuda.current_stream().wait_stream(lane.stream)
    torch.cuda.synchronize()

    assert snapshot.eq(1 + 1j).all()


def test_lanes_stage_through_buffers_of_their_own():
    """Two lanes run at once, so one buffer between them would be a race."""
    lanes = [_Lane(torch.device("cuda"), 256, 1, 8, STATES, 1) for _ in range(2)]
    piece = torch.zeros((256, 8), dtype=torch.complex64)
    for lane in lanes:
        with torch.cuda.stream(lane.stream):
            lane.send(piece)
    torch.cuda.synchronize()

    assert lanes[0].staging.data_ptr() != lanes[1].staging.data_ptr()


def test_a_pass_that_only_brings_signals_home_stages_nothing():
    """Pinned memory is expensive to claim; the forward path never needs it."""
    lane = _Lane(torch.device("cuda"), 256, 1, 8, STATES, 1)

    assert lane.staging is None


def _shim_volume(voxels, trains=1):
    """A spoiled train whose pulses are split across two transmit shims.

    On SPGR for the reason ``_spgr_volume`` gives, and because a shim is a
    thing you would want to design against a gradient-echo readout. Streaming
    cuts the voxel axis, which for the transmit buffers is the last axis
    rather than the whole buffer, so both rows have to stay whole and a chunk
    has to be a slice of each.
    """
    events, prepared, outputs = _spgr_volume(voxels, trains=trains)
    rows = torch.zeros_like(events[6])
    pulses = (events[1] == 1).nonzero().flatten()
    # Every second pulse on the other shim, so both rows drive something.
    rows[pulses[1::2]] = 1
    events = (*events[:6], rows.contiguous(), *events[7:])

    ramp = torch.linspace(0.0, 0.3, voxels)
    tissue = list(prepared)
    tissue[3] = torch.cat((prepared[3], 0.7 + ramp)).contiguous()
    tissue[4] = torch.cat((prepared[4], 0.4 - ramp)).contiguous()
    return events, tuple(tissue), outputs


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_shimmed_volume_matches_the_whole_one(budget, trains):
    voxels = 4000
    events, prepared, outputs = _shim_volume(voxels, trains=trains)
    assert prepared[3].numel() == 2 * voxels

    arguments = dict(real_axis=-1, geometry=SPGR_GEOMETRY)
    expected = _run_packed(prepared, events, STATES, outputs, 0, **arguments)
    with offload(["cuda"], budget_bytes=budget, lanes=2):
        actual = _run_packed(prepared, events, STATES, outputs, 0, **arguments)

    assert actual.device.type == "cpu"
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
def test_a_streamed_shimmed_forward_mode_matches_the_whole_one(budget):
    from torchsim.sequence._accelerators import _run_packed_jvp

    events, prepared, outputs = _shim_volume(4000)
    # Seed the transmit magnitude, so the tangent travels the shim rows too.
    tissue_seed, event_seed = _seeds(events, prepared, 3)
    arguments = (prepared, events, tissue_seed, event_seed, STATES, outputs, 0)
    expected = _run_packed_jvp(*arguments, real_axis=-1, geometry=SPGR_GEOMETRY)
    with offload(["cuda"], budget_bytes=budget, lanes=2):
        actual = _run_packed_jvp(*arguments, real_axis=-1, geometry=SPGR_GEOMETRY)

    assert expected.abs().max() > 0
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_shimmed_adjoint_matches_the_whole_one(budget, trains):
    """A gradient row is a whole volume; a chunk writes a slice of each."""
    voxels = 2000
    events, prepared, outputs = _shim_volume(voxels, trains=trains)
    expected = _adjoint(
        events, prepared, outputs, voxels, trains, -1, None, SPGR_GEOMETRY
    )
    actual = _adjoint(
        events, prepared, outputs, voxels, trains, -1, budget, SPGR_GEOMETRY
    )

    for side in expected:
        assert side[3].numel() == 2 * voxels
    for index in OUTSIDE_THE_SUBSPACE:
        assert expected[0][index].abs().max() > 0
    assert _compare_gradients(expected, actual) < 1e-4


@pytest.mark.parametrize("kind", ["forward", "jvp", "adjoint"])
def test_a_shim_widens_what_a_voxel_costs_the_budget(kind):
    """Chunking has to know, or the transmit rows overrun the lane buffers."""
    shape = (1, 40, 20, STATES, None)
    one = _bytes_per_voxel(kind, *shape, 1)
    four = _bytes_per_voxel(kind, *shape, 4)

    # Three extra rows for each of the two transmit buffers, and for a pass
    # that also carries tangents or gradients, again for each of those.
    copies = 1 if kind == "forward" else 2 if kind == "jvp" else 4
    assert four - one == copies * 2 * 3 * 4


# --- the first-order adjoint, which autograd reaches instead ---


def _inside_the_subspace():
    """``wanted`` with the four gradients a real adjoint cannot produce left
    out, which is what lets it be chosen at all.
    """
    return tuple(
        position not in OUTSIDE_THE_SUBSPACE for position in range(len(_FLOAT_INPUTS))
    )


def _only_wanted(gradients, wanted):
    """The gradients the caller declared it would read.

    A real-subspace adjoint returns zero for the rest, which is what asking
    for fewer of them buys; a host run computes them all regardless.
    """
    if wanted is None:
        return gradients
    return tuple(
        gradient for gradient, asked in zip(gradients, wanted, strict=True) if asked
    )


def _first_order_adjoint(
    events,
    prepared,
    outputs,
    voxels,
    trains,
    wanted,
    budget,
    geometry=NO_GEOMETRY,
):
    """The route ``torch.autograd`` takes for a plain ``.backward()``."""
    from torchsim.sequence._accelerators import _run_packed_vjp

    generator = torch.Generator().manual_seed(7)
    shape = (trains, voxels, outputs) if trains > 1 else (voxels, outputs)
    cotangent = torch.randn(shape, generator=generator, dtype=torch.complex64)
    arguments = (prepared, events, cotangent, STATES, outputs, 0, wanted)
    if budget is None:
        return _run_packed_vjp(*arguments, geometry=geometry)
    with offload(["cuda"], budget_bytes=budget, lanes=2):
        return _run_packed_vjp(*arguments, geometry=geometry)


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_first_order_adjoint_matches_the_cpu_run(budget, trains):
    """Tissue gradients are per voxel; event gradients sum over all of them."""
    voxels = 2000
    events, prepared, outputs = _volume(voxels, trains=trains)
    wanted = _inside_the_subspace()
    expected = _first_order_adjoint(
        events, prepared, outputs, voxels, trains, wanted, None
    )
    actual = _first_order_adjoint(
        events, prepared, outputs, voxels, trains, wanted, budget
    )

    # Leaving four gradients out is what lets the streamed pass take the
    # real-subspace kernel, which returns zero for exactly those.
    for index in OUTSIDE_THE_SUBSPACE:
        assert actual[index].abs().max() == 0
    worst = _compare_gradients(
        (_only_wanted(expected, wanted),), (_only_wanted(actual, wanted),)
    )
    assert worst < 1e-4


@pytest.mark.parametrize("budget", [1 << 20, 512 << 20])
@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_first_order_complex_adjoint_matches_the_cpu_run(budget, trains):
    """On SPGR, so b0 and the RF phase are live rather than refocused away."""
    voxels = 2000
    events, prepared, outputs = _spgr_volume(voxels, trains=trains)
    expected = _first_order_adjoint(
        events, prepared, outputs, voxels, trains, None, None, SPGR_GEOMETRY
    )
    actual = _first_order_adjoint(
        events, prepared, outputs, voxels, trains, None, budget, SPGR_GEOMETRY
    )

    # The gradients the real-subspace kernel cannot produce, checked here.
    for index in OUTSIDE_THE_SUBSPACE:
        assert expected[index].abs().max() > 0
    assert _compare_gradients((expected,), (actual,)) < 1e-4


def test_the_first_order_adjoint_footprint_stays_inside_the_budget():
    """Only a chunked pass fits; a host run or a resident card would not."""
    voxels = 20_000
    events, prepared, outputs = _volume(voxels)
    budget = 32 << 20
    resident = _peak_over_baseline(
        lambda: _first_order_adjoint(
            events, prepared, outputs, voxels, 1, _inside_the_subspace(), budget
        )
    )

    assert resident <= budget * 1.1


def test_a_host_resident_first_order_adjoint_follows_the_execution_target():
    """The forward moves to the card under this policy, so the backward has to
    move with it rather than stay behind.
    """
    from torchsim.sequence import execution

    voxels = 20_000
    events, prepared, outputs = _volume(voxels)
    wanted = _inside_the_subspace()
    expected = _first_order_adjoint(events, prepared, outputs, voxels, 1, wanted, None)
    with execution("cuda"):
        actual = _first_order_adjoint(
            events, prepared, outputs, voxels, 1, wanted, None
        )

    assert all(value.device.type == "cpu" for value in actual)
    worst = _compare_gradients(
        (_only_wanted(expected, wanted),), (_only_wanted(actual, wanted),)
    )
    assert worst < 1e-4


def test_a_host_adjoint_with_no_policy_stays_on_the_host_kernel():
    """The forward-over-reverse kernel carries a direction the first-order
    pass has no use for, so reaching the devices through it must not become
    what an unqualified call does.
    """
    voxels = 64
    events, prepared, outputs = _volume(voxels)
    taken = []
    forwarded = accelerators._run_packed_vjp_jvp

    def spy(*arguments, **options):
        taken.append(True)
        return forwarded(*arguments, **options)

    accelerators._run_packed_vjp_jvp = spy
    try:
        _first_order_adjoint(
            events, prepared, outputs, voxels, 1, _inside_the_subspace(), None
        )
    finally:
        accelerators._run_packed_vjp_jvp = forwarded

    assert not taken


def test_a_backward_through_the_public_api_streams():
    """The level the gap was reported at.

    ``torch.autograd`` reaches the adjoint by a route of its own, and it is
    the one an ordinary user's ``.backward()`` takes.
    """
    from torchsim.sequence import TissueProperties

    voxels = 20_000
    reached = []
    forwarded = accelerators._run_offloaded_vjp
    second_order = []
    around = accelerators._run_offloaded_vjp_jvp

    def spy(*arguments, **options):
        reached.append(True)
        return forwarded(*arguments, **options)

    def watch(*arguments, **options):
        second_order.append(True)
        return around(*arguments, **options)

    def gradients():
        t1 = torch.linspace(600.0, 1400.0, voxels, requires_grad=True)
        t2 = torch.linspace(40.0, 120.0, voxels, requires_grad=True)
        signal = (
            EpgEngine()
            .simulate(
                fse_description(
                    torch.deg2rad(torch.full((ECHOES,), 120.0)),
                    echo_spacing_s=5e-3,
                    phases_rad=torch.pi / 2,
                    excitation_phase_rad=torch.pi / 2,
                ),
                TissueProperties(t1_ms=t1, t2_ms=t2),
                nstates=STATES,
            )
            .signal
        )
        signal.abs().square().sum().backward()
        return t1.grad, t2.grad

    expected = gradients()
    accelerators._run_offloaded_vjp = spy
    accelerators._run_offloaded_vjp_jvp = watch
    try:
        with offload(["cuda"], budget_bytes=1 << 24):
            actual = gradients()
    finally:
        accelerators._run_offloaded_vjp = forwarded
        accelerators._run_offloaded_vjp_jvp = around

    assert reached, "the backward never reached the streamed route"
    # And took the kernel written for it, rather than the pass that carries a
    # forward direction nobody asked for.
    assert not second_order
    assert _compare_gradients((expected,), (actual,)) < 1e-4


@pytest.mark.parametrize("trains", [1, 3])
def test_a_streamed_first_order_adjoint_takes_its_own_kernel(trains):
    """A chunk of a first-order adjoint is a first-order adjoint, so streaming
    must not cost the kernel written for it.
    """
    from torchsim.sequence import _accelerators as accelerators

    voxels = 2000
    events, prepared, outputs = _volume(voxels, trains=trains)
    wanted = _inside_the_subspace()
    second_order = []
    around = accelerators._run_offloaded_vjp_jvp

    def watch(*arguments, **options):
        second_order.append(True)
        return around(*arguments, **options)

    accelerators._run_offloaded_vjp_jvp = watch
    try:
        streamed = _first_order_adjoint(
            events, prepared, outputs, voxels, trains, wanted, 1 << 20
        )
    finally:
        accelerators._run_offloaded_vjp_jvp = around

    assert not second_order
    assert max(float(value.abs().max()) for value in streamed) > 0.0


def test_streaming_a_first_order_adjoint_makes_wider_chunks():
    """Half the trajectory is the second saving: for one budget the chunks come
    out wider than the pass carrying a forward direction would allow.
    """
    from torchsim.sequence._accelerators import _bytes_per_voxel

    shape = (4, 40, 8, 16, None, 1)
    around = _bytes_per_voxel("adjoint", *shape)
    direct = _bytes_per_voxel("first-order adjoint", *shape)

    assert direct < around
    assert around / direct > 1.8
