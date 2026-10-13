"""Cutting a pass into chunks of voxels and dealing them to several devices.

Every route to the cards goes through one loop: a chunk is the same pass over
a smaller volume, what has a voxel axis is cut along it, what every voxel
shares is moved once per device, and the answers are put back in place or
added up. The loop is held here on the host, where the devices are stood in
for by host lanes, so that every physics a pass can carry -- shims, slice
tables, per-voxel rotations, semisolid and exchanging pools, tabulated pools --
is held to the whole volume on any machine. The card's kernels are held the
same way through their host build.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from test_offload import (
    SPGR_GEOMETRY,
    _seeds,
    _shim_volume,
    _spgr_volume,
    _volume,
)

import blochsim._execution as _policy
from blochsim._execution import _Choice
from blochsim.sequence import _accelerators, _pools
from blochsim.sequence._accelerators import (
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp,
    _run_packed_vjp_jvp,
    _tables,
)
from blochsim.sequence._lineshape import lineshape_table
from blochsim.sequence._transition import DynamicPairs

STATES = 10
PASSES = ("forward", "jvp", "vjp", "vjp_jvp")
HOST = torch.device("cpu")

card_build = pytest.mark.skipif(
    importlib.util.find_spec("blochsim._gpu_host") is None,
    reason="the host build of the card's kernels is Linux only",
)


@pytest.fixture
def chunked(monkeypatch):
    """Route every pass through the chunk loop, ``width`` voxels at a time.

    A chunk runs as written, so the routing steps aside while one is in
    flight, as it does on the cards. ``card`` runs each chunk on the card's
    kernels, through their host build.
    """

    def force(width: int, devices: int = 2, card: bool = False) -> None:
        choice = _Choice((HOST,) * devices, width)

        def route(*_arguments, **_options):
            return None if _policy.policy() is None else choice

        monkeypatch.setattr(_accelerators, "_choose", route)
        if card:
            monkeypatch.setattr(_accelerators, "_on_card", lambda device: True)

    return force


def _flat(answer):
    if isinstance(answer, torch.Tensor):
        return [answer]
    return [value for part in answer for value in _flat(part)]


def _agree(expected, actual, tolerance=1e-5):
    expected, actual = _flat(expected), _flat(actual)
    assert len(expected) == len(actual)
    for reference, result in zip(expected, actual, strict=True):
        assert result.shape == reference.shape
        assert result.device == reference.device
        scale = float(reference.abs().max()) if reference.numel() else 0.0
        if scale == 0.0:
            assert float(result.abs().max() if result.numel() else 0.0) == 0.0
            continue
        assert float((reference - result).abs().max()) / scale < tolerance


def _cotangent(trains, voxels, outputs, seed=7):
    generator = torch.Generator().manual_seed(seed)
    shape = (trains, voxels, outputs) if trains > 1 else (voxels, outputs)
    return torch.randn(shape, generator=generator, dtype=torch.complex64)


def _pass(name, tissue, events, outputs, trains, **physics):
    """One pass over a volume, as the dispatcher would be asked for it."""
    voxels = tissue[0].numel()
    seeds, event_seeds = _seeds(events, tissue, 0)
    direction = physics.pop("dynamic_direction", None)
    if name == "forward":
        return _run_packed(tissue, events, STATES, outputs, 1, **physics)
    if name == "jvp":
        return _run_packed_jvp(
            tissue,
            events,
            seeds,
            event_seeds,
            STATES,
            outputs,
            1,
            dynamic_direction=direction,
            **physics,
        )
    cotangent = _cotangent(trains, voxels, outputs)
    if name == "vjp":
        return _run_packed_vjp(tissue, events, cotangent, STATES, outputs, 1, **physics)
    return _run_packed_vjp_jvp(
        tissue,
        events,
        (*seeds, *event_seeds),
        cotangent,
        STATES,
        outputs,
        1,
        dynamic_direction=direction,
        **physics,
    )


def _volumes(voxels=50):
    return {
        "echo train": lambda trains: (*_volume(voxels, trains), {}),
        "spoiled": lambda trains: (
            *_spgr_volume(voxels, trains),
            {"geometry": SPGR_GEOMETRY},
        ),
        "shimmed": lambda trains: (
            *_shim_volume(voxels, trains),
            {"geometry": SPGR_GEOMETRY},
        ),
    }


@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("volume", sorted(_volumes()))
@pytest.mark.parametrize("trains", [1, 3])
@pytest.mark.parametrize(("width", "devices"), [(1, 1), (7, 2), (64, 2)])
def test_a_chunked_pass_matches_the_whole_volume(
    chunked, name, volume, trains, width, devices
):
    """Widths that do not divide the volume, one voxel at a time, and one
    chunk covering it all, on one device and dealt between two."""
    events, tissue, outputs, physics = _volumes()[volume](trains)
    expected = _pass(name, tissue, events, outputs, trains, **dict(physics))
    chunked(width, devices)
    actual = _pass(name, tissue, events, outputs, trains, **dict(physics))

    _agree(expected, actual)


# --- the physics a voxel carries beside its tissue ---


def _tabulated(pools: str, rotation: str, following: bool):
    """A tissue of tabulated pools, and the physics it is simulated with."""
    from sequence.test_many_pools import (
        _PAIR_ROW,
        GEOMETRY,
        OUTPUTS,
        _case,
        _events,
        _extras,
        _instantaneous_table,
        _tissue,
    )

    exchanging, semisolid = int(pools[0]), pools[1] == "s"
    leaves = _case(exchanging, semisolid, rotation)
    tissue = tuple(value.float().contiguous() for value in _tissue(leaves))
    events = _events(leaves, rotation)
    values, index, layout = _pools.tabulate(
        tissue,
        _extras(leaves["pools"]) if exchanging > 1 else None,
        events[0],
        exchanging=exchanging,
        semisolid=semisolid,
        slopes=True,
    )
    generator = torch.Generator().manual_seed(5)
    dynamic = None
    direction = None
    if "pairs" in leaves:
        dynamic = DynamicPairs.from_packed(
            leaves["pairs"], torch.tensor(_PAIR_ROW, dtype=torch.int32)
        )
        if following:
            direction = torch.randn(leaves["pairs"].shape, generator=generator) * 0.1
    tables = _pools.PoolTables(
        layout=layout,
        values=values,
        index=index,
        direction=(
            torch.randn(values.shape, generator=generator) * 0.01 if following else None
        ),
    )
    physics = dict(
        geometry=GEOMETRY,
        profile=_tables(
            _instantaneous_table() if rotation == "profile" else None, events, dynamic
        ),
        lineshape=lineshape_table() if semisolid else None,
        exchanging=layout,
        dynamic=dynamic,
        pools=tables,
    )
    if direction is not None:
        physics["dynamic_direction"] = direction
    return tissue, events, OUTPUTS, physics


@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize(
    ("pools", "rotation"),
    [("2f", "instant"), ("3s", "profile"), ("4s", "dynamic"), ("4s", "shimmed")],
)
@pytest.mark.parametrize("width", [1, 2])
def test_tabulated_pools_chunk_like_the_whole_volume(
    chunked, name, pools, rotation, width
):
    """Exchanging and semisolid pools, a slice table, a rotation per voxel and
    a transmit row per shim: every table a voxel carries is cut with it."""
    following = name in ("jvp", "vjp_jvp")
    tissue, events, outputs, physics = _tabulated(pools, rotation, following)
    expected = _pass(name, tissue, events, outputs, 1, **dict(physics))
    chunked(width)
    actual = _pass(name, tissue, events, outputs, 1, **dict(physics))

    _agree(expected, actual)


def _profiled(locations=3, voxels=8):
    from blochsim.sequence._transition import transition_table
    from sequence.test_slice_table import BANDWIDTH, RASTER, _packed, _shaped

    table = transition_table(
        _shaped(BANDWIDTH),
        torch.linspace(-0.5, 0.5, locations),
        bins=64,
        rf_raster_time_s=RASTER,
    )
    tissue, events, outputs = _packed(voxels * locations)
    return tissue, events, outputs, {"profile": table}


@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("width", [1, 4, 7])
def test_a_chunk_holds_whole_voxels_when_a_table_is_read(chunked, name, width):
    """The kernel takes a voxel's slice position from its index within the
    chunk, so a width that is not a multiple of the positions is rounded to
    one, never cut inside a voxel's copies."""
    tissue, events, outputs, physics = _profiled()
    expected = _pass(name, tissue, events, outputs, 1, **dict(physics))
    chunked(width)
    actual = _pass(name, tissue, events, outputs, 1, **dict(physics))

    _agree(expected, actual)


@pytest.mark.parametrize("name", ["forward", "vjp"])
def test_a_rotation_per_voxel_is_cut_with_its_voxels(chunked, name):
    """The pair's cotangent comes back per voxel, so its chunks are put back
    in place rather than added up."""
    from sequence.test_dynamic_transmit import ECHOES, _train

    _, tissue, events, pairs = _train()
    expected = _pass(name, tissue, events, ECHOES, 1, dynamic=pairs)
    chunked(2)
    actual = _pass(name, tissue, events, ECHOES, 1, dynamic=pairs)

    _agree(expected, actual)


# --- the card's kernels, through their host build ---


@card_build
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("volume", sorted(_volumes()))
def test_the_card_kernels_chunk_like_one_launch(chunked, name, volume):
    """Each chunk takes the card's kernel, whose tissue gradients are per
    voxel whatever the tissue buffer held."""
    events, tissue, outputs, physics = _volumes(12)[volume](3)
    chunked(12, devices=1, card=True)
    expected = _pass(name, tissue, events, outputs, 3, **dict(physics))
    chunked(5, card=True)
    actual = _pass(name, tissue, events, outputs, 3, **dict(physics))

    _agree(expected, actual, 1e-4)


@card_build
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize(("pools", "rotation"), [("3s", "profile"), ("4s", "dynamic")])
def test_the_card_kernels_chunk_tabulated_pools_like_one_launch(
    chunked, name, pools, rotation
):
    following = name in ("jvp", "vjp_jvp")
    tissue, events, outputs, physics = _tabulated(pools, rotation, following)
    chunked(3, devices=1, card=True)
    expected = _pass(name, tissue, events, outputs, 1, **dict(physics))
    chunked(1, card=True)
    actual = _pass(name, tissue, events, outputs, 1, **dict(physics))

    _agree(expected, actual, 1e-4)


# --- the default policy ---


def test_by_default_a_small_problem_runs_as_written():
    events, tissue, outputs = _volume(8)
    assert _accelerators._choose("forward", tissue, events, outputs, STATES, 1) is None


def test_by_default_a_host_problem_stays_on_a_machine_without_a_card(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    events, tissue, outputs = _volume(50_000)

    assert _accelerators._choose("forward", tissue, events, outputs, STATES, 1) is None


def test_a_gradient_through_a_simulator_runs_where_its_tensors_are(monkeypatch):
    """A design loop differentiates a small problem many times; it is never
    moved between devices on each step, whatever the default would do."""
    from blochsim.simulators import FSESimulator

    seen = []
    monkeypatch.setattr(
        _accelerators,
        "_choose",
        lambda *arguments, **options: seen.append(_policy.policy()),
    )
    flip = torch.full((8,), 150.0, dtype=torch.float64, requires_grad=True)
    FSESimulator(ESP=5.0, T1=1000.0, T2=80.0).simulate(flip=flip).abs().sum().backward()

    assert seen and all(policy is None for policy in seen)
    assert flip.grad is not None


def test_no_device_is_used_for_a_volume_with_no_voxels():
    assert (
        _policy.choose(
            work=10**9,
            voxels=0,
            bytes_per_voxel=1,
            crossover=1.0,
            home=HOST,
        )
        is None
    )


@pytest.mark.parametrize("devices", [1, 2, 3])
def test_a_mapping_is_dealt_between_devices_and_put_back_in_order(devices):
    values = torch.arange(37.0)[:, None] * torch.ones(1, 4)
    choice = _Choice((HOST,) * devices, 5)
    doubled, total = _policy.in_chunks(
        choice,
        37,
        HOST,
        lambda begin, end, device: _policy.cut(values, 0, begin, end, device),
        lambda chunk, device: (2 * chunk, chunk.sum(0)),
        lambda position, answer, width: 0 if position == 0 else None,
    )

    assert torch.equal(doubled, 2 * values)
    assert torch.equal(total, values.sum(0))
