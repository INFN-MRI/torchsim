"""Spreading echo trains over several CUDA devices.

Sharding has to be invisible: the same call, split any number of ways, must
give back the same answer in the same shape. The partitioning itself is checked
on its own, because it is pure arithmetic and does not need a second GPU to be
wrong.

This machine has one GPU, so the execution tests name ``cuda:0`` repeatedly.
That exercises the partition, the per-shard slicing, the gather and the
gradient reduction; it does not exercise a real transfer between two devices.
"""

import pytest
import torch

from torchsim.sequence import distribute
from torchsim.sequence._accelerators import (
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp_jvp,
    _shard_bounds,
)
import torchsim.sequence._accelerators as accelerators

from test_cuda_parity import (  # noqa: E402
    _real_case,
    _seeds,
    _spgr_case,
    _worst_disagreement,
)

STATES = 10
TRAINS = 17
ATOMS = 5


@pytest.fixture
def devices(monkeypatch):
    """Pretend to have as many devices as a test asks for."""

    def use(count):
        named = tuple(torch.device("cuda:0") for _ in range(count))
        monkeypatch.setattr(accelerators, "_DEVICES", named)

    return use


@pytest.mark.parametrize(
    "train_count, count, expected",
    [
        (7, 2, [(0, 4), (4, 7)]),
        (8, 3, [(0, 3), (3, 6), (6, 8)]),
        (8, 2, [(0, 4), (4, 8)]),
        (2, 4, [(0, 1), (1, 2)]),
        (1, 4, []),
        (17, 1, []),
    ],
)
def test_the_partition_covers_every_train_once(
    devices, train_count, count, expected
):
    """More devices than trains must not produce empty spans."""
    devices(count)
    bounds = [(begin, end) for begin, end, _ in _shard_bounds(train_count)]

    assert bounds == expected
    if bounds:
        assert bounds[0][0] == 0
        assert bounds[-1][1] == train_count
        for (_, end), (begin, _) in zip(bounds, bounds[1:], strict=False):
            assert end == begin


def test_a_single_device_is_left_alone(devices):
    """Nothing to spread over, so nothing should be sliced or copied."""
    devices(1)
    assert _shard_bounds(64) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("count", [2, 3, 4, TRAINS])
def test_a_sharded_forward_matches_the_whole_one(count):
    """Bitwise: the trains are independent, so nothing should reassociate."""
    events, prepared, outputs = _real_case(TRAINS, "cuda", atoms=ATOMS)
    expected = _run_packed(prepared, events, STATES, outputs, 1)
    with distribute(["cuda:0"] * count):
        actual = _run_packed(prepared, events, STATES, outputs, 1)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_shard_holding_one_train_keeps_the_train_axis():
    """A single-train call returns no train axis; a single-train shard must."""
    events, prepared, outputs = _real_case(TRAINS, "cuda", atoms=ATOMS)
    with distribute(["cuda:0"] * TRAINS):
        actual = _run_packed(prepared, events, STATES, outputs, 1)

    assert actual.shape == (TRAINS, ATOMS, outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("count", [2, 4])
def test_a_sharded_forward_mode_matches_the_whole_one(count):
    """The seeds split with the events they belong to."""
    events, prepared, outputs = _real_case(TRAINS, "cuda", atoms=ATOMS)
    tissue_seed, event_seed = _seeds(events, prepared, tissue_index=1)
    expected = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, outputs, 1
    )
    with distribute(["cuda:0"] * count):
        actual = _run_packed_jvp(
            prepared, events, tissue_seed, event_seed, STATES, outputs, 1
        )

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def _second_order(events, prepared, outputs, trains, atoms, real_axis):
    tissue_seed, event_seed = _seeds(events, prepared, tissue_index=1)
    generator = torch.Generator().manual_seed(7)
    cotangent = torch.randn(
        (trains, atoms, outputs), generator=generator, dtype=torch.complex64
    )
    return _run_packed_vjp_jvp(
        prepared,
        events,
        (*tissue_seed, *event_seed),
        cotangent.to(prepared[0].device),
        state_count=STATES,
        output_count=outputs,
        threads=1,
        real_axis=real_axis,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("real_axis", [1, -1])
def test_a_sharded_adjoint_matches_the_whole_one(count, real_axis):
    """Tissue gradients are partial sums per shard; event gradients are pieces."""
    if real_axis == 1:
        events, prepared, outputs = _real_case(TRAINS, "cuda", atoms=ATOMS)
    else:
        events, prepared, outputs = _spgr_case("cuda", TRAINS, ATOMS)
    expected = _second_order(events, prepared, outputs, TRAINS, ATOMS, real_axis)
    with distribute(["cuda:0"] * count):
        actual = _second_order(events, prepared, outputs, TRAINS, ATOMS, real_axis)

    for side, other in zip(expected, actual, strict=True):
        for reference, result in zip(side, other, strict=True):
            assert reference.shape == result.shape
    assert _worst_disagreement(expected, actual) < 1e-5


def test_an_empty_device_list_is_refused():
    """Silently running everywhere would hide a caller's mistake."""
    with pytest.raises(ValueError, match="at least one device"):
        with distribute([]):
            pass


def test_a_host_device_is_refused():
    """The CPU kernels have their own thread pool and do not shard this way."""
    with pytest.raises(ValueError, match="CUDA devices"):
        with distribute(["cpu"]):
            pass


def test_the_previous_setting_comes_back_after_a_failure():
    """A raising body must not leave later calls sharding."""
    with pytest.raises(RuntimeError):
        with distribute(["cuda:0", "cuda:0"]):
            raise RuntimeError("boom")

    assert accelerators._DEVICES == ()


def test_blocks_nest():
    """An inner block restores the outer one, not the empty default."""
    with distribute(["cuda:0", "cuda:0"]):
        outer = accelerators._DEVICES
        with distribute(["cuda:0"]):
            assert len(accelerators._DEVICES) == 1
        assert accelerators._DEVICES == outer
    assert accelerators._DEVICES == ()
