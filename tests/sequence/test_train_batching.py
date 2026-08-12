"""A batch of echo trains must behave exactly like the trains run one by one."""

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence._accelerators import _pack_events, _run_packed, _run_packed_vjp
from torchsim.sequence._simulation import _prepare_tissue

ECHO_SPACING_S = 5e-3
PHASES_RAD = torch.pi / 2


def _tissue() -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]), t2_ms=torch.tensor([45.0, 120.0])
    )


def _schedules(trains: int, echoes: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    degrees = 80.0 + 80.0 * torch.rand(trains, echoes, generator=generator)
    return torch.deg2rad(degrees)


def _describe(flip: torch.Tensor):
    return fse_description(flip, echo_spacing_s=ECHO_SPACING_S, phases_rad=PHASES_RAD)


def _pack(flip: torch.Tensor):
    return _pack_events(
        "fse",
        _describe(flip),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )


def _buffers(packed):
    return (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )


def test_packing_matches_stacked_single_trains():
    flip = _schedules(5, 12)
    batched = _pack(flip)
    singles = [_pack(flip[train]) for train in range(flip.shape[0])]

    assert batched.train_count == 5
    assert singles[0].train_count == 1
    for name in ("duration", "flip", "phase"):
        expected = torch.stack([getattr(single, name) for single in singles])
        assert torch.equal(getattr(batched, name), expected)
    # Structure is shared, so it never grows a train axis.
    for name in ("kind", "action", "output_index"):
        assert torch.equal(getattr(batched, name), getattr(singles[0], name))


def test_forward_matches_single_trains():
    flip = _schedules(6, 12)
    tissue = _tissue()
    expected = torch.stack(
        [FSE().simulate(_describe(row), tissue).signal for row in flip]
    )
    actual = FSE().simulate(_describe(flip), tissue).signal
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_event_gradients_stay_per_train():
    """Each train's gradient must depend only on that train's flip angles."""
    flip = _schedules(4, 12)
    tissue = _tissue()

    first = flip.clone().requires_grad_(True)
    FSE().simulate(_describe(first), tissue).signal.abs().sum().backward()

    moved = flip.clone()
    moved[2] += 0.3
    second = moved.requires_grad_(True)
    FSE().simulate(_describe(second), tissue).signal.abs().sum().backward()

    assert first.grad.shape == flip.shape
    for train in (0, 1, 3):
        assert torch.equal(first.grad[train], second.grad[train])
    assert not torch.equal(first.grad[2], second.grad[2])


def test_tissue_gradients_sum_over_trains():
    """Tissue parameters are shared, so their gradient accumulates over trains."""
    flip = _schedules(4, 12)
    prepared, _, _ = _prepare_tissue(_tissue(), "cpu")
    singles = [_buffers(_pack(row)) for row in flip]
    batched = _buffers(_pack(flip))
    outputs = int(singles[0][5].max()) + 1
    seed = torch.randn((flip.shape[0], prepared[0].numel(), outputs), dtype=torch.complex64)

    per_train = [
        _run_packed_vjp(
            prepared, events, seed[train], state_count=10, output_count=outputs, threads=1
        )
        for train, events in enumerate(singles)
    ]
    fused = _run_packed_vjp(
        prepared, batched, seed, state_count=10, output_count=outputs, threads=1
    )

    for index in range(7):  # tissue gradients
        assert torch.equal(fused[index], sum(g[index] for g in per_train))
    for index in (7, 8, 9):  # duration, flip, phase stay per train
        assert torch.equal(fused[index], torch.stack([g[index] for g in per_train]))


def _vjp(prepared, events, seed, outputs, threads):
    return _run_packed_vjp(
        prepared, events, seed, state_count=10, output_count=outputs, threads=threads
    )


@pytest.mark.parametrize("threads", [1, 2, 4, 0])
def test_batched_reduction_is_scheduling_independent(threads):
    """At a fixed thread count the result must not depend on scheduling.

    Workers accumulate into private buffers that are summed in ascending thread
    order, so which worker finishes first cannot change the answer.
    """
    flip = _schedules(8, 12)
    prepared, _, _ = _prepare_tissue(_tissue(), "cpu")
    events = _buffers(_pack(flip))
    outputs = int(events[5].max()) + 1
    seed = torch.randn((8, prepared[0].numel(), outputs), dtype=torch.complex64)

    reference = _vjp(prepared, events, seed, outputs, threads)
    for _ in range(5):
        for expected, got in zip(
            reference, _vjp(prepared, events, seed, outputs, threads), strict=True
        ):
            assert torch.equal(expected, got)


@pytest.mark.parametrize("threads", [2, 4, 8, 0])
def test_thread_count_only_reassociates(threads):
    """Changing the thread count regroups the sum, so it may move by an ulp."""
    flip = _schedules(8, 12)
    prepared, _, _ = _prepare_tissue(_tissue(), "cpu")
    events = _buffers(_pack(flip))
    outputs = int(events[5].max()) + 1
    seed = torch.randn((8, prepared[0].numel(), outputs), dtype=torch.complex64)

    reference = _vjp(prepared, events, seed, outputs, 1)
    actual = _vjp(prepared, events, seed, outputs, threads)
    for expected, got in zip(reference, actual, strict=True):
        scale = expected.abs().max().clamp_min(1e-30)
        assert ((expected - got).abs().max() / scale) < 1e-6


def test_mismatched_train_widths_are_rejected():
    """A width mismatch would stride out of bounds inside the kernel."""
    flip = _schedules(3, 12)
    prepared, _, _ = _prepare_tissue(_tissue(), "cpu")
    duration, kind, flips, phase, action, output_index = _buffers(_pack(flip))
    broken = (duration[0], kind, flips, phase, action, output_index)
    with pytest.raises(ValueError, match="disagree on train count"):
        _run_packed(prepared, broken, 10, int(output_index.max()) + 1, 1)
