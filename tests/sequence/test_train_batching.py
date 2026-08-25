"""A batch of echo trains must behave exactly like the trains run one by one.

The comparisons here are bitwise, so both arms have to take the same kernel.
A batch is several times the work of one train and the subspace verdict is
reached for by size, so the threshold is pinned rather than relied on.
"""

import pytest
import torch

from torchsim import EpgEngine, fse_description, TissueProperties
from torchsim.sequence._accelerators import _pack_events, _run_packed, _run_packed_vjp
from torchsim.sequence._parameters import TISSUE_COUNT
from torchsim.sequence._simulation import _prepare_tissue

# Where duration, flip and phase land in a gradient tuple.
_EVENT_GRADIENTS = (TISSUE_COUNT, TISSUE_COUNT + 1, TISSUE_COUNT + 2)

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
                _describe(flip),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )


def _buffers(packed):
    return packed.buffers


def test_packing_matches_stacked_single_trains(always_worth_detecting):
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


def test_forward_matches_single_trains(always_worth_detecting):
    flip = _schedules(6, 12)
    tissue = _tissue()
    expected = torch.stack(
        [EpgEngine().simulate(_describe(row), tissue).signal for row in flip]
    )
    actual = EpgEngine().simulate(_describe(flip), tissue).signal
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_event_gradients_stay_per_train(always_worth_detecting):
    """Each train's gradient must depend only on that train's flip angles."""
    flip = _schedules(4, 12)
    tissue = _tissue()

    first = flip.clone().requires_grad_(True)
    EpgEngine().simulate(_describe(first), tissue).signal.abs().sum().backward()

    moved = flip.clone()
    moved[2] += 0.3
    second = moved.requires_grad_(True)
    EpgEngine().simulate(_describe(second), tissue).signal.abs().sum().backward()

    assert first.grad.shape == flip.shape
    for train in (0, 1, 3):
        assert torch.equal(first.grad[train], second.grad[train])
    assert not torch.equal(first.grad[2], second.grad[2])


def test_tissue_gradients_sum_over_trains(always_worth_detecting):
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

    for index in range(TISSUE_COUNT):  # tissue gradients
        assert torch.equal(fused[index], sum(g[index] for g in per_train))
    for index in _EVENT_GRADIENTS:  # duration, flip, phase stay per train
        assert torch.equal(fused[index], torch.stack([g[index] for g in per_train]))


def _vjp(prepared, events, seed, outputs, threads):
    return _run_packed_vjp(
        prepared, events, seed, state_count=10, output_count=outputs, threads=threads
    )


@pytest.mark.parametrize("threads", [1, 2, 4, 0])
def test_batched_reduction_is_scheduling_independent(
    threads, always_worth_detecting
):
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
def test_thread_count_only_reassociates(threads, always_worth_detecting):
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


def test_mismatched_train_widths_are_rejected(always_worth_detecting):
    """A width mismatch would stride out of bounds inside the kernel."""
    flip = _schedules(3, 12)
    prepared, _, _ = _prepare_tissue(_tissue(), "cpu")
    (
        duration, kind, flips, phase, action, output_index, shim, saturation,
        frequency,
    ) = _buffers(_pack(flip))
    broken = (
        duration[0], kind, flips, phase, action, output_index, shim,
        saturation, frequency,
    )
    with pytest.raises(ValueError, match="disagree on train count"):
        _run_packed(prepared, broken, 10, int(output_index.max()) + 1, 1)


TRAINS, ATOMS = 19, 8


def _second_order(threads):
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    flip = _schedules(TRAINS, 12)
    tissue = TissueProperties(
        t1_ms=torch.full((ATOMS,), 800.0), t2_ms=torch.full((ATOMS,), 45.0)
    )
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    events = _buffers(_pack(flip))
    outputs = int(events[5].max()) + 1
    # A generator of its own: several threads call this at once, and seeding
    # the global one is two steps that another caller can get between.
    generator = torch.Generator().manual_seed(0)
    cotangent = torch.randn(
        (TRAINS, ATOMS, outputs), generator=generator, dtype=torch.complex64
    )
    tangents = (
        *(
            torch.ones_like(value) if index == 1 else torch.zeros_like(value)
            for index, value in enumerate(prepared)
        ),
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    gradients, _ = _run_packed_vjp_jvp(
        prepared,
        events,
        tangents,
        cotangent,
        state_count=10,
        output_count=outputs,
        threads=threads,
    )
    return gradients


# Odd counts force a partition that does not divide the trains evenly, which is
# where a boundary would fall inside a train.
@pytest.mark.parametrize("threads", [2, 3, 5, 0])
def test_event_gradients_do_not_depend_on_the_worker_count(
    threads, always_worth_detecting
):
    """Every slot of an event gradient has exactly one writer.

    Workers take whole trains, and an event gradient belongs to one train, so
    these slots are summed in the same order however the work is divided --
    bitwise, not merely to tolerance. A partition splitting a train would put
    two workers on one slot, and they would race.
    """
    expected = _second_order(1)
    actual = _second_order(threads)
    for index in _EVENT_GRADIENTS:  # duration, flip, phase
        assert torch.equal(expected[index], actual[index])


@pytest.mark.parametrize("threads", [2, 3, 5, 0])
def test_tissue_gradients_survive_the_worker_count(threads, always_worth_detecting):
    """Every train contributes to these, so workers sum private copies.

    Reassociating that sum moves the last bits, so the agreement here is to
    float tolerance rather than bitwise.
    """
    expected = _second_order(1)
    actual = _second_order(threads)
    for index in range(TISSUE_COUNT):
        scale = expected[index].abs().max()
        if scale == 0:
            assert actual[index].abs().max() == 0, index
            continue
        assert ((expected[index] - actual[index]).abs().max() / scale) < 1e-5, index


def test_workers_are_reusable_across_concurrent_callers(always_worth_detecting):
    """The pool serves one job at a time, and callers may arrive together.

    The kernels release the GIL, so several Python threads really can submit at
    once. Slots are numbered rather than owned, so a caller must get the same
    answer whichever pool workers happen to serve it.
    """
    import threading

    expected = _second_order(1)
    mismatches = []

    def call():
        for _ in range(10):
            actual = _second_order(6)
            if any(
                not torch.equal(expected[index], actual[index])
                for index in _EVENT_GRADIENTS
            ):
                mismatches.append(1)

    callers = [threading.Thread(target=call) for _ in range(4)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join()
    assert not mismatches
