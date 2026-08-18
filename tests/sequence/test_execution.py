"""Choosing where a simulation runs.

``execution`` collapses three separate decisions into one: host or device, how
many devices, and whether the volume can be resident or has to be streamed. The
tests split accordingly -- what gets chosen, and whether every route that could
be chosen produces the same numbers.

Deciding reads the free memory on the card, so the size tests pin the reserve
rather than trusting whatever else happens to be resident when they run.
"""

import pytest
import torch

from torchsim.sequence import execution
from torchsim.sequence._calibration import crossover
from torchsim.sequence._parameters import OUTSIDE_THE_SUBSPACE
from torchsim.sequence._accelerators import (
    _FLOAT_INPUTS,
    _bytes_per_voxel,
    _choose,
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp,
    _run_packed_vjp_jvp,
)
import torchsim.sequence._accelerators as accelerators

from test_offload import _seeds, _volume  # noqa: E402

STATES = 10
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


def _decide(kind, voxels, **arguments):
    events, prepared, outputs = _volume(voxels)
    with execution(**arguments):
        return _choose(kind, prepared, events, outputs, STATES, 1)


def _voxels_worth(kind, share):
    """A volume carrying ``share`` of the work the crossover sits at."""
    events, _prepared, _outputs = _volume(1)
    per_voxel = int(events[1].numel())
    floor = crossover(kind, torch.device("cuda", 0), STATES)
    return max(1, int(floor * share / per_voxel))


def test_without_a_block_a_call_is_left_where_it_is():
    """Simulations must not start moving themselves unasked."""
    events, prepared, outputs = _volume(64)
    assert _choose("forward", prepared, events, outputs, STATES, 1) is None


@pytest.mark.parametrize("kind", ["forward", "jvp", "adjoint"])
def test_work_too_small_to_repay_a_launch_stays_on_the_host(kind):
    """Where that line falls is measured, so the size is taken from it."""
    voxels = _voxels_worth(kind, 1 / 8) if torch.cuda.is_available() else 4
    assert _decide(kind, voxels, target="auto").where == "cpu"


@cuda_only
@pytest.mark.parametrize("kind", ["forward", "jvp", "adjoint"])
def test_work_that_fits_goes_across_in_one_piece(kind):
    choice = _decide(kind, 4096, target="auto")

    assert choice.where == "upfront"
    assert choice.devices


@cuda_only
def test_the_forward_pass_needs_more_work_than_the_adjoint_to_leave_the_host():
    """Its arithmetic is cheapest, so a launch takes longest to repay.

    Both sides of this are measured on whatever card is present, so it is a
    claim about the two passes rather than about one machine.
    """
    card = torch.device("cuda", 0)

    assert crossover("forward", card, STATES) > crossover("adjoint", card, STATES)


@cuda_only
@pytest.mark.parametrize("kind", ["forward", "jvp", "adjoint"])
def test_work_well_past_the_crossover_goes_to_the_card(kind):
    assert _decide(kind, _voxels_worth(kind, 8), target="auto").where != "cpu"


@cuda_only
def test_a_volume_larger_than_the_card_is_streamed():
    """Reserving everything leaves nothing resident, so it has to chunk."""
    free, _total = torch.cuda.mem_get_info(0)
    choice = _decide("forward", 4096, target="auto", reserve_bytes=free)

    assert choice.where == "stream"
    assert choice.offload.budget_bytes > 0


@cuda_only
def test_streaming_can_be_demanded_for_a_volume_that_would_fit():
    choice = _decide("forward", 4096, target="cuda", stream=True)

    assert choice.where == "stream"


@cuda_only
def test_residency_can_be_demanded_for_a_volume_that_will_not_fit():
    """Better to fail on the allocation than to quietly do something else."""
    free, _total = torch.cuda.mem_get_info(0)
    choice = _decide("forward", 4096, target="cuda", stream=False, reserve_bytes=free)

    assert choice.where == "upfront"


@cuda_only
def test_the_budget_can_be_set_outright():
    free, _total = torch.cuda.mem_get_info(0)
    choice = _decide(
        "forward", 4096, target="cuda", stream=True, budget_bytes=7 << 20, lanes=3
    )

    assert choice.offload.budget_bytes == 7 << 20
    assert choice.offload.lanes == 3


@cuda_only
def test_the_reserve_is_left_alone():
    """A recon server's product pipeline keeps its own memory resident."""
    free, _total = torch.cuda.mem_get_info(0)
    choice = _decide(
        "forward", 4096, target="cuda", stream=True, reserve_bytes=free // 2
    )

    assert choice.offload.budget_bytes <= free - free // 2


def test_the_host_can_be_demanded_whatever_the_size():
    assert _decide("adjoint", 4096, target="cpu").where == "cpu"


@cuda_only
def test_a_named_device_is_used_however_small_the_work():
    """Naming one is an instruction, not a hint, so the threshold is skipped."""
    choice = _decide("forward", 4, target="cuda:0")

    assert choice.where == "upfront"
    assert choice.devices == (torch.device("cuda:0"),)


def test_the_size_of_a_voxel_grows_with_the_pass():
    """The adjoint records a trajectory; the forward pass keeps only a signal."""
    shape = (1, 200, 100, STATES, 1)
    forward = _bytes_per_voxel("forward", *shape)
    jvp = _bytes_per_voxel("jvp", *shape)
    adjoint = _bytes_per_voxel("adjoint", *shape)

    assert forward < jvp < adjoint
    assert adjoint > 10 * jvp


# --- every route has to agree ---


_INSIDE_THE_SUBSPACE = tuple(
    position not in OUTSIDE_THE_SUBSPACE for position in range(len(_FLOAT_INPUTS))
)


def _all_four(prepared, events, outputs, seeds, cotangent):
    tissue_seed, event_seed = seeds
    return (
        _run_packed(prepared, events, STATES, outputs, 0, real_axis=1),
        _run_packed_jvp(
            prepared, events, tissue_seed, event_seed, STATES, outputs, 0, real_axis=1
        ),
        _run_packed_vjp_jvp(
            prepared,
            events,
            (*tissue_seed, *event_seed),
            cotangent,
            STATES,
            outputs,
            0,
            real_axis=1,
        ),
        # The route autograd takes for a plain backward. It settles its own
        # real axis rather than being handed one, from what the caller says it
        # will read -- so leaving the four out is how it reaches the same
        # subspace the pass above was told to take.
        _run_packed_vjp(
            prepared, events, cotangent, STATES, outputs, 0, _INSIDE_THE_SUBSPACE
        ),
    )


@cuda_only
@pytest.mark.parametrize(
    "arguments",
    [
        {"target": "auto"},
        {"target": "cpu"},
        {"target": "cuda:0"},
        {"target": "cuda", "stream": False},
        {"target": "cuda", "stream": True, "budget_bytes": 1 << 20},
    ],
    ids=["auto", "cpu", "named", "upfront", "streamed"],
)
def test_every_route_gives_the_same_answer(arguments):
    voxels = 3000
    events, prepared, outputs = _volume(voxels)
    seeds = _seeds(events, prepared, 1)
    generator = torch.Generator().manual_seed(7)
    cotangent = torch.randn(
        (voxels, outputs), generator=generator, dtype=torch.complex64
    )
    expected = _all_four(prepared, events, outputs, seeds, cotangent)
    with execution(**arguments):
        actual = _all_four(prepared, events, outputs, seeds, cotangent)

    for reference, result in zip(expected[:2], actual[:2], strict=True):
        assert result.device == reference.device
        assert ((reference - result).abs().max() / reference.abs().max()) < 1e-5
    # The fourth route is compared on the gradients it was told the caller
    # would read: leaving four out is what lets a device run take the reduced
    # kernel, and a host run computes them anyway.
    wanted_expected = tuple(
        gradient for gradient, asked in zip(
            expected[3], _INSIDE_THE_SUBSPACE, strict=True
        ) if asked
    )
    wanted_actual = tuple(
        gradient for gradient, asked in zip(
            actual[3], _INSIDE_THE_SUBSPACE, strict=True
        ) if asked
    )
    for side, other in zip((*expected[2], wanted_expected),
                           (*actual[2], wanted_actual), strict=True):
        for reference, result in zip(side, other, strict=True):
            scale = reference.abs().max()
            if scale == 0:
                assert result.abs().max() == 0
                continue
            assert ((reference - result).abs().max() / scale) < 1e-4


@cuda_only
def test_a_device_resident_call_can_be_forced_back_to_the_host():
    """Forcing the host means running there, wherever the tensors started."""
    events, prepared, outputs = _volume(3000)
    device_events = tuple(value.cuda() for value in events)
    device_tissue = tuple(value.cuda() for value in prepared)
    expected = _run_packed(prepared, events, STATES, outputs, 0, real_axis=1)
    with execution("cpu"):
        actual = _run_packed(
            device_tissue, device_events, STATES, outputs, 0, real_axis=1
        )

    # The answer comes back where the caller's tensors live.
    assert actual.device.type == "cuda"
    assert ((expected - actual.cpu()).abs().max() / expected.abs().max()) < 1e-5


# --- the block itself ---


def test_an_empty_device_list_is_refused():
    with pytest.raises(ValueError, match="at least one device"):
        with execution([]):
            pass


def test_a_device_that_is_neither_cuda_nor_the_host_is_refused():
    with pytest.raises(ValueError, match="CUDA devices or 'cpu'"):
        with execution(["meta"]):
            pass


@pytest.mark.parametrize("arguments", [{"lanes": 0}, {"budget_bytes": 0}])
def test_nonsense_settings_are_refused(arguments):
    with pytest.raises(ValueError):
        with execution("cuda", **arguments):
            pass


def test_naming_the_host_as_a_device_means_the_host():
    with execution(["cpu"]):
        assert accelerators._EXECUTION.target == "cpu"


def test_the_previous_setting_comes_back_after_a_failure():
    with pytest.raises(RuntimeError):
        with execution("cpu"):
            raise RuntimeError("boom")

    assert accelerators._EXECUTION is None


def test_blocks_nest():
    with execution("cpu"):
        with execution("auto"):
            assert accelerators._EXECUTION.target == "auto"
        assert accelerators._EXECUTION.target == "cpu"
    assert accelerators._EXECUTION is None
