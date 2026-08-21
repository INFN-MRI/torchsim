"""Multi-channel transmit, resolved into the field the array produces.

A pulse driven on several channels excites a voxel through the sum of what
each channel puts there, and the RF operator only ever sees the magnitude and
phase of that sum. So an array reduces to the two per-voxel buffers the state
machine already carries, and every backend and every derivative reaches it
without knowing channels exist.

The check that matters is cancellation: two channels driven in antiphase must
leave the voxel untouched. Summing magnitudes and phases separately cannot do
that, and is the reason ``epg.phased_multidrive_rf_pulse_op`` is not used here
as the reference.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence._description import EventType, SequenceEvent, ShimDefinition
from torchsim.sequence._accelerators import (
    _across_the_table,
    _pack_events,
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp,
    _run_packed_vjp_jvp,
    offload,
)
from torchsim.sequence._parameters import TISSUE_COUNT
from torchsim.sequence._simulation import _prepare_tissue, _resolve_transmit
from torchsim.sequence._transmit import (
    channel_count,
    shim_rows,
    transmit_field,
)
from utils.packed_reference import simulate_packed

ECHOES = 4
CHANNELS = 8


def _description(shim: ShimDefinition | None = None, shim_ids: tuple[int, ...] = ()):
    """An echo train, optionally driven through a transmit array."""
    description = fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )
    if shim is None:
        return description
    events = description.events
    if shim_ids:
        pulses = [
            index
            for index, event in enumerate(events)
            if event.type is EventType.RF
        ]
        events = list(events)
        for position, identifier in zip(pulses, shim_ids, strict=False):
            event = events[position]
            events[position] = SequenceEvent(
                event.type, event.timestamp_us, (*event.params[:5], identifier,
                                                 *event.params[6:])
            )
        events = tuple(events)
    return replace(
        description, events=events, shim_definitions={shim.id: shim}
    )


def _two_shim_description():
    """Excitation on one shim, every refocusing pulse on another."""
    quadrature = ShimDefinition(
        0, (1.0 / CHANNELS,) * CHANNELS, (0.0,) * CHANNELS
    )
    shaped = ShimDefinition(
        1,
        tuple(0.05 * (index + 1) for index in range(CHANNELS)),
        tuple(0.3 * index for index in range(CHANNELS)),
    )
    description = _description(quadrature, shim_ids=(0, 1, 1, 1, 1, 1, 1))
    return replace(
        description, shim_definitions={0: quadrature, 1: shaped}
    )


def _uniform_shim(identifier: int = 0) -> ShimDefinition:
    """Every channel driven equally, which is circularly polarized drive."""
    return ShimDefinition(
        identifier, (1.0 / CHANNELS,) * CHANNELS, (0.0,) * CHANNELS
    )


def _sensitivity(voxels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A transmit array whose channels differ in both magnitude and phase."""
    generator = torch.Generator().manual_seed(0)
    magnitude = 0.6 + 0.8 * torch.rand((CHANNELS, voxels), generator=generator)
    phase = torch.linspace(0.0, 2.0 * torch.pi, CHANNELS)[:, None].expand(
        CHANNELS, voxels
    )
    return magnitude, phase.contiguous()


def _tissue(voxels: int = 3, **transmit) -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, voxels),
        t2_ms=torch.linspace(40.0, 120.0, voxels),
        **transmit,
    )


def test_a_sequence_without_shims_declares_one_channel() -> None:
    assert channel_count(_description()) == 1


def test_channels_in_antiphase_cancel() -> None:
    """The check that separates a complex sum from two independent ones."""
    shim = ShimDefinition(0, (1.0, 1.0), (0.0, torch.pi))
    description = _description(shim)
    magnitude, phase = transmit_field(
        description,
        torch.ones((2, 3)),
        torch.zeros((2, 3)),
        torch.device("cpu"),
    )
    assert magnitude.abs().max() < 1e-6


def test_one_channel_driven_at_unit_weight_is_the_field_itself() -> None:
    shim = ShimDefinition(0, (1.0,), (0.0,))
    b1 = torch.tensor([[0.8, 1.1, 0.95]])
    b1_phase = torch.tensor([[0.1, -0.2, 0.3]])
    magnitude, phase = transmit_field(
        _description(shim), b1, b1_phase, torch.device("cpu")
    )
    assert torch.allclose(magnitude, b1.reshape(-1), atol=1e-6)
    assert torch.allclose(phase, b1_phase.reshape(-1), atol=1e-6)


def test_the_field_is_the_complex_sum_written_out() -> None:
    voxels = 3
    shim = ShimDefinition(
        0,
        tuple(0.1 * (index + 1) for index in range(CHANNELS)),
        tuple(0.4 * index for index in range(CHANNELS)),
    )
    b1, b1_phase = _sensitivity(voxels)
    magnitude, phase = transmit_field(
        _description(shim), b1, b1_phase, torch.device("cpu")
    )

    weights = torch.polar(
        torch.tensor(shim.magnitudes), torch.tensor(shim.phases_rad)
    )
    expected = (torch.polar(b1, b1_phase) * weights[:, None]).sum(dim=0)
    assert torch.allclose(magnitude, expected.abs(), atol=1e-6)
    assert torch.allclose(phase, expected.angle(), atol=1e-6)


def test_a_scalar_sensitivity_is_the_same_on_every_channel() -> None:
    shim = _uniform_shim()
    magnitude, _phase = transmit_field(
        _description(shim), 1.0, 0.0, torch.device("cpu")
    )
    # Eight channels each at an eighth, all in phase.
    assert torch.allclose(magnitude, torch.tensor(1.0), atol=1e-6)


def test_a_sensitivity_that_ignores_the_channel_axis_is_refused() -> None:
    shim = _uniform_shim()
    with pytest.raises(ValueError, match="transmit channels"):
        transmit_field(
            _description(shim), torch.ones(5), torch.zeros(5), torch.device("cpu")
        )


def test_shims_of_different_widths_are_refused() -> None:
    description = fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)), echo_spacing_s=5e-3
    )
    description = replace(
        description,
        shim_definitions={
            0: ShimDefinition(0, (1.0, 1.0), (0.0, 0.0)),
            1: ShimDefinition(1, (1.0,), (0.0,)),
        },
    )
    with pytest.raises(ValueError, match="channel count"):
        channel_count(description)


def test_pulses_driving_different_shims_get_a_row_each() -> None:
    """Excitation on one shim and refocusing on another is two fields."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    description = _two_shim_description()
    magnitude, phase = transmit_field(
        description, b1, b1_phase, torch.device("cpu")
    )

    assert magnitude.shape == (2, voxels)
    assert phase.shape == (2, voxels)
    assert not torch.allclose(magnitude[0], magnitude[1])


def test_the_shim_rows_follow_the_ids_the_pulses_name() -> None:
    description = _two_shim_description()
    assert shim_rows(description) == {0: 0, 1: 1}
    assert shim_rows(_description(_uniform_shim())) == {0: 0}
    assert shim_rows(_description()) == {}


def test_a_pulse_naming_an_undefined_shim_is_refused() -> None:
    description = _description(_uniform_shim(), shim_ids=(0, 7, 0, 7, 0))
    with pytest.raises(KeyError, match="no shim definition with id 7"):
        shim_rows(description)


def _packed(description, voxels: int = 3, device: str = "cpu"):
    """Prepared tissue and packed events for a train, shimmed or not."""
    b1, b1_phase = _sensitivity(voxels)
    tissue, shims = _resolve_transmit(
        _tissue(voxels, b1=b1, b1_phase_rad=b1_phase), description, device
    )
    prepared, _, resolved = _prepare_tissue(tissue, device, shims)
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=resolved,
        rf_raster_time_s=1e-6,
        shim_rows=shim_rows(description),
    )
    return prepared, packed.buffers, packed.output_count


def _two_shim_packed(voxels: int = 3, device: str = "cpu"):
    """Prepared tissue and packed events for the two-shim train."""
    return _packed(_two_shim_description(), voxels, device)


def test_each_pulse_reads_the_shim_it_names() -> None:
    """Against the oracle, which indexes the rows independently."""
    tissue, events, output_count = _two_shim_packed()
    assert tissue[3].numel() == 2 * tissue[0].numel()
    expected = simulate_packed(
        tissue, events, state_count=8, output_count=output_count
    )
    actual = _run_packed(tissue, events, state_count=8, output_count=output_count,
                         threads=1)
    assert (expected - actual).abs().max() / expected.abs().max() < 1e-5


def test_the_shim_a_pulse_names_is_the_one_that_reaches_it() -> None:
    """Changing a row no pulse reads leaves the signal alone, bit for bit."""
    tissue, events, output_count = _two_shim_packed()
    atoms = tissue[0].numel()
    arguments = dict(state_count=8, output_count=output_count, threads=1)
    reference = _run_packed(tissue, events, **arguments)

    # Every pulse here names shim 0 or shim 1, so a third row is dead weight.
    padded = list(tissue)
    for index in (3, 4):
        padded[index] = torch.cat(
            (tissue[index], torch.full((atoms,), 7.0))
        ).contiguous()
    assert torch.equal(_run_packed(tuple(padded), events, **arguments), reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_forward_agrees_with_the_cpu_one_across_shims() -> None:
    tissue, events, output_count = _two_shim_packed()
    device = torch.device("cuda")
    arguments = dict(state_count=8, output_count=output_count, threads=1)
    expected = _run_packed(tissue, events, **arguments)
    actual = _run_packed(
        tuple(value.to(device) for value in tissue),
        tuple(value.to(device) for value in events),
        **arguments,
    )
    assert (expected - actual.cpu()).abs().max() / expected.abs().max() < 1e-5


def test_two_identical_shims_are_the_same_as_one() -> None:
    """The row a pulse reads is the only thing the second shim changes."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    shim = _uniform_shim()
    doubled = replace(
        _description(shim, shim_ids=(0, 1, 0, 1, 0)),
        shim_definitions={
            0: shim,
            1: ShimDefinition(1, shim.magnitudes, shim.phases_rad),
        },
    )
    tissue = _tissue(voxels, b1=b1, b1_phase_rad=b1_phase)

    assert torch.equal(
        FSE().simulate(doubled, tissue, nstates=8).signal,
        FSE().simulate(_description(shim), tissue, nstates=8).signal,
    )


def test_the_slice_spreads_every_shim_row_and_keeps_them_apart() -> None:
    """The positions spread a voxel into copies; each row has to follow it.

    ``_across_the_table`` widens every buffer alike, which lands the shim rows
    in the order the kernels index only because the rows are the outermost
    axis. Checked through the kernels, against the oracle reading the same
    spread buffers.
    """
    voxels, locations = 3, 3
    tissue, events, output_count = _two_shim_packed(voxels)
    spread = _across_the_table(tissue, locations)

    assert spread[3].numel() == 2 * voxels * locations
    # Each row is that row's field, once per position.
    assert torch.equal(
        spread[3].view(2, voxels, locations),
        tissue[3].view(2, voxels)[:, :, None].expand(-1, -1, locations),
    )

    arguments = dict(state_count=8, output_count=output_count)
    expected = simulate_packed(spread, events, **arguments)
    actual = _run_packed(spread, events, threads=1, **arguments)
    assert (expected - actual).abs().max() < 1e-5 * expected.abs().max()


def _seed(tissue, output_count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(
        (tissue[0].numel(), output_count),
        generator=generator,
        dtype=torch.complex64,
    )


def _oracle_adjoint(tissue, events, seed, output_count: int):
    """What autograd through the reference makes of the same pass."""
    leaves = tuple(value.clone().requires_grad_(True) for value in tissue)
    signal = simulate_packed(
        leaves, events, state_count=8, output_count=output_count
    )
    (signal.real * seed.real + signal.imag * seed.imag).sum().backward()
    return tuple(
        torch.zeros_like(value) if value.grad is None else value.grad
        for value in leaves
    )


def test_the_adjoint_gives_each_shim_a_row_of_its_own() -> None:
    """Two shims, two transmit gradients, against the oracle."""
    tissue, events, output_count = _two_shim_packed()
    atoms = tissue[0].numel()
    seed = _seed(tissue, output_count)
    expected = _oracle_adjoint(tissue, events, seed, output_count)
    actual = _run_packed_vjp(
        tissue, events, seed, state_count=8, output_count=output_count, threads=1
    )

    assert actual[3].numel() == 2 * atoms
    for index in (3, 4):
        reference = expected[index]
        assert (reference - actual[index]).abs().max() < 1e-5 * reference.abs().max()


def test_forward_mode_follows_each_shim() -> None:
    tissue, events, output_count = _two_shim_packed()
    generator = torch.Generator().manual_seed(3)
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
            state_count=8,
            output_count=output_count,
        )

    _, expected = torch.func.jvp(
        forward,
        (*tissue, events[0], events[2], events[3]),
        (*tissue_dot, *event_dot),
    )
    actual = _run_packed_jvp(
        tissue, events, tissue_dot, event_dot,
        state_count=8, output_count=output_count, threads=1,
    )
    assert (expected - actual).abs().max() < 1e-5 * expected.abs().max()


def test_the_second_order_pass_follows_each_shim() -> None:
    """Forward-over-reverse, which is also how CUDA reaches a first adjoint."""
    tissue, events, output_count = _two_shim_packed()
    generator = torch.Generator().manual_seed(4)
    primals = (*tissue, events[0], events[2], events[3])
    tangents = tuple(
        0.05 * torch.randn(value.shape, generator=generator) for value in primals
    )
    seed = _seed(tissue, output_count)

    leaves = tuple(value.clone().requires_grad_(True) for value in primals)
    signal = simulate_packed(
        leaves[:TISSUE_COUNT],
        (
            leaves[TISSUE_COUNT], events[1], leaves[TISSUE_COUNT + 1],
            leaves[TISSUE_COUNT + 2], *events[4:],
        ),
        state_count=8,
        output_count=output_count,
    )
    first = torch.autograd.grad(
        (signal.real * seed.real + signal.imag * seed.imag).sum(),
        leaves,
        create_graph=True,
        materialize_grads=True,
    )
    expected = torch.autograd.grad(
        sum((grad * step).sum() for grad, step in zip(first, tangents)),
        leaves,
        materialize_grads=True,
    )
    actual, _ = _run_packed_vjp_jvp(
        tissue, events, tangents, seed,
        state_count=8, output_count=output_count, threads=1,
    )
    for index in (3, 4):
        reference = expected[index]
        assert (reference - actual[index]).abs().max() < 1e-4 * reference.abs().max()


def test_a_shim_no_pulse_drives_gets_no_gradient() -> None:
    """A row nothing reads is a row nothing can change, exactly."""
    tissue, events, output_count = _two_shim_packed()
    atoms = tissue[0].numel()
    padded = list(tissue)
    for index in (3, 4):
        padded[index] = torch.cat(
            (tissue[index], torch.full((atoms,), 0.7))
        ).contiguous()

    gradients = _run_packed_vjp(
        tuple(padded), events, _seed(tissue, output_count),
        state_count=8, output_count=output_count, threads=1,
    )
    for index in (3, 4):
        assert torch.equal(
            gradients[index][2 * atoms :], torch.zeros(atoms)
        )
        assert gradients[index][:atoms].abs().max() > 0


def test_two_identical_shims_carry_between_them_what_one_carries() -> None:
    """Splitting a field across rows only splits where its gradient lands."""
    voxels = 3
    shim = _uniform_shim()
    doubled = replace(
        _description(shim, shim_ids=(0, 1, 0, 1, 0)),
        shim_definitions={
            0: shim,
            1: ShimDefinition(1, shim.magnitudes, shim.phases_rad),
        },
    )
    single, events, output_count = _packed(_description(shim), voxels)
    seed = _seed(single, output_count)
    arguments = dict(state_count=8, output_count=output_count, threads=1)
    whole = _run_packed_vjp(single, events, seed, **arguments)

    split_tissue, split_events, _ = _packed(doubled, voxels)
    split = _run_packed_vjp(split_tissue, split_events, seed, **arguments)

    for index in (3, 4):
        summed = split[index][:voxels] + split[index][voxels:]
        assert (summed - whole[index]).abs().max() < 1e-5 * whole[index].abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_cuda_adjoint_agrees_with_the_cpu_one_across_shims() -> None:
    tissue, events, output_count = _two_shim_packed()
    seed = _seed(tissue, output_count)
    arguments = dict(state_count=8, output_count=output_count, threads=1)
    expected = _run_packed_vjp(tissue, events, seed, **arguments)
    card = torch.device("cuda")
    actual = _run_packed_vjp(
        tuple(value.to(card) for value in tissue),
        tuple(value.to(card) for value in events),
        seed.to(card),
        **arguments,
    )
    for index in (3, 4):
        reference = expected[index]
        moved = actual[index].cpu()
        assert (reference - moved).abs().max() < 1e-5 * reference.abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_streaming_cuts_the_voxels_and_not_the_shim_rows() -> None:
    """A chunk is a slice of every row, not a slice of the buffer."""
    tissue, events, output_count = _two_shim_packed(voxels=400)
    arguments = dict(state_count=8, output_count=output_count, threads=1)
    expected = _run_packed(tissue, events, **arguments)
    with offload(["cuda"], budget_bytes=1 << 16):
        actual = _run_packed(tissue, events, **arguments)

    assert (expected - actual).abs().max() < 1e-5 * expected.abs().max()


def test_a_gradient_reaches_the_weights_of_each_shim() -> None:
    """The point of the pass: a per-pulse shim is a thing to design."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    excitation = torch.full((CHANNELS,), 1.0 / CHANNELS, requires_grad=True)
    refocusing = torch.linspace(0.05, 0.4, CHANNELS).requires_grad_(True)
    shims = {
        0: ShimDefinition(0, excitation, (0.0,) * CHANNELS),
        1: ShimDefinition(1, refocusing, tuple(0.3 * i for i in range(CHANNELS))),
    }
    description = replace(
        _description(shims[0], shim_ids=(0, 1, 1, 1, 1, 1, 1)),
        shim_definitions=shims,
    )

    signal = FSE().simulate(
        description, _tissue(voxels, b1=b1, b1_phase_rad=b1_phase), nstates=8
    ).signal
    signal.abs().square().sum().backward()

    assert excitation.grad is not None and excitation.grad.abs().max() > 0
    assert refocusing.grad is not None and refocusing.grad.abs().max() > 0


def test_the_weight_gradients_match_a_finite_difference_shim_by_shim() -> None:
    """A row swapped between shims stays nonzero, so nonzero is not enough."""
    voxels = 2
    b1, b1_phase = _sensitivity(voxels)
    phases = tuple(0.3 * index for index in range(CHANNELS))

    def loss(excitation: torch.Tensor, refocusing: torch.Tensor) -> torch.Tensor:
        shims = {
            0: ShimDefinition(0, excitation, (0.0,) * CHANNELS),
            1: ShimDefinition(1, refocusing, phases),
        }
        description = replace(
            _description(shims[0], shim_ids=(0, 1, 1, 1, 1, 1, 1)),
            shim_definitions=shims,
        )
        signal = FSE().simulate(
            description, _tissue(voxels, b1=b1, b1_phase_rad=b1_phase), nstates=8
        ).signal
        return signal.abs().square().sum()

    weights = (
        torch.full((CHANNELS,), 1.0 / CHANNELS),
        torch.linspace(0.05, 0.4, CHANNELS),
    )
    leaves = tuple(value.clone().requires_grad_(True) for value in weights)
    loss(*leaves).backward()

    step = 1e-3
    for shim in (0, 1):
        for channel in (0, CHANNELS // 2):
            bump = torch.zeros(CHANNELS)
            bump[channel] = step
            moved = tuple(
                value + bump if index == shim else value
                for index, value in enumerate(weights)
            )
            back = tuple(
                value - bump if index == shim else value
                for index, value in enumerate(weights)
            )
            numeric = (loss(*moved) - loss(*back)) / (2.0 * step)
            assert abs(numeric - leaves[shim].grad[channel]) < 3e-3 * abs(numeric)


def test_a_uniform_array_reproduces_the_single_channel_signal() -> None:
    """Bitwise: the array resolves to exactly the buffers it would have had."""
    voxels = 3
    plain = FSE().simulate(_description(), _tissue(voxels), nstates=8).signal
    array = FSE().simulate(
        _description(_uniform_shim()),
        _tissue(voxels, b1=torch.full((CHANNELS, voxels), 1.0)),
        nstates=8,
    ).signal
    assert torch.equal(plain, array)


def test_the_array_reaches_the_signal() -> None:
    """Without this, the parity checks would hold on an inert array."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    shim = _uniform_shim()
    uniform = FSE().simulate(
        _description(shim),
        _tissue(voxels, b1=torch.full((CHANNELS, voxels), 1.0)),
        nstates=8,
    ).signal
    shaped = FSE().simulate(
        _description(shim), _tissue(voxels, b1=b1, b1_phase_rad=b1_phase), nstates=8
    ).signal
    relative = (uniform - shaped).abs().max() / uniform.abs().max()
    assert relative > 0.01


def test_the_resolved_field_matches_a_hand_built_single_channel_run() -> None:
    """The array path and the equivalent scalar path give the same signal."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    shim = ShimDefinition(
        0,
        tuple(0.1 * (index + 1) for index in range(CHANNELS)),
        tuple(0.4 * index for index in range(CHANNELS)),
    )
    weights = torch.polar(
        torch.tensor(shim.magnitudes), torch.tensor(shim.phases_rad)
    )
    field = (torch.polar(b1, b1_phase) * weights[:, None]).sum(dim=0)

    array = FSE().simulate(
        _description(shim), _tissue(voxels, b1=b1, b1_phase_rad=b1_phase), nstates=8
    ).signal
    equivalent = FSE().simulate(
        _description(),
        _tissue(voxels, b1=field.abs(), b1_phase_rad=field.angle()),
        nstates=8,
    ).signal
    assert torch.equal(array, equivalent)


def test_a_common_phase_on_every_channel_only_turns_the_axis() -> None:
    """A static shim is a flip scaling and an axis turn, nothing more.

    Turning the whole array together cannot change how far a voxel tips, only
    which way it tips -- which is what makes the array reducible to the pair of
    buffers the state machine carries.
    """
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    turn = 0.7
    plain = transmit_field(
        _description(_uniform_shim()), b1, b1_phase, torch.device("cpu")
    )
    turned = transmit_field(
        _description(
            ShimDefinition(0, (1.0 / CHANNELS,) * CHANNELS, (turn,) * CHANNELS)
        ),
        b1,
        b1_phase,
        torch.device("cpu"),
    )
    assert torch.allclose(plain[0], turned[0], atol=1e-6)
    drift = turned[1] - plain[1] - turn
    drift = torch.remainder(drift + torch.pi, 2.0 * torch.pi) - torch.pi
    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-5)


def test_the_array_matches_the_operator_library() -> None:
    """Two implementations of the same superposition, written out separately."""
    from torchsim.epg import phased_multidrive_rf_pulse_op

    voxels = 1
    b1, b1_phase = _sensitivity(voxels)
    flip = 0.6
    shim = ShimDefinition(
        0,
        tuple(0.1 * (index + 1) for index in range(CHANNELS)),
        tuple(0.4 * index for index in range(CHANNELS)),
    )
    magnitude, phase = transmit_field(
        _description(shim), b1, b1_phase, torch.device("cpu")
    )

    _rotation, _net = phased_multidrive_rf_pulse_op(
        flip * torch.tensor(shim.magnitudes),
        torch.tensor(shim.phases_rad),
        1.0,
        b1.reshape(-1),
        b1_phase.reshape(-1),
    )
    field = (
        torch.polar(b1.reshape(-1), b1_phase.reshape(-1))
        * flip
        * torch.polar(torch.tensor(shim.magnitudes), torch.tensor(shim.phases_rad))
    ).sum()
    assert torch.allclose(flip * magnitude.reshape(()), field.abs(), atol=1e-6)
    assert torch.allclose(phase.reshape(()), field.angle(), atol=1e-6)


def test_a_gradient_reaches_each_channel_of_the_array() -> None:
    """Autograd carries it through the sum, so no kernel has to know."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    b1 = b1.clone().requires_grad_(True)
    b1_phase = b1_phase.clone().requires_grad_(True)

    signal = FSE().simulate(
        _description(_uniform_shim()),
        _tissue(voxels, b1=b1, b1_phase_rad=b1_phase),
        nstates=8,
    ).signal
    signal.abs().square().sum().backward()

    assert b1.grad is not None and b1.grad.shape == (CHANNELS, voxels)
    assert b1.grad.abs().max() > 0
    assert b1_phase.grad is not None and b1_phase.grad.abs().max() > 0


def test_a_gradient_reaches_the_shim_weights() -> None:
    """A shim is a thing to design, so it has to be differentiable too."""
    voxels = 3
    b1, b1_phase = _sensitivity(voxels)
    magnitudes = torch.full((CHANNELS,), 1.0 / CHANNELS, requires_grad=True)
    shim = ShimDefinition(0, magnitudes, (0.0,) * CHANNELS)

    signal = FSE().simulate(
        _description(shim), _tissue(voxels, b1=b1, b1_phase_rad=b1_phase), nstates=8
    ).signal
    signal.abs().square().sum().backward()

    assert magnitudes.grad is not None
    assert magnitudes.grad.abs().max() > 0


def test_the_channel_gradient_matches_a_finite_difference() -> None:
    """An independent check on the chain, one channel through to signal."""
    voxels = 2
    b1, b1_phase = _sensitivity(voxels)
    shim = _uniform_shim()

    def loss(magnitude: torch.Tensor) -> torch.Tensor:
        signal = FSE().simulate(
            _description(shim),
            _tissue(voxels, b1=magnitude, b1_phase_rad=b1_phase),
            nstates=8,
        ).signal
        return signal.abs().square().sum()

    leaf = b1.clone().requires_grad_(True)
    loss(leaf).backward()

    step = 1e-3
    for channel in (0, CHANNELS // 2):
        for voxel in range(voxels):
            bump = torch.zeros_like(b1)
            bump[channel, voxel] = step
            numeric = (loss(b1 + bump) - loss(b1 - bump)) / (2.0 * step)
            assert abs(numeric - leaf.grad[channel, voxel]) < 2e-3 * abs(numeric)
