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
from torchsim.sequence._transmit import channel_count, transmit_field

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


def test_pulses_driving_different_shims_are_refused() -> None:
    """One field per voxel cannot describe a shim that changes mid-sequence."""
    shim = _uniform_shim()
    description = _description(shim, shim_ids=(0, 1, 0, 1, 0))
    with pytest.raises(NotImplementedError, match="same shim"):
        transmit_field(description, 1.0, 0.0, torch.device("cpu"))


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
