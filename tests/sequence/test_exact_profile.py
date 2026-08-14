"""Asking for the exact slice profile from the public API.

``slice_profile=`` has taken a tensor of flip-angle scalings: a Bloch response
treated as proportional to the pulse driving it, which it is not. Passing
:func:`exact_slice_profile` instead asks for the rotation the sequence's own
pulse performs at each position, built from the RF definition the description
already carries.

The anchor is a pulse with no gradient across it, whose rotation is the same
everywhere along the slice and must therefore reproduce a simulation with no
profile at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import FSE, exact_slice_profile, fse_description
from torchsim.sequence._description import RfDefinition, RfShape
from torchsim.sequence._simulation import TissueProperties

ECHOES = 6
STATES = 10


def _describe(flip=None, definition=None):
    description = fse_description(
        torch.deg2rad(torch.full((ECHOES,), 150.0)) if flip is None else flip,
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    if definition is None:
        return description
    from dataclasses import replace

    return replace(description, rf_definitions={definition.id: definition})


def _sinc(bandwidth_hz: float, samples: int = 128, rf_id: int = 0) -> RfDefinition:
    grid = np.linspace(-2.0, 2.0, samples)
    envelope = np.sinc(grid) * (0.54 + 0.46 * np.cos(np.pi * grid / 2.0))
    envelope = envelope / np.abs(envelope).max()
    return RfDefinition(
        id=rf_id,
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


def _tissue(**overrides):
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        **overrides,
    )


def _signal(description, profile, tissue=None, backend="native"):
    return FSE().simulate(
        description,
        _tissue() if tissue is None else tissue,
        slice_profile=profile,
        nstates=STATES,
        backend=backend,
    ).signal


def test_a_pulse_with_no_gradient_across_it_is_no_profile_at_all() -> None:
    """The anchor: same rotation everywhere, so the mean over it is itself."""
    description = _describe()
    plain = _signal(description, 1.0)
    exact = _signal(description, exact_slice_profile(5))

    assert exact.shape == plain.shape
    assert (exact - plain).abs().max() < 1e-5 * plain.abs().max()


def test_the_slice_shows_up_when_the_pulse_selects_one() -> None:
    """A real sinc under its gradient is not the pulse at the slice centre."""
    description = _describe(definition=_sinc(4.0e3))
    flat = _signal(description, 1.0)
    exact = _signal(description, exact_slice_profile(15))

    assert (exact - flat).abs().max() > 0.05 * flat.abs().max()


def test_the_exact_profile_follows_the_transmit_where_a_scaling_cannot() -> None:
    """The point of the stage, reached through the public API.

    The scaling model fits the profile once at nominal amplitude, so it tracks
    the transmit field only where that field is one. Two transmit values that
    the exact profile separates must not move the scaled one the same way.
    """
    description = _describe(definition=_sinc(4.0e3))
    positions = torch.linspace(-1.0, 1.0, 9)
    # The profile the scaling model would fit: on-axis flip at nominal drive.
    from torchsim.sequence._transition import transition_table

    table = transition_table(
        description.rf_definitions[0], positions, bins=64, rf_raster_time_s=1e-6
    )
    nominal = float(torch.deg2rad(torch.tensor(150.0)))
    fitted = torch.tensor(
        [
            2.0
            * float(
                torch.arccos(table.a[index, :].abs().clamp(max=1.0)[
                    min(63, int(round(nominal / table.step)))
                ])
            )
            for index in range(len(positions))
        ]
    )
    fitted = fitted / fitted[len(positions) // 2]

    scaled = {}
    exact = {}
    for b1 in (1.0, 1.4):
        tissue = _tissue(b1=torch.full((2,), b1))
        scaled[b1] = _signal(description, fitted, tissue=tissue)
        exact[b1] = _signal(description, exact_slice_profile(positions), tissue=tissue)

    # At nominal transmit the two models are close; away from it they part.
    near = (scaled[1.0] - exact[1.0]).abs().max()
    far = (scaled[1.4] - exact[1.4]).abs().max()
    assert far > 2.0 * near


def test_the_gradients_reach_the_flip_angles_through_the_table() -> None:
    flip = torch.deg2rad(torch.full((ECHOES,), 150.0)).requires_grad_(True)
    description = _describe(flip=flip, definition=_sinc(4.0e3))
    signal = _signal(description, exact_slice_profile(5))
    (gradient,) = torch.autograd.grad(signal.abs().square().sum(), flip)

    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0.0


def _two_shapes(second: RfDefinition):
    """An FSE whose excitation and refocusing are shaped differently."""
    from dataclasses import replace

    from torchsim.sequence._description import RfUse, SequenceEvent

    description = _describe(definition=_sinc(4.0e3))
    events = []
    seen_rf = 0
    for event in description.events:
        if event.type.name == "RF" and event.rf_use is not RfUse.INVERSION:
            seen_rf += 1
            if seen_rf > 1:
                event = SequenceEvent.rf(
                    event.timestamp_us,
                    second.id,
                    event.rf_use,
                    event.rf_amplitude_hz,
                    event.rf_phase_rad,
                )
        events.append(event)
    return replace(
        description,
        events=tuple(events),
        rf_definitions={0: _sinc(4.0e3), second.id: second},
    )


def test_each_pulse_reads_the_table_of_its_own_shape() -> None:
    """The excitation and the refocusing need not be the same pulse.

    A sequence playing two shapes gets a table each, and the event says which
    it drives. If the index were ignored every pulse would read one of them,
    so the mixed sequence must sit apart from both sequences that play only
    one shape.
    """
    mixed = _two_shapes(_sinc(1.0e3, rf_id=1))
    only_wide = _describe(definition=_sinc(4.0e3))
    only_narrow = _describe(definition=_sinc(1.0e3))

    profile = exact_slice_profile(9)
    both = _signal(mixed, profile)
    wide = _signal(only_wide, profile)
    narrow = _signal(only_narrow, profile)

    assert (both - wide).abs().max() > 1e-3 * wide.abs().max()
    assert (both - narrow).abs().max() > 1e-3 * narrow.abs().max()


def test_two_shapes_that_are_the_same_shape_are_the_one_table_twice() -> None:
    """The index is what changed, so pointing it at equal pulses changes nothing."""
    twinned = _two_shapes(_sinc(4.0e3, rf_id=1))
    single = _describe(definition=_sinc(4.0e3))

    profile = exact_slice_profile(9)
    assert (
        _signal(twinned, profile) - _signal(single, profile)
    ).abs().max() < 1e-5 * _signal(single, profile).abs().max()


def test_a_sequence_with_no_shaped_pulse_is_refused() -> None:
    from dataclasses import replace

    description = _describe()
    description = replace(
        description,
        events=tuple(
            event for event in description.events if event.type.name != "RF"
        ),
    )
    with pytest.raises(ValueError, match="no shaped pulse"):
        _signal(description, exact_slice_profile(5))


def test_the_operator_loop_refuses_an_exact_profile() -> None:
    """It applies a flip and a phase, which cannot name a general rotation."""
    description = _describe(definition=_sinc(4.0e3))
    with pytest.raises(NotImplementedError, match="tabulated rotation"):
        _signal(description, exact_slice_profile(5), backend="torch")


def test_a_slice_needs_at_least_one_position() -> None:
    with pytest.raises(ValueError, match="at least one position"):
        exact_slice_profile(0).positions()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_card_reads_each_shape_the_host_reads() -> None:
    mixed = _two_shapes(_sinc(1.0e3, rf_id=1))
    profile = exact_slice_profile(9)
    tissue = _tissue(b1=torch.tensor([0.8, 1.2]))
    host = _signal(mixed, profile, tissue=tissue)
    card = FSE().simulate(
        mixed,
        tissue,
        slice_profile=profile,
        nstates=STATES,
        backend="native",
        device="cuda",
    ).signal

    assert (host - card.cpu()).abs().max() < 1e-4 * host.abs().max()
