"""Integrating a sequence's own pulse across the slice, from the public API.

A slice profile is a Bloch response, so it is worked out from the RF definition
the description already carries rather than named by the caller.
:func:`exact_slice_profile` says only where across the slice to sample it.

The anchor is a pulse with no gradient across it, whose rotation is the same
everywhere along the slice and must therefore reproduce a simulation that
integrates nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim import (
    EpgEngine,
    exact_slice_profile,
    fse_description,
    rf_definition,
)
from torchsim.sequence._description import RfDefinition
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


DWELL = 1.0e-5


def _sinc(bandwidth_hz: float, samples: int = 128, rf_id: int = 0) -> RfDefinition:
    """A windowed sinc, scaled so an event's flip is the flip it turns.

    Built through :func:`rf_definition` rather than assembled by hand, because
    a definition whose envelope is not scaled against its own integral turns a
    small fraction of what the event asks for. Every assertion below is then
    against a signal of order one, where a shaped pulse quietly turning nothing
    would show up.
    """
    grid = np.linspace(-2.0, 2.0, samples)
    envelope = np.sinc(grid) * (0.54 + 0.46 * np.cos(np.pi * grid / 2.0))
    return rf_definition(
        envelope.astype(np.complex128),
        dwell_s=DWELL,
        bandwidth_hz=bandwidth_hz,
        definition_id=rf_id,
    )


def _tissue(**overrides):
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        **overrides,
    )


def _signal(description, profile=None, tissue=None):
    return (
        EpgEngine()
        .simulate(
            description,
            _tissue() if tissue is None else tissue,
            slice_profile=profile,
            nstates=STATES,
        )
        .signal
    )


def test_a_pulse_with_no_gradient_across_it_is_no_profile_at_all() -> None:
    """The anchor: same rotation everywhere, so the mean over it is itself."""
    description = _describe()
    plain = _signal(description)
    exact = _signal(description, exact_slice_profile(5))

    assert exact.shape == plain.shape
    assert (exact - plain).abs().max() < 1e-5 * plain.abs().max()


def test_the_slice_shows_up_when_the_pulse_selects_one() -> None:
    """A real sinc under its gradient is not the pulse at the slice centre."""
    description = _describe(definition=_sinc(4.0e3))
    centre = _signal(description)
    exact = _signal(description, exact_slice_profile(15))

    assert (exact - centre).abs().max() > 0.05 * centre.abs().max()


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

    from torchsim.sequence._description import RfUse

    description = _describe(definition=_sinc(4.0e3))
    events = []
    seen_rf = 0
    for event in description.events:
        if event.type.name == "RF" and event.rf_use is not RfUse.INVERSION:
            seen_rf += 1
            if seen_rf > 1:
                event = replace(event, params=(second.id, *event.params[1:]))
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


def test_a_sequence_with_no_pulse_at_all_is_refused() -> None:
    from dataclasses import replace

    description = _describe()
    description = replace(
        description,
        events=tuple(event for event in description.events if event.type.name != "RF"),
    )
    with pytest.raises(ValueError, match="no pulse to take it from"):
        _signal(description, exact_slice_profile(5))


def test_a_slice_needs_at_least_one_position() -> None:
    with pytest.raises(ValueError, match="at least one position"):
        exact_slice_profile(0).positions()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_card_reads_each_shape_the_host_reads() -> None:
    mixed = _two_shapes(_sinc(1.0e3, rf_id=1))
    profile = exact_slice_profile(9)
    tissue = _tissue(b1=torch.tensor([0.8, 1.2]))
    host = _signal(mixed, profile, tissue=tissue)
    card = (
        EpgEngine()
        .simulate(
            mixed,
            tissue,
            slice_profile=profile,
            nstates=STATES,
            device="cuda",
        )
        .signal
    )

    assert (host - card.cpu()).abs().max() < 1e-4 * host.abs().max()


# --- the waveform decides, not the call ---


def test_a_shaped_pulse_is_integrated_without_being_asked() -> None:
    """A definition carrying a waveform worth integrating is tabulated on its
    own, so the caller never names a mode -- only where to sample it.
    """
    from torchsim.sequence._accelerators import _wants_a_table

    shaped = _describe(definition=_sinc(4.0e3))
    hard = _describe()

    assert _wants_a_table(shaped, None)
    assert not _wants_a_table(hard, None)
    assert _wants_a_table(hard, exact_slice_profile(5))


def test_a_builder_pulse_stays_the_hard_pulse_to_the_bit() -> None:
    """Everything the builders emit is a rectangle played without slice
    selection, which the instant operator turns exactly. Tabulating one and
    reading it at a flip between knots would not, so it must not happen by
    itself.
    """
    from torchsim.sequence._accelerators import _run_packed

    description = _describe()
    seen = {}
    signal = None

    from torchsim.sequence import _accelerators

    original = _accelerators._run_packed

    def record(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    _accelerators._run_packed = record
    try:
        signal = _signal(description)
    finally:
        _accelerators._run_packed = original

    assert seen["profile"] is None
    assert seen["dynamic"] is None
    assert float(signal.abs().max()) > 0.0
    assert _run_packed is original


def test_a_flip_angle_scaling_is_refused() -> None:
    """A response is not proportional to the pulse driving it, so a tensor of
    scalings is not a slice profile and is not taken for one.
    """
    with pytest.raises(TypeError, match="exact_slice_profile"):
        _signal(_describe(), torch.linspace(0.4, 1.0, 5))


def test_the_table_lays_its_copies_out_voxel_major() -> None:
    """The mean at the end folds the last axis, so the copies have to be there
    and each voxel's have to be adjacent.
    """
    from torchsim.sequence._accelerators import _across_the_table

    tissue = tuple(torch.tensor([800.0, 1400.0], dtype=torch.float32) for _ in range(7))
    spread = _across_the_table(tissue, 3)

    assert spread[0].tolist() == [800.0, 800.0, 800.0, 1400.0, 1400.0, 1400.0]
    # A position is not a scaling of anything, so the transmit is left alone.
    assert torch.equal(spread[3], spread[0])
    assert _across_the_table(tissue, 1)[0].numel() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_positions_are_counted_by_the_memory_policy() -> None:
    """Spreading happens before the launch, so a streamed run has to see the
    copies as the voxels rather than budget for the volume it started with.
    """
    from test_offload import _peak_over_baseline

    from torchsim.sequence import offload

    voxels, points = 40_000, 5
    budget = 8 << 20
    description = _describe(definition=_sinc(4.0e3))
    tissue = TissueProperties(
        t1_ms=torch.linspace(300.0, 2000.0, voxels),
        t2_ms=torch.linspace(20.0, 200.0, voxels),
    )
    profile = exact_slice_profile(points)
    expected = _signal(description, profile, tissue=tissue)
    streamed = []

    def run():
        with offload(["cuda"], budget_bytes=budget):
            streamed.append(_signal(description, profile, tissue=tissue))

    resident = _peak_over_baseline(run)

    assert resident <= budget * 1.1
    worst = (streamed[0] - expected).abs().max() / expected.abs().max()
    assert worst < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("state_count", [8, 12, 17])
def test_a_table_takes_the_first_order_kernel_on_the_card(state_count) -> None:
    """A tabulated rotation does not cost the kernel written for a gradient.

    The flip gradient comes off the table's own slope here rather than off a
    differentiated rotation, which is the part a single width would not check.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import (
        _pack_events,
        _run_packed_vjp,
        geometry_of,
    )
    from torchsim.sequence._simulation import _prepare_tissue
    from torchsim.sequence._transition import transition_table

    description = _describe(definition=_sinc(4.0e3))
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cuda"),
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
        packed.shim_index,
        packed.saturation,
        packed.rf_frequency_hz,
    )
    prepared, _, _ = _prepare_tissue(_tissue(), "cuda")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    table = transition_table(
        description.rf_definitions[0],
        torch.zeros(1, dtype=torch.float64),
        bins=64,
        rf_raster_time_s=1e-6,
        device="cuda",
    )
    outputs = int(packed.output_count)
    seed = torch.ones(
        prepared[0].numel(), outputs, dtype=torch.complex64, device="cuda"
    )
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )
    arguments = dict(
        state_count=state_count,
        output_count=outputs,
        threads=1,
        profile=table,
        geometry=geometry_of(description),
    )

    reached = []
    original = _accelerators._run_packed_vjp_jvp

    def record(*args, **kwargs):
        reached.append(True)
        return original(*args, **kwargs)

    _accelerators._run_packed_vjp_jvp = record
    try:
        fast = _run_packed_vjp(prepared, events, seed, **arguments)
    finally:
        _accelerators._run_packed_vjp_jvp = original
    _, expected = original(prepared, events, still, seed, **arguments)

    assert not reached
    largest = max(float(value.abs().max()) for value in expected)
    assert largest > 0.0
    for reference, result in zip(expected, fast, strict=True):
        assert float((reference - result).abs().max()) / largest < 1e-5
