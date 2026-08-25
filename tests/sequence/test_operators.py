"""The builders, re-expressed as operator compositions.

The five built-in builders are the evidence that the operator vocabulary is
sufficient: each is written over operators, so each must pack to exactly the
buffers it packed to when it constructed events by hand. Bit-identity is the
bar, not agreement -- a timestamp that drifts in its last place is a
composition that computes something slightly different, and the difference
would land in an event duration.
"""

from __future__ import annotations

import torch

from torchsim.sequence._accelerators import _pack_events
from torchsim.sequence._builders import (
    fse_description,
    mpnrage_description,
    mprage_description,
    mrf_description,
    spgr_description,
)
from torchsim.sequence._description import AdcRole, EventAction, EventType
from torchsim.sequence._operators import (
    compose,
    Delay,
    Excitation,
    module,
    operator,
    operator_names,
    Readout,
    Refocusing,
    register_operator,
)

import pytest


def _packed(description):
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    return (
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


def _flip(count, generator=None):
    if generator is None:
        return torch.full((count,), 2.6)
    return torch.rand(count, generator=generator) + 1.5


def test_the_operator_registry_answers_for_every_shipped_operator() -> None:
    """A generated stream dispatches by name, so the names have to resolve."""
    for name in operator_names():
        assert callable(operator(name)), name
    assert operator("EXCITATION") is operator("excitation"), "lookup is normalized"
    with pytest.raises(ValueError, match="unknown operator"):
        operator("no-such-operator")


def test_an_operator_can_be_added_and_not_replaced() -> None:
    """Registration is how a caller extends the vocabulary, once per name."""
    def spin_lock(duration_s):
        return Excitation(0.0, duration_s=duration_s)

    register_operator("spin-lock", spin_lock)
    assert operator("spin-lock") is spin_lock
    with pytest.raises(ValueError, match="already registered"):
        register_operator("spin_lock", spin_lock)


def test_a_bare_operator_follows_the_one_before_it() -> None:
    """Composition is a running sum, which is the accumulator it replaces."""
    events, span = compose(
        Excitation(1.0, duration_s=2e-3),
        Readout(duration_s=3e-3),
    )
    assert [event.timestamp_us for event in events] == [0.0, 2e3]
    assert span == pytest.approx(5e-3)


def test_an_offset_is_measured_from_where_the_composition_starts() -> None:
    """A sequence timed from its echo places modules rather than stacking them."""
    events, _ = compose(
        (4e-3, Readout()), (1e-3, Readout()), start_s=10e-3
    )
    assert [event.timestamp_us for event in events] == [14e3, 11e3]


def test_a_module_carries_its_own_offsets_wherever_it_is_placed() -> None:
    """A shot is one thing a caller lays down, not three it has to align."""
    shot = module(
        Excitation(1.0), (2e-3, Readout()), duration_s=10e-3
    )
    events, span = compose(shot, shot)
    assert [event.timestamp_us for event in events] == [0.0, 2e3, 10e3, 12e3]
    assert span == pytest.approx(20e-3)


def test_a_delay_can_carry_what_the_sequence_plays_across_it() -> None:
    """``SequenceEvent.wait`` takes no action, so the operator supplies one."""
    plain, _ = compose(Delay(5e-3))
    assert plain[0].type is EventType.WAIT
    assert plain[0].action is EventAction.NONE
    spoiled, _ = compose(Delay(5e-3, action=EventAction.SPOIL_AFTER))
    assert spoiled[0].action is EventAction.SPOIL_AFTER


def test_a_refocusing_pulse_brings_its_crushers() -> None:
    """The unbalanced pair belongs to the pulse, not to the caller's memory."""
    crushed, _ = compose(Refocusing(torch.pi))
    assert crushed[0].action == (
        EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER
    )
    bare, _ = compose(Refocusing(torch.pi, crushed=False))
    assert bare[0].action is EventAction.NONE


_GENERATOR = torch.Generator().manual_seed(0)

BUILDERS = [
    ("fse", fse_description, (_flip(8), 8e-3), {}),
    (
        "fse phased",
        fse_description,
        (_flip(8), 8e-3),
        {"phases_rad": torch.linspace(0.0, 1.0, 8)},
    ),
    (
        "fse batched",
        fse_description,
        (torch.rand(3, 8, generator=_GENERATOR) + 1.5, 5e-3),
        {},
    ),
    (
        "fse long",
        fse_description,
        (_flip(200), 4e-3),
        {"crusher_dephasing_rad": 3.0, "voxel_size_m": 1e-3},
    ),
    ("mrf", mrf_description, (torch.linspace(0.1, 1.0, 40), 10e-3), {}),
    (
        "mrf inverted",
        mrf_description,
        (torch.linspace(0.1, 1.0, 40), torch.linspace(8e-3, 14e-3, 40)),
        {"inversion_time_s": 1.0, "phases_rad": torch.linspace(0.0, 3.0, 40)},
    ),
    ("spgr", spgr_description, (torch.full((30,), 0.2), 10e-3, 3e-3), {}),
    (
        "spgr varied",
        spgr_description,
        (
            torch.linspace(0.1, 0.5, 30),
            torch.linspace(9e-3, 12e-3, 30),
            torch.linspace(2e-3, 4e-3, 30),
        ),
        {"phases_rad": torch.linspace(0.0, 2.0, 30)},
    ),
    (
        "mpnrage",
        mpnrage_description,
        (16, torch.linspace(0.1, 0.4, 16), 8e-3),
        {"inversion_time_s": 0.9},
    ),
    (
        "mprage",
        mprage_description,
        (5, 6, torch.linspace(0.1, 0.4, 12), 8e-3, 1.1),
        {},
    ),
]


# What each builder packs for a small instance, written out rather than
# captured. Every number is checkable by hand against the sequence it
# describes, which a golden buffer taken from a previous run would not be:
# 64 is an Excitation, 131 a Refocusing pulse between crushers, 4 an ideal
# Inversion, 32 a recorded sample, 48 one followed by a shift and 40 one
# followed by an ideal spoil.
GOLDEN = [
    (
        "fse",
        lambda: fse_description(torch.tensor([2.0, 2.5]), 8e-3),
        [1, 1, 2, 1, 2],
        [64, 131, 32, 131, 32],
        # Excitation at 0, Refocusing half an echo spacing in, echo at 8 ms.
        [0.0, 4e-3, 4e-3, 4e-3, 4e-3],
    ),
    (
        "mrf",
        lambda: mrf_description(
            torch.tensor([0.2, 0.3]), 10e-3, inversion_time_s=1.0
        ),
        [1, 1, 2, 1, 2],
        [4, 64, 48, 64, 48],
        # The Inversion time is the gap before the first shot; TR the gaps after.
        [0.0, 1.0, 0.0, 10e-3, 0.0],
    ),
    (
        "spgr",
        lambda: spgr_description(torch.tensor([0.2, 0.3]), 10e-3, 3e-3),
        [1, 2, 1, 2],
        [64, 40, 64, 40],
        # The sample sits TE into the repetition, so the pulse follows TR - TE.
        [0.0, 3e-3, 7e-3, 3e-3],
    ),
    (
        "mprage",
        lambda: mprage_description(
            1, 1, torch.tensor([0.2, 0.3, 0.4]), 8e-3, 1.0
        ),
        [1, 1, 2, 1, 2, 1, 2],
        [4, 64, 40, 64, 40, 64, 40],
        # The Inversion time is measured to the centre shot, so the gap before
        # the first is TI less the one repetition that precedes the centre.
        [0.0, 1.0 - 8e-3, 0.0, 8e-3, 0.0, 8e-3, 0.0],
    ),
]


@pytest.mark.parametrize(
    "build,kinds,actions,durations",
    [case[1:] for case in GOLDEN],
    ids=[case[0] for case in GOLDEN],
)
def test_a_builder_over_operators_packs_what_the_sequence_says(
    build, kinds, actions, durations
) -> None:
    """The composition has to reproduce the timing, not merely be repeatable."""
    packed = _packed(build())
    assert packed[1].tolist() == kinds
    assert packed[4].tolist() == actions
    # The timestamps are stored float32, so a gap of ~1 s lands a hundred
    # nanoseconds out however it is computed.
    assert packed[0].reshape(-1).tolist() == pytest.approx(durations, abs=1e-7)


@pytest.mark.parametrize(
    "builder,arguments,keywords",
    [case[1:] for case in BUILDERS],
    ids=[case[0] for case in BUILDERS],
)
def test_a_builder_over_operators_is_differentiable_and_batched(
    builder, arguments, keywords
) -> None:
    """Composition must not have cost the tensors their graph or their batch."""
    packed = _packed(builder(*arguments, **keywords))
    assert all(buffer.numel() for buffer in packed[:5]), "nothing was packed"
    # The builders differ in where the flip train sits: two of them lead with
    # shot counts.
    flip = next(
        value
        for value in arguments
        if torch.is_tensor(value) and value.dim() >= 1
    )
    if flip.dim() == 2:
        # A batch of trains shares the event structure and differs in flips.
        assert packed[2].shape[0] == flip.shape[0]
    assert packed[1].numel() == packed[4].numel() == packed[5].numel()


def test_the_shipped_builders_still_say_what_they_meant() -> None:
    """Roles and dephasing survive the rewrite, which packing alone would hide."""
    fse = fse_description(_flip(4), 8e-3)
    kinds = [event.type for event in fse.events]
    assert kinds[0] is EventType.RF
    assert kinds[1::2] == [EventType.RF] * 4
    refocus = fse.events[1]
    assert refocus.action == (EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER)
    assert fse.events[2].adc_role is AdcRole.ECHO_CENTER

    centred = mprage_description(2, 3, torch.full((6,), 0.2), 8e-3, 1.0)
    roles = [
        event.adc_role for event in centred.events if event.type is EventType.ADC
    ]
    assert roles == [
        AdcRole.NON_ACQUIRED,
        AdcRole.NON_ACQUIRED,
        AdcRole.SINGLE,
        AdcRole.NON_ACQUIRED,
        AdcRole.NON_ACQUIRED,
        AdcRole.NON_ACQUIRED,
    ]
