"""The terms a launch carries, chosen per property.

:mod:`test_features` pins what a tissue asks for; this pins what the kernels
then do with that answer. The gated kernel and the full one must give the same
numbers, and the terms that were left out must be the ones whose derivative is
genuinely zero -- so each case here either compares the two kernels or shows a
gradient the gate would have had to lie about.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import EpgEngine, TissueProperties, fse_description
from torchsim.sequence import _epg_triton
from torchsim.sequence._epg_triton import _feature_flags
from torchsim.sequence._parameters import (
    TISSUE_NAMES,
    TISSUE_PARAMETERS,
    Geometry,
    at_identity,
)

ECHOES = 8
STATES = 8

WINDING = Geometry(flow_scale=120.0, washout_scale=4.0)
STILL = Geometry()


def _parameter(name: str):
    return TISSUE_PARAMETERS[TISSUE_NAMES.index(name)]


def _description(crusher_rad: float = 0.0):
    """The train the kernels are compared on.

    A crusher is what gives a diffusion coefficient anything to attenuate:
    the rate the kernels read is the coefficient times the winding across a
    voxel, squared, so a train with no unbalanced gradient leaves it at zero
    however large the coefficient.
    """
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        crusher_dephasing_rad=crusher_rad,
        voxel_size_m=None if crusher_rad == 0.0 else 1e-3,
    )


CRUSHED = 2.0 * torch.pi


# --- what the flags say ---


def test_a_caller_who_declares_nothing_keeps_every_term():
    """The reading every existing entry point takes until it declares."""
    flags = _feature_flags(None, WINDING)
    assert all(flags.values())


def test_each_absent_property_takes_its_term_with_it():
    live = frozenset({"T1", "T2"})
    assert _feature_flags(live, WINDING) == {
        "off_axis": False,
        "moving": False,
        "diffusing": False,
        "transmit": False,
        "density": False,
        "inverting": False,
    }


def test_a_scalar_at_its_identity_is_named_one_property_at_a_time():
    """The three per-voxel scalars are three switches, so declaring a transmit
    map must not turn a proton density on with it.
    """
    flags = _feature_flags(frozenset({"T1", "T2", "B1"}), WINDING)

    assert flags["transmit"]
    assert not flags["density"]
    assert not flags["inverting"]


def test_the_attenuation_answers_to_itself_alone():
    """Diffusion shares no switch: neither static phase turns it on, and a
    sequence with no gradient geometry to speak of still attenuates.
    """
    assert _feature_flags(frozenset({"T1", "DIFFUSION"}), STILL)["diffusing"]
    assert not _feature_flags(frozenset({"T1", "B0", "FLOW"}), WINDING)["diffusing"]


@pytest.mark.parametrize("name", ["B0", "B1_PHASE"])
def test_either_static_phase_switches_the_same_term_on(name: str) -> None:
    """Off-resonance and transmit phase turn the states through one operator,
    so one switch answers for both -- a tissue with either pays for both.
    """
    flags = _feature_flags(frozenset({"T1", "T2", name}), WINDING)
    assert flags["off_axis"]
    assert not flags["moving"]


def test_a_moving_voxel_in_a_sequence_that_declares_no_geometry_stands_still():
    """Velocity reaches the state machine only through the two scales, so a
    sequence that winds no phase and draws in no fresh spins has nothing to do
    with it however fast the voxel moves.
    """
    assert not _feature_flags(frozenset({"FLOW"}), STILL)["moving"]


@pytest.mark.parametrize(
    "geometry", [Geometry(flow_scale=120.0), Geometry(washout_scale=4.0)]
)
def test_either_geometry_scale_is_enough_to_set_the_voxel_moving(geometry):
    """The two scales drive different terms but one switch carries both, so a
    sequence declaring either keeps the velocity arithmetic.
    """
    assert _feature_flags(frozenset({"FLOW"}), geometry)["moving"]


# --- the trap forward mode sets ---


def test_a_value_carrying_a_forward_direction_asks_for_its_term():
    """A dual number sits at its primal, and its primal may be the identity.

    Dropping the term there would return zero for a directional derivative
    that is not zero, which is the same lie the reverse-mode guard refuses --
    and a forward tangent is not a ``requires_grad`` flag, so it needs asking
    for separately.
    """
    from torch.autograd.forward_ad import dual_level, make_dual

    parameter = _parameter("b1_phase_rad")
    with dual_level():
        held = make_dual(torch.tensor([0.0]), torch.tensor([1.0]))
        assert float(held.item()) == parameter.identity
        assert not held.requires_grad
        assert not at_identity(parameter, held)


def test_the_transmit_phase_derivative_survives_at_zero_phase():
    """The end of that trap: forward mode through the whole simulation, along
    a property sitting exactly where its term would be dropped.
    """
    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    def signal(phase):
        return (
            EpgEngine()
            .simulate(
                _description(),
                TissueProperties(
                    t1_ms=torch.tensor([1000.0]),
                    t2_ms=torch.tensor([80.0]),
                    b1_phase_rad=phase,
                ),
                nstates=STATES,
            )
            .signal
        )

    with dual_level():
        held = make_dual(torch.tensor([0.0]), torch.tensor([1.0]))
        derivative = unpack_dual(signal(held)).tangent
    assert derivative is not None

    step = 1e-3
    difference = (signal(torch.tensor([step])) - signal(torch.tensor([-step]))) / (
        2.0 * step
    )
    assert float(derivative.abs().max()) > 0.0
    assert torch.allclose(derivative, difference, rtol=2e-3, atol=1e-5)


# --- the gated kernel against the full one ---


def _adjoint(crusher_rad=0.0, **properties):
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (("t1_ms", 1000.0), ("t2_ms", 80.0))
    }
    held = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in properties.items()
    }
    signal = (
        EpgEngine()
        .simulate(
            _description(crusher_rad),
            TissueProperties(
                **{name: value.cuda() for name, value in leaves.items()}, **held
            ),
            nstates=STATES,
        )
        .signal
    )
    signal.abs().square().sum().backward()
    return tuple(leaves[name].grad.clone() for name in leaves)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("properties", "crusher_rad"),
    [
        ({}, 0.0),
        ({"b0_hz": torch.tensor([30.0])}, 0.0),
        ({"b1_phase_rad": torch.tensor([0.4])}, 0.0),
        ({"diffusion_um2_per_ms": torch.tensor([2.0])}, CRUSHED),
    ],
    ids=["bare", "off-resonance", "transmit-phase", "diffusion"],
)
def test_the_answer_does_not_depend_on_the_gate(
    monkeypatch, properties, crusher_rad
) -> None:
    """Whatever the mask leaves out, the numbers it leaves behind are the ones
    the full kernel gives.
    """
    gated = _adjoint(crusher_rad, **properties)
    monkeypatch.setattr(
        _epg_triton,
        "_feature_flags",
        lambda features, geometry: _feature_flags(None, geometry),
    )
    whole = _adjoint(crusher_rad, **properties)
    for one, other in zip(gated, whole, strict=True):
        assert torch.allclose(one, other, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_an_off_resonance_asked_for_its_gradient_gets_a_real_one():
    """The gate is a choice, not a coincidence: the term it can drop is one
    that moves the answer when it is there.
    """
    b0 = torch.tensor([30.0], device="cuda", requires_grad=True)
    signal = (
        EpgEngine()
        .simulate(
            _description(),
            TissueProperties(
                t1_ms=torch.tensor([1000.0], device="cuda"),
                t2_ms=torch.tensor([80.0], device="cuda"),
                b0_hz=b0,
            ),
            nstates=STATES,
        )
        .signal
    )
    signal.abs().square().sum().backward()
    assert float(b0.grad.abs().max()) > 0.0


# --- the gated forward and forward-mode kernels against the full ones ---

# Enough voxels for the launch to look like a real one; the comparison is
# between two kernels on the same input, so the count only has to be honest.
GATE_VOXELS = 4096

# Each case leaves out a term the ungated launch still carries, so the two
# arms compile to different kernels rather than to the same one twice.
GATE_CASES = [
    ({}, STILL),
    ({}, WINDING),
    ({"b0_hz": 42.0}, WINDING),
    ({"b1_phase_rad": 0.35}, WINDING),
    ({"velocity_m_per_s": 0.08}, STILL),
    ({"diffusion_um2_per_ms": 2.0}, WINDING),
]
GATE_IDS = [
    "bare",
    "still-voxel",
    "off-resonance",
    "transmit-phase",
    "flow",
    "diffusion",
]


def _gate_inputs(extra):
    from torchsim.sequence._accelerators import _pack_events
    from torchsim.sequence._simulation import _prepare_tissue

    generator = torch.Generator().manual_seed(7)

    def spread(low, high):
        return low + (high - low) * torch.rand(GATE_VOXELS, generator=generator)

    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=spread(600.0, 1800.0), t2_ms=spread(30.0, 150.0), **extra
        ),
        "cuda",
    )
    packed = _pack_events(
        _description(),
        repetitions=1,
        record="all",
        device=torch.device("cuda"),
        rf_raster_time_s=1e-6,
    )
    return prepared, packed.buffers, packed.output_count


def _seeds(prepared, events, live_names):
    """A forward direction along each declared property and each event field.

    An undeclared property is not an input, so seeding one would be asking for
    a derivative that does not exist -- and the two arms would part over the
    seed rather than over the kernel.
    """
    generator = torch.Generator(device="cuda").manual_seed(11)
    directions = []
    for name, value in zip(TISSUE_NAMES, prepared, strict=True):
        directions.append(
            torch.rand(
                value.shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            if name in live_names
            else torch.zeros_like(value)
        )
    for value in (events[0], events[2], events[3]):
        directions.append(
            torch.rand(
                value.shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
        )
    return tuple(directions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(("extra", "geometry"), GATE_CASES, ids=GATE_IDS)
def test_the_forward_answer_does_not_depend_on_the_gate(extra, geometry):
    """The forward kernel, run once with the terms the tissue asks for and
    once with every term, on the same input.
    """
    from torchsim.sequence._accelerators import _run_packed
    from torchsim.sequence._parameters import features_of

    prepared, events, output_count = _gate_inputs(extra)
    live = features_of(
        TissueProperties(t1_ms=torch.tensor([1.0]), t2_ms=torch.tensor([1.0]), **extra)
    )

    def run(features):
        # The real-subspace kernel would answer both arms with one body, so
        # the complex kernel -- the one carrying the flags -- is asked for.
        return _run_packed(
            prepared,
            events,
            STATES,
            output_count,
            1,
            0,
            geometry=geometry,
            features=features,
        )

    assert torch.equal(run(live), run(None))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(("extra", "geometry"), GATE_CASES, ids=GATE_IDS)
def test_the_forward_direction_does_not_depend_on_the_gate(extra, geometry):
    """The same for forward mode, seeded along every declared input at once so
    a term the gate drops has somewhere to show up.

    Held to a tolerance where the forward is held to the bit, because the
    terms the gate drops enter a direction as addends rather than as the
    factors of one they are in the value. Dropping an addend leaves the
    product beside it to be rounded on its own, which moves the last bits: the
    value half of this same pair is exact.
    """
    from torchsim.sequence._accelerators import _run_packed_jvp
    from torchsim.sequence._parameters import features_of

    prepared, events, output_count = _gate_inputs(extra)
    seeds = _seeds(prepared, events, {"t1_ms", "t2_ms", *extra})
    live = features_of(
        TissueProperties(t1_ms=torch.tensor([1.0]), t2_ms=torch.tensor([1.0]), **extra)
    )

    def run(features):
        return _run_packed_jvp(
            prepared,
            events,
            seeds[: len(prepared)],
            seeds[len(prepared) :],
            STATES,
            output_count,
            1,
            0,
            geometry=geometry,
            features=features,
        )

    gated, whole = run(live), run(None)
    drift = float((gated - whole).abs().max())
    assert drift <= 2e-6 * float(whole.abs().max()), drift


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_diffusion_coefficient_asked_for_its_gradient_gets_a_real_one():
    """The attenuation gate is a choice too: the term it drops is one that
    moves the answer when the tissue declares it.
    """
    coefficient = torch.tensor([2.0], device="cuda", requires_grad=True)
    signal = (
        EpgEngine()
        .simulate(
            _description(CRUSHED),
            TissueProperties(
                t1_ms=torch.tensor([1000.0], device="cuda"),
                t2_ms=torch.tensor([80.0], device="cuda"),
                diffusion_um2_per_ms=coefficient,
            ),
            nstates=STATES,
        )
        .signal
    )
    signal.abs().square().sum().backward()
    assert float(coefficient.grad.abs().max()) > 0.0
