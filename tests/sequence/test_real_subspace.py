"""Detection of sequences whose states never leave a real subspace.

Each case pairs the predicate against the signal it describes, so the rule
cannot drift away from the behaviour it is meant to predict.
"""

from dataclasses import replace
from unittest import mock

import pytest
import torch

from torchsim import EpgEngine, fse_description
from torchsim.sequence import _accelerators as accelerators
from torchsim.sequence._accelerators import (
    _one_axis,
    _pack_events,
    _run_packed,
    real_subspace_axis,
)
from torchsim.sequence._parameters import FLOAT_NAMES, OUTSIDE_THE_SUBSPACE
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

# The gradients a real adjoint does produce: every differentiable input the
# subspace contains. Derived rather than listed, so a new parameter cannot
# leave a stale index pointing at whatever moved into its slot.
INSIDE_THE_SUBSPACE = tuple(
    index for index in range(len(FLOAT_NAMES)) if index not in OUTSIDE_THE_SUBSPACE
)

ECHO_SPACING_S = 5e-3
ECHOES = 10


def _flip() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.deg2rad(80.0 + 80.0 * torch.rand(3, ECHOES, generator=generator))


def _tissue(b0_hz: float, b1_phase_rad: float) -> TissueProperties:
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        b0_hz=torch.tensor([b0_hz, b0_hz]),
        b1_phase_rad=torch.tensor([b1_phase_rad, b1_phase_rad]),
    )


def _axis(phases, excitation, b0_hz=0.0, b1_phase_rad=0.0):
    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=phases,
        excitation_phase_rad=excitation,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    prepared, _, _ = _prepare_tissue(_tissue(b0_hz, b1_phase_rad), "cpu")
    signal = EpgEngine().simulate(description, _tissue(b0_hz, b1_phase_rad)).signal
    return real_subspace_axis(events, prepared), signal


@pytest.mark.parametrize("phase", [0.0, torch.pi / 4, torch.pi / 2])
def test_uniform_rf_phase_stays_imaginary(phase):
    axis, signal = _axis(phase, phase)
    assert axis == 1
    # The predicate claims it; the signal must show it, at every echo.
    assert signal.real.abs().max() < 1e-12 * signal.imag.abs().max().clamp_min(1e-30)


def test_transmit_phase_breaks_the_subspace():
    axis, signal = _axis(torch.pi / 2, torch.pi / 2, b1_phase_rad=0.3)
    assert axis is None
    assert signal.real.abs().max() > 0.1 * signal.imag.abs().max()


def test_alternating_refocusing_phase_breaks_the_subspace():
    phases = torch.full((ECHOES,), torch.pi / 2)
    phases[::2] = 0.0
    axis, signal = _axis(phases, torch.pi / 2)
    assert axis is None
    assert signal.real.abs().max() > 0.1 * signal.imag.abs().max()


def test_off_resonance_is_rejected_despite_real_looking_echoes():
    """The echoes refocus off-resonance; the states in between do not."""
    axis, signal = _axis(torch.pi / 2, torch.pi / 2, b0_hz=20.0)
    assert axis is None
    # The recorded samples alone would wrongly suggest the subspace holds.
    assert signal.real.abs().max() < 1e-5 * signal.imag.abs().max()


def test_quarter_turn_excitation_is_rejected_despite_a_real_signal():
    """CPMG: excitation a quarter turn from the refocusing pulses.

    Every recorded echo is real, and the states are not. Splitting the state
    into real and imaginary parts and asking how far each configuration sits
    off a single line puts them 97% off it, so there is no real subspace here
    to run a cheaper kernel on -- only a real projection of a complex one.
    """
    axis, signal = _axis(torch.pi / 2, 0.0)
    assert axis is None
    # The signal alone would wrongly suggest a subspace, at every echo.
    assert signal.imag.abs().max() < 1e-6 * signal.real.abs().max()


def _buffers(packed):
    return packed.buffers


@pytest.mark.parametrize("phase", [0.0, torch.pi / 4, torch.pi / 2])
def test_real_kernel_reproduces_the_complex_one(phase):
    """The real kernel is an optimization, not an approximation."""
    from torchsim.sequence._accelerators import _run_packed

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=phase,
        excitation_phase_rad=phase,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    complex_signal = _run_packed(prepared, events, 10, packed.output_count, 1)
    real_signal = _run_packed(prepared, events, 10, packed.output_count, 1, real_axis=1)
    scale = complex_signal.abs().max()
    assert ((complex_signal - real_signal).abs().max() / scale) < 1e-6


def test_real_jvp_reproduces_the_complex_one():
    """Forward mode along T2 stays inside the subspace, so the kernels agree."""
    from torchsim.sequence._accelerators import _run_packed_jvp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    event_seed = (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    arguments = (prepared, events, t2_seed, event_seed, 10, packed.output_count, 1)
    expected = _run_packed_jvp(*arguments)
    actual = _run_packed_jvp(*arguments, real_axis=1)
    scale = expected.abs().max()
    assert ((expected - actual).abs().max() / scale) < 1e-6


@pytest.mark.parametrize("phase", [0.0, torch.pi / 4, torch.pi / 2])
def test_real_adjoint_reproduces_the_complex_one(phase, always_worth_detecting):
    """The first-order adjoint through the real subspace, against the kernel it
    specializes: every gradient the subspace contains, and zero for the four it
    divides out.

    The caller has to say what it will read -- the four are not produced at all
    -- so the verdict is reached through ``wanted`` rather than handed over,
    which is the route the public adjoint takes.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=phase,
        excitation_phase_rad=phase,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    torch.manual_seed(0)
    seed = torch.randn(
        (events[2].shape[0], prepared[0].numel(), packed.output_count),
        dtype=torch.complex64,
    )
    arguments = dict(state_count=10, output_count=packed.output_count, threads=1)
    expected = _run_packed_vjp(prepared, events, seed, **arguments)
    wanted = tuple(index in INSIDE_THE_SUBSPACE for index in range(len(FLOAT_NAMES)))
    actual = _run_packed_vjp(prepared, events, seed, wanted=wanted, **arguments)

    compared = 0
    for index in INSIDE_THE_SUBSPACE:
        scale = expected[index].abs().max()
        if scale == 0:
            assert actual[index].abs().max() == 0
            continue
        assert ((expected[index] - actual[index]).abs().max() / scale) < 1e-5
        compared += 1
    assert compared > 3

    for index in OUTSIDE_THE_SUBSPACE:
        assert actual[index].abs().max() == 0


def test_the_real_adjoint_reaches_the_inversion_gradient():
    """An FSE never inverts, so the trains the other cases use leave the
    inversion efficiency at zero in both kernels and prove nothing about it.

    The train is a builder's, with an inversion and a delay spliced in front:
    the representation describes a sequence the builders produce, and a train
    assembled from scratch is not one.
    """
    from torchsim.sequence._accelerators import (
        _INVERSION,
        _run_packed_vjp,
    )

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    trains = packed.buffers[0].shape[0]

    def ahead(values, first):
        """The two prepended events, on whichever axis the buffer carries."""
        head = torch.tensor(first, dtype=values.dtype)
        if values.dim() == 2:
            return torch.cat([head.expand(trains, 2), values], dim=1).contiguous()
        return torch.cat([head, values]).contiguous()

    duration, kind, flip, phase, action, output, shim, saturation, frequency = (
        packed.buffers
    )
    events = (
        ahead(duration, [0.0, 20e-3]),
        ahead(kind, [1, 0]),
        ahead(flip, [0.0, 0.0]),
        ahead(phase, [0.0, 0.0]),
        ahead(action, [_INVERSION, 0]),
        ahead(output, [-1, -1]),
        ahead(shim, [0, 0]),
        ahead(saturation, [0.0, 0.0]),
        ahead(frequency, [0.0, 0.0]),
    )
    tissue = TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        inversion_efficiency=torch.tensor([0.92, 0.97]),
    )
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    assert real_subspace_axis(events, prepared) == 1

    torch.manual_seed(0)
    seed = torch.randn(
        (trains, prepared[0].numel(), packed.output_count), dtype=torch.complex64
    )
    arguments = dict(state_count=10, output_count=packed.output_count, threads=1)
    expected = _run_packed_vjp(prepared, events, seed, **arguments)
    wanted = tuple(index in INSIDE_THE_SUBSPACE for index in range(len(FLOAT_NAMES)))
    actual = _run_packed_vjp(prepared, events, seed, wanted=wanted, **arguments)

    position = FLOAT_NAMES.index("inversion_efficiency")
    scale = float(expected[position].abs().max())
    assert scale > 0.0
    assert float((expected[position] - actual[position]).abs().max()) / scale < 1e-5


def test_real_second_order_kernel_reproduces_the_complex_one():
    """Forward-over-reverse agrees on every gradient the subspace contains."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    tangents = (
        *t2_seed,
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    torch.manual_seed(0)
    seed = torch.randn(
        (events[2].shape[0], prepared[0].numel(), packed.output_count),
        dtype=torch.complex64,
    )
    arguments = dict(state_count=10, output_count=packed.output_count, threads=1)
    expected, _ = _run_packed_vjp_jvp(prepared, events, tangents, seed, **arguments)
    actual, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, real_axis=1, **arguments
    )

    for index in INSIDE_THE_SUBSPACE:
        scale = expected[index].abs().max()
        if scale == 0:
            assert actual[index].abs().max() == 0
            continue
        assert ((expected[index] - actual[index]).abs().max() / scale) < 1e-5

    # b1_phase, b0 and RF phase leave the subspace and are not produced.
    for index in OUTSIDE_THE_SUBSPACE:
        assert actual[index].abs().max() == 0


def _packed(trains):
    generator = torch.Generator().manual_seed(0)
    flip = torch.deg2rad(80.0 + 80.0 * torch.rand(trains, ECHOES, generator=generator))
    description = fse_description(
        flip,
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    return _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )


# The real kernels process a fixed-width block of trains at a time, so a train
# count that does not fill the last block leaves lanes that carry a repeat of
# the block's first train. These counts straddle that boundary.
@pytest.mark.parametrize("trains", [1, 7, 8, 9, 17])
def test_partial_train_blocks_match_the_complex_kernel(trains):
    from torchsim.sequence._accelerators import _run_packed_jvp, _run_packed_vjp_jvp

    packed = _packed(trains)
    events = _buffers(packed)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(events, prepared) == 1

    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    event_seed = (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    arguments = (prepared, events, t2_seed, event_seed, 10, packed.output_count, 1)
    expected = _run_packed_jvp(*arguments)
    actual = _run_packed_jvp(*arguments, real_axis=1)
    scale = expected.abs().max()
    assert ((expected - actual).abs().max() / scale) < 1e-6

    torch.manual_seed(0)
    seed = torch.randn(
        (trains, prepared[0].numel(), packed.output_count), dtype=torch.complex64
    )
    tangents = (*t2_seed, *event_seed)
    keywords = dict(state_count=10, output_count=packed.output_count, threads=1)
    reference, _ = _run_packed_vjp_jvp(prepared, events, tangents, seed, **keywords)
    result, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, real_axis=1, **keywords
    )
    # The per-atom gradients sum across the whole block, so a repeated lane
    # would show up here as one counted more than once.
    for index in INSIDE_THE_SUBSPACE:
        magnitude = reference[index].abs().max()
        if magnitude == 0:
            assert result[index].abs().max() == 0
            continue
        assert ((reference[index] - result[index]).abs().max() / magnitude) < 1e-4


def _tissue_events(trains, echoes=20, atoms=64, b0_hz=0.0):
    generator = torch.Generator().manual_seed(0)
    flip = torch.deg2rad(80.0 + 80.0 * torch.rand(trains, echoes, generator=generator))
    packed = _pack_events(
        fse_description(
            flip,
            echo_spacing_s=ECHO_SPACING_S,
            phases_rad=torch.pi / 2,
            excitation_phase_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    tissue = TissueProperties(
        t1_ms=torch.full((atoms,), 800.0),
        t2_ms=torch.full((atoms,), 45.0),
        b0_hz=torch.full((atoms,), b0_hz),
        b1_phase_rad=torch.zeros(atoms),
    )
    prepared, _, _ = _prepare_tissue(tissue, "cpu")
    prepared = tuple(v.to(torch.float32).contiguous() for v in prepared)
    return events, prepared, packed.output_count


def _trains_worth(share, echoes=20, atoms=64):
    """Enough trains to carry ``share`` of the work the test is worth at."""
    from torchsim.sequence._calibration import detection

    events, _tissue, _count = _tissue_events(1, echoes=echoes, atoms=atoms)
    per_train = atoms * int(events[1].numel())
    floor = detection("forward", torch.device("cpu"), 10)
    return max(1, int(floor * share / per_train))


def test_the_fast_path_is_chosen_without_being_asked():
    """A caller should not have to know the subspace rule to benefit from it."""
    from torchsim.sequence._accelerators import _auto_real_axis, _run_packed

    events, tissue, count = _tissue_events(_trains_worth(8))
    assert _auto_real_axis("forward", events, tissue, 10) == 1
    automatic = _run_packed(tissue, events, 10, count, 1)
    complex_path = _run_packed(tissue, events, 10, count, 1, real_axis=-1)
    scale = complex_path.abs().max()
    assert ((automatic - complex_path).abs().max() / scale) < 1e-6


def test_off_resonance_is_not_chosen():
    from torchsim.sequence._accelerators import _auto_real_axis

    events, tissue, _ = _tissue_events(_trains_worth(8), b0_hz=15.0)
    assert _auto_real_axis("forward", events, tissue, 10) is None


def test_a_tiny_problem_skips_the_test_that_would_cost_more_than_it_saves(monkeypatch):
    """The rule, against a threshold this test names rather than measures.

    What a machine measures moves with the machine, and deciding is cheap
    enough now that the measured threshold sits within a small multiple of the
    smallest problem this file can build -- so a probe taken on a loaded
    machine could land either side of it. The rule is the same whatever the
    number: work under the threshold is not worth the test.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _auto_real_axis

    events, tissue, _ = _tissue_events(1, atoms=2)
    work = 2 * int(events[1].numel())
    monkeypatch.setattr(_accelerators, "detection", lambda *_: 8.0 * work)
    assert _auto_real_axis("forward", events, tissue, 10) is None
    monkeypatch.setattr(_accelerators, "detection", lambda *_: work / 8.0)
    assert _auto_real_axis("forward", events, tissue, 10) == 1


@pytest.mark.parametrize("direction", [4, 5])
def test_a_seed_that_leaves_the_subspace_is_not_chosen(direction):
    """b1_phase and off-resonance seeds have no derivative in the real kernels.

    The primal can sit in the subspace while the tangent leaves it, so the
    verdict has to look at the seed as well; otherwise the fast path would
    silently return zero for exactly the direction that was asked for.
    """
    from torchsim.sequence._accelerators import _auto_real_axis

    events, tissue, _ = _tissue_events(_trains_worth(8))
    seed = tuple(
        torch.ones_like(value) if index == direction else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    )
    phase_seed = torch.zeros_like(events[3])
    assert (
        _auto_real_axis("jvp", events, tissue, 10, (seed[4], seed[5], phase_seed))
        is None
    )


def test_an_rf_phase_seed_is_not_chosen():
    from torchsim.sequence._accelerators import _auto_real_axis

    events, tissue, _ = _tissue_events(_trains_worth(8))
    zeros = tuple(torch.zeros_like(value) for value in tissue)
    assert (
        _auto_real_axis(
            "jvp", events, tissue, 10, (zeros[4], zeros[5], torch.ones_like(events[3]))
        )
        is None
    )


# --- the adjoint, where the verdict also depends on what is being asked for ---


def _adjoint_case(trains):
    """An echo train, its forward directions, and a cotangent to pull back."""
    events, tissue, count = _tissue_events(trains)
    t2_seed = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(tissue)
    )
    tangents = (
        *t2_seed,
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    generator = torch.Generator().manual_seed(0)
    cotangent = torch.randn(
        (trains, tissue[0].numel(), count),
        generator=generator,
        dtype=torch.complex64,
    )
    return events, tissue, tangents, cotangent, count


def _every_gradient():
    return tuple(True for _ in FLOAT_NAMES)


def _only(*names):
    """A mask over the differentiable inputs, named rather than positional."""
    positions = {
        name if isinstance(name, int) else FLOAT_NAMES.index(name) for name in names
    }
    return tuple(index in positions for index in range(len(FLOAT_NAMES)))


_FLIP = FLOAT_NAMES.index("flip")


@pytest.mark.parametrize(
    "wanted, expected",
    [(None, None), (_every_gradient(), None), (_only("t2_ms", "flip"), 1)],
    ids=["unsaid", "all ten", "t2 and flip"],
)
def test_the_adjoint_verdict_follows_what_the_caller_will_read(wanted, expected):
    """A real adjoint is three gradients short, so it depends who is asking."""
    from torchsim.sequence._accelerators import _auto_real_axis_adjoint

    events, tissue, tangents, _cotangent, _count = _adjoint_case(_trains_worth(8))

    assert _auto_real_axis_adjoint(events, tissue, 10, tangents, wanted) == expected


@pytest.mark.parametrize(
    "position", OUTSIDE_THE_SUBSPACE, ids=["b1_phase", "b0", "velocity", "phase"]
)
def test_wanting_a_gradient_outside_the_subspace_keeps_the_complex_kernel(position):
    """Each is genuinely non-zero; returning zero would be wrong."""
    from torchsim.sequence._accelerators import (
        _auto_real_axis_adjoint,
        _run_packed_vjp_jvp,
    )
    from torchsim.sequence._parameters import Geometry

    events, tissue, tangents, cotangent, count = _adjoint_case(_trains_worth(8))
    wanted = _only(_FLIP, position)

    assert _auto_real_axis_adjoint(events, tissue, 10, tangents, wanted) is None

    gradients, _ = _run_packed_vjp_jvp(
        tissue,
        events,
        tangents,
        cotangent,
        10,
        count,
        1,
        wanted=wanted,
        # The velocity gradient is imaginary at any velocity, this case's zero
        # included -- but only where the sequence declares a gradient for the
        # spins to wind across.
        geometry=Geometry(flow_scale=8.0 * torch.pi / 5e-4),
    )
    assert gradients[position].abs().max() > 0.0


def test_an_adjoint_that_stays_in_the_subspace_reaches_the_real_kernel():
    """Bitwise, because only the same kernel gives the same bits."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    events, tissue, tangents, cotangent, count = _adjoint_case(_trains_worth(8))
    shared = (tissue, events, tangents, cotangent, 10, count, 1)
    automatic, _ = _run_packed_vjp_jvp(*shared, wanted=_only("t2_ms", "flip"))
    real, _ = _run_packed_vjp_jvp(*shared, real_axis=1)

    for chosen, reference in zip(automatic, real, strict=True):
        assert torch.equal(chosen, reference)


def _forward_over_reverse(atoms):
    """A directional derivative, differentiated: what backward fuses."""
    from torchsim import EpgEngine
    from torchsim.sequence._simulation import TissueProperties

    generator = torch.Generator().manual_seed(0)
    flip = torch.deg2rad(80.0 + 80.0 * torch.rand(20, generator=generator))
    description = fse_description(
        flip,
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    t2 = torch.linspace(40.0, 120.0, atoms).requires_grad_(True)

    def simulate(values):
        tissue = TissueProperties(t1_ms=torch.full((atoms,), 900.0), t2_ms=values)
        return EpgEngine().simulate(description, tissue, nstates=10).signal

    _signal, derivative = torch.func.jvp(simulate, (t2,), (torch.ones_like(t2),))
    return torch.autograd.grad(derivative.abs().square().sum(), t2)[0]


def test_autograd_asks_for_what_the_graph_needs(monkeypatch, always_worth_detecting):
    """The verdict is reached inside backward, off ``needs_input_grad``.

    Differentiating T2 stays inside the subspace, so the fast adjoint is
    available -- and the answer must not depend on it being taken.
    """
    from torchsim.sequence import _accelerators

    chosen = []
    original = _accelerators._auto_real_axis_adjoint
    monkeypatch.setattr(
        _accelerators,
        "_auto_real_axis_adjoint",
        lambda *arguments: chosen.append(original(*arguments)) or chosen[-1],
    )
    atoms = max(2, _trains_worth(8, echoes=20, atoms=1))
    automatic = _forward_over_reverse(atoms)
    monkeypatch.setattr(
        _accelerators, "_auto_real_axis_adjoint", lambda *arguments: None
    )
    reference = _forward_over_reverse(atoms)

    assert chosen and all(axis == 1 for axis in chosen)
    assert reference.abs().max() > 0.0
    assert ((automatic - reference).abs().max() / reference.abs().max()) < 1e-4


# Enough voxels that the subspace verdict is worth reaching for on a device:
# below the detection threshold the adjoint takes the complex kernel whatever
# the states do, and the fast path would never be exercised.
CUDA_VOXELS = 8192


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("state_count", [8, 10, 16, 17])
def test_the_cuda_real_adjoint_agrees_with_the_second_order_kernel(
    state_count, always_worth_detecting
):
    """The device first-order adjoint against the forward-over-reverse pass it
    specializes, which is what produced this gradient before it existed.

    Widths are swept because a reverse kernel has miscompiled silently at one
    state count before, and a single width would not have caught it.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _run_packed_vjp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cuda"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    generator = torch.Generator().manual_seed(11)
    wide = TissueProperties(
        t1_ms=600.0 + 1200.0 * torch.rand(CUDA_VOXELS, generator=generator),
        t2_ms=30.0 + 120.0 * torch.rand(CUDA_VOXELS, generator=generator),
    )
    prepared, _, _ = _prepare_tissue(wide, "cuda")
    assert real_subspace_axis(events, prepared) == 1

    seed = torch.randn(
        (events[2].shape[0], CUDA_VOXELS, packed.output_count),
        dtype=torch.complex64,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(3),
    )
    wanted = tuple(index in INSIDE_THE_SUBSPACE for index in range(len(FLOAT_NAMES)))
    actual = _run_packed_vjp(
        prepared,
        events,
        seed,
        state_count=state_count,
        output_count=packed.output_count,
        threads=1,
        wanted=wanted,
    )

    # The route this gradient took before the first-order kernel existed: the
    # forward-over-reverse pass, handed a direction of zeros.
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )
    _, expected = _accelerators._run_packed_vjp_jvp(
        prepared,
        events,
        still,
        seed,
        state_count=state_count,
        output_count=packed.output_count,
        threads=1,
        real_axis=1,
        wanted=wanted,
    )

    compared = 0
    for index in INSIDE_THE_SUBSPACE:
        scale = expected[index].abs().max()
        if scale == 0:
            assert actual[index].abs().max() == 0
            continue
        assert ((expected[index] - actual[index]).abs().max() / scale) < 1e-4
        compared += 1
    assert compared > 3
    for index in OUTSIDE_THE_SUBSPACE:
        assert actual[index].abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_device_adjoint_stops_short_of_the_forward_over_reverse_pass(
    always_worth_detecting,
):
    """The first-order kernel is the point of this route, so the test above has
    to be reaching it rather than agreeing with itself.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _run_packed_vjp

    description = fse_description(
        _flip(),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cuda"),
        rf_raster_time_s=1e-6,
    )
    events = _buffers(packed)
    generator = torch.Generator().manual_seed(11)
    wide = TissueProperties(
        t1_ms=600.0 + 1200.0 * torch.rand(CUDA_VOXELS, generator=generator),
        t2_ms=30.0 + 120.0 * torch.rand(CUDA_VOXELS, generator=generator),
    )
    prepared, _, _ = _prepare_tissue(wide, "cuda")
    seed = torch.zeros(
        (events[2].shape[0], CUDA_VOXELS, packed.output_count),
        dtype=torch.complex64,
        device="cuda",
    )
    wanted = tuple(index in INSIDE_THE_SUBSPACE for index in range(len(FLOAT_NAMES)))

    reached = []
    original = _accelerators._run_packed_vjp_jvp
    _accelerators._run_packed_vjp_jvp = lambda *arguments, **keywords: (
        reached.append(True) or original(*arguments, **keywords)
    )
    try:
        _run_packed_vjp(
            prepared,
            events,
            seed,
            state_count=10,
            output_count=packed.output_count,
            threads=1,
            wanted=wanted,
        )
    finally:
        _accelerators._run_packed_vjp_jvp = original

    assert not reached


def _spoiled_axis(phases_rad):
    """The verdict and the signal for an unbalanced train of the given phases."""
    from torchsim import mrf_description

    flip = torch.deg2rad(torch.linspace(5.0, 60.0, ECHOES))
    description = mrf_description(flip, 10e-3, phases_rad=phases_rad)
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    signal = EpgEngine().simulate(description, _tissue(0.0, 0.0)).signal
    return real_subspace_axis(packed.buffers, prepared), signal, packed, prepared


@pytest.mark.parametrize("phase", [0.0, torch.pi / 3, torch.pi])
def test_a_train_with_no_refocusing_pulse_stays_on_the_axis(phase):
    """Winding the states on does not take them off the axis a pulse turns about.

    A fingerprinting schedule has an inversion and a train of excitations and
    no refocusing pulse at all, so the arrangement is one axis rather than two.
    """
    axis, signal, _, _ = _spoiled_axis(phase)
    assert axis == 1
    # The engine runs the complex kernel on a problem this small, so what is
    # left on the real axis is the round-off of demodulating by the pulse
    # phase rather than anything the states did.
    assert signal.real.abs().max() < 1e-6 * signal.imag.abs().max()


def test_rf_spoiling_breaks_the_subspace():
    """Quadratic phase increments turn about a different axis every repetition.

    Each sample is demodulated by the phase of the pulse that made it, so the
    echoes still look nearly imaginary; the states do not, and the verdict
    follows the states.
    """
    index = torch.arange(ECHOES, dtype=torch.float32)
    axis, signal, _, _ = _spoiled_axis(torch.deg2rad(0.5 * 117.0 * index * (index + 1)))
    assert axis is None
    assert signal.real.abs().max() > 1e-3 * signal.imag.abs().max()


def test_the_real_kernel_reproduces_the_complex_one_without_refocusing():
    """The widened verdict is an optimization, not an approximation."""
    from torchsim.sequence._accelerators import _run_packed

    axis, _, packed, prepared = _spoiled_axis(0.0)
    assert axis == 1
    events = _buffers(packed)
    complex_signal = _run_packed(prepared, events, 10, packed.output_count, 1)
    real_signal = _run_packed(prepared, events, 10, packed.output_count, 1, real_axis=1)
    scale = complex_signal.abs().max()
    assert ((complex_signal - real_signal).abs().max() / scale) < 1e-6


def test_a_declared_tissue_settles_the_question_without_a_buffer():
    """What the caller passed already says whether the three terms are there.

    A property left at its identity is absent from the feature set, and absent
    is the whole answer: the verdict is the same one the reduction gives, and
    it is reached without reading the buffer that reduction would read.
    """
    from torchsim.sequence._parameters import features_of

    description = fse_description(
        _flip(), echo_spacing_s=ECHO_SPACING_S, phases_rad=0.0, excitation_phase_rad=0.0
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    plain = TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]), t2_ms=torch.tensor([45.0, 120.0])
    )
    prepared, _, _ = _prepare_tissue(plain, "cpu")
    assert real_subspace_axis(packed.buffers, prepared) == 1
    assert (
        real_subspace_axis(packed.buffers, prepared, features=features_of(plain)) == 1
    )

    off_resonant = _tissue(40.0, 0.0)
    prepared, _, _ = _prepare_tissue(off_resonant, "cpu")
    assert real_subspace_axis(packed.buffers, prepared) is None
    assert (
        real_subspace_axis(packed.buffers, prepared, features=features_of(off_resonant))
        is None
    )


def test_a_map_of_zeros_keeps_the_axis_it_declares_it_has_left():
    """A declared term is still reduced over rather than taken at its word.

    A caller who passes off-resonance as a full map has declared the term
    whatever the map holds, and a map of zeros is one a run should not lose the
    fast path to.
    """
    from torchsim.sequence._parameters import features_of

    description = fse_description(
        _flip(), echo_spacing_s=ECHO_SPACING_S, phases_rad=0.0, excitation_phase_rad=0.0
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    zeroed = _tissue(0.0, 0.0)
    assert "B0" in features_of(zeroed)
    prepared, _, _ = _prepare_tissue(zeroed, "cpu")
    assert (
        real_subspace_axis(packed.buffers, prepared, features=features_of(zeroed)) == 1
    )


def test_a_rewritten_phase_buffer_is_read_again():
    """The remembered summary follows the buffer it was read from.

    The event stream is the half of the verdict a binding holds fixed, so it is
    read once and reused. A caller who writes new phases into the same buffer
    gets the verdict those phases earn, not the one the old ones did.
    """
    description = fse_description(
        _flip(), echo_spacing_s=ECHO_SPACING_S, phases_rad=0.0, excitation_phase_rad=0.0
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    plain = TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]), t2_ms=torch.tensor([45.0, 120.0])
    )
    prepared, _, _ = _prepare_tissue(plain, "cpu")
    events = packed.buffers
    assert real_subspace_axis(events, prepared) == 1

    kind, phase = events[1], events[3]
    turning = kind == 1
    quarter = torch.where(turning, torch.full_like(phase, 0.25 * torch.pi), phase)
    phase.copy_(quarter)
    phase[turning.nonzero()[0]] = 0.0
    assert real_subspace_axis(events, prepared) is None


# The laned adjoint fills a block from the atom axis, so an atom count that
# does not fill the last block leaves lanes carrying a repeat of the block's
# first atom. These counts straddle that boundary; every per-atom gradient the
# block writes and every per-event gradient it sums has to ignore them.
@pytest.mark.parametrize("atoms", [7, 8, 9, 17])
def test_partial_atom_blocks_of_the_adjoint_match_the_complex_kernel(atoms):
    from torchsim.sequence._accelerators import (
        _auto_real_axis_adjoint,
        _run_packed_vjp,
    )

    trains = _trains_worth(8, atoms=atoms)
    events, tissue, count = _tissue_events(trains, atoms=atoms)
    wanted = tuple(index in INSIDE_THE_SUBSPACE for index in range(len(FLOAT_NAMES)))
    assert _auto_real_axis_adjoint(events, tissue, 10, (), wanted) == 1

    torch.manual_seed(0)
    seed = torch.randn(
        (trains, atoms, count) if trains > 1 else (atoms, count),
        dtype=torch.complex64,
    )
    keywords = dict(state_count=10, output_count=count, threads=1)
    real = _run_packed_vjp(tissue, events, seed, wanted=wanted, **keywords)
    complex_path = _run_packed_vjp(tissue, events, seed, wanted=None, **keywords)

    floor = 1e-6 * max(float(value.abs().max()) for value in complex_path)
    compared = 0
    for index in INSIDE_THE_SUBSPACE:
        scale = float(complex_path[index].abs().max())
        drift = float((complex_path[index] - real[index]).abs().max())
        assert drift <= 2e-4 * scale + floor, f"{FLOAT_NAMES[index]}: {drift}"
        if scale > floor:
            compared += 1
    assert compared >= 3


# An axis is not a direction. Pulses a half turn apart lie on one axis and turn
# about it in opposite senses, and a flip angle carries no sign to say which --
# so a packing subtracts the half turns, negating the flips it turns round and
# the samples it demodulates the other way. What the kernels then see is a
# stream on one axis, and the reduced ones can carry it.
HALF_A_TURN = {
    "an excitation a half turn from its refocusing pulses": (torch.pi, 0.0),
    "refocusing pulses a half turn from the excitation": (0.0, torch.pi),
    "a train alternating its phase, as a phase-cycled one does": (
        0.0,
        torch.tensor([0.0, torch.pi] * 3),
    ),
}


def _half_a_turn(name):
    excitation, refocusing = HALF_A_TURN[name]
    description = fse_description(
        torch.deg2rad(torch.full((6,), 140.0)),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=refocusing,
        excitation_phase_rad=excitation,
    )
    return _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )


@pytest.mark.parametrize("name", list(HALF_A_TURN))
def test_pulses_a_half_turn_apart_are_brought_onto_one_axis(name):
    packed = _half_a_turn(name)
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(packed.buffers, prepared) is None

    settled, samples = _one_axis(packed)
    assert real_subspace_axis(settled.buffers, prepared) == 1

    # And the reduced kernel answers what the full one does, which is what the
    # verdict has just claimed of it.
    arguments = (prepared, settled.buffers, 8, packed.output_count, 1)
    sign = 1.0 if samples is None else samples
    complex_path = _run_packed(*arguments, real_axis=-1) * sign
    reduced = _run_packed(*arguments, real_axis=1) * sign
    scale = complex_path.abs().max()
    assert (complex_path - reduced).abs().max() <= 1e-5 * scale

    # The rewrite is an identity: the sequence it describes is unchanged.
    original = _run_packed(
        prepared, packed.buffers, 8, packed.output_count, 1, real_axis=-1
    )
    assert (original - complex_path).abs().max() <= 1e-5 * scale


@pytest.mark.parametrize("name", list(HALF_A_TURN))
def test_a_half_turn_apart_reaches_the_engine(name):
    """End to end, against the same sequence with the packing stood down."""
    excitation, refocusing = HALF_A_TURN[name]
    description = fse_description(
        torch.deg2rad(torch.full((6,), 140.0)),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=refocusing,
        excitation_phase_rad=excitation,
    )
    tissue = _tissue(0.0, 0.0)
    settled = EpgEngine().simulate(description, tissue).signal
    with mock.patch.object(accelerators, "_one_axis", lambda packed: (packed, None)):
        plain = EpgEngine().simulate(description, tissue).signal
    assert (settled - plain).abs().max() <= 1e-5 * plain.abs().max()


def test_a_uniform_train_is_left_exactly_as_it_stands():
    """Nothing is a half turn out, so the packing has nothing to subtract."""
    description = fse_description(
        torch.deg2rad(torch.full((6,), 140.0)),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 4,
        excitation_phase_rad=torch.pi / 4,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    settled, samples = _one_axis(packed)
    assert settled is packed
    assert samples is None


def test_pulses_a_quarter_turn_apart_are_left_to_the_full_kernel():
    """No sign brings them together, so the packing does not try."""
    description = fse_description(
        torch.deg2rad(torch.full((6,), 140.0)),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=0.0,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    settled, samples = _one_axis(packed)
    assert settled is packed
    assert samples is None
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(packed.buffers, prepared) is None


def test_a_sample_demodulated_off_the_axis_keeps_the_full_kernel():
    """The reduced kernels demodulate by nothing, so they cannot produce it."""
    description = fse_description(
        torch.deg2rad(torch.full((6,), 140.0)),
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=0.0,
        excitation_phase_rad=0.0,
    )
    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    prepared, _, _ = _prepare_tissue(_tissue(0.0, 0.0), "cpu")
    assert real_subspace_axis(packed.buffers, prepared) == 1

    recording = (packed.kind == 2) & ((packed.action & 32) != 0)
    phase = torch.where(recording, torch.pi / 2, packed.phase)
    turned = replace(packed, phase=phase.contiguous())
    assert real_subspace_axis(turned.buffers, prepared) is None

    # And the refusal is not caution: demodulating a quarter turn round is a
    # signal the reduced kernel does not carry.
    arguments = (prepared, turned.buffers, 8, packed.output_count, 1)
    complex_path = _run_packed(*arguments, real_axis=-1)
    reduced = _run_packed(*arguments, real_axis=1)
    assert (complex_path - reduced).abs().max() > 0.5 * complex_path.abs().max()


@pytest.mark.parametrize("name", list(HALF_A_TURN))
def test_the_rewrite_carries_derivatives_as_well_as_values(name):
    """Subtracting a half turn is affine, so the phase gradient survives it."""
    excitation, refocusing = HALF_A_TURN[name]

    def gradients(settle):
        flip = torch.deg2rad(torch.full((6,), 140.0)).requires_grad_(True)
        phases = (
            torch.as_tensor(refocusing, dtype=torch.float32)
            .expand(6)
            .clone()
            .requires_grad_(True)
        )
        tissue = _tissue(0.0, 0.0)
        tissue.t2_ms.requires_grad_(True)
        description = fse_description(
            flip,
            echo_spacing_s=ECHO_SPACING_S,
            phases_rad=phases,
            excitation_phase_rad=excitation,
        )
        with mock.patch.object(
            accelerators,
            "_one_axis",
            accelerators._one_axis if settle else (lambda packed: (packed, None)),
        ):
            signal = EpgEngine().simulate(description, tissue).signal
        weight = torch.linspace(0.3, 1.9, signal.shape[-1])
        loss = (weight * signal.real).sum() + (1.7 * weight * signal.imag).sum()
        return torch.autograd.grad(loss, [flip, phases, tissue.t2_ms])

    for settled, plain in zip(gradients(True), gradients(False), strict=True):
        assert (settled - plain).abs().max() <= 1e-5 * plain.abs().max()
