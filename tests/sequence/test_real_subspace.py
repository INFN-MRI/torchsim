"""Detection of sequences whose states never leave a real subspace.

Each case pairs the predicate against the signal it describes, so the rule
cannot drift away from the behaviour it is meant to predict.
"""

import pytest
import torch

from torchsim import FSE, fse_description
from torchsim.sequence._parameters import FLOAT_NAMES, OUTSIDE_THE_SUBSPACE

# The gradients a real adjoint does produce: every differentiable input the
# subspace contains. Derived rather than listed, so a new parameter cannot
# leave a stale index pointing at whatever moved into its slot.
INSIDE_THE_SUBSPACE = tuple(
    index for index in range(len(FLOAT_NAMES)) if index not in OUTSIDE_THE_SUBSPACE
)
from torchsim.sequence._accelerators import _pack_events, real_subspace_axis
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

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
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    events = packed.buffers
    prepared, _, _ = _prepare_tissue(_tissue(b0_hz, b1_phase_rad), "cpu")
    signal = FSE().simulate(description, _tissue(b0_hz, b1_phase_rad)).signal
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
        "fse",
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
    real_signal = _run_packed(
        prepared, events, 10, packed.output_count, 1, real_axis=1
    )
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
        "fse",
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
        "fse",
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
    arguments = dict(
        state_count=10, output_count=packed.output_count, threads=1
    )
    expected, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, **arguments
    )
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
    flip = torch.deg2rad(
        80.0 + 80.0 * torch.rand(trains, ECHOES, generator=generator)
    )
    description = fse_description(
        flip,
        echo_spacing_s=ECHO_SPACING_S,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    return _pack_events(
        "fse",
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
    reference, _ = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, **keywords
    )
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
    flip = torch.deg2rad(
        80.0 + 80.0 * torch.rand(trains, echoes, generator=generator)
    )
    packed = _pack_events(
        "fse",
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


def test_a_tiny_problem_skips_the_test_that_would_cost_more_than_it_saves():
    from torchsim.sequence._accelerators import _auto_real_axis

    events, tissue, _ = _tissue_events(_trains_worth(1 / 8, atoms=2), atoms=2)
    assert _auto_real_axis("forward", events, tissue, 10) is None


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
    assert _auto_real_axis(
        "jvp", events, tissue, 10, (seed[4], seed[5], phase_seed)
    ) is None


def test_an_rf_phase_seed_is_not_chosen():
    from torchsim.sequence._accelerators import _auto_real_axis

    events, tissue, _ = _tissue_events(_trains_worth(8))
    zeros = tuple(torch.zeros_like(value) for value in tissue)
    assert _auto_real_axis(
        "jvp", events, tissue, 10, (zeros[4], zeros[5], torch.ones_like(events[3]))
    ) is None


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
        name if isinstance(name, int) else FLOAT_NAMES.index(name)
        for name in names
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
    from torchsim import FSE
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
        return FSE().simulate(description, tissue, nstates=10).signal

    _signal, derivative = torch.func.jvp(simulate, (t2,), (torch.ones_like(t2),))
    return torch.autograd.grad(derivative.abs().square().sum(), t2)[0]


def test_autograd_asks_for_what_the_graph_needs(monkeypatch):
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
