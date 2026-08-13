"""The CUDA kernels against the CPU kernels they stand in for.

The two share no code, so agreement across a batch of echo trains is what keeps
the Triton grid indexing honest: a train axis dropped there reads one train's
flip angles for every train, which is a wrong answer rather than an error.
"""

import pytest
import torch

from torchsim.sequence._parameters import OUTSIDE_THE_SUBSPACE
from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed,
    _run_packed_jvp,
    _run_packed_vjp_jvp,
)
from torchsim.sequence._builders import fse_description
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)

ECHOES = 20
STATES = 10

def _probed_atoms(kind="forward", trains=64, share=8):
    """Atoms enough to carry ``share`` of the work the probe repays itself at.

    That work is measured on whatever card is present, so a size written down
    here would be a claim about one machine.
    """
    from torchsim.sequence._calibration import detection

    events, _prepared, _count = _real_case(trains, "cpu", atoms=1)
    per_atom = trains * int(events[1].numel())
    floor = detection(kind, torch.device("cuda", 0), STATES)
    return max(1, int(floor * share / per_atom))


def _case(trains, device):
    generator = torch.Generator().manual_seed(0)
    flip = torch.deg2rad(
        80.0 + 80.0 * torch.rand(trains, ECHOES, generator=generator)
    )
    description = fse_description(
        flip,
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device(device),
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )
    # Off-resonance and transmit phase keep both kernels on their complex path.
    tissue = TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]),
        t2_ms=torch.tensor([45.0, 120.0]),
        b0_hz=torch.tensor([0.0, 13.0]),
        b1_phase_rad=torch.tensor([0.0, 0.2]),
    )
    prepared, _, _ = _prepare_tissue(tissue, device)
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return events, prepared, packed.output_count


def _real_case(trains, device, atoms=2):
    """The same sequence over tissue that keeps the states in a real subspace."""
    generator = torch.Generator().manual_seed(0)
    flip = torch.deg2rad(
        80.0 + 80.0 * torch.rand(trains, ECHOES, generator=generator)
    )
    description = fse_description(
        flip,
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device(device),
        rf_raster_time_s=1e-6,
    )
    events = (
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
    )
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, atoms),
        t2_ms=torch.linspace(40.0, 120.0, atoms),
        b0_hz=torch.zeros(atoms),
        b1_phase_rad=torch.zeros(atoms),
    )
    prepared, _, _ = _prepare_tissue(tissue, device)
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return events, prepared, packed.output_count


def _seeds(events, prepared, tissue_index=None, event_index=None):
    tissue = tuple(
        torch.ones_like(value) if index == tissue_index else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    event = tuple(
        torch.ones_like(value) if index == event_index else torch.zeros_like(value)
        for index, value in enumerate((events[0], events[2], events[3]))
    )
    return tissue, event


def _t2_seeds(events, prepared):
    tissue = tuple(
        torch.ones_like(value) if index == 1 else torch.zeros_like(value)
        for index, value in enumerate(prepared)
    )
    event = (
        torch.zeros_like(events[0]),
        torch.zeros_like(events[2]),
        torch.zeros_like(events[3]),
    )
    return tissue, event


@pytest.mark.parametrize("trains", [1, 4, 17, 64])
def test_forward_matches_the_cpu_kernel(trains):
    events, prepared, count = _case(trains, "cpu")
    expected = _run_packed(prepared, events, STATES, count, 1)
    events, prepared, count = _case(trains, "cuda")
    actual = _run_packed(prepared, events, STATES, count, 1)

    assert actual.shape == expected.shape
    scale = expected.abs().max()
    assert ((expected - actual.cpu()).abs().max() / scale) < 1e-5


@pytest.mark.parametrize("trains", [1, 4, 17, 64])
def test_forward_mode_matches_the_cpu_kernel(trains):
    events, prepared, count = _case(trains, "cpu")
    tissue_seed, event_seed = _t2_seeds(events, prepared)
    expected = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, count, 1
    )
    events, prepared, count = _case(trains, "cuda")
    tissue_seed, event_seed = _t2_seeds(events, prepared)
    actual = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, count, 1
    )

    assert actual.shape == expected.shape
    scale = expected.abs().max()
    assert ((expected - actual.cpu()).abs().max() / scale) < 1e-5


def test_each_train_gets_its_own_flip_angles():
    """A dropped train axis would make every train agree with the first."""
    events, prepared, count = _case(8, "cuda")
    signal = _run_packed(prepared, events, STATES, count, 1)
    first = signal[0]
    assert all(
        not torch.allclose(signal[index], first) for index in range(1, 8)
    )


@pytest.mark.parametrize("trains", [1, 4, 17, 64])
def test_the_real_kernel_agrees_with_the_complex_one(trains):
    """The subspace claim, checked against the kernel that assumes nothing."""
    events, prepared, count = _real_case(trains, "cuda")
    expected = _run_packed(prepared, events, STATES, count, 1, real_axis=-1)
    actual = _run_packed(prepared, events, STATES, count, 1, real_axis=1)

    assert actual.shape == expected.shape
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


@pytest.mark.parametrize("tissue_index, event_index", [(1, None), (None, 1)])
def test_the_real_forward_mode_kernel_agrees_with_the_complex_one(
    tissue_index, event_index
):
    """Seeded along T2 and along the flip angle: both stay in the subspace."""
    events, prepared, count = _real_case(17, "cuda")
    tissue_seed, event_seed = _seeds(events, prepared, tissue_index, event_index)
    expected = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, count, 1, real_axis=-1
    )
    actual = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, count, 1, real_axis=1
    )

    assert expected.abs().max() > 0.0
    assert ((expected - actual).abs().max() / expected.abs().max()) < 1e-5


@pytest.mark.parametrize("trains", [1, 4, 17, 64])
def test_the_real_kernel_matches_the_cpu_real_kernel(trains):
    """Two independent derivations of the same real 3x3 recursion."""
    events, prepared, count = _real_case(trains, "cpu")
    expected = _run_packed(prepared, events, STATES, count, 1, real_axis=1)
    events, prepared, count = _real_case(trains, "cuda")
    actual = _run_packed(prepared, events, STATES, count, 1, real_axis=1)

    assert actual.shape == expected.shape
    assert ((expected - actual.cpu()).abs().max() / expected.abs().max()) < 1e-5


def test_a_subspace_sequence_reaches_the_real_kernel_unasked():
    """Bitwise, because only the same kernel gives the same bits."""
    events, prepared, count = _real_case(64, "cuda", atoms=_probed_atoms())
    automatic = _run_packed(prepared, events, STATES, count, 1)
    real = _run_packed(prepared, events, STATES, count, 1, real_axis=1)

    assert torch.equal(automatic, real)


def test_an_off_resonance_seed_keeps_the_complex_kernel_on_cuda():
    """The real kernel has no derivative along b0 and would return zeros."""
    events, prepared, count = _real_case(
        64, "cuda", atoms=_probed_atoms("jvp")
    )
    tissue_seed, event_seed = _seeds(events, prepared, tissue_index=5)
    automatic = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, count, 1
    )
    complex_kernel = _run_packed_jvp(
        prepared, events, tissue_seed, event_seed, STATES, count, 1, real_axis=-1
    )

    assert automatic.abs().max() > 0.0
    assert torch.equal(automatic, complex_kernel)


@pytest.mark.parametrize("total", [0, 1, 1024, 3071, 3072, 5000, 100_000])
@pytest.mark.parametrize("block_states", [4, 16])
def test_the_packing_width_is_a_power_of_two(total, block_states):
    """It indexes a ``tl.arange``, which rejects anything else."""
    from torchsim.sequence._epg_triton import _problems_per_program

    width = _problems_per_program(total, block_states)
    assert width >= 1
    assert width & (width - 1) == 0


@pytest.mark.parametrize("atoms", [3, 16, 21])
def test_awkward_problem_counts_still_match_the_cpu_kernel(atoms):
    """Counts whose packing width does not fall on a power of two on its own."""
    events, prepared, count = _real_case(200, "cpu", atoms=atoms)
    expected = _run_packed(prepared, events, STATES, count, 1, real_axis=1)
    events, prepared, count = _real_case(200, "cuda", atoms=atoms)
    actual = _run_packed(prepared, events, STATES, count, 1, real_axis=1)

    assert actual.shape == expected.shape
    assert ((expected - actual.cpu()).abs().max() / expected.abs().max()) < 1e-5


def test_a_small_problem_skips_the_probe_on_cuda():
    """The probe costs the same everywhere; what it buys does not.

    A GPU clears the work behind the verdict fast enough that a probe worth
    running on the CPU is pure overhead here.
    """
    from torchsim.sequence._calibration import detection

    assert detection("forward", torch.device("cuda", 0), STATES) > detection(
        "forward", torch.device("cpu"), STATES
    )


def _second_order(device, trains, atoms, seed_index=1, inversion=None):
    """CPU and CUDA agree here or they do not; nothing else is asserted."""
    events, prepared, count = _real_case(trains, device, atoms=atoms)
    if inversion is not None:
        action = events[4].clone()
        pulses = ((events[1] == 1) & ((action & 4) == 0)).nonzero().flatten()
        action[pulses[inversion]] |= 4
        events = (*events[:4], action.contiguous(), events[5])
    tissue_seed, event_seed = _seeds(events, prepared, tissue_index=seed_index)
    generator = torch.Generator().manual_seed(7)
    shape = (trains, atoms, count) if trains > 1 else (atoms, count)
    cotangent = torch.randn(shape, generator=generator, dtype=torch.complex64)
    return _run_packed_vjp_jvp(
        prepared,
        events,
        (*tissue_seed, *event_seed),
        cotangent.to(device),
        state_count=STATES,
        output_count=count,
        threads=1,
        real_axis=1,
    )


# A per-voxel gradient sums the trains that reach that voxel, and on CUDA those
# terms land through ``tl.atomic_add``, so the order of the sum is not fixed.
# On a quiescent machine the order happens to repeat and the disagreement
# measures 1.1e-06, but it shifts when compilation perturbs how blocks retire,
# and a sum this wide leaves room for cancellation. The looser bound absorbs a
# reordered sum; it is still three orders below anything that would count as
# the kernels actually disagreeing.
_ACCUMULATED = 1e-3
_ONE_TERM = 1e-4


def _tolerance(trains: int) -> float:
    """One train per voxel is a one-term sum, which has no order to vary."""
    return _ONE_TERM if trains == 1 else _ACCUMULATED


def _worst_disagreement(expected, actual):
    worst = 0.0
    for expected_side, actual_side in zip(expected, actual, strict=True):
        for reference, result in zip(expected_side, actual_side, strict=True):
            reference, result = reference.cpu(), result.cpu()
            scale = reference.abs().max()
            if scale == 0:
                assert result.abs().max() == 0
                continue
            worst = max(worst, ((reference - result).abs().max() / scale).item())
    return worst


# Index 6 is inversion_efficiency, which only a sequence with an inversion
# pulse can produce a gradient for.
@pytest.mark.parametrize("seed_index", [0, 1, 3, 6])
def test_the_second_order_kernel_matches_the_cpu_kernel(seed_index):
    """Every gradient, along four different forward-mode seeds."""
    expected = _second_order("cpu", 4, 3, seed_index=seed_index)
    actual = _second_order("cuda", 4, 3, seed_index=seed_index)

    assert _worst_disagreement(expected, actual) < _tolerance(4)


@pytest.mark.parametrize("trains, atoms", [(1, 1), (17, 5), (64, 32)])
def test_the_second_order_kernel_matches_across_shapes(trains, atoms):
    """One problem, a partial tile, and several full tiles."""
    expected = _second_order("cpu", trains, atoms)
    actual = _second_order("cuda", trains, atoms)

    assert _worst_disagreement(expected, actual) < _tolerance(trains)


@pytest.mark.parametrize("inversion", [3, 8])
def test_an_inversion_pulse_reaches_the_same_gradients(inversion):
    """The inversion branch is unreachable in a plain echo train."""
    expected = _second_order("cpu", 4, 3, inversion=inversion)
    actual = _second_order("cuda", 4, 3, inversion=inversion)

    # inversion_efficiency, the gradient the branch exists to produce.
    assert expected[0][6].abs().max() > 0
    assert _worst_disagreement(expected, actual) < _tolerance(4)


def test_a_trajectory_too_large_for_one_launch_is_split(monkeypatch):
    """The grid rounds up past a wave, onto rows the next launch owns."""
    from torchsim.sequence import _epg_triton

    expected = _second_order("cpu", 17, 5)
    monkeypatch.setattr(_epg_triton, "_TRAJECTORY_BUDGET_BYTES", 40_000)
    actual = _second_order("cuda", 17, 5)

    assert _worst_disagreement(expected, actual) < _tolerance(17)


def test_the_directions_outside_the_subspace_stay_zero():
    """b1_phase, b0 and the RF phase divide out of the representation."""
    gradients = _second_order("cuda", 4, 3)

    for side in gradients:
        for index in OUTSIDE_THE_SUBSPACE:
            assert side[index].abs().max() == 0


def _spgr_case(device, trains, atoms):
    """A spoiled train, which unlike FSE leaves off-resonance in the signal.

    An echo train refocuses b0 exactly at the samples it records, so its b0
    gradient is zero and says nothing about whether a kernel computes it. SPGR
    records at an echo time instead, and every gradient is live there.

    The builder emits one train at a time, so the float buffers are stacked to
    batch them; the structural buffers are shared, as the kernels expect.
    """
    from torchsim.sequence._builders import spgr_description

    generator = torch.Generator().manual_seed(0)
    packed = [
        _pack_events(
            "spgr",
            spgr_description(
                torch.deg2rad(5.0 + 20.0 * torch.rand(12, generator=generator)),
                repetition_time_s=10e-3,
                echo_time_s=4e-3,
                phases_rad=torch.pi / 3,
            ),
            repetitions=1,
            record="all",
            device=torch.device(device),
            rf_raster_time_s=1e-6,
        )
        for _ in range(trains)
    ]

    def stack(name):
        values = [getattr(value, name) for value in packed]
        return (values[0] if trains == 1 else torch.stack(values)).contiguous()

    events = (
        stack("duration"),
        packed[0].kind,
        stack("flip"),
        stack("phase"),
        packed[0].action,
        packed[0].output_index,
    )
    tissue = TissueProperties(
        t1_ms=torch.linspace(600.0, 1400.0, atoms),
        t2_ms=torch.linspace(40.0, 120.0, atoms),
        b0_hz=torch.linspace(20.0, 200.0, atoms),
        b1_phase_rad=torch.linspace(0.0, 0.3, atoms),
    )
    prepared, _, _ = _prepare_tissue(tissue, device)
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    return events, prepared, packed[0].output_count


def _complex_second_order(device, trains, atoms, seed_index=1):
    events, prepared, count = _spgr_case(device, trains, atoms)
    tissue_seed, event_seed = _seeds(events, prepared, tissue_index=seed_index)
    generator = torch.Generator().manual_seed(7)
    shape = (trains, atoms, count) if trains > 1 else (atoms, count)
    cotangent = torch.randn(shape, generator=generator, dtype=torch.complex64)
    return _run_packed_vjp_jvp(
        prepared,
        events,
        (*tissue_seed, *event_seed),
        cotangent.to(device),
        state_count=STATES,
        output_count=count,
        threads=1,
        real_axis=-1,
    )


# b0 and b1_phase are the directions the real-subspace kernel cannot follow, so
# they are the ones worth seeding here.
@pytest.mark.parametrize("seed_index", [0, 1, 4, 5])
def test_the_complex_second_order_kernel_matches_the_cpu_kernel(seed_index):
    """All ten gradients, including the three the real kernel leaves at zero."""
    expected = _complex_second_order("cpu", 4, 3, seed_index=seed_index)
    actual = _complex_second_order("cuda", 4, 3, seed_index=seed_index)

    # b1_phase, b0 and phase separate this from the real-subspace kernel.
    for index in OUTSIDE_THE_SUBSPACE:
        assert expected[0][index].abs().max() > 0
    assert _worst_disagreement(expected, actual) < _tolerance(4)


@pytest.mark.parametrize("trains, atoms", [(1, 3), (17, 5), (64, 32)])
def test_the_complex_second_order_kernel_matches_across_shapes(trains, atoms):
    """One train, a partial tile, and several full tiles."""
    expected = _complex_second_order("cpu", trains, atoms)
    actual = _complex_second_order("cuda", trains, atoms)

    assert _worst_disagreement(expected, actual) < _tolerance(trains)


def test_the_complex_trajectory_splits_into_waves(monkeypatch):
    """Twice the planes of the real one, so it reaches the budget sooner."""
    from torchsim.sequence import _epg_triton

    expected = _complex_second_order("cpu", 17, 5)
    monkeypatch.setattr(_epg_triton, "_TRAJECTORY_BUDGET_BYTES", 40_000)
    actual = _complex_second_order("cuda", 17, 5)

    assert _worst_disagreement(expected, actual) < _tolerance(17)


def test_an_echo_train_has_no_off_resonance_gradient():
    """Why the complex kernel is checked on SPGR rather than on FSE.

    The refocusing pulses undo off-resonance exactly at the sample points, so
    both kernels compute rounding noise around zero for b0 and agreeing there
    would mean nothing.
    """
    events, prepared, count = _real_case(4, "cpu", atoms=3)
    signal = _run_packed(prepared, events, STATES, count, 1, real_axis=-1)
    shifted = (*prepared[:5], prepared[5] + 300.0, *prepared[6:])
    detuned = _run_packed(shifted, events, STATES, count, 1, real_axis=-1)

    assert ((signal - detuned).abs().max() / signal.abs().max()) < 1e-5


def test_the_second_order_path_needs_no_subspace_verdict_on_cuda():
    """Left unasked, the adjoint still has to reach a kernel and agree."""
    events, prepared, count = _case(4, "cuda")
    tissue_seed, event_seed = _t2_seeds(events, prepared)
    cotangent = torch.randn(
        (4, prepared[0].numel(), count), dtype=torch.complex64, device="cuda"
    )
    gradients = _run_packed_vjp_jvp(
        prepared,
        events,
        (*tissue_seed, *event_seed),
        cotangent,
        state_count=STATES,
        output_count=count,
        threads=1,
    )

    assert gradients[0][1].abs().max() > 0


def test_device_tensors_are_refused_by_the_cpu_kernels():
    """A CPU kernel must never be handed a device pointer."""
    from torchsim.sequence._accelerators import _pointers

    events, prepared, _ = _real_case(4, "cuda", atoms=2)
    with pytest.raises(ValueError, match="CPU tensors"):
        _pointers((*prepared, *events))


def test_repeated_second_order_runs_agree_to_tolerance():
    """Gradients land through atomics, so the order of accumulation varies.

    The CPU adjoint is bitwise reproducible; this one is not, and the contract
    is agreement to floating-point tolerance instead.
    """
    first = _second_order("cuda", 17, 5)
    second = _second_order("cuda", 17, 5)

    assert _worst_disagreement(first, second) < 1e-5
