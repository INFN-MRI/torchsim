"""The CUDA kernels against the CPU kernels they stand in for.

The two share no code, so agreement across a batch of echo trains is what keeps
the Triton grid indexing honest: a train axis dropped there reads one train's
flip angles for every train, which is a wrong answer rather than an error.
"""

import pytest
import torch

from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed,
    _run_packed_jvp,
)
from torchsim.sequence._builders import fse_description
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)

ECHOES = 20
STATES = 10

# Enough atoms that a CUDA run is worth probing for the real subspace at all;
# below the threshold the probe is skipped and no verdict is reached.
PROBED_ATOMS = 512


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
    events, prepared, count = _real_case(64, "cuda", atoms=PROBED_ATOMS)
    automatic = _run_packed(prepared, events, STATES, count, 1)
    real = _run_packed(prepared, events, STATES, count, 1, real_axis=1)

    assert torch.equal(automatic, real)


def test_an_off_resonance_seed_keeps_the_complex_kernel_on_cuda():
    """The real kernel has no derivative along b0 and would return zeros."""
    events, prepared, count = _real_case(64, "cuda", atoms=PROBED_ATOMS)
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
    from torchsim.sequence._accelerators import _auto_real_axis

    events, prepared, _ = _real_case(200, "cuda", atoms=2)
    assert _auto_real_axis(events, prepared) is None

    events, prepared, _ = _real_case(200, "cpu", atoms=2)
    assert _auto_real_axis(events, prepared) == 1


def test_device_tensors_are_refused_by_the_cpu_kernels():
    """No CUDA second-order kernel exists, so it must not reach the CPU one.

    The CPU entry points take raw addresses, and a device address is valid to
    take and meaningless to dereference on the host.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp

    events, prepared, count = _case(4, "cuda")
    tissue_seed, event_seed = _t2_seeds(events, prepared)
    tangents = (*tissue_seed, *event_seed)
    cotangent = torch.randn(
        (4, prepared[0].numel(), count), dtype=torch.complex64, device="cuda"
    )
    with pytest.raises(ValueError, match="CPU tensors"):
        _run_packed_vjp_jvp(
            prepared,
            events,
            tangents,
            cotangent,
            state_count=STATES,
            output_count=count,
            threads=1,
        )
