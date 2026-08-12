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
