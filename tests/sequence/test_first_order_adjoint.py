"""The device adjoint that carries no forward direction.

A first-order adjoint needs no dual, and on a device it is its own kernel
rather than the forward-over-reverse pass handed a direction of zeros. These
pin it against that pass -- the route the gradient took before it existed --
because two kernels agreeing is the only evidence that stripping the dual left
the answer alone.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import fse_description
from torchsim.sequence import _accelerators
from torchsim.sequence._accelerators import (
    _pack_events,
    _run_packed_vjp,
    _train_count,
)
from torchsim.sequence._parameters import FLOAT_NAMES, Geometry
from torchsim.sequence._simulation import TissueProperties, _prepare_tissue

ECHOES = 12
ECHO_SPACING_S = 5e-3
# Enough voxels that a device adjoint is worth specializing: below `detection`
# the fast path is not reached at all, and the test would pass without ever
# running the kernel it names.
VOXELS = 8192

WINDING = Geometry(flow_scale=90.0, washout_scale=3.0)


def _flip() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.deg2rad(80.0 + 80.0 * torch.rand(ECHOES, generator=generator))


def _events(phases):
    description = fse_description(
        _flip(), echo_spacing_s=ECHO_SPACING_S, phases_rad=phases
    )
    packed = _pack_events(
        "fse",
        description,
        repetitions=1,
        record="all",
        device=torch.device("cuda"),
        rf_raster_time_s=1e-6,
    )
    return packed.buffers, packed.output_count


def _tissue(**extra):
    generator = torch.Generator().manual_seed(7)

    def spread(low, high):
        return low + (high - low) * torch.rand(VOXELS, generator=generator)

    return TissueProperties(
        t1_ms=spread(600.0, 1800.0), t2_ms=spread(30.0, 150.0), **extra
    )


def _both(prepared, events, seed, output_count, state_count, geometry):
    """The specialized adjoint and the forward-over-reverse pass, same inputs."""
    actual = _run_packed_vjp(
        prepared,
        events,
        seed,
        state_count=state_count,
        output_count=output_count,
        threads=1,
        geometry=geometry,
    )
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
        output_count=output_count,
        threads=1,
        real_axis=None,
        geometry=geometry,
    )
    return actual, expected


def _agree(actual, expected, relative=2e-4):
    """How many gradients were compared, having checked every one of them.

    The floor is tied to the largest gradient in the set rather than to each
    parameter's own, because a gradient that vanishes by symmetry -- an
    off-resonance derivative at zero off-resonance in a CPMG train, which is
    round-off some ten orders below its siblings -- carries no relative
    precision to hold a kernel to.
    """
    floor = 1e-6 * max(float(value.abs().max()) for value in expected)
    compared = 0
    for index, name in enumerate(FLOAT_NAMES):
        scale = float(expected[index].abs().max())
        drift = float((expected[index] - actual[index]).abs().max())
        assert drift <= relative * scale + floor, f"{name}: {drift} vs {scale}"
        if scale > floor:
            compared += 1
    return compared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("state_count", [8, 12, 16, 17, 32])
def test_the_complex_adjoint_agrees_at_every_width(state_count: int) -> None:
    """Widths are swept because a reverse kernel has miscompiled silently at
    one state count before, and a single width would not have caught it.
    """
    events, output_count = _events(torch.pi / 2)
    prepared, _, _ = _prepare_tissue(_tissue(b0_hz=30.0), "cuda")
    seed = torch.randn(
        (_train_count(events), VOXELS, output_count),
        dtype=torch.complex64,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(3),
    )
    actual, expected = _both(
        prepared, events, seed, output_count, state_count, _accelerators.NO_GEOMETRY
    )
    assert _agree(actual, expected) > 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("extra", "geometry"),
    [
        ({}, _accelerators.NO_GEOMETRY),
        ({"b0_hz": 42.0}, _accelerators.NO_GEOMETRY),
        ({"b1_phase_rad": 0.35}, _accelerators.NO_GEOMETRY),
        ({"diffusion_um2_per_ms": 2.0}, _accelerators.NO_GEOMETRY),
        ({"velocity_m_per_s": 0.08}, WINDING),
        ({"b0_hz": 42.0, "velocity_m_per_s": 0.08}, WINDING),
    ],
    ids=["bare", "off-resonance", "transmit-phase", "diffusion", "flow", "both"],
)
def test_every_term_the_kernel_carries_agrees(extra, geometry) -> None:
    """One case per gated term, so a term dropped or doubled shows up as its
    own failure rather than hiding in a run that never switched it on.
    """
    events, output_count = _events(torch.pi / 2)
    prepared, _, _ = _prepare_tissue(_tissue(**extra), "cuda")
    seed = torch.randn(
        (_train_count(events), VOXELS, output_count),
        dtype=torch.complex64,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(5),
    )
    actual, expected = _both(
        prepared, events, seed, output_count, 12, geometry
    )
    assert _agree(actual, expected) > 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_drifting_phase_train_agrees_too() -> None:
    """A train whose refocusing phase moves echo to echo, which is what keeps
    the states off the real axis without any tissue property doing it.
    """
    events, output_count = _events(torch.pi / 2 + 0.07 * torch.arange(ECHOES))
    prepared, _, _ = _prepare_tissue(_tissue(), "cuda")
    seed = torch.randn(
        (_train_count(events), VOXELS, output_count),
        dtype=torch.complex64,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(9),
    )
    actual, expected = _both(
        prepared, events, seed, output_count, 12, _accelerators.NO_GEOMETRY
    )
    assert _agree(actual, expected) > 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_adjoint_stops_short_of_the_forward_over_reverse_pass() -> None:
    """The specialized kernel is the point of this route, so the cases above
    have to be reaching it rather than agreeing with themselves.
    """
    events, output_count = _events(torch.pi / 2)
    prepared, _, _ = _prepare_tissue(_tissue(b0_hz=30.0), "cuda")
    seed = torch.zeros(
        (_train_count(events), VOXELS, output_count),
        dtype=torch.complex64,
        device="cuda",
    )
    reached = []
    original = _accelerators._run_packed_vjp_jvp
    _accelerators._run_packed_vjp_jvp = (
        lambda *arguments, **keywords: reached.append(True)
        or original(*arguments, **keywords)
    )
    try:
        _run_packed_vjp(
            prepared,
            events,
            seed,
            state_count=12,
            output_count=output_count,
            threads=1,
        )
    finally:
        _accelerators._run_packed_vjp_jvp = original
    assert not reached
