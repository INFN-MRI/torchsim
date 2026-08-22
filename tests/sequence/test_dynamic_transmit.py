"""The rotation a dynamically shimmed pulse performs, per voxel.

A static array reduces to a flip angle and a phase; weights that vary while the
pulse plays do not. These pin the generalization against the model it extends,
against an independent integration, and against autograd.
"""

from __future__ import annotations

import math
from itertools import repeat

import numpy as np
import pytest
import torch

from torchsim.sequence._description import RfDefinition, RfShape
from torchsim.sequence._transition import (
    dynamic_pair,
    transition_table,
)
from utils.packed_reference import simulate_packed

RASTER = 1e-6


def _pulse(samples: int = 128, *, bandwidth_hz: float = 2000.0) -> RfDefinition:
    """A sinc, which is what a slice-selective sequence actually drives."""
    grid = np.linspace(-2.0, 2.0, samples)
    envelope = np.sinc(grid) * (0.54 + 0.46 * np.cos(np.pi * grid / 2.0))
    envelope = envelope / np.abs(envelope).max()
    return RfDefinition(
        id=0,
        bandwidth_hz=bandwidth_hz,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=bandwidth_hz,
        total_b1sq_power=1.0,
        magnitude=RfShape(
            num_uncompressed=samples, samples=envelope.astype(np.float32)
        ),
    )


def _stepwise(drive, turn_z):
    """The same rotation composed as explicit 2x2 matrices, by matrix_exp.

    ``compose_spinor`` reaches each sample's element from closed forms for
    ``cos(t/2)`` and ``sin(t/2)/t`` and composes the Cayley-Klein pair by its
    own rule. This shares neither: it exponentiates the Pauli combination
    outright and multiplies the matrices. Agreement is therefore evidence
    about both halves of that algebra rather than a restatement of it.

    ``expm(-i t/2 (n . sigma))`` puts ``a`` and ``b`` in the top row, so the
    element is ``[[a, b], [-conj(b), conj(a)]]`` and the pulse is the product
    taken in the order the scanner plays it.
    """
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    pauli_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128)
    pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    net = torch.eye(2, dtype=torch.complex128)
    turns = repeat(turn_z) if isinstance(turn_z, float) else iter(turn_z)
    for sample, turn in zip(drive, turns):
        field = (
            complex(sample).real * pauli_x
            + complex(sample).imag * pauli_y
            + turn * pauli_z
        )
        net = torch.matrix_exp(-0.5j * field) @ net
    return net


# --- the model it generalizes ---


def test_weights_that_do_not_vary_reproduce_the_static_table():
    """A static array is a dynamic one whose weights are constant.

    The table is built over effective flip because a constant weight enters
    only through its product with the flip; feeding the same array to the
    per-voxel integrator has to land on the same rotation.
    """
    definition = _pulse()
    scaling = torch.tensor([0.6, 1.0, 1.4], dtype=torch.float64)
    flip = 0.7 * torch.pi

    table = transition_table(
        definition,
        torch.zeros(1, dtype=torch.float64),
        bins=512,
        rf_raster_time_s=RASTER,
    )
    expected = table.at(
        torch.zeros(3, dtype=torch.int64), scaling * flip
    )

    measured = dynamic_pair(
        definition,
        scaling.to(torch.complex128)[:, None],
        weights=torch.ones(1, dtype=torch.complex128),
        flip=flip,
        rf_raster_time_s=RASTER,
    )

    for reference, result in zip(expected, measured, strict=True):
        assert float((reference.to(torch.complex128) - result).abs().max()) < 1e-5


def test_a_channel_phase_turns_the_axis_and_nothing_else():
    """Turning the whole array by an angle turns the rotation axis with it,
    which is ``b -> b * exp(-i phi)`` and leaves ``a`` alone.
    """
    definition = _pulse()
    phi = 0.9
    plain = dynamic_pair(
        definition,
        torch.ones(1, 1, dtype=torch.complex128),
        weights=torch.ones(1, dtype=torch.complex128),
        flip=1.1,
        rf_raster_time_s=RASTER,
    )
    turned = dynamic_pair(
        definition,
        torch.ones(1, 1, dtype=torch.complex128),
        weights=torch.full(
            (1,), complex(math.cos(phi), math.sin(phi)), dtype=torch.complex128
        ),
        flip=1.1,
        rf_raster_time_s=RASTER,
    )

    assert abs(complex(turned[0][0]) - complex(plain[0][0])) < 1e-6
    expected = complex(plain[1][0]) * complex(math.cos(phi), -math.sin(phi))
    assert abs(complex(turned[1][0]) - expected) < 1e-6


def test_two_channels_in_antiphase_leave_the_voxel_alone():
    """The complex sum is what makes this true, and summing magnitudes would
    not: a voxel both channels reach equally and oppositely sees no field.
    """
    definition = _pulse()
    weights = torch.tensor([1.0 + 0.0j, -1.0 + 0.0j], dtype=torch.complex128)
    sensitivities = torch.ones(1, 2, dtype=torch.complex128)

    a, b = dynamic_pair(
        definition,
        sensitivities,
        weights=weights,
        flip=torch.pi,
        rf_raster_time_s=RASTER,
    )

    assert abs(complex(a[0]) - 1.0) < 1e-9
    assert abs(complex(b[0])) < 1e-9


# --- against an integration that shares no algebra ---


def test_the_pair_is_what_exponentiating_each_sample_gives():
    """A pulse whose weights genuinely vary, against matrix_exp per sample."""
    definition = _pulse(samples=64)
    samples = 64
    generator = torch.Generator().manual_seed(11)
    weights = torch.complex(
        torch.rand(2, samples, generator=generator, dtype=torch.float64) * 2.0 - 1.0,
        torch.rand(2, samples, generator=generator, dtype=torch.float64) * 2.0 - 1.0,
    )
    sensitivities = torch.tensor([[0.8 + 0.2j, 0.5 - 0.3j]], dtype=torch.complex128)
    offset = 130.0

    a, b = dynamic_pair(
        definition,
        sensitivities,
        weights=weights,
        flip=1.3,
        off_resonance_hz=torch.tensor([offset]),
        rf_raster_time_s=RASTER,
    )

    envelope = torch.as_tensor(definition.complex_envelope(), dtype=torch.complex128)
    shape = envelope / envelope.sum()
    drive = [
        1.3 * shape[sample] * (sensitivities @ weights[:, sample])[0]
        for sample in range(samples)
    ]
    expected = _stepwise(drive, 2.0 * torch.pi * offset * RASTER)
    measured = torch.tensor(
        [
            [complex(a[0]), complex(b[0])],
            [-complex(b[0]).conjugate(), complex(a[0]).conjugate()],
        ],
        dtype=torch.complex128,
    )

    assert float((expected - measured).abs().max()) < 1e-6


def test_the_pair_stays_on_the_unit_sphere():
    """``|a|^2 + |b|^2 == 1`` is what makes it a rotation at all, and it has to
    survive a thousand composed samples in the presence of off-resonance.
    """
    definition = _pulse(samples=512)
    generator = torch.Generator().manual_seed(5)
    weights = torch.complex(
        torch.rand(4, 512, generator=generator, dtype=torch.float64) * 2.0 - 1.0,
        torch.rand(4, 512, generator=generator, dtype=torch.float64) * 2.0 - 1.0,
    )
    sensitivities = torch.complex(
        torch.rand(64, 4, generator=generator, dtype=torch.float64),
        torch.rand(64, 4, generator=generator, dtype=torch.float64),
    )

    a, b = dynamic_pair(
        definition,
        sensitivities,
        weights=weights,
        flip=2.0,
        off_resonance_hz=torch.linspace(-300.0, 300.0, 64),
        rf_raster_time_s=RASTER,
    )

    norm = a.to(torch.complex128).abs() ** 2 + b.to(torch.complex128).abs() ** 2
    assert float((norm - 1.0).abs().max()) < 1e-6


# --- what a gradient reaches through it ---


@pytest.mark.parametrize("name", ["weights", "sensitivities"])
def test_the_pair_differentiates_back_to_the_array(name: str) -> None:
    """A cotangent on the pair has to reach the channel weights and the
    sensitivity maps, since that is the whole point of resolving the array in
    torch rather than in a kernel.
    """
    definition = _pulse(samples=32)
    generator = torch.Generator().manual_seed(3)
    weights = torch.complex(
        torch.rand(2, 32, generator=generator, dtype=torch.float64),
        torch.rand(2, 32, generator=generator, dtype=torch.float64),
    )
    sensitivities = torch.complex(
        torch.rand(3, 2, generator=generator, dtype=torch.float64),
        torch.rand(3, 2, generator=generator, dtype=torch.float64),
    )
    leaves = {"weights": weights, "sensitivities": sensitivities}
    leaves[name] = leaves[name].clone().requires_grad_(True)

    def reading(**overrides):
        a, b = dynamic_pair(
            definition,
            overrides.get("sensitivities", leaves["sensitivities"]),
            weights=overrides.get("weights", leaves["weights"]),
            flip=1.0,
            rf_raster_time_s=RASTER,
        )
        return (a.to(torch.complex128).real + 2.0 * b.to(torch.complex128).imag).sum()

    reading().backward()
    gradient = leaves[name].grad
    assert gradient is not None
    assert float(gradient.abs().max()) > 0.0

    # Central differences along one live entry. The pair is stored in
    # ``complex64``, as the tabulated one is, so a step small enough to make
    # truncation negligible would sit under the reference's own resolution --
    # the error grows as the step shrinks. This one is above that floor.
    step = 1e-2
    probe = leaves[name].detach().clone()
    index = (0, probe.shape[-1] // 2) if name == "weights" else (1, 1)
    ahead = probe.clone()
    ahead[index] = ahead[index] + step
    behind = probe.clone()
    behind[index] = behind[index] - step
    difference = float(
        (reading(**{name: ahead}) - reading(**{name: behind})).real
    ) / (2.0 * step)

    assert abs(difference) > 1e-9
    assert abs(float(gradient[index].real) - difference) / abs(difference) < 1e-4


# --- through the state machine ---


ECHOES = 6
VOXELS = 5
STATES = 16


def _train():
    """A refocused train on a voxel-varying transmit field, and the rotation
    each of its pulses performs there.

    The array holds still while every pulse plays, which is the one case both
    the tabulated route and the per-voxel one describe -- so it is the case
    that can hold them against each other.
    """
    from torchsim.sequence._accelerators import _pack_events, _shim_count
    from torchsim.sequence._builders import fse_description
    from torchsim.sequence._simulation import TissueProperties, _prepare_tissue
    from torchsim.sequence._transition import DynamicPairs

    definition = _pulse(samples=96)
    flips = torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    packed = _pack_events(
                fse_description(
            flips,
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
            excitation_phase_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=RASTER,
    )
    events = (
        packed.duration, packed.kind, packed.flip, packed.phase, packed.action,
        packed.output_index, packed.shim_index, packed.saturation,
        packed.rf_frequency_hz,
    )
    scaling = torch.linspace(0.7, 1.3, VOXELS, dtype=torch.float64)
    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
            t2_ms=torch.linspace(50.0, 110.0, VOXELS),
            b1=scaling.to(torch.float32),
        ),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    assert _shim_count(prepared) == 1

    count = int(packed.kind.numel())
    halves = [
        dynamic_pair(
            definition,
            scaling.to(torch.complex128)[:, None],
            weights=torch.ones(1, dtype=torch.complex128),
            flip=float(packed.flip.reshape(-1, count)[0, event]),
            rf_raster_time_s=RASTER,
        )
        for event in range(count)
    ]
    pairs = DynamicPairs(
        a=torch.stack([half[0] for half in halves]),
        b=torch.stack([half[1] for half in halves]),
        index=torch.arange(count, dtype=torch.int32),
    )
    return definition, prepared, events, pairs



def test_the_reference_reads_a_dynamic_pair_where_it_reads_a_table():
    """A pulse whose weights hold still is a pulse the exact slice profile
    already describes, so the two routes through the state machine have to
    record the same train.

    The pair is integrated at each pulse's own flip, so a row per pulse is what
    the table's flip axis stands in for.
    """
    from torchsim.sequence._transition import transition_table

    definition, prepared, events, pairs = _train()
    table = transition_table(
        definition,
        torch.zeros(1, dtype=torch.float64),
        bins=1024,
        theta_max=4.0,
        rf_raster_time_s=RASTER,
    )

    options = dict(state_count=16, output_count=ECHOES)
    tabulated = simulate_packed(prepared, events, profile=table, **options)
    integrated = simulate_packed(prepared, events, dynamic=pairs, **options)

    assert float(tabulated.abs().max()) > 0.0
    worst = float((tabulated - integrated).abs().max() / tabulated.abs().max())
    assert worst < 1e-4, worst


def test_the_host_kernel_reads_the_pair_the_reference_does():
    """The C++ forward through the dynamic mode, against the oracle.

    The two share no code: the reference turns a pulse in torch and the kernel
    reads four floats per voxel out of a packed buffer.
    """
    from torchsim.sequence._accelerators import _run_packed

    _, prepared, events, pairs = _train()

    expected = simulate_packed(
        prepared, events, state_count=16, output_count=ECHOES, dynamic=pairs
    )
    measured = _run_packed(
        prepared, events, 16, ECHOES, 1, dynamic=pairs
    )

    assert float(expected.abs().max()) > 0.0
    worst = float((expected - measured).abs().max() / expected.abs().max())
    assert worst < 1e-6, worst


def test_the_host_kernel_still_reads_a_table_where_one_is_given():
    """The mode is picked from which buffer the caller filled, so a sequence
    with a table has to be untouched by the pair's arrival.
    """
    from torchsim.sequence._accelerators import _run_packed
    from torchsim.sequence._transition import SliceTables, transition_table

    definition, prepared, events, _ = _train()
    table = SliceTables.alone(
        transition_table(
            definition,
            torch.zeros(1, dtype=torch.float64),
            bins=256,
            theta_max=4.0,
            rf_raster_time_s=RASTER,
        ),
        int(events[1].numel()),
    )

    expected = simulate_packed(
        prepared, events, state_count=16, output_count=ECHOES,
        profile=table.tables[0],
    )
    measured = _run_packed(prepared, events, 16, ECHOES, 1, profile=table)

    assert float(expected.abs().max()) > 0.0
    worst = float((expected - measured).abs().max() / expected.abs().max())
    assert worst < 1e-6, worst


def test_the_host_forward_mode_follows_a_direction_along_the_pair():
    """Under the dynamic mode the array is resolved outside the kernel, so a
    direction along a channel weight or a sensitivity arrives already carried
    through the pulse integral -- and the kernel's job is to carry it on.

    The reference is a central difference on the pair itself. The pair is
    stored in ``complex64``, so the difference stops improving below a step of
    about 1e-2; this one sits at that floor rather than under it.
    """
    from torchsim.sequence._accelerators import _run_packed, _run_packed_jvp
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(13)
    direction = (
        torch.randn(pairs.a.shape + (4,), generator=generator) * 0.05
    ).contiguous()
    still = tuple(torch.zeros_like(value) for value in prepared)
    still_events = tuple(
        torch.zeros_like(value) for value in (events[0], events[2], events[3])
    )

    measured = _run_packed_jvp(
        prepared, events, still, still_events, 16, ECHOES, 1, -1,
        dynamic=pairs, dynamic_direction=direction,
    )

    step = 1e-2

    def moved(sign):
        packed = pairs.packed() + sign * step * direction
        return DynamicPairs(
            a=torch.complex(packed[..., 0], packed[..., 1]).to(torch.complex64),
            b=torch.complex(packed[..., 2], packed[..., 3]).to(torch.complex64),
            index=pairs.index,
        )

    difference = (
        _run_packed(prepared, events, 16, ECHOES, 1, dynamic=moved(+1))
        - _run_packed(prepared, events, 16, ECHOES, 1, dynamic=moved(-1))
    ) / (2.0 * step)

    assert float(difference.abs().max()) > 0.0
    worst = float((difference - measured).abs().max() / difference.abs().max())
    assert worst < 1e-3, worst


def test_a_direction_along_nothing_moves_nothing():
    """Seeding no direction at all has to leave the forward-mode result at
    zero, which is what catches a buffer read where none was given.
    """
    from torchsim.sequence._accelerators import _run_packed_jvp

    _, prepared, events, pairs = _train()
    still = tuple(torch.zeros_like(value) for value in prepared)
    still_events = tuple(
        torch.zeros_like(value) for value in (events[0], events[2], events[3])
    )
    quiet = torch.zeros(pairs.a.shape + (4,), dtype=torch.float32)

    measured = _run_packed_jvp(
        prepared, events, still, still_events, 16, ECHOES, 1, -1,
        dynamic=pairs, dynamic_direction=quiet,
    )

    assert float(measured.abs().max()) == 0.0


def test_the_host_adjoint_returns_the_cotangent_on_the_pair():
    """Under the dynamic mode the flip is inside the pair rather than read
    against it, so it has no gradient in the kernel: the cotangent comes out on
    the rotation itself and whatever integrated it carries the rest.

    A row belongs to one train and a work item is one (train, atom), so every
    entry has exactly one writer and the reverse pass accumulates nothing
    across threads.
    """
    from torchsim.sequence._accelerators import _run_packed, _run_packed_vjp
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(21)
    seed = torch.complex(
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
    )

    gradient = _run_packed_vjp(
        prepared, events, seed, state_count=16, output_count=ECHOES,
        threads=1, dynamic=pairs,
    )[-1]

    assert gradient.shape == pairs.a.shape + (4,)
    assert float(gradient.abs().max()) > 0.0

    direction = (
        torch.randn(pairs.a.shape + (4,), generator=generator) * 0.05
    ).contiguous()
    step = 3e-2

    def reading(pair):
        recorded = _run_packed(prepared, events, 16, ECHOES, 1, dynamic=pair)
        return float((seed.conj() * recorded).real.sum())

    def moved(sign):
        packed = pairs.packed() + sign * step * direction
        return DynamicPairs(
            a=torch.complex(packed[..., 0], packed[..., 1]).to(torch.complex64),
            b=torch.complex(packed[..., 2], packed[..., 3]).to(torch.complex64),
            index=pairs.index,
        )

    difference = (reading(moved(+1)) - reading(moved(-1))) / (2.0 * step)
    along = float((gradient * direction).sum())

    assert abs(difference) > 0.0
    assert abs(along - difference) / abs(difference) < 1e-3


def test_a_sequence_with_no_pair_still_gets_the_gradients_it_did():
    """The pair's arrival adds a buffer to every entry point's tail, so a run
    that fills none of it has to come back exactly as it was.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp

    _, prepared, events, _ = _train()
    generator = torch.Generator().manual_seed(23)
    seed = torch.complex(
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
    )

    gradients = _run_packed_vjp(
        prepared, events, seed, state_count=16, output_count=ECHOES, threads=1
    )

    assert len(gradients) == len(prepared) + 3
    assert any(float(value.abs().max()) > 0.0 for value in gradients)


def test_the_second_order_pass_returns_the_adjoint_given_no_direction():
    """The forward-over-reverse kernel with nothing to follow has to return
    what the first-order one does -- including on the pair, which the two reach
    by different code.
    """
    from torchsim.sequence._accelerators import (
        _run_packed_vjp, _run_packed_vjp_jvp,
    )

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(21)
    seed = torch.complex(
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
    )
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )
    quiet = torch.zeros(pairs.a.shape + (4,), dtype=torch.float32)

    first = _run_packed_vjp(
        prepared, events, seed, state_count=16, output_count=ECHOES,
        threads=1, dynamic=pairs,
    )
    # A direction of zero and no direction at all are different arguments:
    # one is a buffer of zeros, the other a null the kernels must not read.
    for direction in (quiet, None):
        _, second = _run_packed_vjp_jvp(
            prepared, events, still, seed, state_count=16, output_count=ECHOES,
            threads=1, dynamic=pairs, dynamic_direction=direction,
        )

        compared = 0
        for expected, measured in zip(first, second, strict=True):
            scale = float(expected.abs().max())
            if scale < 1e-6:
                continue
            assert float((expected - measured).abs().max()) / scale < 1e-5
            compared += 1
        assert compared > 3
    assert float(first[-1].abs().max()) > 0.0


def test_the_second_order_pass_differentiates_the_pair_gradient():
    """Given a direction along the pair, the curvature is what the first-order
    gradient's own derivative is.
    """
    from torchsim.sequence._accelerators import (
        _run_packed_vjp, _run_packed_vjp_jvp,
    )
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(21)
    seed = torch.complex(
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
    )
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )
    direction = (
        torch.randn(pairs.a.shape + (4,), generator=generator) * 0.05
    ).contiguous()

    curvature, _ = _run_packed_vjp_jvp(
        prepared, events, still, seed, state_count=16, output_count=ECHOES,
        threads=1, dynamic=pairs, dynamic_direction=direction,
    )

    def adjoint(pair):
        return _run_packed_vjp(
            prepared, events, seed, state_count=16, output_count=ECHOES,
            threads=1, dynamic=pair,
        )[-1]

    step = 3e-2

    def moved(sign):
        packed = pairs.packed() + sign * step * direction
        return DynamicPairs(
            a=torch.complex(packed[..., 0], packed[..., 1]).to(torch.complex64),
            b=torch.complex(packed[..., 2], packed[..., 3]).to(torch.complex64),
            index=pairs.index,
        )

    difference = (adjoint(moved(+1)) - adjoint(moved(-1))) / (2.0 * step)

    assert float(difference.abs().max()) > 0.0
    worst = float(
        (curvature[-1] - difference).abs().max() / difference.abs().max()
    )
    assert worst < 1e-3, worst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_forward_reads_the_pair_the_host_does():
    """The two backends share no code, so agreement is what keeps the per-voxel
    read honest on the card: a row taken from the wrong train would still
    produce a plausible train.
    """
    from torchsim.sequence._accelerators import _run_packed
    from torchsim.sequence._epg_triton import simulate
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    host = _run_packed(prepared, events, 16, ECHOES, 1, dynamic=pairs)
    card = simulate(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in events),
        state_count=16,
        output_count=ECHOES,
        dynamic=DynamicPairs(
            a=pairs.a.cuda(), b=pairs.b.cuda(), index=pairs.index.cuda()
        ),
    ).cpu()

    assert float(host.abs().max()) > 0.0
    assert float((host - card).abs().max() / host.abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("state_count", [8, 15, 16, 17, 32])
def test_the_cuda_reverse_agrees_at_every_width(state_count: int) -> None:
    """The adjoint and the cotangent on the pair, against the host.

    Swept over the state axis because the width is a compile-time constant on
    the card: each one is a kernel of its own, and a rotation read per voxel
    rather than built from a flip angle is the largest of them.
    """
    from torchsim.sequence._accelerators import _run_packed_vjp
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(21)
    seed = torch.complex(
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
    )
    options = dict(state_count=state_count, output_count=ECHOES, threads=1)

    host = _run_packed_vjp(prepared, events, seed, dynamic=pairs, **options)
    card = _run_packed_vjp(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in events),
        seed.cuda(),
        dynamic=DynamicPairs(
            a=pairs.a.cuda(), b=pairs.b.cuda(), index=pairs.index.cuda()
        ),
        **options,
    )

    assert float(host[-1].abs().max()) > 0.0
    for expected, measured in zip(host, card, strict=True):
        scale = float(expected.abs().max())
        if scale < 1e-6:
            continue
        assert float((expected - measured.cpu()).abs().max()) / scale < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("state_count", [15, 16, 17])
def test_the_cuda_second_order_agrees_at_every_width(state_count: int) -> None:
    """The curvature along a direction on the pair, against the host."""
    from torchsim.sequence._accelerators import _run_packed_vjp_jvp
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(21)
    seed = torch.complex(
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
        torch.rand(VOXELS, ECHOES, generator=generator) * 2.0 - 1.0,
    )
    tangents = tuple(
        torch.randn(value.shape, generator=generator) * 0.01
        for value in (*prepared, events[0], events[2], events[3])
    )
    direction = (
        torch.randn(pairs.a.shape + (4,), generator=generator) * 0.05
    ).contiguous()

    host = _run_packed_vjp_jvp(
        prepared, events, tangents, seed, state_count, ECHOES, 1,
        dynamic=pairs, dynamic_direction=direction,
    )
    card = _run_packed_vjp_jvp(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in events),
        tuple(value.cuda() for value in tangents),
        seed.cuda(),
        state_count,
        ECHOES,
        1,
        dynamic=DynamicPairs(
            a=pairs.a.cuda(), b=pairs.b.cuda(), index=pairs.index.cuda()
        ),
        dynamic_direction=direction.cuda(),
    )

    for expected_side, measured_side in zip(host, card, strict=True):
        assert float(expected_side[-1].abs().max()) > 0.0
        for expected, measured in zip(expected_side, measured_side, strict=True):
            scale = float(expected.abs().max())
            if scale < 1e-6:
                continue
            assert float((expected - measured.cpu()).abs().max()) / scale < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_cuda_forward_mode_follows_the_direction_the_host_does():
    """Forward mode along a direction on the pair, against the host."""
    from torchsim.sequence._accelerators import _run_packed_jvp
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    generator = torch.Generator().manual_seed(29)
    tissue = tuple(
        torch.randn(value.shape, generator=generator) * 0.01 for value in prepared
    )
    per_event = tuple(
        torch.randn(value.shape, generator=generator) * 0.01
        for value in (events[0], events[2], events[3])
    )
    direction = (
        torch.randn(pairs.a.shape + (4,), generator=generator) * 0.05
    ).contiguous()

    host = _run_packed_jvp(
        prepared, events, tissue, per_event, 16, ECHOES, 1,
        dynamic=pairs, dynamic_direction=direction,
    )
    card = _run_packed_jvp(
        tuple(value.cuda() for value in prepared),
        tuple(value.cuda() for value in events),
        tuple(value.cuda() for value in tissue),
        tuple(value.cuda() for value in per_event),
        16,
        ECHOES,
        1,
        dynamic=DynamicPairs(
            a=pairs.a.cuda(), b=pairs.b.cuda(), index=pairs.index.cuda()
        ),
        dynamic_direction=direction.cuda(),
    ).cpu()

    assert float(host.abs().max()) > 0.0
    assert float((host - card).abs().max() / host.abs().max()) < 1e-5


# --- the routes that would drop the pair rather than carry it ---


def _real_subspace_train(trains: int = 1):
    """A train that satisfies every real-subspace condition, and its pairs.

    One refocusing phase, an excitation sharing it, no off-resonance, no
    transmit phase and no velocity -- so the verdict is 1 unless the pair
    itself rules it out. That is the whole point: the reduced kernels take no
    pair argument, so a verdict of 1 does not slow this train down, it plays a
    different pulse.
    """
    from torchsim.sequence._accelerators import _pack_events
    from torchsim.sequence._builders import fse_description
    from torchsim.sequence._simulation import TissueProperties, _prepare_tissue
    from torchsim.sequence._transition import DynamicPairs

    definition = _pulse(samples=96)
    flips = torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    if trains > 1:
        flips = flips[None, :] * torch.linspace(0.9, 1.1, trains)[:, None]
    packed = _pack_events(
                fse_description(
            flips,
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
            excitation_phase_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=RASTER,
    )
    events = (
        packed.duration, packed.kind, packed.flip, packed.phase, packed.action,
        packed.output_index, packed.shim_index, packed.saturation,
        packed.rf_frequency_hz,
    )
    scaling = torch.linspace(0.7, 1.3, VOXELS, dtype=torch.float64)
    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
            t2_ms=torch.linspace(50.0, 110.0, VOXELS),
            b1=scaling.to(torch.float32),
        ),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)

    count = int(packed.kind.numel())
    flat = packed.flip.reshape(-1, count)
    halves = [
        dynamic_pair(
            definition,
            scaling.to(torch.complex128)[:, None],
            weights=torch.ones(1, dtype=torch.complex128),
            flip=float(flat[0, event]),
            rf_raster_time_s=RASTER,
        )
        for event in range(count)
    ]
    pairs = DynamicPairs(
        a=torch.stack([half[0] for half in halves]),
        b=torch.stack([half[1] for half in halves]),
        index=torch.arange(count, dtype=torch.int32),
    )
    return prepared, events, pairs, packed.output_count


def test_the_subspace_verdict_refuses_a_per_voxel_pair():
    """The predicate itself, held to the reason: this train is inside the real
    subspace, and a pair still has to rule the reduced kernels out.
    """
    from torchsim.sequence._accelerators import real_subspace_axis

    prepared, events, pairs, _ = _real_subspace_train()
    assert real_subspace_axis(events, prepared) == 1
    assert real_subspace_axis(events, prepared, dynamic=pairs) is None


def test_a_pair_survives_a_train_the_real_kernel_would_have_taken(monkeypatch):
    """And the verdict is reached rather than skipped.

    ``detection`` is what decides whether the subspace is even tested for, and
    it is measured per machine -- so a test that hopes to clear it by carrying
    enough voxels is testing the machine. Forcing it to zero asks the question
    directly.
    """
    from torchsim.sequence import _accelerators

    monkeypatch.setattr(
        _accelerators, "detection", lambda kind, device, state_count: 0.0
    )
    prepared, events, pairs, output_count = _real_subspace_train()
    settled = _accelerators._run_packed(
        prepared, events, 16, output_count, 1, dynamic=pairs
    )
    overruled = _accelerators._run_packed(
        prepared, events, 16, output_count, 1, real_axis=0, dynamic=pairs
    )
    assert torch.equal(settled, overruled)


def test_the_lane_forward_leaves_a_pair_to_the_scalar_kernel(monkeypatch):
    """The lane kernel carries no rotation to read a pair into, so a run that
    would otherwise vectorize has to fall back rather than play a hard pulse.
    """
    from torchsim.sequence import _accelerators

    monkeypatch.setenv("TORCHSIM_LANES", "1")
    prepared, events, pairs, output_count = _real_subspace_train(trains=4)
    lanes = _accelerators._run_packed(
        prepared, events, 16, output_count, 1, dynamic=pairs
    )
    monkeypatch.delenv("TORCHSIM_LANES")
    scalar = _accelerators._run_packed(
        prepared, events, 16, output_count, 1, dynamic=pairs
    )
    assert torch.equal(lanes, scalar)


# --- the pulse's own channels, and where across the slice it is integrated ---


def _per_channel(samples: int = 128, *, bandwidth_hz: float = 2000.0) -> RfDefinition:
    """A sinc on one channel and a Gaussian on the other."""
    grid = np.linspace(-2.0, 2.0, samples)
    first = np.sinc(grid)
    second = np.exp(-2.0 * grid**2)
    return RfDefinition(
        id=1,
        bandwidth_hz=bandwidth_hz,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=bandwidth_hz,
        total_b1sq_power=1.0,
        magnitude=(
            RfShape(samples, first.astype(np.float32)),
            RfShape(samples, second.astype(np.float32)),
        ),
    )


def test_a_per_channel_envelope_drives_what_per_sample_weights_would():
    """The two ways of saying a channel plays its own waveform must agree: as
    the definition's own envelope, or as one flat envelope with the waveform
    put in the weights.
    """
    definition = _per_channel(samples=64)
    channels = torch.as_tensor(
        definition.complex_envelope(), dtype=torch.complex128
    )
    sensitivities = torch.tensor([[0.8 + 0.2j, 0.5 - 0.3j]], dtype=torch.complex128)

    combined = definition.combined_envelope()
    flat = RfDefinition(
        id=2,
        bandwidth_hz=definition.bandwidth_hz,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=definition.band_bandwidth_hz,
        total_b1sq_power=1.0,
        magnitude=RfShape(combined.size, np.ones(combined.size, dtype=np.float32)),
    )

    own = dynamic_pair(
        definition, sensitivities, flip=1.4, rf_raster_time_s=RASTER
    )
    # The flat pulse normalizes by its own area, so scale the weights by the
    # area the per-channel one is read against to drive the same flip.
    weighted = dynamic_pair(
        flat,
        sensitivities,
        weights=channels / channels.sum() * combined.size,
        flip=1.4,
        rf_raster_time_s=RASTER,
    )

    for reference, result in zip(own, weighted, strict=True):
        assert float((reference - result).abs().max()) < 1e-6


def test_a_pulse_of_no_bandwidth_turns_every_position_alike():
    definition = _pulse(samples=64, bandwidth_hz=0.0)
    sensitivities = torch.ones(1, 1, dtype=torch.complex128)
    positions = torch.linspace(-0.5, 0.5, 5)

    a, b = dynamic_pair(
        definition,
        sensitivities,
        positions=positions,
        flip=1.0,
        rf_raster_time_s=RASTER,
    )

    assert a.numel() == positions.numel()
    assert torch.equal(a, a[:1].expand_as(a))
    assert torch.equal(b, b[:1].expand_as(b))


def test_a_selective_pulse_falls_off_outside_the_slice():
    """A four-lobe sinc played over ``4 / bandwidth_hz`` seconds, so that the
    definition's bandwidth is the one the waveform actually has and a
    normalized position of 0.5 lands on the edge of the passband.
    """
    definition = _pulse(samples=256, bandwidth_hz=2000.0)
    sensitivities = torch.ones(1, 1, dtype=torch.complex128)
    positions = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0])

    _, b = dynamic_pair(
        definition,
        sensitivities,
        positions=positions,
        flip=torch.pi / 2.0,
        rf_raster_time_s=4.0 / (2000.0 * 256.0),
    )
    turned = b.abs()

    centre = float(turned[0])
    assert centre > 0.7
    assert float(turned[1]) > 0.8 * centre
    # Half the thickness out is the half-amplitude edge of the slice.
    assert 0.35 * centre < float(turned[2]) < 0.6 * centre
    assert float(turned[3]) < 0.05 * centre
    assert float(turned[4]) < float(turned[3])


def test_positions_run_fastest_and_sensitivities_repeat_across_them():
    """The kernels read an atom as ``voxel * locations + location``, so a voxel
    the array reaches twice as hard must own a contiguous run of locations.
    """
    definition = _pulse(samples=64, bandwidth_hz=0.0)
    sensitivities = torch.tensor([[1.0 + 0.0j], [0.5 + 0.0j]], dtype=torch.complex128)
    positions = torch.tensor([-0.25, 0.0, 0.25])

    _, b = dynamic_pair(
        definition,
        sensitivities,
        positions=positions,
        flip=1.0,
        rf_raster_time_s=RASTER,
    )

    assert b.numel() == 6
    first, second = b[:3], b[3:]
    assert torch.equal(first, first[:1].expand_as(first))
    assert torch.equal(second, second[:1].expand_as(second))
    assert float(second[0].abs()) < float(first[0].abs())


def test_a_position_and_an_off_resonance_reach_the_same_turn():
    """Position enters through the slice-select gradient as an offset in Hz, so
    a voxel a thickness off centre is a voxel that far off resonance.
    """
    definition = _pulse(samples=64)
    sensitivities = torch.ones(1, 1, dtype=torch.complex128)
    offset = 0.3

    placed = dynamic_pair(
        definition,
        sensitivities,
        positions=torch.tensor([offset]),
        flip=1.0,
        rf_raster_time_s=RASTER,
    )
    detuned = dynamic_pair(
        definition,
        sensitivities,
        off_resonance_hz=torch.tensor([offset * definition.bandwidth_hz]),
        flip=1.0,
        rf_raster_time_s=RASTER,
    )

    for reference, result in zip(placed, detuned, strict=True):
        assert float((reference - result).abs().max()) < 1e-7


def test_a_pulse_reaches_its_rotation_one_way_or_the_other():
    """Handed both, the kernels read the pair and the table says nothing, so a
    caller who built one is owed the news rather than a silent choice.
    """
    from torchsim.sequence._accelerators import _run_packed
    from torchsim.sequence._transition import transition_table

    definition, prepared, events, pairs = _train()
    table = transition_table(
        definition,
        torch.zeros(1, dtype=torch.float64),
        bins=32,
        rf_raster_time_s=RASTER,
    )

    with pytest.raises(ValueError, match="not through both"):
        _run_packed(
            prepared, events, 16, ECHOES, 1, profile=table, dynamic=pairs
        )


# --- end to end: the waveform picks the mode ---


def _split_across_two_channels(flips: torch.Tensor):
    """An FSE train whose pulses drive half their rectangle on each channel.

    The channels sum back to the pulse every builder emits, so the rotation
    integrated per voxel is the one a flip angle and a phase already name --
    which is what lets the dynamic route be held against the ordinary one.
    """
    from dataclasses import replace

    from torchsim.sequence._builders import fse_description

    base = fse_description(
        flips,
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    half = RfShape(2, np.full(2, 0.5, dtype=np.float32))
    twin = replace(base.rf_definitions[0], magnitude=(half, half))
    return replace(base, rf_definitions={0: twin}), base


def _sensitivities():
    generator = torch.Generator().manual_seed(5)
    magnitude = 0.6 + 0.8 * torch.rand(2, VOXELS, generator=generator)
    phase = torch.pi * (2.0 * torch.rand(2, VOXELS, generator=generator) - 1.0)
    return magnitude, phase


def test_a_split_pulse_records_the_train_its_sum_records():
    """The dispatch, end to end: the pulse's own waveform sends the run through
    the per-voxel rotation, and the answer is the one the flip-and-phase route
    gives for the field the channels sum to.
    """
    from torchsim.sequence import FSE, TissueProperties

    flips = torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    split, plain = _split_across_two_channels(flips)
    magnitude, phase = _sensitivities()
    # Each channel carries half the rectangle, so the field a voxel sees is the
    # mean of what the two put there -- which is a transmit map the ordinary
    # flip-and-phase route reads.
    combined = torch.polar(magnitude, phase).mean(dim=0)
    tissue = dict(
        t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
        t2_ms=torch.linspace(50.0, 110.0, VOXELS),
    )

    integrated = FSE().simulate(
        split,
        TissueProperties(**tissue, b1=magnitude, b1_phase_rad=phase),
        nstates=16,
    ).signal
    turned = FSE().simulate(
        plain,
        TissueProperties(
            **tissue, b1=combined.abs(), b1_phase_rad=combined.angle()
        ),
        nstates=16,
    ).signal

    assert float(turned.abs().max()) > 0.0
    worst = float((integrated - turned).abs().max() / turned.abs().max())
    assert worst < 1e-5, worst


def test_a_split_pulse_leaves_the_transmit_phase_to_the_rotation():
    """The pair is integrated against the complex sensitivities, so the phase
    is already in it. Left on the tissue as well it would be turned by twice,
    and the run would not answer the summed field.
    """
    from torchsim.sequence._simulation import _dynamic_transmit

    split, _ = _split_across_two_channels(torch.deg2rad(torch.full((ECHOES,), 140.0)))
    magnitude, phase = _sensitivities()
    tissue = _tissue_properties(magnitude, phase)

    left, sensitivities = _dynamic_transmit(tissue, split, None)

    assert sensitivities.shape == (VOXELS, 2)
    assert left.b1_phase_rad == 0.0
    torch.testing.assert_close(
        left.b1, torch.polar(magnitude, phase).sum(dim=0).abs(), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        sensitivities.to(torch.complex64), torch.polar(magnitude, phase).mT
    )


def _tissue_properties(magnitude, phase):
    from torchsim.sequence import TissueProperties

    return TissueProperties(
        t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
        t2_ms=torch.linspace(50.0, 110.0, VOXELS),
        b1=magnitude,
        b1_phase_rad=phase,
    )


def test_a_single_channel_train_never_reaches_the_pair():
    """Nothing a builder emits is to be diverted onto the integrated route."""
    from torchsim.sequence._builders import fse_description
    from torchsim.sequence._simulation import _dynamic_transmit

    plain = fse_description(torch.deg2rad(torch.full((ECHOES,), 140.0)), 5e-3)
    magnitude, phase = _sensitivities()

    _, sensitivities = _dynamic_transmit(
        _tissue_properties(magnitude[:1, 0], phase[:1, 0]), plain, None
    )

    assert sensitivities is None

def test_a_static_shim_beside_a_dynamic_pulse_is_refused():
    from dataclasses import replace

    from torchsim.sequence import ShimDefinition
    from torchsim.sequence._simulation import _dynamic_transmit

    split, _ = _split_across_two_channels(torch.deg2rad(torch.full((ECHOES,), 140.0)))
    shimmed = replace(
        split,
        shim_definitions={0: ShimDefinition(0, (1.0, 1.0), (0.0, 0.0))},
    )
    magnitude, phase = _sensitivities()

    with pytest.raises(NotImplementedError, match="second answer"):
        _dynamic_transmit(_tissue_properties(magnitude, phase), shimmed, None)


def test_a_transmit_map_short_of_the_channels_is_refused():
    from torchsim.sequence._simulation import _dynamic_transmit

    split, _ = _split_across_two_channels(torch.deg2rad(torch.full((ECHOES,), 140.0)))
    magnitude, phase = _sensitivities()

    with pytest.raises(ValueError, match="transmit map apiece"):
        _dynamic_transmit(_tissue_properties(magnitude[:1], phase[:1]), split, None)


def test_the_dispatch_hands_the_kernels_a_pair_and_no_table(monkeypatch):
    """Agreement with the flip-and-phase route is only evidence about the pair
    if the pair is what ran.
    """
    from torchsim.sequence import FSE
    from torchsim.sequence import _accelerators

    split, _ = _split_across_two_channels(torch.deg2rad(torch.full((ECHOES,), 140.0)))
    magnitude, phase = _sensitivities()
    seen = {}
    original = _accelerators._run_packed

    def record(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(_accelerators, "_run_packed", record)
    FSE().simulate(
        split, _tissue_properties(magnitude, phase), nstates=8
    )

    assert seen["profile"] is None
    pairs = seen["dynamic"]
    assert pairs is not None
    assert pairs.voxels == VOXELS
    # One row per distinct pulse: the excitation and the flip every refocusing
    # shares, rather than one per event.
    assert pairs.rows == 2
    assert int(pairs.index.numel()) == len(split.events)


def test_a_dynamic_pulse_is_integrated_across_the_slice():
    """Asking for positions spreads each voxel over them and averages the
    recorded signal, so a selective pulse records less than an ideal one.
    """
    from torchsim.sequence import FSE, exact_slice_profile

    flips = torch.deg2rad(torch.full((ECHOES,), 150.0))
    split, _ = _split_across_two_channels(flips)
    magnitude, phase = _sensitivities()
    tissue = _tissue_properties(magnitude, phase)

    centred = FSE().simulate(
        split, tissue, nstates=8
    ).signal
    across = FSE().simulate(
        split,
        tissue,
        nstates=8,
        slice_profile=exact_slice_profile(9, extent=2.0),
    ).signal

    assert centred.shape == across.shape
    assert float(centred.abs().max()) > 0.0
    # The pulse is a rectangle at zero bandwidth, so every position sees the
    # same rotation and the mean over them is the centre.
    worst = float((centred - across).abs().max() / centred.abs().max())
    assert worst < 1e-6, worst


def test_a_relaxation_gradient_reaches_through_the_pair():
    """The pair fixes the rotation and the kernels differentiate everything
    around it, so a relaxation gradient is the one the flip-and-phase route
    returns for the same field.
    """
    from torchsim.sequence import FSE, TissueProperties

    flips = torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    split, plain = _split_across_two_channels(flips)
    magnitude, phase = _sensitivities()
    combined = torch.polar(magnitude, phase).mean(dim=0)

    def gradients(description, b1, b1_phase):
        t1 = torch.linspace(700.0, 1300.0, VOXELS).requires_grad_(True)
        t2 = torch.linspace(50.0, 110.0, VOXELS).requires_grad_(True)
        signal = FSE().simulate(
            description,
            TissueProperties(t1_ms=t1, t2_ms=t2, b1=b1, b1_phase_rad=b1_phase),
            nstates=16,
        ).signal
        return torch.autograd.grad(signal.abs().square().sum(), (t1, t2))

    integrated = gradients(split, magnitude, phase)
    turned = gradients(plain, combined.abs(), combined.angle())

    for reference, result in zip(turned, integrated, strict=True):
        assert float(reference.abs().max()) > 0.0
        assert float((reference - result).abs().max() / reference.abs().max()) < 1e-5


def test_the_flip_gradient_comes_back_through_the_pulse_integral():
    """The rotation is integrated before the kernels run, so what they return
    for it is a cotangent on the pair. Carried the rest of the way by autograd,
    it has to be the gradient the flip-and-phase route gives for the field the
    channels sum to.
    """
    from torchsim.sequence import FSE, TissueProperties

    magnitude, phase = _sensitivities()
    combined = torch.polar(magnitude, phase).mean(dim=0)

    def gradient(splitting: bool):
        flips = torch.deg2rad(
            torch.linspace(100.0, 170.0, ECHOES)
        ).requires_grad_(True)
        split, plain = _split_across_two_channels(flips)
        tissue = (
            _tissue_properties(magnitude, phase)
            if splitting
            else TissueProperties(
                t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
                t2_ms=torch.linspace(50.0, 110.0, VOXELS),
                b1=combined.abs(),
                b1_phase_rad=combined.angle(),
            )
        )
        signal = FSE().simulate(
            split if splitting else plain, tissue, nstates=16
        ).signal
        return torch.autograd.grad(signal.abs().square().sum(), flips)[0]

    integrated = gradient(True)
    turned = gradient(False)

    assert float(turned.abs().max()) > 0.0
    assert float((integrated - turned).abs().max() / turned.abs().max()) < 1e-5


@pytest.mark.parametrize("name", ["b1", "b1_phase_rad"])
def test_a_transmit_gradient_comes_back_through_the_pulse_integral(name: str):
    """Against central differences, because there is no second route to the
    per-channel maps: the pair is the only thing that reads them.
    """
    from torchsim.sequence import FSE, TissueProperties

    magnitude, phase = _sensitivities()
    split, _ = _split_across_two_channels(
        torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    )

    def loss(b1, b1_phase):
        signal = FSE().simulate(
            split,
            TissueProperties(
                t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
                t2_ms=torch.linspace(50.0, 110.0, VOXELS),
                b1=b1,
                b1_phase_rad=b1_phase,
            ),
            nstates=16,
        ).signal
        return signal.abs().square().sum()

    leaves = {"b1": magnitude.clone(), "b1_phase_rad": phase.clone()}
    leaves[name] = leaves[name].requires_grad_(True)
    analytic = torch.autograd.grad(
        loss(leaves["b1"], leaves["b1_phase_rad"]), leaves[name]
    )[0]

    step = 2e-3
    assert float(analytic.abs().max()) > 0.0
    for cell in ((0, 1), (1, 2)):
        moved = {key: value.detach().clone() for key, value in leaves.items()}
        moved[name][cell] += step
        forward = float(loss(moved["b1"], moved["b1_phase_rad"]))
        moved[name][cell] -= 2.0 * step
        backward = float(loss(moved["b1"], moved["b1_phase_rad"]))
        measured = (forward - backward) / (2.0 * step)
        assert abs(measured - float(analytic[cell])) < 3e-3 * max(
            abs(measured), 1.0
        )


def test_a_forward_direction_follows_the_pair():
    """Forward mode sends a direction along the pair the same way reverse mode
    brings a cotangent back from it, so a directional derivative through the
    per-channel maps is the one differencing the simulation gives.
    """
    from torchsim.sequence import FSE, TissueProperties

    magnitude, phase = _sensitivities()
    split, _ = _split_across_two_channels(
        torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    )

    def loss(b1):
        signal = FSE().simulate(
            split,
            TissueProperties(
                t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
                t2_ms=torch.linspace(50.0, 110.0, VOXELS),
                b1=b1,
                b1_phase_rad=phase,
            ),
            nstates=12,
        ).signal
        return signal.abs().square().sum()

    generator = torch.Generator().manual_seed(7)
    direction = torch.randn(magnitude.shape, generator=generator) * 0.1
    _, followed = torch.func.jvp(loss, (magnitude,), (direction,))

    step = 1e-3
    measured = (
        float(loss(magnitude + step * direction))
        - float(loss(magnitude - step * direction))
    ) / (2.0 * step)

    assert abs(measured) > 0.0
    assert abs(float(followed) - measured) < 5e-3 * abs(measured)


def test_a_hessian_vector_product_reaches_through_the_pair():
    """The second derivative is the first one differenced, and the pair carries
    both halves: a direction along it going in, a curvature coming back.
    """
    from torchsim.sequence import FSE, TissueProperties

    magnitude, phase = _sensitivities()
    nominal = torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))

    def loss(flips):
        split, _ = _split_across_two_channels(flips)
        signal = FSE().simulate(
            split,
            TissueProperties(
                t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
                t2_ms=torch.linspace(50.0, 110.0, VOXELS),
                b1=magnitude,
                b1_phase_rad=phase,
            ),
            nstates=12,
        ).signal
        return signal.abs().square().sum()

    def gradient(flips):
        leaf = flips.clone().requires_grad_(True)
        return torch.autograd.grad(loss(leaf), leaf, create_graph=True)[0]

    generator = torch.Generator().manual_seed(11)
    direction = torch.randn(ECHOES, generator=generator) * 0.1
    leaf = nominal.clone().requires_grad_(True)
    (product,) = torch.autograd.grad(
        (gradient(leaf) * direction).sum(), leaf
    )

    step = 1e-3
    measured = (
        gradient(nominal + step * direction).detach()
        - gradient(nominal - step * direction).detach()
    ) / (2.0 * step)

    assert float(measured.abs().max()) > 0.0
    worst = float((product - measured).abs().max() / measured.abs().max())
    assert worst < 5e-3, worst


# --- a gradient the scanner moves while the pulse plays ---


def _under_gradient(waveform, *, samples: int = 64, bandwidth_hz: float = 2000.0):
    """The sinc of :func:`_pulse`, played under a gradient that is not held.

    The waveform is in units of the gradient ``bandwidth_hz`` is quoted at, so
    a shape of ones is the pulse itself.
    """
    from dataclasses import replace

    definition = _pulse(samples=samples, bandwidth_hz=bandwidth_hz)
    return replace(
        definition,
        gradient=RfShape(len(waveform), np.asarray(waveform, dtype=np.float32)),
    )


def test_a_gradient_of_ones_is_the_gradient_held():
    """The moving path has to land on the held one to the bit where the
    waveform says the gradient never moved, or the two are different physics
    wearing the same units.
    """
    samples = 64
    held = _pulse(samples=samples)
    moved = _under_gradient(np.ones(samples), samples=samples)
    sensitivities = torch.tensor([[0.9 + 0.1j]], dtype=torch.complex128)
    positions = torch.linspace(-0.5, 0.5, 5)
    arguments = dict(
        positions=positions,
        flip=1.2,
        off_resonance_hz=torch.tensor([40.0]),
        rf_raster_time_s=RASTER,
    )

    for reference, result in zip(
        dynamic_pair(held, sensitivities, **arguments),
        dynamic_pair(moved, sensitivities, **arguments),
        strict=True,
    ):
        assert torch.equal(reference, result)

    table = dict(bins=32, rf_raster_time_s=RASTER)
    tabulated = transition_table(held, positions, **table)
    integrated = transition_table(moved, positions, **table)
    assert torch.equal(tabulated.a, integrated.a)
    assert torch.equal(tabulated.b, integrated.b)
    assert torch.equal(tabulated.slope_a, integrated.slope_a)


def test_a_moving_gradient_is_what_exponentiating_each_sample_gives():
    """The oracle carries the gradient sample by sample too, so agreement is
    evidence about where the turn is applied and not only about its size.
    """
    samples = 48
    ramp = np.linspace(-1.0, 1.0, samples)
    definition = _under_gradient(ramp, samples=samples)
    sensitivities = torch.tensor([[0.8 + 0.2j]], dtype=torch.complex128)
    position = 0.4
    offset = 90.0

    a, b = dynamic_pair(
        definition,
        sensitivities,
        flip=1.3,
        off_resonance_hz=torch.tensor([offset]),
        positions=torch.tensor([position]),
        rf_raster_time_s=RASTER,
    )

    envelope = torch.as_tensor(definition.complex_envelope(), dtype=torch.complex128)
    shape = envelope / envelope.sum()
    drive = [1.3 * shape[sample] * sensitivities[0, 0] for sample in range(samples)]
    turns = [
        2.0 * torch.pi * RASTER * (offset + 2000.0 * position * float(ramp[sample]))
        for sample in range(samples)
    ]
    expected = _stepwise(drive, turns)
    measured = torch.tensor(
        [
            [complex(a[0]), complex(b[0])],
            [-complex(b[0]).conjugate(), complex(a[0]).conjugate()],
        ],
        dtype=torch.complex128,
    )

    assert float((expected - measured).abs().max()) < 1e-6


def test_a_gradient_switched_off_mid_pulse_moves_the_profile():
    """The whole point of the axis: where the gradient is off, the pulse
    excites without selecting, so the slice is not the one the held gradient
    would have cut.
    """
    samples = 128
    blipped = np.ones(samples)
    blipped[samples // 3 : 2 * samples // 3] = 0.0
    positions = torch.linspace(-1.0, 1.0, 9)
    sensitivities = torch.ones(1, 1, dtype=torch.complex128)
    # Played over ``4 / bandwidth_hz``, so the sinc's four lobes are the
    # bandwidth the definition declares and a position is a real thickness.
    arguments = dict(
        positions=positions,
        flip=torch.pi / 2.0,
        rf_raster_time_s=4.0 / (2000.0 * samples),
    )

    _, held = dynamic_pair(_pulse(samples=samples), sensitivities, **arguments)
    _, moved = dynamic_pair(
        _under_gradient(blipped, samples=samples), sensitivities, **arguments
    )

    assert float(held.abs().max()) > 0.1
    assert float((held - moved).abs().max()) > 0.05 * float(held.abs().max())


def test_a_gradient_that_is_never_on_selects_nothing():
    """No gradient across the pulse is no slice, however wide the bandwidth."""
    samples = 64
    definition = _under_gradient(np.zeros(samples), samples=samples)
    positions = torch.linspace(-2.0, 2.0, 5)

    a, b = dynamic_pair(
        definition,
        torch.ones(1, 1, dtype=torch.complex128),
        positions=positions,
        flip=1.0,
        rf_raster_time_s=RASTER,
    )

    assert torch.equal(a, a[:1].expand_as(a))
    assert torch.equal(b, b[:1].expand_as(b))


def test_a_gradient_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="the gradient carries"):
        _under_gradient(np.ones(9), samples=64).gradient_waveform()


def test_a_moving_gradient_has_to_last_the_pulse():
    """``compose_spinor`` takes the two waveforms in the same form, so a
    gradient that runs out before the RF does is a caller's mistake, not a
    shorter pulse.
    """
    from torchsim.sequence._transition import compose_spinor

    drive = [torch.tensor(0.1, dtype=torch.complex128) for _ in range(4)]
    turns = [torch.zeros(2, dtype=torch.float64) for _ in range(3)]

    with pytest.raises(ValueError):
        compose_spinor(drive, turns)


def test_the_table_reads_the_moving_gradient_too():
    """The tabulated route composes the same spinor, so it has to carry the
    gradient sample by sample where one is given -- shown by a waveform that
    changes the answer rather than by one that cannot.
    """
    samples = 128
    blipped = np.ones(samples)
    blipped[samples // 3 : 2 * samples // 3] = 0.0
    positions = torch.linspace(-1.0, 1.0, 9)
    arguments = dict(bins=64, rf_raster_time_s=4.0 / (2000.0 * samples))

    held = transition_table(_pulse(samples=samples), positions, **arguments)
    moved = transition_table(
        _under_gradient(blipped, samples=samples), positions, **arguments
    )

    assert float(held.b.abs().max()) > 0.1
    assert float((held.b - moved.b).abs().max()) > 0.05 * float(held.b.abs().max())


# --- the routes that cannot cut a per-voxel rotation say so ---


class _Elsewhere:
    """Stands in for a streaming plan, so the refusal can be reached without a
    second device to stream to. Nothing reads it: the guard runs first.
    """

    devices = ()
    budget_bytes = 1 << 20
    lanes = 1


def _streamed(monkeypatch):
    from torchsim.sequence import _accelerators

    monkeypatch.setattr(_accelerators, "_OFFLOAD", _Elsewhere())


@pytest.mark.parametrize("pass_name", ["forward", "forward-mode", "adjoint"])
def test_the_streamed_route_refuses_a_per_voxel_pair(monkeypatch, pass_name):
    """A table is the same for every voxel so a chunked run can broadcast one.
    The pair is per voxel, so a route that moves the volume a piece at a time
    would have to cut it the same way -- and is told to say so rather than
    quietly reach for a hard pulse.
    """
    from torchsim.sequence._accelerators import (
        _run_packed,
        _run_packed_jvp,
        _run_packed_vjp,
    )

    _, prepared, events, pairs = _train()
    _streamed(monkeypatch)
    still = tuple(torch.zeros_like(value) for value in prepared)
    quiet = tuple(torch.zeros_like(events[index]) for index in (0, 2, 3))
    seed = torch.ones(VOXELS, ECHOES, dtype=torch.complex64)
    arguments = dict(state_count=16, output_count=ECHOES, dynamic=pairs)

    with pytest.raises(NotImplementedError, match="streamed route does not cut"):
        if pass_name == "forward":
            _run_packed(prepared, events, 16, ECHOES, 1, dynamic=pairs)
        elif pass_name == "forward-mode":
            _run_packed_jvp(
                prepared, events, still, quiet, threads=1, **arguments
            )
        else:
            _run_packed_vjp(prepared, events, seed, threads=1, **arguments)


def test_the_streamed_route_still_takes_a_train_with_no_pair(monkeypatch):
    """The guard has to be about the pair and not about the route, or a plain
    sequence would lose streaming with it.
    """
    from torchsim.sequence._accelerators import _carries_the_pair

    assert _carries_the_pair(None, "streamed") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_the_sharded_route_refuses_a_per_voxel_pair():
    """Shards cut the trains rather than the voxels, but each device gets its
    own launch and the pair would have to travel with it.
    """
    from torchsim.sequence._accelerators import _run_packed, distribute
    from torchsim.sequence._transition import DynamicPairs

    prepared, events, pairs, echoes = _real_subspace_train(trains=4)
    prepared = tuple(value.cuda() for value in prepared)
    events = tuple(value.cuda() for value in events)
    moved = DynamicPairs(
        a=pairs.a.cuda(), b=pairs.b.cuda(), index=pairs.index.cuda()
    )

    with distribute(["cuda", "cuda"]):
        with pytest.raises(NotImplementedError, match="sharded route does not cut"):
            _run_packed(prepared, events, 16, echoes, 1, dynamic=moved)


# --- the rotation mode against every pool the kernels carry ---


def _pooled_train(**properties):
    """The dynamic train of :func:`_train`, on a tissue carrying a second pool.

    The kernels are templated on the rotation mode and the pool count together,
    so a pair beside a pool is an instantiation of its own -- and one nothing
    reached until now.
    """
    from torchsim.sequence._accelerators import _pack_events
    from torchsim.sequence._builders import fse_description
    from torchsim.sequence._simulation import TissueProperties, _prepare_tissue
    from torchsim.sequence._transition import DynamicPairs

    definition = _pulse(samples=96)
    flips = torch.deg2rad(torch.linspace(100.0, 170.0, ECHOES))
    packed = _pack_events(
                fse_description(
            flips,
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
            excitation_phase_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=RASTER,
    )
    events = (
        packed.duration, packed.kind, packed.flip, packed.phase, packed.action,
        packed.output_index, packed.shim_index, packed.saturation,
        packed.rf_frequency_hz,
    )
    scaling = torch.linspace(0.7, 1.3, VOXELS, dtype=torch.float64)
    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(700.0, 1300.0, VOXELS),
            t2_ms=torch.linspace(50.0, 110.0, VOXELS),
            b1=scaling.to(torch.float32),
            **properties,
        ),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)

    count = int(packed.kind.numel())
    flat = packed.flip.reshape(-1, count)
    halves = [
        dynamic_pair(
            definition,
            scaling.to(torch.complex128)[:, None],
            weights=torch.ones(1, dtype=torch.complex128),
            flip=float(flat[0, event]),
            rf_raster_time_s=RASTER,
        )
        for event in range(count)
    ]
    pairs = DynamicPairs(
        a=torch.stack([half[0] for half in halves]),
        b=torch.stack([half[1] for half in halves]),
        index=torch.arange(count, dtype=torch.int32),
    )
    return prepared, events, pairs, int(packed.output_count)


POOLS = {
    "semisolid": dict(bound_fraction=0.1, bound_exchange_hz=30.0, t1_bound_ms=1000.0),
    "exchanging": dict(
        pool_b_fraction=0.15, pool_b_exchange_hz=20.0, t1_pool_b_ms=400.0,
        t2_pool_b_ms=20.0, pool_b_shift_hz=420.0,
    ),
    "three": dict(
        bound_fraction=0.1, bound_exchange_hz=30.0, t1_bound_ms=1000.0,
        pool_b_fraction=0.15, pool_b_exchange_hz=20.0, t1_pool_b_ms=400.0,
        t2_pool_b_ms=20.0, pool_b_shift_hz=420.0,
    ),
}


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_a_pair_reaches_every_pool_the_kernels_carry(pool: str):
    """One instantiation of the kernel template per (rotation, pools), against
    the oracle reading the same buffers.
    """
    from torchsim.sequence._accelerators import _run_packed
    from torchsim.sequence._lineshape import lineshape_table

    prepared, events, pairs, echoes = _pooled_train(**POOLS[pool])
    carried = dict(
        lineshape=lineshape_table() if pool in ("semisolid", "three") else None,
        exchanging=pool in ("exchanging", "three"),
    )

    expected = simulate_packed(
        prepared, events, state_count=STATES, output_count=echoes,
        dynamic=pairs, **carried,
    )
    measured = _run_packed(
        prepared, events, STATES, echoes, 1, dynamic=pairs, **carried
    )

    assert float(expected.abs().max()) > 0.0
    worst = float((expected - measured).abs().max() / expected.abs().max())
    assert worst < 1e-5, worst


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_a_pool_moves_the_answer_a_pair_gives(pool: str):
    """The agreement above is only worth having if the pool is doing something,
    so the same train without one must record a different signal.
    """
    from torchsim.sequence._accelerators import _run_packed
    from torchsim.sequence._lineshape import lineshape_table

    prepared, events, pairs, echoes = _pooled_train(**POOLS[pool])
    bare, _, _, _ = _pooled_train()

    pooled = _run_packed(
        prepared, events, STATES, echoes, 1, dynamic=pairs,
        lineshape=lineshape_table() if pool in ("semisolid", "three") else None,
        exchanging=pool in ("exchanging", "three"),
    )
    alone = _run_packed(bare, events, STATES, echoes, 1, dynamic=pairs)

    assert float(alone.abs().max()) > 0.0
    assert float((pooled - alone).abs().max()) > 1e-3 * float(alone.abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("state_count", [8, 12, 17])
def test_a_pair_takes_the_first_order_kernel_on_the_card(state_count):
    """A per-voxel rotation does not cost the kernel written for a gradient.

    Widths are swept because a reverse kernel has miscompiled silently at one
    state count before, and the pair adds a per-event load and an atomic the
    others do not make.
    """
    from torchsim.sequence import _accelerators
    from torchsim.sequence._accelerators import _run_packed_vjp
    from torchsim.sequence._transition import DynamicPairs

    _, prepared, events, pairs = _train()
    prepared = tuple(value.cuda() for value in prepared)
    events = tuple(value.cuda() for value in events)
    moved = DynamicPairs(
        a=pairs.a.cuda(), b=pairs.b.cuda(), index=pairs.index.cuda()
    )
    seed = torch.ones(
        prepared[0].numel(), ECHOES, dtype=torch.complex64, device="cuda"
    )
    still = tuple(
        torch.zeros_like(value)
        for value in (*prepared, events[0], events[2], events[3])
    )
    arguments = dict(
        state_count=state_count, output_count=ECHOES, threads=1, dynamic=moved
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
    # The cotangent on the rotation itself comes back last, and is what
    # carries the gradient to whatever integrated the pair.
    assert fast[-1].shape == moved.packed().shape
    for reference, result in zip(expected, fast, strict=True):
        assert reference.shape == result.shape
        assert float((reference - result).abs().max()) / largest < 1e-5
