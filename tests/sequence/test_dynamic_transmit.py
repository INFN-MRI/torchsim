"""The rotation a dynamically shimmed pulse performs, per voxel.

A static array reduces to a flip angle and a phase; weights that vary while the
pulse plays do not. These pin the generalization against the model it extends,
against an independent integration, and against autograd.
"""

from __future__ import annotations

import math

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
    for sample in drive:
        field = (
            complex(sample).real * pauli_x
            + complex(sample).imag * pauli_y
            + turn_z * pauli_z
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
        torch.ones(1, dtype=torch.complex128),
        scaling.to(torch.complex128)[:, None],
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
        torch.ones(1, dtype=torch.complex128),
        torch.ones(1, 1, dtype=torch.complex128),
        flip=1.1,
        rf_raster_time_s=RASTER,
    )
    turned = dynamic_pair(
        definition,
        torch.full((1,), complex(math.cos(phi), math.sin(phi)), dtype=torch.complex128),
        torch.ones(1, 1, dtype=torch.complex128),
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
        definition, weights, sensitivities, flip=torch.pi, rf_raster_time_s=RASTER
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
        weights,
        sensitivities,
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
        weights,
        sensitivities,
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
            overrides.get("weights", leaves["weights"]),
            overrides.get("sensitivities", leaves["sensitivities"]),
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


def test_the_reference_reads_a_dynamic_pair_where_it_reads_a_table():
    """A pulse whose weights hold still is a pulse the exact slice profile
    already describes, so the two routes through the state machine have to
    record the same train.

    The pair is integrated at each pulse's own flip, so a row per pulse is what
    the table's flip axis stands in for.
    """
    from torchsim.sequence._accelerators import _pack_events, _shim_count
    from torchsim.sequence._builders import fse_description
    from torchsim.sequence._simulation import TissueProperties, _prepare_tissue
    from torchsim.sequence._transition import DynamicPairs, transition_table

    echoes = 6
    voxels = 5
    definition = _pulse(samples=96)
    flips = torch.deg2rad(torch.linspace(100.0, 170.0, echoes, dtype=torch.float64))
    description = fse_description(
        flips.to(torch.float32),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        excitation_phase_rad=torch.pi / 2,
    )
    packed = _pack_events(
        "fse",
        description,
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
    scaling = torch.linspace(0.7, 1.3, voxels, dtype=torch.float64)
    prepared, _, _ = _prepare_tissue(
        TissueProperties(
            t1_ms=torch.linspace(700.0, 1300.0, voxels),
            t2_ms=torch.linspace(50.0, 110.0, voxels),
            b1=scaling.to(torch.float32),
        ),
        "cpu",
    )
    prepared = tuple(value.to(torch.float32).contiguous() for value in prepared)
    assert _shim_count(prepared) == 1

    table = transition_table(
        definition,
        torch.zeros(1, dtype=torch.float64),
        bins=1024,
        theta_max=4.0,
        rf_raster_time_s=RASTER,
    )

    # One row per event, holding whatever rotation that event's own flip drives.
    rows_a = []
    rows_b = []
    for event in range(int(packed.kind.numel())):
        flip = float(packed.flip.reshape(-1, int(packed.kind.numel()))[0, event])
        pair = dynamic_pair(
            definition,
            torch.ones(1, dtype=torch.complex128),
            scaling.to(torch.complex128)[:, None],
            flip=flip,
            rf_raster_time_s=RASTER,
        )
        rows_a.append(pair[0])
        rows_b.append(pair[1])
    pairs = DynamicPairs(
        a=torch.stack(rows_a),
        b=torch.stack(rows_b),
        index=torch.arange(int(packed.kind.numel()), dtype=torch.int32),
    )

    options = dict(state_count=16, output_count=echoes)
    tabulated = simulate_packed(prepared, events, profile=table, **options)
    integrated = simulate_packed(prepared, events, dynamic=pairs, **options)

    assert float(tabulated.abs().max()) > 0.0
    worst = float((tabulated - integrated).abs().max() / tabulated.abs().max())
    assert worst < 1e-4, worst
