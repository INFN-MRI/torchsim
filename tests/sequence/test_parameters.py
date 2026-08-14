"""The parameter table has to describe the buffers the kernels actually take.

Its whole purpose is that a count appears once rather than in the Python
dispatch, the CPU extension and the Triton kernels separately. That only holds
while the table and the things it describes agree, which is what these check --
a new parameter added to the table but not to the dataclass, or to the kernels'
pointer list, fails here rather than by reading past the end of a buffer.
"""

import dataclasses

import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence import _accelerators
from torchsim.sequence._accelerators import _pack_events
from torchsim.sequence._parameters import (
    EVENT_PARAMETERS,
    FLOAT_INPUTS,
    PACKED_COUNT,
    SEED_INPUT,
    TISSUE_COUNT,
    TISSUE_NAMES,
    TISSUE_PARAMETERS,
)
from torchsim.sequence._simulation import _prepare_tissue


def test_the_table_names_the_tissue_dataclass_fields():
    """``_prepare_tissue`` reads the table, so a typo would silently drop one."""
    fields = tuple(field.name for field in dataclasses.fields(TissueProperties))

    assert TISSUE_NAMES == fields


def test_every_tissue_parameter_is_prepared():
    """A short tuple here would leave a kernel reading an unset pointer."""
    prepared, _, _ = _prepare_tissue(
        TissueProperties(t1_ms=torch.tensor([800.0]), t2_ms=torch.tensor([45.0])),
        "cpu",
    )

    assert len(prepared) == TISSUE_COUNT


def test_every_event_parameter_is_packed():
    """The packed events carry a field per event parameter the table names."""
    packed = _pack_events(
        "fse",
        fse_description(
            torch.deg2rad(torch.full((4,), 140.0)),
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )

    for parameter in EVENT_PARAMETERS:
        assert hasattr(packed, parameter.name), parameter.name


def test_the_seed_follows_the_packed_buffers():
    """A differentiable adjoint takes the packed inputs and then its seed."""
    assert SEED_INPUT == PACKED_COUNT
    assert PACKED_COUNT == TISSUE_COUNT + len(EVENT_PARAMETERS)


def test_only_the_integer_buffers_are_undifferentiable():
    """Gradient tuples are ordered by this, so the split has to be right."""
    undifferentiated = {
        parameter.name
        for parameter in EVENT_PARAMETERS
        if not parameter.differentiable
    }

    assert undifferentiated == {"kind", "action", "output_index"}
    assert all(parameter.differentiable for parameter in TISSUE_PARAMETERS)


def test_the_gradient_order_matches_what_the_adjoint_returns():
    """Every tissue property, then the float event buffers, skipping ``kind``."""
    tissue = tuple(range(TISSUE_COUNT))
    events = tuple(
        TISSUE_COUNT + offset
        for offset, parameter in enumerate(EVENT_PARAMETERS)
        if parameter.differentiable
    )

    assert FLOAT_INPUTS == (*tissue, *events)


def test_autograd_asks_for_exactly_the_differentiable_inputs():
    """``needs_input_grad`` is indexed by the table, so its width must match."""
    widths = []
    original = _accelerators._wanted

    def record(needs_input_grad):
        widths.append(len(needs_input_grad))
        return original(needs_input_grad)

    _accelerators._wanted = record
    try:
        t2 = torch.tensor([45.0, 120.0], requires_grad=True)
        signal = FSE().simulate(
            fse_description(
                torch.deg2rad(torch.full((4,), 140.0)),
                echo_spacing_s=5e-3,
                phases_rad=torch.pi / 2,
            ),
            TissueProperties(t1_ms=torch.tensor([800.0, 1400.0]), t2_ms=t2),
            nstates=8,
        ).signal
        torch.autograd.grad(signal.abs().square().sum(), t2)
    finally:
        _accelerators._wanted = original

    # The four trailing arguments -- state count, output count, thread count
    # and the sequence geometry -- follow the packed buffers.
    assert widths and all(width == PACKED_COUNT + 4 for width in widths)
