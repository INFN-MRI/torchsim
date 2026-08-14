"""The state machine's parameters, as data.

The kernels take a flat list of buffers: the tissue properties, then the packed
per-event buffers. Their order is an ABI shared by the Python dispatch, the
CPU extension and the Triton kernels, and the counts derived from it -- how
many buffers are saved for backward, which of them carry gradients, how wide
the raw pointer array is -- appear in all three. This module is where that
order is written down, so those counts are read from one place instead of
being repeated as literals.

``identity`` is the value at which a parameter has no effect on the answer.
It is what lets a run decide which terms the kernel actually has to execute.
"""

from __future__ import annotations

__all__ = [
    "Parameter",
    "Geometry",
    "EVENT_PARAMETERS",
    "TISSUE_PARAMETERS",
    "at_identity",
    "tissue_gradient_bases",
    "tissue_gradient_rows",
    "tissue_gradient_height",
]

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Parameter:
    """One buffer in the kernels' input list.

    Parameters
    ----------
    name
        Attribute on :class:`TissueProperties` for a tissue parameter, or the
        field of the packed events for an event parameter.
    differentiable
        Whether a gradient is defined for it. Integer buffers are not.
    identity
        The value at which the parameter has no effect, or ``None`` when it
        always does.
    feature
        The term of the state machine it switches on.
    transmit
        Whether the buffer carries one row of voxels per shim rather than a
        single row. Only the transmit field does: a pulse reads the shim it
        drives, so both the buffer and its gradient need a row for each.
    """

    name: str
    differentiable: bool = True
    identity: float | None = None
    feature: str | None = None
    transmit: bool = False


@dataclass(frozen=True)
class Geometry:
    """The sequence geometry a velocity has to be read through.

    A spin velocity drives two unrelated terms. Flow dephasing turns each
    order through ``k0 * v``, so it needs the winding the unbalanced gradient
    puts across a metre. Washout replaces the voxel's spins at ``|v| / L``,
    which happens whether or not a gradient is playing and so needs the voxel
    size on its own. Neither can be folded into the other, and the absolute
    value cannot be folded into the buffer at all without losing the sign flow
    dephasing needs, so both scales travel to the kernels as they are.

    Both zero is a sequence that declares no geometry, which leaves every
    velocity-driven term out.
    """

    flow_scale: float = 0.0
    washout_scale: float = 0.0


NO_GEOMETRY = Geometry()


# Order is the ABI. Anything appended here is appended to the pointer arrays
# the kernels index, so the two move together.
TISSUE_PARAMETERS: tuple[Parameter, ...] = (
    Parameter("t1_ms", feature="T1"),
    Parameter("t2_ms", feature="T2"),
    Parameter("m0", identity=1.0, feature="M0"),
    Parameter("b1", identity=1.0, feature="B1", transmit=True),
    Parameter("b1_phase_rad", identity=0.0, feature="B1_PHASE", transmit=True),
    Parameter("b0_hz", identity=0.0, feature="B0"),
    Parameter("inversion_efficiency", identity=1.0, feature="INVERSION"),
    Parameter("diffusion_um2_per_ms", identity=0.0, feature="DIFFUSION"),
    Parameter("velocity_m_per_s", identity=0.0, feature="FLOW"),
)

EVENT_PARAMETERS: tuple[Parameter, ...] = (
    Parameter("duration"),
    Parameter("kind", differentiable=False),
    Parameter("flip"),
    Parameter("phase"),
    Parameter("action", differentiable=False),
    Parameter("output_index", differentiable=False),
    # Which row of the transmit buffers this pulse drives. The shim itself is
    # differentiable, but through the field it produces rather than through the
    # row it is stored in, so the index carries no gradient.
    Parameter("shim_index", differentiable=False),
)

PACKED_PARAMETERS: tuple[Parameter, ...] = (*TISSUE_PARAMETERS, *EVENT_PARAMETERS)

TISSUE_COUNT = len(TISSUE_PARAMETERS)
EVENT_COUNT = len(EVENT_PARAMETERS)
PACKED_COUNT = len(PACKED_PARAMETERS)

# Where the differentiable buffers sit among the packed ones. Gradient tuples
# are ordered by this throughout: every tissue property, then event duration,
# flip and phase.
FLOAT_INPUTS: tuple[int, ...] = tuple(
    index
    for index, parameter in enumerate(PACKED_PARAMETERS)
    if parameter.differentiable
)
FLOAT_COUNT = len(FLOAT_INPUTS)

# The seed of a first-order adjoint follows the packed buffers, so a
# differentiable adjoint sees it at this position.
SEED_INPUT = PACKED_COUNT

TISSUE_NAMES: tuple[str, ...] = tuple(
    parameter.name for parameter in TISSUE_PARAMETERS
)

# Which tissue buffers hold a row per shim.
TRANSMIT_INPUTS: tuple[int, ...] = tuple(
    index for index, parameter in enumerate(TISSUE_PARAMETERS) if parameter.transmit
)


FLOAT_NAMES: tuple[str, ...] = tuple(
    PACKED_PARAMETERS[index].name for index in FLOAT_INPUTS
)

# Where the differentiable-input order carries the gradients a real adjoint
# leaves at zero. Transmit phase, off-resonance and RF phase all point out of
# the subspace, and so does velocity at any value the sequence gives it a
# gradient to wind across: flow turns each dephasing order through a phase, so
# the derivative is imaginary even where the states themselves are still real.
OUTSIDE_THE_SUBSPACE: tuple[int, ...] = tuple(
    FLOAT_NAMES.index(name)
    for name in ("b1_phase_rad", "b0_hz", "velocity_m_per_s", "phase")
)


def at_identity(parameter: Parameter, value: Any) -> bool:
    """Whether this value leaves the parameter's term with nothing to do.

    Read from what the caller passed, before broadcasting, so the common case
    -- a property left at its default -- is answered without touching a buffer.
    A property given as a full tensor is reported as mattering rather than
    reduced over: on a device that reduction costs a synchronization, and at
    the sizes where the answer would pay for itself it is the rarer case.
    """
    if parameter.identity is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) == parameter.identity
    numel = getattr(value, "numel", None)
    if numel is not None and numel() == 1:
        return float(value.item()) == parameter.identity
    return False


def tissue_gradient_rows(shims: int) -> tuple[int, ...]:
    """Rows of one voxel each that every tissue parameter's gradient takes.

    A pulse reaches only the shim it drives, so the transmit pair takes a row
    per shim; every other property belongs to the voxel alone and takes one.
    """
    return tuple(
        shims if parameter.transmit else 1 for parameter in TISSUE_PARAMETERS
    )


def tissue_gradient_bases(shims: int) -> tuple[int, ...]:
    """Where each tissue parameter's gradient starts, in those rows.

    At a single shim each base is its own parameter index, which is the flat
    plane a sequence without a transmit array uses.
    """
    bases = []
    height = 0
    for rows in tissue_gradient_rows(shims):
        bases.append(height)
        height += rows
    return tuple(bases)


def tissue_gradient_height(shims: int) -> int:
    """How many rows of one voxel each the whole tissue gradient plane takes."""
    return TISSUE_COUNT + (shims - 1) * len(TRANSMIT_INPUTS)
