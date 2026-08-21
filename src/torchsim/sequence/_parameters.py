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
    "features_of",
    "wants_bound_pool",
    "wants_exchange_pool",
    "tissue_gradient_bases",
    "tissue_gradient_rows",
    "tissue_gradient_height",
]

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.forward_ad import unpack_dual


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
    gate
        Whether this parameter is the one whose value decides that term. Every
        feature has exactly one. A term with several parameters is switched by
        one of them and described by the rest: a bound pool's exchange rate and
        relaxation time say what the pool does, but its fraction says whether
        there is a pool at all.
    transmit
        Whether the buffer carries one row of voxels per shim rather than a
        single row. Only the transmit field does: a pulse reads the shim it
        drives, so both the buffer and its gradient need a row for each.
    """

    name: str
    differentiable: bool = True
    identity: float | None = None
    feature: str | None = None
    gate: bool = False
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
    Parameter("t1_ms", feature="T1", gate=True),
    Parameter("t2_ms", feature="T2", gate=True),
    Parameter("m0", identity=1.0, feature="M0", gate=True),
    Parameter("b1", identity=1.0, feature="B1", gate=True, transmit=True),
    Parameter(
        "b1_phase_rad", identity=0.0, feature="B1_PHASE", gate=True, transmit=True
    ),
    Parameter("b0_hz", identity=0.0, feature="B0", gate=True),
    Parameter("inversion_efficiency", identity=1.0, feature="INVERSION", gate=True),
    Parameter("diffusion_um2_per_ms", identity=0.0, feature="DIFFUSION", gate=True),
    Parameter("velocity_m_per_s", identity=0.0, feature="FLOW", gate=True),
    # The semisolid pool, which magnetization transfer drives. Its fraction is
    # the gate: at zero the exchange matrix is diagonal and the pool starts
    # empty, so nothing it drives can reach the free water, and the kernels
    # leave the whole second pool out.
    Parameter("bound_fraction", identity=0.0, feature="MT", gate=True),
    Parameter("bound_exchange_hz", identity=0.0, feature="MT"),
    Parameter("t1_bound_ms", feature="MT"),
    # A second pool that exchanges chemically rather than by saturation. It
    # carries transverse magnetization, so it has a T2 the semisolid pool does
    # not, and it sits at its own offset from the free water. Its fraction is
    # the gate on the same terms.
    Parameter("pool_b_fraction", identity=0.0, feature="BM", gate=True),
    Parameter("pool_b_exchange_hz", identity=0.0, feature="BM"),
    Parameter("t1_pool_b_ms", feature="BM"),
    Parameter("t2_pool_b_ms", feature="BM"),
    Parameter("pool_b_shift_hz", identity=0.0, feature="BM"),
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
    # What a pulse deposits in the bound pool, per unit of flip angle squared:
    # ``-pi gamma**2 b1rms_1rad**2 tau``. The flip itself is differentiable and
    # reaches the saturation through the square the kernels take of it, so this
    # buffer holds only the shape's power and carries no gradient of its own.
    Parameter("saturation", differentiable=False),
    # Where the pulse is played, in Hz off the scanner's centre frequency. The
    # lineshape is read at this less the voxel's own off-resonance.
    Parameter("rf_frequency_hz", differentiable=False),
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

# Which tissue buffer gates each second pool. Read before broadcasting, so a
# sequence that never mentions one decides on a Python float rather than by
# reducing over a buffer.
BOUND_FRACTION_INPUT: int = TISSUE_NAMES.index("bound_fraction")
POOL_B_FRACTION_INPUT: int = TISSUE_NAMES.index("pool_b_fraction")

# The second pools' properties, as positions among the packed buffers. They
# sit past everything the free pool alone accounts for, which is what lets a
# single-pool kernel index the ones it takes without knowing they are there.
BOUND_POOL_INPUTS: tuple[int, ...] = tuple(
    index
    for index, parameter in enumerate(TISSUE_PARAMETERS)
    if parameter.feature == "MT"
)
EXCHANGE_POOL_INPUTS: tuple[int, ...] = tuple(
    index
    for index, parameter in enumerate(TISSUE_PARAMETERS)
    if parameter.feature == "BM"
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

    A value carrying a derivative matters whatever it holds, in either mode:
    a gradient to be accumulated into, or a forward direction to be followed.
    Leaving out a term leaves both at zero, which is the truth for a property
    the run does not take and a lie for one it differentiates: the derivative
    at an identity is a number like any other, and a B1 map fitted from unity
    or a diffusion coefficient fitted from zero both start there.
    """
    if parameter.identity is None:
        return False
    if getattr(value, "requires_grad", False):
        return False
    if isinstance(value, torch.Tensor) and unpack_dual(value).tangent is not None:
        return False
    if isinstance(value, (int, float)):
        return float(value) == parameter.identity
    numel = getattr(value, "numel", None)
    if numel is not None and numel() == 1:
        return float(value.item()) == parameter.identity
    return False


def features_of(tissue: Any) -> frozenset[str]:
    """Which terms of the state machine this tissue gives anything to do.

    Each feature is decided by its gate alone, on the terms
    :func:`at_identity` sets: from what the caller passed, before broadcasting,
    without touching a buffer. The parameters that describe a term rather than
    switch it are not consulted, so a bound pool's exchange rate left at some
    non-zero value does not conjure a pool the fraction says is not there.

    ``t1_ms`` and ``t2_ms`` have no identity and so are always present, which
    is what makes relaxation the floor rather than a feature.
    """
    return frozenset(
        parameter.feature
        for parameter in TISSUE_PARAMETERS
        if parameter.gate
        and not at_identity(parameter, getattr(tissue, parameter.name))
    )


def feature_flags(features: Any, geometry: Geometry) -> dict[str, bool]:
    """Which optional terms a launch is to carry.

    ``features`` is the set :func:`torchsim.sequence._parameters.features_of`
    reads off the tissue; ``None`` is a caller who did not declare, and every
    term stays.

    Fewer switches than properties, because each Triton flag multiplies how
    many kernels the cache holds and these groups are what the arithmetic
    actually splits into. ``off_axis`` is the static phase a tissue puts on the
    states -- off-resonance and transmit phase reach the interval and the pulse
    through the same turn. ``moving`` is what a voxel's velocity drives, which
    it does only through the sequence geometry: flow winding and washout are
    two readings of one property through two scales, so a sequence that winds
    no phase and draws in no fresh spins drops both however fast the voxel
    moves. ``diffusing`` stands alone: an attenuation per dephasing order is a
    factor where the other two are phases, and it reaches the second pools that
    have no coefficient of their own.

    Kept here rather than beside either backend's launcher, because both read
    it and a launcher must not be able to describe the tissue one way while the
    kernel reads it another.
    """
    undeclared = features is None
    return {
        "off_axis": undeclared or bool({"B0", "B1_PHASE"} & features),
        "moving": (undeclared or "FLOW" in features)
        and (geometry.flow_scale != 0.0 or geometry.washout_scale != 0.0),
        "diffusing": undeclared or "DIFFUSION" in features,
    }


# The order is the C++ ABI: ``feature_mask`` packs these bits and
# ``_epg_cpu.cpp`` unpacks them by the same names, so the two move together.
FEATURE_BITS: tuple[str, ...] = ("off_axis", "moving", "diffusing")


def feature_mask(features: Any, geometry: Geometry) -> int:
    """The same answer as :func:`feature_flags`, as the host kernels read it.

    Triton takes a flag per term because each one compiles a kernel of its own;
    the host kernels take one integer and branch on it at run time, which is
    the same choice the pool count already makes on each side.
    """
    flags = feature_flags(features, geometry)
    return sum(
        1 << bit for bit, name in enumerate(FEATURE_BITS) if flags[name]
    )


def wants_bound_pool(bound_fraction: Any) -> bool:
    """Whether this bound fraction gives the semisolid pool anything to do.

    Read from what the caller passed, before broadcasting, on the same terms
    as :func:`at_identity`: a fraction given as a full tensor is taken to
    matter rather than reduced over.
    """
    return not at_identity(TISSUE_PARAMETERS[BOUND_FRACTION_INPUT], bound_fraction)


def wants_exchange_pool(pool_b_fraction: Any) -> bool:
    """Whether this fraction gives the chemically exchanging pool anything to do."""
    return not at_identity(
        TISSUE_PARAMETERS[POOL_B_FRACTION_INPUT], pool_b_fraction
    )


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
