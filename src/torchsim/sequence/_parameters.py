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

__all__ = ["Parameter", "EVENT_PARAMETERS", "TISSUE_PARAMETERS", "at_identity"]

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
    """

    name: str
    differentiable: bool = True
    identity: float | None = None
    feature: str | None = None


# Order is the ABI. Anything appended here is appended to the pointer arrays
# the kernels index, so the two move together.
TISSUE_PARAMETERS: tuple[Parameter, ...] = (
    Parameter("t1_ms", feature="T1"),
    Parameter("t2_ms", feature="T2"),
    Parameter("m0", identity=1.0, feature="M0"),
    Parameter("b1", identity=1.0, feature="B1"),
    Parameter("b1_phase_rad", identity=0.0, feature="B1_PHASE"),
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
)

PACKED_PARAMETERS: tuple[Parameter, ...] = (*TISSUE_PARAMETERS, *EVENT_PARAMETERS)

TISSUE_COUNT = len(TISSUE_PARAMETERS)
PACKED_COUNT = len(PACKED_PARAMETERS)

# Where the differentiable buffers sit among the packed ones. Gradient tuples
# are ordered by this throughout: seven tissue properties, then event duration,
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
FLOAT_NAMES: tuple[str, ...] = tuple(
    PACKED_PARAMETERS[index].name for index in FLOAT_INPUTS
)

# Where the differentiable-input order carries the three gradients a real
# adjoint leaves at zero: transmit phase, off-resonance and RF phase all point
# out of the subspace.
OUTSIDE_THE_SUBSPACE: tuple[int, ...] = tuple(
    FLOAT_NAMES.index(name) for name in ("b1_phase_rad", "b0_hz", "phase")
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
