"""Keeping a parameter inside its range by changing what is solved for.

A relaxation time is positive, a fraction lies in the unit interval, an
inversion efficiency cannot exceed one. None of those are preferences that
should trade against the residual -- they are what the parameter *is*, and a
solver that returns something outside them has not returned a slightly worse
answer, it has returned no answer at all.

The way to impose that is to solve for a variable the bound is a function of,
so the iterate lives in the unconstrained space and the parameter it stands
for is inside the interval by construction. A two-sided bound is a scaled
sigmoid, a one-sided bound a softplus. The chain rule factor -- the derivative
of the parameter with respect to its free variable, which :func:`widen`
returns -- multiplies the Jacobian, so the solver never sees the natural
parameter and never has to be told about the bound again.

This matters more under an encoding operator than in a voxel-wise fit. A fit
that wanders to a negative relaxation time spoils that voxel; a model-based
reconstruction evaluates the model at every voxel to predict every k-space
sample, so one voxel out of range corrupts the whole residual.

The second thing the transform does is scale. A two-sided bound maps the whole
interval onto a variable of order one, whichever units the parameter is in, so
a millisecond relaxation time and a dimensionless fraction take steps of
comparable size. That is the preconditioning a Gauss-Newton step on a mixed
parameter set needs, and it comes from the bound rather than from a table of
weights someone has to keep.

**Equality constraints are not here, and cannot be.** A constraint that fixes
one parameter in terms of the others removes a degree of freedom, so the way
to impose it is to not have that freedom: write the model on the parameters
that remain. For a fat-water fit whose two fractions must sum to one, make the
fat fraction the only unknown and write water as ``1 - f`` inside the model.
The constraint then holds identically at every iterate rather than being
restored after each one.
"""

from __future__ import annotations

__all__ = ["bound_of", "to_free", "to_natural", "widen"]

from collections.abc import Mapping, Sequence
from typing import Any

import torch

#: Keeps the transform's argument finite at the very edge of an interval.
EDGE = 1e-6


def bound_of(
    bounds: Mapping[str, Any], name: str
) -> tuple[float | None, float | None]:
    """The pair for one property, absent meaning unbounded either way.

    Parameters
    ----------
    bounds:
        ``{name: (low, high)}``, either end ``None`` for unbounded.
    name:
        The property to look up.

    Returns
    -------
    tuple
        ``(low, high)``, either ``None``.

    Raises
    ------
    ValueError
        If the pair is not increasing.
    """
    low, high = bounds.get(name, (None, None))
    if low is not None and high is not None and not high > low:
        raise ValueError(f"{name}: the bound ({low}, {high}) is not increasing")
    return low, high


def to_natural(
    free: torch.Tensor, bounds: Mapping[str, Any], names: Sequence[str]
) -> torch.Tensor:
    """Map the unconstrained variables back to the properties they stand for.

    Parameters
    ----------
    free:
        ``(..., parameters)``, in the order ``names`` gives.
    bounds:
        ``{name: (low, high)}``. Empty leaves everything as it is.
    names:
        One name per column of ``free``.

    Returns
    -------
    torch.Tensor
        The properties, each inside its interval.
    """
    if not bounds:
        return free
    columns = []
    for index, name in enumerate(names):
        low, high = bound_of(bounds, name)
        value = free[..., index]
        if low is not None and high is not None:
            columns.append(low + (high - low) * torch.sigmoid(value))
        elif low is not None:
            columns.append(low + torch.nn.functional.softplus(value))
        elif high is not None:
            columns.append(high - torch.nn.functional.softplus(-value))
        else:
            columns.append(value)
    return torch.stack(columns, dim=-1)


def to_free(
    natural: torch.Tensor,
    bounds: Mapping[str, Any],
    names: Sequence[str],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map properties to the unconstrained variables that stand for them.

    The inverse of :func:`to_natural`. The clamp guards the arithmetic at the
    very edge of the interval; a starting value actually sitting on a bound is
    something the caller should refuse where it is stated, because the free
    variable that stands for it is infinite.

    Parameters
    ----------
    natural:
        ``(..., parameters)``, each inside its interval.
    bounds:
        ``{name: (low, high)}``.
    names:
        One name per column.
    device:
        Where to put the answer. Left out, it stays where it was.

    Returns
    -------
    torch.Tensor
        The unconstrained variables.
    """
    if not bounds:
        return natural if device is None else natural.to(device)
    columns = []
    for index, name in enumerate(names):
        low, high = bound_of(bounds, name)
        value = natural[..., index]
        if low is not None and high is not None:
            span = high - low
            inside = ((value - low) / span).clamp(EDGE, 1.0 - EDGE)
            columns.append(torch.log(inside) - torch.log1p(-inside))
        elif low is not None:
            columns.append(_softplus_inverse((value - low).clamp_min(EDGE)))
        elif high is not None:
            columns.append(-_softplus_inverse((high - value).clamp_min(EDGE)))
        else:
            columns.append(value)
    stacked = torch.stack(columns, dim=-1)
    return stacked if device is None else stacked.to(device)


def widen(
    bounds: Mapping[str, Any], names: Sequence[str], free: torch.Tensor
) -> torch.Tensor:
    """The derivative of each property with respect to its free variable.

    What multiplies the Jacobian so a solver stepping in the unconstrained
    variable takes the right step.

    Parameters
    ----------
    bounds:
        ``{name: (low, high)}``.
    names:
        One name per column of ``free``.
    free:
        ``(..., parameters)``, the unconstrained variables.

    Returns
    -------
    torch.Tensor
        The chain-rule factor, shaped like ``free``.
    """
    if not bounds:
        return torch.ones_like(free)
    columns = []
    for index, name in enumerate(names):
        low, high = bound_of(bounds, name)
        value = free[..., index]
        if low is not None and high is not None:
            opened = torch.sigmoid(value)
            columns.append((high - low) * opened * (1.0 - opened))
        elif low is not None:
            columns.append(torch.sigmoid(value))
        elif high is not None:
            columns.append(torch.sigmoid(-value))
        else:
            columns.append(torch.ones_like(value))
    return torch.stack(columns, dim=-1)


# %% private module subroutines


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """``log(exp(x) - 1)``, written so a large ``x`` does not overflow."""
    return value + torch.log(-torch.expm1(-value))
