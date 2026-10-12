"""Sequence design: a cost over bounded parameters, minimized by gradient descent.

The cost is a user function of the designed parameters (by keyword) returning
one number; the gradient is reverse-mode autograd through it.
"""

from __future__ import annotations

__all__ = [
    "Bounded",
    "SequenceDesign",
    "SequenceOptimization",
    "crlb",
]

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch

from ..sequence._array import as_torch

Limits: TypeAlias = tuple[Any, Any] | None


@dataclass(frozen=True)
class Bounded:
    """An initial value and the interval a designed parameter stays in.

    The limits are enforced exactly by optimizing a logit-transformed
    variable, so no iterate leaves them.

    Attributes
    ----------
    initial : array-like
        Starting value.
    lower, upper : float or array-like
        Limits, scalar or per element, in the parameter's units.
    """

    initial: Any
    lower: Any
    upper: Any


@dataclass(frozen=True)
class SequenceOptimization:
    """Result of :meth:`SequenceDesign.minimize`.

    Attributes
    ----------
    parameters : dict
        ``{name: value}`` at the last step, inside their limits.
    loss : torch.Tensor
        Cost at each step.
    """

    parameters: dict[str, torch.Tensor]
    loss: torch.Tensor


class SequenceDesign(torch.nn.Module):
    """A cost, the parameters it is minimized over, and their limits.

    Parameters
    ----------
    cost : callable
        Called with the designed parameters by keyword; returns a one-element
        tensor.
    **parameters : Bounded or torch.Tensor
        One entry per designed parameter. A :class:`Bounded` carries its limits;
        a bare array is unbounded. At least one.

    Raises
    ------
    ValueError
        If no parameter is given, or a lower limit is not below its upper.

    Notes
    -----
    A preference (a soft penalty) belongs in the cost; a hard scanner limit
    belongs in :class:`Bounded`.

    Examples
    --------
    .. code-block:: python

        design = SequenceDesign(
            sharpness, flip=Bounded(torch.full((8, 120), 120.0), 20.0, 180.0)
        )
        result = design.minimize(iterations=120)
    """

    def __init__(self, cost: Callable[..., torch.Tensor], **parameters: Any) -> None:
        super().__init__()
        if not parameters:
            raise ValueError("a design needs at least one parameter")
        self.cost = cost
        self._limits: dict[str, Limits] = {}
        raw: dict[str, torch.nn.Parameter] = {}
        for name, entry in parameters.items():
            value = as_torch(entry.initial if isinstance(entry, Bounded) else entry)
            limits = (entry.lower, entry.upper) if isinstance(entry, Bounded) else None
            self._limits[name] = limits
            raw[name] = torch.nn.Parameter(_unconstrained(value, limits))
        self.raw = torch.nn.ParameterDict(raw)

    def values(self) -> dict[str, torch.Tensor]:
        """Return the parameters as passed to the cost, inside their limits."""
        return {
            name: _constrained(value, self._limits[name])
            for name, value in self.raw.items()
        }

    def forward(self) -> torch.Tensor:
        """Evaluate the cost at the current parameters.

        Raises
        ------
        ValueError
            If the cost does not return one number.
        """
        loss = self.cost(**self.values())
        if loss.numel() != 1:
            raise ValueError("a design cost must return one number")
        return loss

    def minimize(
        self,
        *,
        iterations: int = 200,
        learning_rate: float = 0.05,
        optimizer_factory: Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer]
        | None = None,
        callback: Callable[[int, dict[str, torch.Tensor], torch.Tensor], bool | None]
        | None = None,
    ) -> SequenceOptimization:
        """Minimize the cost over the designed parameters.

        Parameters
        ----------
        iterations : int, optional
            Number of updates. Defaults to 200.
        learning_rate : float, optional
            Adam step size, used when ``optimizer_factory`` is omitted.
            Defaults to 0.05.
        optimizer_factory : callable, optional
            Maps the list of unconstrained variables to a
            :class:`torch.optim.Optimizer`.
        callback : callable, optional
            Called after each update as ``callback(step, parameters, loss)``
            with detached values; returning ``True`` stops early.

        Returns
        -------
        SequenceOptimization

        Raises
        ------
        ValueError
            If ``iterations`` or ``learning_rate`` is not positive.
        """
        if iterations < 1 or learning_rate <= 0:
            raise ValueError("iterations and learning_rate must be positive")
        variables = list(self.raw.values())
        optimizer = (
            torch.optim.Adam(variables, lr=learning_rate)
            if optimizer_factory is None
            else optimizer_factory(variables)
        )
        history = []
        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = self()
            loss.backward()
            optimizer.step()
            history.append(loss.detach())
            if callback is not None and callback(
                step, _detached(self.values()), loss.detach()
            ):
                break
        return SequenceOptimization(
            parameters=_detached(self.values()), loss=torch.stack(history)
        )


def crlb(
    jacobian: torch.Tensor,
    *,
    noise_variance: float = 1.0,
    singular: Literal["raise", "infinite"] = "raise",
) -> torch.Tensor:
    """Cramer-Rao lower bound on the variance of each parameter.

    Parameters
    ----------
    jacobian : torch.Tensor
        ``(..., parameters, samples)``, derivative of the signal with respect
        to each parameter. Real and imaginary parts count as separate
        measurements.
    noise_variance : float, optional
        Variance of the noise per real measurement, in signal units squared.
        Defaults to 1.
    singular : {"raise", "infinite"}, optional
        Behaviour where the parameters are not jointly identifiable: raise, or
        return an infinite variance at those entries. Defaults to ``"raise"``.

    Returns
    -------
    torch.Tensor
        ``(..., parameters)``, diagonal of the inverse Fisher matrix.

    Raises
    ------
    ValueError
        If the Jacobian has no parameter axis, or ``singular`` is invalid.
    torch.linalg.LinAlgError
        If the Fisher matrix is singular and ``singular="raise"``.
    """
    if jacobian.ndim < 2:
        raise ValueError("a Jacobian is (..., parameters, samples)")
    rows = (
        torch.stack((jacobian.real, jacobian.imag), dim=-1)
        if torch.is_complex(jacobian)
        else jacobian.unsqueeze(-1)
    )
    fisher = torch.einsum("...psc,...qsc->...pq", rows, rows)
    if singular == "raise":
        inverse = torch.linalg.inv(fisher)
        return noise_variance * torch.diagonal(inverse, dim1=-2, dim2=-1).real
    if singular != "infinite":
        raise ValueError(f"singular is 'raise' or 'infinite', got {singular!r}")
    inverse, failed = torch.linalg.inv_ex(fisher)
    bound = noise_variance * torch.diagonal(inverse, dim1=-2, dim2=-1).real
    return bound.masked_fill((failed != 0)[..., None].expand_as(bound), float("inf"))


# %% private module subroutines


def _detached(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in values.items()}


def _unconstrained(value: torch.Tensor, limits: Limits) -> torch.Tensor:
    """The variable whose sigmoid is ``value`` inside ``limits``."""
    if limits is None:
        return value.detach().clone()
    lower, upper = _as_limits(limits, value)
    if bool(torch.any(lower >= upper)):
        raise ValueError("each lower limit must be below its upper limit")
    fraction = ((value - lower) / (upper - lower)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(fraction).detach().clone()


def _constrained(raw: torch.Tensor, limits: Limits) -> torch.Tensor:
    if limits is None:
        return raw
    lower, upper = _as_limits(limits, raw)
    return lower + (upper - lower) * torch.sigmoid(raw)


def _as_limits(
    limits: tuple[Any, Any], reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        for value in limits
    )
