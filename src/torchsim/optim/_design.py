"""Stating a sequence-design problem, and solving it.

A design problem is three things, and this module is one object for each.

The **acquisition** is a simulator with the tissue it is being designed for
already fixed on it, so only the parameters under design are left to give. It
answers the same two questions any simulator does -- what is recorded, and how
that changes with the tissue -- which is what lets a cost be written against
the simulator's own vocabulary.

The **cost** is a plain function the user writes. It takes the parameters
being designed by name, asks its acquisitions what they record, and returns
one number. Nothing about it is registered or declared.

:class:`SequenceDesign` holds the cost and the parameters, each with the
limits it may move between, and :meth:`SequenceDesign.minimize` runs the loop.
The gradient is reverse-mode autograd on the cost: the engine reads which of
its inputs carry one and picks its kernel from that, so a design loop needs no
derivative of its own.

The two families this has to carry look nothing alike, which is the point.
A quantitative sequence is designed for **precision** -- the cost is a
Cramer-Rao bound on what it estimates, and reads the acquisition's Jacobian.
An anatomical sequence is designed for **image quality** -- the cost is a
property of the point spread function the echo train produces, and reads the
signal alone. Both are the same object with a different function in it.
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
from typing import Any, TypeAlias

import torch

from ..sequence._array import as_torch

Limits: TypeAlias = tuple[Any, Any] | None


@dataclass(frozen=True)
class Bounded:
    """An initial value and the range it is allowed to move in.

    Attributes
    ----------
    initial : array-like
        Where the design starts.
    lower, upper : float or array-like
        The limits, either scalars or values per element. They are enforced
        exactly, by optimizing a variable the limits are a sigmoid of, so no
        iterate is ever outside them -- which matters when the bound is what
        the scanner can actually play.
    """

    initial: Any
    lower: Any
    upper: Any


@dataclass(frozen=True)
class SequenceOptimization:
    """The designed parameters and the loss at every step that reached them.

    Attributes
    ----------
    parameters : dict
        ``{name: value}`` at the last step, inside the limits each was given.
    loss : torch.Tensor
        The cost after every step, so convergence is read rather than assumed.
    """

    parameters: dict[str, torch.Tensor]
    loss: torch.Tensor


class SequenceDesign(torch.nn.Module):
    """A cost, the parameters it is minimized over, and the limits on them.

    Parameters
    ----------
    cost : callable
        Called with the designed parameters by keyword, returning one number.
        Everything sequence-specific lives here: what is simulated, what is
        measured about it, and what is penalized.
    parameters : Bounded or torch.Tensor, optional
        One entry per designed parameter. A :class:`Bounded` carries its own
        limits; a bare array is left free.

    Examples
    --------
    .. code-block:: python

        design = SequenceDesign(
            sharpness, flip=Bounded(torch.full((8, 120), 120.0), 20.0, 180.0)
        )
        result = design.minimize(iterations=120)

    Notes
    -----
    A feasibility term the scanner would merely prefer belongs in the cost,
    where it trades against everything else. A limit the scanner cannot exceed
    belongs in :class:`Bounded`, where no iterate can cross it.
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
        """The designed parameters as the cost sees them, inside their limits."""
        return {
            name: _constrained(value, self._limits[name])
            for name, value in self.raw.items()
        }

    def forward(self) -> torch.Tensor:
        """Evaluate the cost at the parameters as they stand."""
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
        """Run the design to ``iterations`` steps.

        Parameters
        ----------
        iterations : int, optional
            How many updates to take.
        learning_rate : float, optional
            Adam step size, used when ``optimizer_factory`` is omitted.
        optimizer_factory : callable, optional
            Called with the unconstrained variables, returning the optimizer
            to drive them.
        callback : callable, optional
            Called after each update with the step number, the parameters and
            the loss, both detached. Returning ``True`` stops early.

        Returns
        -------
        SequenceOptimization
            The parameters as they finished, and the loss at every step.
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


def crlb(jacobian: torch.Tensor, *, noise_variance: float = 1.0) -> torch.Tensor:
    """The lowest variance an unbiased estimate of each parameter can have.

    Parameters
    ----------
    jacobian : torch.Tensor
        ``(..., parameters, samples)`` -- the derivative of the signal with
        respect to each parameter being estimated. Real and imaginary parts
        are counted as separate measurements, which is what independent
        Gaussian noise on the two channels means.
    noise_variance : float, optional
        The variance of that noise, in the units the signal is in.

    Returns
    -------
    torch.Tensor
        ``(..., parameters)``, the diagonal of the inverse Fisher matrix. What
        to do with it is the design's: summing gives A-optimality, dividing by
        the parameter values first makes the sum dimensionless and so weights
        a short and a long relaxation time comparably.

    Raises
    ------
    ValueError
        If the Jacobian has no parameter axis.
    torch.linalg.LinAlgError
        If the parameters are not jointly identifiable from these samples,
        which is a singular Fisher matrix and says the design cannot work at
        all rather than that it is imprecise.
    """
    if jacobian.ndim < 2:
        raise ValueError("a Jacobian is (..., parameters, samples)")
    rows = (
        torch.stack((jacobian.real, jacobian.imag), dim=-1)
        if torch.is_complex(jacobian)
        else jacobian.unsqueeze(-1)
    )
    fisher = torch.einsum("...psc,...qsc->...pq", rows, rows)
    inverse = torch.linalg.inv(fisher)
    return noise_variance * torch.diagonal(inverse, dim1=-2, dim2=-1).real


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
