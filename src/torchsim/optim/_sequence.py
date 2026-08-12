"""General constrained optimization of differentiable sequence parameters."""

from __future__ import annotations

__all__ = ["SequenceOptimization", "SequenceOptimizer"]

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

SequenceParameters: TypeAlias = torch.Tensor | dict[str, torch.Tensor]
Bounds: TypeAlias = (
    tuple[float | torch.Tensor, float | torch.Tensor]
    | dict[str, tuple[float | torch.Tensor, float | torch.Tensor]]
    | None
)


@dataclass(frozen=True)
class SequenceOptimization:
    """Optimized sequence parameters and convergence history."""

    parameters: SequenceParameters
    loss: torch.Tensor


class SequenceOptimizer(torch.nn.Module):
    """Optimize one or more differentiable sequence-parameter tensors.

    Parameters
    ----------
    objective : callable
        Differentiable callable receiving a tensor or parameter dictionary and
        returning a scalar loss.
    bounds : tuple or dict, optional
        Bounds enforced by a sigmoid parameterization. Dictionary bounds may
        constrain only a subset of named parameters.
    iterations : int, optional
        Number of optimizer updates.
    learning_rate : float, optional
        Adam learning rate used when ``optimizer_factory`` is omitted.
    optimizer_factory : callable, optional
        Factory accepting the unconstrained ``torch.nn.Parameter`` objects.

    Notes
    -----
    The objective owns sequence-specific physics and costs. Consequently the
    same optimizer can tune flip angles, phases, timings, preparation pulses,
    or several of them jointly. A callback may log intermediate values or
    request early stopping by returning ``True``. Scanner-specific feasibility
    terms belong in the objective, while hard limits belong in ``bounds``.
    """

    def __init__(
        self,
        objective: Callable[[SequenceParameters], torch.Tensor],
        *,
        bounds: Bounds = None,
        iterations: int = 80,
        learning_rate: float = 0.08,
        optimizer_factory: Callable[
            [list[torch.nn.Parameter]], torch.optim.Optimizer
        ]
        | None = None,
    ) -> None:
        super().__init__()
        if iterations < 1 or learning_rate <= 0:
            raise ValueError("iterations and learning_rate must be positive")
        self.objective = objective
        self.bounds = bounds
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.optimizer_factory = optimizer_factory

    def forward(self, parameters: SequenceParameters) -> torch.Tensor:
        """Evaluate the configured sequence objective."""
        return self.objective(parameters)

    def optimize(
        self,
        initial: SequenceParameters,
        *,
        callback: Callable[
            [int, SequenceParameters, torch.Tensor], bool | None
        ]
        | None = None,
    ) -> SequenceOptimization:
        """Run constrained optimization from ``initial``.

        Parameters
        ----------
        initial : Tensor or dict
            Initial sequence parameters.
        callback : callable, optional
            Called after each update with the iteration, detached parameters,
            and detached loss. Returning ``True`` stops optimization.

        Returns
        -------
        SequenceOptimization
            Final parameters and the loss history through the last update.
        """
        raw, is_mapping = _make_raw_parameters(initial, self.bounds)
        raw_values = list(raw.values()) if is_mapping else [raw]
        optimizer = (
            torch.optim.Adam(raw_values, lr=self.learning_rate)
            if self.optimizer_factory is None
            else self.optimizer_factory(raw_values)
        )
        history = []

        for iteration in range(self.iterations):
            optimizer.zero_grad(set_to_none=True)
            parameters = _transform_parameters(raw, self.bounds, is_mapping)
            loss = self(parameters)
            if loss.numel() != 1:
                raise ValueError("sequence objective must return a scalar")
            loss.backward()
            optimizer.step()
            history.append(loss.detach())
            if callback is not None and callback(
                iteration, _detach_parameters(parameters), loss.detach()
            ):
                break

        parameters = _transform_parameters(raw, self.bounds, is_mapping)
        return SequenceOptimization(
            parameters=_detach_parameters(parameters),
            loss=torch.stack(history),
        )


# %% private module subroutines


def _make_raw_parameters(
    initial: SequenceParameters,
    bounds: Bounds,
) -> tuple[torch.nn.Parameter | torch.nn.ParameterDict, bool]:
    if isinstance(initial, Mapping):
        raw = {
            name: torch.nn.Parameter(
                _inverse_transform(value, _named_bounds(bounds, name))
            )
            for name, value in initial.items()
        }
        return torch.nn.ParameterDict(raw), True
    initial = torch.as_tensor(initial)
    tensor_bounds = bounds if isinstance(bounds, tuple) else None
    return torch.nn.Parameter(_inverse_transform(initial, tensor_bounds)), False


def _transform_parameters(
    raw: torch.nn.Parameter | torch.nn.ParameterDict,
    bounds: Bounds,
    is_mapping: bool,
) -> SequenceParameters:
    if is_mapping:
        assert isinstance(raw, torch.nn.ParameterDict)
        return {
            name: _transform(value, _named_bounds(bounds, name))
            for name, value in raw.items()
        }
    assert isinstance(raw, torch.nn.Parameter)
    return _transform(raw, bounds if isinstance(bounds, tuple) else None)


def _named_bounds(
    bounds: Bounds,
    name: str,
) -> tuple[float | torch.Tensor, float | torch.Tensor] | None:
    if isinstance(bounds, dict):
        return bounds.get(name)
    return None


def _inverse_transform(
    value: torch.Tensor,
    bounds: tuple[float | torch.Tensor, float | torch.Tensor] | None,
) -> torch.Tensor:
    value = torch.as_tensor(value)
    if bounds is None:
        return value.detach().clone()
    lower, upper = _bound_tensors(bounds, value)
    if bool(torch.any(lower >= upper)):
        raise ValueError("each lower bound must be smaller than its upper bound")
    fraction = ((value - lower) / (upper - lower)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(fraction).detach().clone()


def _transform(
    raw: torch.Tensor,
    bounds: tuple[float | torch.Tensor, float | torch.Tensor] | None,
) -> torch.Tensor:
    if bounds is None:
        return raw
    lower, upper = _bound_tensors(bounds, raw)
    return lower + (upper - lower) * torch.sigmoid(raw)


def _bound_tensors(
    bounds: tuple[float | torch.Tensor, float | torch.Tensor],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        for value in bounds
    )


def _detach_parameters(parameters: SequenceParameters) -> SequenceParameters:
    if isinstance(parameters, dict):
        return {name: value.detach() for name, value in parameters.items()}
    return parameters.detach()
