"""Running one model from a flat argument list."""

from __future__ import annotations

__all__ = ["evaluated"]

from collections.abc import Sequence
from typing import Any

import torch

from ..model import SignalModel


def evaluated(
    model: SignalModel,
    diff: str | Sequence[str] | None,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return the signal, and its Jacobian where one was asked for.

    Property and sequence arguments are given together; the model tells them
    apart. Array-like arguments are placed on ``device``, while a plain count
    -- a number of states or of shots -- stays what it is.

    Parameters
    ----------
    model:
        The model to run.
    diff:
        What to differentiate with respect to, or ``None`` for the signal
        alone.
    device:
        Where to run, or ``None`` to follow the inputs.
    values:
        The model's property and sequence arguments.

    Returns
    -------
    torch.Tensor or tuple
        The signal, or the signal and its Jacobian.
    """
    placed = {
        name: value if isinstance(value, (int, bool)) else torch.as_tensor(
            value, device=device
        )
        for name, value in values.items()
        if value is not None
    }
    if diff is None:
        return model.simulate(**placed)
    return model.jacobian(diff, **placed)
