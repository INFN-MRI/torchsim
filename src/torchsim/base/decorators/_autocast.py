"""Automatic conversion of model inputs to Torch tensors."""

from __future__ import annotations

__all__ = ["autocast"]

from functools import wraps
import inspect
from collections.abc import Callable
from typing import Any

import torch


def autocast(func: Callable) -> Callable:
    """
    Force all inputs to be torch tensors of the same size on the same device.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _fill_kwargs(func, args, kwargs)
        args, kwargs = _to_tensors(*args, **kwargs)
        args, kwargs = _enforce_precision(*args, **kwargs)
        device = _leading_device(args, kwargs)
        args, kwargs = _to_device(device, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


# %% subroutines
def _fill_kwargs(func, args, kwargs):
    """This automatically fills missing kwargs with default values."""
    signature = inspect.signature(func)

    # Get number of arguments
    n_args = len(args)

    # Create a dictionary of keyword arguments and their default values
    _kwargs = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            _kwargs[k] = v.default
        else:
            _kwargs[k] = None

    # Merge the default keyword arguments with the provided kwargs
    for k in kwargs.keys():
        _kwargs[k] = kwargs[k]

    # Replace args
    _keys = list(_kwargs.keys())[n_args:]
    _values = list(_kwargs.values())[n_args:]

    return args, dict(zip(_keys, _values, strict=True))


def _enforce_precision(*args, **kwargs):
    """Enforce tensors precision."""
    args = [_to_float32(value) for value in args]
    kwargs = {key: _to_float32(value) for key, value in kwargs.items()}
    return args, kwargs


def _to_tensors(*args, **kwargs):
    """Enforce tensors."""
    args = [_to_tensor(value) for value in args]
    kwargs = {key: _to_tensor(value) for key, value in kwargs.items()}
    return args, kwargs


def _to_device(device, *args, **kwargs):
    """Enforce same device."""
    args = [value.to(device) if isinstance(value, torch.Tensor) else value for value in args]
    kwargs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
    return args, kwargs


def _to_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor) or value is None:
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return value


def _to_float32(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(torch.float32)
    return value


def _leading_device(args: list[Any], kwargs: dict[str, Any]) -> torch.device:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")
