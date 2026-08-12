"""Automatic broadcasting"""

__all__ = ["broadcast", "broadcast_arguments"]

import torch

from functools import wraps
from typing import Callable


def broadcast(func: Callable) -> Callable:
    """
    Force all inputs to be torch tensors of the same size on the same device.
    """

    @wraps(func)
    def wrapper(*args):
        shape = _find_first_nonscalar_shape(args)

        # broadcast
        args, _ = broadcast_arguments(*args)

        # run function
        output = func(*args)
        if isinstance(output, tuple):
            return tuple(_reshape_output(item, shape) for item in output)
        return _reshape_output(output, shape)

    return wrapper


def _reshape_output(output: torch.Tensor, shape) -> torch.Tensor:
    """Drop a vanishing imaginary part and restore the batch shape."""
    if torch.isreal(output).all():
        output = output.real
    return output.reshape(*shape, *output.shape[1:]).squeeze()


def broadcast_arguments(*args, **kwargs) -> tuple[list, dict]:
    """
    Force all inputs to be torch tensors of the same size.
    """
    # enforge mutable
    args = list(args)

    items, kwitems, indexes, keys = _get_tensor_args_kwargs(*args, **kwargs)
    items = [torch.atleast_1d(item) for item in items]
    kwitems = {k: torch.atleast_1d(v) for k, v in kwitems.items()}
    tmp = torch.broadcast_tensors(*items, *list(kwitems.values()))
    tmp = list(tmp)
    for n in range(len(items)):
        items[n] = tmp[0]
        tmp.pop(0)
    kwitems = dict(zip(kwitems.keys(), tmp))

    for idx in indexes:
        args[idx] = items[idx]
    for key in kwitems.keys():
        kwargs[key] = kwitems[key]

    return args, kwargs


# %% subroutines
def _find_first_nonscalar_shape(batched_args):  # noqa
    """Returns shape of first non-scalar tensor."""
    shape = [1]
    for arg in batched_args:
        if arg.ndim != 0:
            return arg.shape
    return shape


def _get_tensor_args_kwargs(*args, **kwargs):
    items = []
    kwitems = {}
    indexes = []
    keys = []
    for n in range(len(args)):
        if isinstance(args[n], torch.Tensor):
            items.append(args[n])
            indexes.append(n)
    for key in kwargs.keys():
        if isinstance(kwargs[key], torch.Tensor):
            kwitems[key] = kwargs[key]
            keys.append(key)

    return items, kwitems, indexes, keys
