"""Wrapper for complex forward jacobian."""

__all__ = ["jacfwd"]

from functools import wraps

from typing import Callable, Optional
import torch


def jacfwd(argnums: tuple[int]) -> Callable:
    """
    Decorator to compute the Jacobian of a function with complex-valued outputs.

    Parameters
    ----------
    argnum : tuple[int]
        The argument indexes to compute the Jacobian with respect to.

    Returns
    -------
    Callable
        Decorated function that computes the Jacobian with appropriate handling of complex outputs.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Wrapper function for differentiation.

            Returns
            -------
            Tuple[torch.Tensor, Optional[torch.Tensor]]
                The Jacobian and the primal output it was linearized about.
            """

            # Forward-mode differentiation evaluates the primal anyway, so
            # return it as auxiliary output rather than simulating twice.
            def wrapped_fn(*wrapped_args):
                output = fn(*wrapped_args, **kwargs)
                return _split_real_imag(output), output

            jacobian, primal = torch.func.jacfwd(
                wrapped_fn, argnums=argnums, has_aux=True
            )(*args)

            return _combine_real_imag(jacobian), primal

        return wrapper

    return decorator


# %% subroutines
def _split_real_imag(tensor: torch.Tensor) -> torch.Tensor:
    """Split complex tensor into real and imaginary components."""
    if torch.is_complex(tensor):
        return torch.stack([tensor.real, tensor.imag], dim=0)
    else:
        return torch.stack([tensor, torch.zeros_like(tensor)], dim=0)


def _combine_real_imag(split_tensor: torch.Tensor) -> torch.Tensor:
    """Combine split real and imaginary components into a complex tensor."""
    if isinstance(split_tensor, tuple):
        output = [tensor[0] + 1j * tensor[1] for tensor in split_tensor]
        output = torch.stack(output, dim=0)
    else:
        output = split_tensor[0] + 1j * split_tensor[1]
    return output
