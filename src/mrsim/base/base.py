"""Base simulator class"""

__all__ = ["AbstractModel"]

import inspect
import functools

from abc import ABC, abstractmethod
from typing import Any, Callable
from types import SimpleNamespace

import torch

from .decorators import autocast, broadcast, jacfwd


class AbstractModel(ABC):
    """Abstract base class for MRI simulation models with automated parameter handling."""

    def __init__(
        self,
        diff: str | tuple[str] | None = None,
        chunk_size: int | None = None,
        device: str | torch.device | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the model with automatic parameter segregation and engine setup.

        Parameters
        ----------
        diff : str | tuple[str] | None, optional
            Parameters to compute the jacobian with respect to.
        device : str | torch.device | None, optional
            Device for computations (e.g., 'cpu', 'cuda').
        chunk_size : int | None, optional
            Number of samples to process in parallel.

        """
        self.chunk_size = chunk_size
        self.device = device if device is not None else torch.device("cpu")
        self.diff = diff

        # Extract broadcastable parameters
        self.broadcastable_params = list(
            inspect.signature(self.set_properties).parameters.keys()
        )
        self.properties = SimpleNamespace()
        self.sequence = SimpleNamespace()

    @autocast
    @abstractmethod
    def set_properties(self, *args, **kwargs):
        """
        Define broadcastable spin/environment parameters.
        """
        raise NotImplementedError("Subclasses must implement `set_properties`.")

    @autocast
    @abstractmethod
    def set_sequence(self, *args, **kwargs):
        """
        Define sequence parameters.
        """
        raise NotImplementedError("Subclasses must implement `set_sequence`.")

    @staticmethod
    @abstractmethod
    def _engine(*args, **kwargs):
        """
        Core computational function for the model.
        """
        raise NotImplementedError("Subclasses must implement `_engine`.")

    @staticmethod
    def _jacobian_engine(*args, **kwargs):
        """
        Manual Jacobian engine.
        """
        raise NotImplementedError("Manual derivative not implemented.")

    def _get_func(self, _func, *args, **kwargs) -> tuple[Callable, dict[str, Any]]:
        """
        Dynamically split parameters into broadcastable and non-broadcastable groups.

        Parameters
        ----------
        _engine : Callable
            Function to be wrapped.
        *args : tuple[Any, ...]
            Positional arguments provided by the user.
        **kwargs : dict[str, Any]
            Keyword arguments provided by the user.

        Returns
        -------
        Callable
            A new engine accepting broadcastable parameters.
        dict[str, Any]
            Captured non-broadcastable parameters.

        """
        # Get the engine's signature
        signature = inspect.signature(_func)

        # Extract default values for all parameters
        default_args = {}
        for k, v in signature.parameters.items():
            if v.default is not inspect.Parameter.empty:
                default_args[k] = v.default
            else:
                default_args[k] = None

        # Merge default values and user-provided keyword arguments
        merged_kwargs = {**default_args, **kwargs}

        # Replace first `len(user_args)` items with positional arguments
        parameter_names = list(signature.parameters.keys())
        for idx, arg in enumerate(args):
            merged_kwargs[parameter_names[idx]] = arg

        # Split into broadcastable and non-broadcastable
        # broadcastable_args = {
        #     k: v for k, v in merged_kwargs.items() if k in self.broadcastable_params
        # }
        non_broadcastable_args = {
            k: v for k, v in merged_kwargs.items() if k not in self.broadcastable_params
        }

        # Define a new engine that takes only broadcastable parameters explicitly
        def func(*args):
            # Map provided positional arguments to broadcastable parameter names
            broadcastable_mapping = dict(zip(self.broadcastable_params, args))
            # Merge broadcastable and non-broadcastable parameters
            combined_args = {**broadcastable_mapping, **non_broadcastable_args}
            # Call the original engine
            return _func(**combined_args)

        # Get argnums for diff
        if self.diff is not None:
            argnums = _get_argnums(self.diff, self.broadcastable_params)
        else:
            argnums = None

        return func, argnums

    def forward(self, *args, **kwargs):
        """
        Return a callable for forward computation. Useful for sequence optimization.

        Parameters
        ----------
        *args : Any
            Positional arguments for the simulation.
        **kwargs : Any
            Keyword arguments for the simulation.

        Returns
        -------
        callable
            A function that evaluates the forward model with the specified arguments.

        """
        engine, _ = self._get_func(self._engine, *args, **kwargs)

        def vmapped_engine(*inputs):
            vmapped = torch.vmap(engine, chunk_size=self.chunk_size)
            broadcast_vmapped = broadcast(vmapped)
            return broadcast_vmapped(*inputs)

        return vmapped_engine

    def jacobian(self, *args, **kwargs):
        """
        Return a callable for the Jacobian computation. Useful for sequence optimization.

        Parameters
        ----------
        *args : Any
            Positional arguments for the simulation.
        **kwargs : Any
            Keyword arguments for the simulation.

        Returns
        -------
        callable
            A function that computes the Jacobian with respect to specified arguments.

        """
        if self.diff is None:
            return None
        if _is_implemented(self._jacobian_engine):
            jac_engine, argnums = self._get_func(self._jacobian_engine, *args, **kwargs)

            def jacobian_engine(*inputs):
                vmapped_jac = torch.vmap(jac_engine, chunk_size=self.chunk_size)
                broadcast_vmapped_jac = broadcast(vmapped_jac)
                return broadcast_vmapped_jac(*inputs)

        else:
            engine, argnums = self._get_func(self._engine, *args, **kwargs)

            def jacobian_engine(*inputs):
                jac_engine = jacfwd(argnums=argnums)(engine)
                vmapped_jac = torch.vmap(jac_engine, chunk_size=self.chunk_size)
                broadcast_vmapped_jac = broadcast(vmapped_jac)
                return broadcast_vmapped_jac(*inputs)

        return jacobian_engine

    def __call__(self):
        """
        Calls both forward and jacobian methods, returning output and jacobian for sequence optimization.

        Returns
        -------
        tuple
            A tuple containing the forward output and jacobian output.

        """
        kwargs = {**vars(self.properties), **vars(self.sequence)}

        broadcastable_kwargs = {
            k: v for k, v in kwargs.items() if k in self.broadcastable_params
        }
        non_broadcastable_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.broadcastable_params
        }

        forward_fn = self.forward(**non_broadcastable_kwargs)

        if self.diff is None:
            with torch.no_grad():
                output = forward_fn(*broadcastable_kwargs.values())
            return output

        output = forward_fn(*broadcastable_kwargs.values())

        jacobian_fn = self.jacobian(**non_broadcastable_kwargs)
        jacobian_output = jacobian_fn(*broadcastable_kwargs.values())

        return output, jacobian_output


# %% TODO: move
def _is_implemented(method) -> bool:
    """Check if the method is implemented or raises NotImplementedError."""
    try:
        # Check if the method raises NotImplementedError or has no implementation
        source_code = inspect.getsource(method)
        return (
            "raise NotImplementedError" not in source_code and "pass" not in source_code
        )
    except AttributeError:
        return False


def _get_args(func, args, kwargs):
    """Convert input args/kwargs mix to a list of positional arguments.

    This automatically fills missing kwargs with default values.
    """
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
    _args = list(_kwargs.values())

    return list(args) + _args[n_args:]


def _get_argnums(diff, ARGS):  # noqa
    """Helper function to get argument indices for differentiation."""
    ARGMAP = dict(zip(ARGS, list(range(len(ARGS)))))

    if isinstance(diff, str):
        return ARGMAP[diff]
    elif isinstance(diff, (tuple, list)):
        return tuple([ARGMAP[d] for d in diff])
    else:
        raise ValueError(f"Unsupported diff type: {diff}")
