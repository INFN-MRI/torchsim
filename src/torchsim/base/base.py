"""Base simulator class."""

from __future__ import annotations

__all__ = ["AbstractModel"]

from abc import ABC, abstractmethod
from collections.abc import Callable
import inspect
from types import SimpleNamespace
from typing import Any

import torch

from .decorators import autocast, broadcast, broadcast_arguments, jacfwd


class AbstractModel(ABC):
    """Abstract base class for MRI simulation models with automated parameter handling."""

    vectorized_engine = False

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
        self.device = device
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

    def _forward(self, *args, **kwargs):
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

        if self.vectorized_engine:

            def vectorized_engine(*inputs):
                broadcast_inputs, _ = broadcast_arguments(*inputs)
                return engine(*broadcast_inputs).squeeze()

            return vectorized_engine

        def vmapped_engine(*inputs):
            vmapped = torch.vmap(engine, chunk_size=self.chunk_size)
            broadcast_vmapped = broadcast(vmapped)
            return broadcast_vmapped(*inputs)

        return vmapped_engine

    def _jacobian(self, *args, **kwargs):
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

        elif self.vectorized_engine:
            engine, argnums = self._get_func(self._engine, *args, **kwargs)

            # A vectorized engine already evaluates a whole batch of voxels,
            # and voxels are independent, so its Jacobian is diagonal across
            # them. One directional derivative per differentiated parameter
            # therefore yields every per-voxel derivative at once, and vmap --
            # whose pytree bookkeeping dominates the runtime here -- is not
            # needed.
            def jacobian_engine(*inputs):
                def differentiate(*broadcast_inputs):
                    return _forward_mode_jacobian(engine, argnums, broadcast_inputs)

                return broadcast(differentiate)(*inputs)

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

        # Get device
        if self.device is None:
            device = _leading_device(kwargs)
        else:
            device = self.device

        # Force device
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor):
                kwargs[k] = kwargs[k].to(device)

        # Split broadcastable and non-broadcastable params
        broadcastable_kwargs = {
            k: v for k, v in kwargs.items() if k in self.broadcastable_params
        }
        non_broadcastable_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.broadcastable_params
        }

        # If no derivative is requested, run forward with explicit `no_grad()` for performance
        if self.diff is None:
            forward_fn = self._forward(**non_broadcastable_kwargs)
            with torch.no_grad():
                output = forward_fn(*broadcastable_kwargs.values())
            return output

        # Forward-mode differentiation evaluates the primal on the way, so the
        # jacobian pass returns both and the model is simulated once.
        jacobian_fn = self._jacobian(**non_broadcastable_kwargs)
        result = jacobian_fn(*broadcastable_kwargs.values())
        if isinstance(result, tuple):
            jacobian_output, output = result
        else:
            # an analytic ``_jacobian_engine`` yields no primal
            jacobian_output = result
            forward_fn = self._forward(**non_broadcastable_kwargs)
            output = forward_fn(*broadcastable_kwargs.values())

        return output, jacobian_output

    def forward(self, compile: bool = False) -> Callable:
        """
        Get forward method.

        Parameters
        ----------
        compile : bool, optional
            Compile function using ``torch.compile``. The default is ``False``.

        Returns
        -------
        Callable
            Forward method.

        """
        _kwargs = {**vars(self.properties), **vars(self.sequence)}

        non_broadcastable_kwargs = {
            k: v for k, v in _kwargs.items() if k not in self.broadcastable_params
        }

        _forward_fn = self._forward(**non_broadcastable_kwargs)

        # Update the signature of the forward_fn
        # Extract the signature of `_engine`
        engine_sig = inspect.signature(self._engine)
        forward_sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=engine_sig.parameters[param_name].default,
                )
                for param_name in self.broadcastable_params
            ]
        )

        def forward_fn(*args, **kwargs):
            # Get number of arguments
            n_args = len(args)

            # Create a dictionary of keyword arguments and their default values
            _kwargs = {}
            for k, v in forward_sig.parameters.items():
                if v.default is not inspect.Parameter.empty:
                    _kwargs[k] = v.default
                else:
                    _kwargs[k] = None

            # Merge the default keyword arguments with the provided kwargs
            for k in kwargs.keys():
                _kwargs[k] = kwargs[k]

            # Replace args
            _args = list(_kwargs.values())
            _args = list(args) + _args[n_args:]

            return _forward_fn(*_args)

        # Bind the new signature to the function
        forward_fn.__signature__ = forward_sig

        forward_fn = autocast(forward_fn)
        if compile:
            return torch.compile(forward_fn)
        return forward_fn

    def jacobian(self, compile: bool = False) -> Callable:
        """
        Get Jacobian method.

        Parameters
        ----------
        compile : bool, optional
            Compile function using ``torch.compile``. The default is ``False``.

        Returns
        -------
        Callable
            Jacobian method.

        """
        kwargs = {**vars(self.properties), **vars(self.sequence)}

        non_broadcastable_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.broadcastable_params
        }

        _jacobian_fn = self._jacobian(**non_broadcastable_kwargs)

        # Update the signature of the forward_fn
        # Extract the signature of `_engine`
        engine_sig = inspect.signature(self._engine)
        jacobian_sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=engine_sig.parameters[param_name].default,
                )
                for param_name in self.broadcastable_params
            ]
        )

        def jacobian_fn(*args, **kwargs):
            # Get number of arguments
            n_args = len(args)

            # Create a dictionary of keyword arguments and their default values
            _kwargs = {}
            for k, v in jacobian_sig.parameters.items():
                if v.default is not inspect.Parameter.empty:
                    _kwargs[k] = v.default
                else:
                    _kwargs[k] = None

            # Merge the default keyword arguments with the provided kwargs
            for k in kwargs.keys():
                _kwargs[k] = kwargs[k]

            # Replace args
            _args = list(_kwargs.values())
            _args = list(args) + _args[n_args:]

            result = _jacobian_fn(*_args)
            # The internal pass also returns the primal it linearized about;
            # this public method is documented to return the Jacobian alone.
            if isinstance(result, tuple):
                return result[0]
            return result

        # Bind the new signature to the function
        jacobian_fn.__signature__ = jacobian_sig

        jacobian_fn = autocast(jacobian_fn)
        if compile:
            return torch.compile(jacobian_fn)
        return jacobian_fn


def _forward_mode_jacobian(
    engine: Callable,
    argnums: int | tuple[int, ...],
    inputs: tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-voxel Jacobian of a vectorized engine, plus its primal output.

    Returns the Jacobian shaped ``(..., nparams, ncontrasts)``, alongside the
    primal the derivative was taken about. As with ``torch.func``, a bare
    integer ``argnums`` omits the parameter axis while a tuple keeps it.
    """
    single = isinstance(argnums, int)
    argnums = (argnums,) if single else tuple(argnums)
    # Broadcasting leaves zero-stride views behind, which forward-mode AD
    # cannot attach a tangent to, so materialize them first.
    primals = tuple(
        value.contiguous()
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value)
        for value in inputs
    )
    # Only floating-point inputs can carry a tangent; everything else is held
    # fixed by closure so that jvp never sees it.
    is_differentiable = [
        torch.is_tensor(value) and torch.is_floating_point(value)
        for value in primals
    ]
    tangent_primals = tuple(
        value for value, flag in zip(primals, is_differentiable) if flag
    )
    position = {}
    for index, flag in enumerate(is_differentiable):
        if flag:
            position[index] = len(position)

    def restricted(*values: torch.Tensor) -> torch.Tensor:
        remaining = iter(values)
        rebuilt = [
            next(remaining) if flag else value
            for value, flag in zip(primals, is_differentiable)
        ]
        return engine(*rebuilt)

    primal_output = None
    columns = []
    for argnum in argnums:
        if argnum not in position:
            raise ValueError(
                "cannot differentiate with respect to a non-floating input"
            )
        selected = position[argnum]
        tangents = tuple(
            torch.ones_like(value) if index == selected else torch.zeros_like(value)
            for index, value in enumerate(tangent_primals)
        )
        primal_output, directional = torch.func.jvp(
            restricted, tangent_primals, tangents
        )
        columns.append(directional)

    jacobian = columns[0] if single else torch.stack(columns, dim=-2)
    return jacobian, primal_output


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


def _leading_device(values: dict[str, Any]) -> torch.device:
    for value in values.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")
