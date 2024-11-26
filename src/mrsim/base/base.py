"""Base simulator class"""

__all__ = ["AbstractModel"]

import inspect
import functools

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from .decorators import autocast, jacfwd


class AbstractModel(ABC):
    """Abstract base class for MRI simulation models with automated parameter handling."""

    def __init__(
        self,
        chunk_size: int,
        device: str | torch.device | None = None,
        diff: str | tuple[str] | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the model with automatic parameter segregation and engine setup.

        Parameters
        ----------
        chunk_size : int
            Number of samples to process in parallel.
        device : str | torch.device | None
            Device for computations (e.g., 'cpu', 'cuda').
        diff : str | tuple[str] | None
            Parameters to compute the jacobian with respect to.
        """
        self.chunk_size = chunk_size
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = torch.float32
        self.diff = diff

        # Extract broadcastable parameters
        self.broadcastable_params = set(
            inspect.signature(self.set_properties).parameters.keys()
        )

    @autocast
    def set_properties(self, *args, **kwargs):
        """
        Define broadcastable spin/environment parameters.
        """
        raise NotImplementedError("Subclasses must implement `set_properties`.")

    @autocast
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
        default_args = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        # Merge default values and user-provided keyword arguments
        merged_kwargs = {**default_args, **kwargs}

        # Replace first `len(user_args)` items with positional arguments
        parameter_names = list(signature.parameters.keys())
        for idx, arg in enumerate(args):
            merged_kwargs[parameter_names[idx]] = arg

        # Split into broadcastable and non-broadcastable
        broadcastable_args = {
            k: v for k, v in merged_kwargs.items() if k in self.broadcastable_params
        }
        non_broadcastable_args = {
            k: v for k, v in merged_kwargs.items() if k not in self.broadcastable_params
        }

        # Define a new engine that takes only broadcastable parameters explicitly
        def func(*args):
            # Map provided positional arguments to broadcastable parameter names
            broadcastable_mapping = dict(zip(broadcastable_args.keys(), args))
            # Merge broadcastable and non-broadcastable parameters
            combined_args = {**broadcastable_mapping, **non_broadcastable_args}
            # Call the original engine
            return _func(**combined_args)

        # Get argnums for diff
        if self.diff is not None:
            argnums = _get_argnums(self.diff, broadcastable_args)
        else:
            argnums = None

        return func, list(broadcastable_args.values()), argnums

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
        engine, broadcastable_args, _ = self._get_func(self._engine, *args, **kwargs)

        def vmapped_engine(*inputs):
            vmapped = torch.vmap(engine, chunk_size=self.chunk_size)
            return vmapped(*inputs)

        return functools.partial(vmapped_engine, *broadcastable_args)

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
            jac_engine, broadcastable_args, argnums = self._get_func(
                self._jacobian_engine, *args, **kwargs
            )

            def jacobian_engine(*inputs):
                vmapped_jac = torch.vmap(jac_engine, chunk_size=self.chunk_size)
                return vmapped_jac(*inputs)

        else:
            engine, broadcastable_args, argnums = self._get_func(
                self._engine, *args, **kwargs
            )

            def jacobian_engine(*inputs):
                jac_engine = jacfwd(argnums=argnums)(engine)
                vmapped_jac = torch.vmap(jac_engine, chunk_size=self.chunk_size)
                return vmapped_jac(*inputs)

        return functools.partial(jacobian_engine, *broadcastable_args)

    def __call__(self, *args, **kwargs):
        """
        Calls both forward and jacobian methods, returning output and jacobian for sequence optimization.

        Parameters
        ----------
        *args : Any
            Positional arguments for the simulation.
        **kwargs : Any
            Keyword arguments for the simulation.

        Returns
        -------
        tuple
            A tuple containing the forward output and jacobian output.
        """
        forward_fn = self.forward(*args, **kwargs)

        if self.diff is None:
            with torch.no_grad():
                output = forward_fn(*args, **kwargs)
            return output

        output = forward_fn(*args, **kwargs)

        jacobian_fn = self.jacobian(*args, **kwargs)
        jacobian_output = jacobian_fn(*args, **kwargs)

        return output, jacobian_output


# %% todo: move
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


# %%
# def SPGRModel(AbstractModel):

#     def __init__(
#         self,
#         batch_size: int,
#         device: str | torch.device | None = None,
#         diff: str | tuple[str] | None = None,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(batch_size, device, diff)

#         self._forward = self._engine
#         self._jacobian = complex_jacfwd(argmap)(self._engine)

#     @autocast
#     def set_properties(self, T1, T2star, M0, field_map=None, delta_cs=None): ...

#     @autocast
#     def set_sequence(self, alpha, TR, TE): ...


# # %% subroutines
# def _get_spgr_phase(T2star, TE, field_map, delta_cs):
#     """Additional SPGR phase factors."""
#     # Enable broadcasting
#     T2star = T2star.unsqueeze(-1)
#     delta_cs = delta_cs.unsqueeze(-1)
#     field_map = field_map.unsqueeze(-1)
#     TE = TE.unsqueeze(0)

#     # Compute total phase accrual
#     phi = 2 * math.pi * (delta_cs + field_map) * TE

#     # Compute signal dampening
#     exp_term = torch.exp(-TE / T2star)
#     exp_term = torch.nan_to_num(exp_term, nan=0.0, posinf=0.0, neginf=0.0)

#     return torch.exp(1j * phi) * exp_term


# def _spgr_engine(
#     T1,
#     T2star,
#     M0,
#     field_map,
#     delta_cs,
#     TR,
#     TE,
#     alpha,
# ):
#     # Unit conversion
#     T1 = T1 * 1e-3  # ms -> s
#     T2star = T2star * 1e-3  # ms -> s
#     TE = TE * 1e-3  # ms -> s
#     TR = TR * 1e-3  # ms -> s
#     alpha = torch.deg2rad(alpha)

#     # We are assuming Freeman-Hill convention for off-resonance map,
#     # so we need to negate to make use with this Ernst-Anderson-based implementation from Hoff
#     field_map = -1 * field_map

#     # divide-by-zero risk with PyTorch's nan_to_num
#     E1 = torch.exp(
#         -1
#         * torch.nan_to_num(
#             TR.unsqueeze(0) / T1.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0
#         )
#     )

#     # Precompute cos, sin
#     ca = torch.cos(alpha).unsqueeze(0)
#     sa = torch.sin(alpha).unsqueeze(0)

#     # Main calculation
#     den = 1 - E1 * ca
#     Mxy = M0 * ((1 - E1) * sa) / den
#     Mxy = torch.nan_to_num(Mxy, nan=0.0, posinf=0.0, neginf=0.0)

#     # Add additional phase factor for readout at TE.
#     signal = Mxy * _get_spgr_phase(T2star, TE, field_map, delta_cs)

#     # Move multi-contrast in front
#     signal = signal.unsqueeze(0)
#     signal = signal.swapaxes(0, -1)

#     return signal.squeeze().to(torch.complex64)
