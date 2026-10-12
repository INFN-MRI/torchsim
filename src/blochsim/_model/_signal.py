"""What a signal model owes, and what it is spared.

A model says three things: which properties it exposes, how it turns them into
a signal, and nothing else. Which kernel runs, how the work is cut across
memory and devices, and how derivatives are taken belong below.

Three rules earn their place here.

**A property stays as the caller gave it.** A model reaching the fused state
machine gets the terms it needs and no more, and the kernels decide that from
what was passed rather than by reducing over a buffer -- so a default widened
to the voxel count before the simulation sees it reads as a live map, and the
run pays for off-resonance, diffusion or flow it does not have. Nothing is
broadcast here except a property being differentiated, which has to be wide
enough for one directional derivative to cover every voxel.

**The caller's array library is the caller's.** Properties and sequence
arguments arrive in whatever they were written in -- NumPy, CuPy, torch -- and
are read through DLPack, over the same memory rather than a copy. The signal
goes back the same way. Everything between is torch, which is what the kernels
and the autograd graph are written against; see :mod:`blochsim.sequence._array`
for what that costs, and for the one thing it cannot carry back.

**Derivatives are taken in the mode that suits what is differentiated.** A
Bloch simulation records far more samples than it takes parameters, so a
derivative with respect to tissue is cheapest forward: :meth:`jacobian` issues
one directional derivative per property and gets every voxel from each. A
derivative of a scalar cost with respect to a sequence -- the acquisition
optimization problem -- is the other way round, and is plain autograd on the
returned signal. That path is deliberately not wrapped: the engine reads which
inputs carry a gradient and picks its kernel from that, and a layer here would
only hide it.
"""

from __future__ import annotations

__all__ = ["_SignalModel"]

import numbers
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import copy as shallow_copy
from types import MappingProxyType
from typing import Any

import torch

from .._derivative import forward_mode
from ..sequence._array import as_torch, brought, is_array, like
from ..sequence._parameters import PUBLIC_PROPERTIES

_NOTHING: Mapping[str, Any] = MappingProxyType({})


class _SignalModel(ABC):
    """Interface of a signal model: declared properties and a signal.

    A subclass supplies :attr:`properties` and :meth:`evaluate`.
    :meth:`simulate` returns the signal and :meth:`jacobian` its derivative
    with respect to tissue properties.

    Parameters
    ----------
    **values : float or array-like
        Tissue properties or sequence arguments to fix, under the names
        :meth:`simulate` takes. A call overrides them.

    Attributes
    ----------
    properties : mapping or sequence of str, default=()
        Names of the exposed tissue properties. A mapping is read for its
        keys.
    bound : mapping
        Arguments fixed on this model; a call may override them.
    """

    properties: Mapping[str, str | None] | Sequence[str] = ()
    bound: Mapping[str, Any] = _NOTHING

    def __init__(self, **values: Any) -> None:
        """Fix arguments on the model; see the class."""
        self.bound = self._fix(values)

    @abstractmethod
    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the signal recorded for these properties and sequence arguments.

        Parameters
        ----------
        properties : mapping
            ``{name: torch.Tensor}`` for the declared properties the caller
            gave, in public units (ms, degrees, Hz).
        **sequence
            Sequence arguments, as given.

        Returns
        -------
        torch.Tensor
            Signal, with the voxel axis first when properties are maps.
        """

    # -- what is fixed on the model -----------------------------------------

    @property
    def exposes(self) -> tuple[str, ...]:
        """Names of the declared properties."""
        return tuple(self.properties)

    @property
    def accepts(self) -> tuple[str, ...]:
        """Names of every property a call may give: the declared ones and all tissue fields."""
        return tuple(dict.fromkeys((*self.exposes, *PUBLIC_PROPERTIES)))

    def bind(self, **values: Any) -> Any:
        """Return a copy with further arguments fixed.

        Parameters
        ----------
        **values : float or array-like
            Tissue properties or sequence arguments, under the names
            :meth:`simulate` takes. They replace earlier fixed values.

        Returns
        -------
        Simulator
            The copy; ``self`` is unchanged.
        """
        held = shallow_copy(self)
        held.bound = {**self.bound, **self._fix(values)}
        return held

    def to(self, device: torch.device | str) -> Any:
        """Return a copy with the fixed arguments on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        Simulator
            The copy; ``self`` is unchanged.
        """
        moved = shallow_copy(self)
        moved.bound = _moved(self.bound, torch.device(device))
        return moved

    # -- the two derivative modes -------------------------------------------

    def simulate(self, **values: Any) -> torch.Tensor:
        """Return the recorded signal.

        Differentiating a cost of the result with :meth:`torch.Tensor.backward`
        gives gradients with respect to sequence arguments by the adjoint.

        Parameters
        ----------
        **values : float or array-like
            Tissue properties and sequence arguments. Properties broadcast to
            the voxel shape; values override those in :attr:`bound`.

        Returns
        -------
        torch.Tensor
            Complex signal, shaped ``(*voxel_shape, samples)`` for property
            maps. Returned in the caller's array library.
        """
        backend = self._backend(values)
        held, sequence = self._split({**self.bound, **values})
        signal = self._shaped(self.evaluate(held, **sequence), self._batch(held))
        return like(signal, backend)

    def jacobian(
        self, diff: str | Sequence[str], **values: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the signal and its derivative with respect to tissue properties.

        Forward mode, with one pass per differentiated property.

        Parameters
        ----------
        diff : str or sequence of str
            Property name(s). A single name gives no parameter axis; a
            sequence stacks one before the samples.
        **values : float or array-like
            Tissue properties and sequence arguments, as for :meth:`simulate`.

        Returns
        -------
        signal, jacobian : torch.Tensor
            The signal and its derivative, shaped
            ``(*voxel_shape, [parameters,] samples)`` for the Jacobian.

        Raises
        ------
        ValueError
            If ``diff`` names something the model does not expose, or
            something that carries no derivative.
        """
        names = (diff,) if isinstance(diff, str) else tuple(diff)
        backend = self._backend(values)
        held, sequence = self._split({**self.bound, **values})
        for name in names:
            if name not in held:
                raise ValueError(
                    f"{name!r} is not among the properties this call gave "
                    f"{type(self).__name__}: {sorted(held)}. Forward mode "
                    f"differentiates a property the tissue carries; for a "
                    f"sequence argument, build a cost and call backward()."
                )
            if not torch.is_floating_point(held[name]):
                raise ValueError(f"{name!r} does not carry a derivative")
        batch = self._batch(held)

        # Only the differentiated properties are widened, and only because one
        # directional derivative has to cover every voxel. Widening the rest
        # would report them as maps and put their terms back in the kernel.
        # Broadcasting also leaves zero-stride views, which forward mode cannot
        # attach a tangent to, hence the copy.
        primals = tuple(
            held[name].expand(batch).contiguous() if batch else held[name]
            for name in names
        )
        rest = {name: value for name, value in held.items() if name not in names}

        def along(*inputs: torch.Tensor) -> torch.Tensor:
            declared = {**rest, **dict(zip(names, inputs, strict=True))}
            return self._shaped(self.evaluate(declared, **sequence), batch)

        forward_mode()
        columns = []
        signal = None
        for name in names:
            tangents = tuple(
                torch.ones_like(value) if other == name else torch.zeros_like(value)
                for other, value in zip(names, primals, strict=True)
            )
            signal, column = torch.func.jvp(along, primals, tangents)
            columns.append(column)
        jacobian = columns[0] if isinstance(diff, str) else torch.stack(columns, dim=-2)
        return like(signal, backend), like(jacobian, backend)

    # -- what a subclass does not have to write -----------------------------

    def _backend(self, values: Mapping[str, Any]) -> Any:
        """Return the array namespace the caller's own arrays belong to.

        The first argument carrying a buffer decides; ``None`` -- a torch
        tensor, or nothing but plain numbers -- asks for no conversion.
        """
        return brought(values.values())

    def _fix(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Read values to hold on the model, as a call would read them.

        A declared property is read as torch here rather than at the call: one
        left as NumPy would decide the answer's array library, and the gradient
        a design loop needs does not survive the trip out of torch and back. A
        sequence argument keeps whatever it was given as, since a plain number
        carries more precision than a default-dtype tensor and a sequence's
        event times are accumulated from exactly these.
        """
        declared = set(self.exposes)
        return {
            name: as_torch(value) if name in declared or is_array(value) else value
            for name, value in values.items()
        }

    def _split(self, values: Mapping[str, Any]) -> tuple[dict, dict]:
        """Tell the property arguments from the sequence ones."""
        declared = self.accepts
        held = {name: as_torch(values[name]) for name in declared if name in values}
        sequence = {
            name: value for name, value in values.items() if name not in declared
        }
        return held, sequence

    def _batch(self, held: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        """Return the voxel shape the declared maps agree on.

        Empty when every property is a scalar, which is what makes a
        single-voxel call give one signal rather than a batch of one.
        """
        if not held:
            return ()
        return tuple(torch.broadcast_shapes(*(value.shape for value in held.values())))

    def _shaped(self, signal: torch.Tensor, batch: tuple[int, ...]) -> torch.Tensor:
        """Give the voxel axis the shape the caller's properties had."""
        voxels = 1
        for size in batch:
            voxels *= size
        if signal.ndim and signal.shape[0] == voxels:
            return signal.reshape(*batch, *signal.shape[1:])
        return signal


# %% private module subroutines


def _moved(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """The mapping with every array in it on ``device``, the rest untouched.

    Off the host, a list or tuple of numbers arrives as the tensor
    :func:`as_torch` makes of it; on the host it stays as the caller wrote it.
    """
    return {name: _on(value, device) for name, value in values.items()}


def _on(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    numbers_given = (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(entry, numbers.Real) for entry in value)
    )
    if numbers_given and device.type != "cpu":
        return as_torch(value).to(device)
    return value
