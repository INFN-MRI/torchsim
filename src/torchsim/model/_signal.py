"""Authoring a signal model.

A model says three things: which properties it exposes, how it turns them into
a signal, and nothing else. Which kernel runs, how the work is cut across
memory and devices, and how derivatives are taken belong below.

Two rules earn their place here.

**A property stays as the caller gave it.** A model reaching the fused state
machine gets the terms it needs and no more, and the kernels decide that from
what was passed rather than by reducing over a buffer -- so a default widened
to the voxel count before the simulation sees it reads as a live map, and the
run pays for off-resonance, diffusion or flow it does not have. Nothing is
broadcast here except a property being differentiated, which has to be wide
enough for one directional derivative to cover every voxel.

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

__all__ = ["SignalModel"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class SignalModel(ABC):
    """One signal model: declared properties, and a signal.

    Subclasses set :attr:`properties` and implement :meth:`evaluate`.

    Attributes
    ----------
    properties:
        What the model exposes, as names. A mapping is read for its keys, so a
        subclass that also needs to say where each property goes -- as
        :class:`~torchsim.model.StateMachineModel` does, naming the tissue
        field each fills -- declares both at once.
    """

    properties: Mapping[str, str] | Sequence[str] = ()

    @abstractmethod
    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Return the signal these properties and this sequence record.

        ``properties`` holds the declared values the caller passed, under the
        names :attr:`properties` gives them, and holds nothing for those the
        caller left out. Public units are the caller's -- milliseconds and
        degrees -- and converting them belongs here, with the sequence.
        """

    # -- the two derivative modes -------------------------------------------

    def simulate(self, **values: Any) -> torch.Tensor:
        """Return the recorded signal.

        Property and sequence arguments are given together and told apart by
        :attr:`properties`. A cost built on this and differentiated with
        :meth:`torch.Tensor.backward` reaches the engine's adjoint, which is
        the mode that suits a sequence parameter.
        """
        held, sequence = self._split(values)
        return self._shaped(self.evaluate(held, **sequence), self._batch(held))

    def jacobian(
        self, diff: str | Sequence[str], **values: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the signal and its derivative with respect to ``diff``.

        Forward mode, one directional derivative per differentiated property.
        Voxels are independent, so each pass yields every voxel's derivative at
        once and the cost is one pass per property rather than per voxel.

        Parameters
        ----------
        diff:
            One property name, or several. A single name collapses the
            parameter axis; a sequence keeps it, stacked before the samples.
        values:
            The property and sequence arguments, as for :meth:`simulate`.

        Returns
        -------
        tuple
            The signal, and its Jacobian.

        Raises
        ------
        ValueError
            If ``diff`` names something the model does not expose, or
            something that carries no derivative.
        """
        names = (diff,) if isinstance(diff, str) else tuple(diff)
        held, sequence = self._split(values)
        for name in names:
            if name not in held:
                raise ValueError(
                    f"{name!r} is not a property {type(self).__name__} exposes"
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

        columns = []
        signal = None
        for name in names:
            tangents = tuple(
                torch.ones_like(value) if other == name else torch.zeros_like(value)
                for other, value in zip(names, primals, strict=True)
            )
            signal, column = torch.func.jvp(along, primals, tangents)
            columns.append(column)
        if isinstance(diff, str):
            return signal, columns[0]
        return signal, torch.stack(columns, dim=-2)

    # -- what a subclass does not have to write -----------------------------

    def _names(self) -> tuple[str, ...]:
        """Return the property names, however :attr:`properties` declares them."""
        return tuple(self.properties)

    def _split(self, values: Mapping[str, Any]) -> tuple[dict, dict]:
        """Tell the declared property arguments from the sequence ones."""
        declared = self._names()
        held = {
            name: torch.as_tensor(values[name])
            for name in declared
            if name in values
        }
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
        return tuple(
            torch.broadcast_shapes(*(value.shape for value in held.values()))
        )

    def _shaped(
        self, signal: torch.Tensor, batch: tuple[int, ...]
    ) -> torch.Tensor:
        """Give the voxel axis the shape the caller's properties had."""
        voxels = 1
        for size in batch:
            voxels *= size
        if signal.ndim and signal.shape[0] == voxels:
            return signal.reshape(*batch, *signal.shape[1:])
        return signal
