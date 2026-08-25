"""Authoring a signal model over the fused state machine.

A model says three things: which tissue properties it exposes, what sequence
it plays, and how the two turn into a signal. Everything else -- which kernel
runs, how the work is cut across memory and devices, how derivatives are
taken -- belongs to the engine and is not the model author's to manage.

Two rules earn their place here.

**A property the model does not declare stays a scalar.** The kernels leave
out the terms a tissue gives nothing to do, and they decide that from what the
caller passed rather than by reducing over a buffer. A default broadcast to
the voxel count before the simulation sees it is therefore reported as live,
and the run pays for off-resonance, diffusion or flow it does not have. Only
the declared properties are broadcast.

**Derivatives are taken in the mode that suits what is being differentiated.**
A Bloch simulation records far more samples than it takes parameters, so a
derivative with respect to tissue is cheapest forward: :meth:`jacobian` issues
one directional derivative per property and gets every voxel from each. A
derivative of a scalar cost with respect to a sequence -- the acquisition
optimization problem -- is the other way round, and is plain autograd on the
returned signal. That path is deliberately not wrapped: the engine reads which
inputs carry a gradient and picks its kernel from that, and an extra layer
here would only hide it.
"""

from __future__ import annotations

__all__ = ["SignalModel"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..sequence import EpgSimulator, SequenceDescription, TissueProperties
from ..sequence._parameters import TISSUE_NAMES


class SignalModel(ABC):
    """One signal model: declared physics, a sequence, and a signal.

    Subclasses set :attr:`properties` and implement :meth:`describe`.

    Attributes
    ----------
    properties:
        The tissue this model exposes, as ``{public name: tissue field}``. A
        bare sequence of names is read as exposing those fields under their own
        names. Every field must be one of
        :data:`torchsim.sequence._parameters.TISSUE_NAMES`.
    simulator:
        The policy whose state matrix the sequence needs. The base policy
        reads what the description declares and serves anything.
    """

    properties: Mapping[str, str] | Sequence[str] = ()
    simulator: EpgSimulator = EpgSimulator()

    @abstractmethod
    def describe(self, **sequence: Any) -> SequenceDescription:
        """Return the sequence this model plays.

        Public units are the caller's -- milliseconds and degrees -- and the
        conversion to the seconds and radians a description carries belongs
        here, where the sequence is written.
        """

    # -- the two derivative modes -------------------------------------------

    def simulate(self, **values: Any) -> torch.Tensor:
        """Return the recorded signal.

        Tissue and sequence arguments may be given together; they are told
        apart by :attr:`properties`. A cost built on this and differentiated
        with :meth:`torch.Tensor.backward` reaches the engine's adjoint, which
        is the right mode for sequence parameters.
        """
        tissue, sequence = self._split(values)
        held = {name: torch.as_tensor(value) for name, value in tissue.items()}
        return self._simulate(held, sequence, self._batch(held))

    def jacobian(
        self, diff: str | Sequence[str], **values: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the signal and its derivative with respect to ``diff``.

        Forward mode, one directional derivative per differentiated property.
        Voxels are independent, so each pass yields every voxel's derivative
        at once and the cost is one pass per property rather than per voxel.

        Parameters
        ----------
        diff:
            One property name, or several. A single name collapses the
            parameter axis; a sequence keeps it, stacked before the sample
            axis.
        values:
            The tissue and sequence arguments, as for :meth:`simulate`.

        Returns
        -------
        tuple
            The signal, and its Jacobian.
        """
        wanted = (diff,) if isinstance(diff, str) else tuple(diff)
        single = isinstance(diff, str)
        tissue, sequence = self._split(values)
        held = {name: torch.as_tensor(value) for name, value in tissue.items()}
        for name in wanted:
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
        names = tuple(wanted)
        primals = tuple(
            held[name].expand(batch).reshape(-1).contiguous()
            if batch
            else held[name].reshape(()).contiguous()
            for name in names
        )
        rest = {name: value for name, value in held.items() if name not in wanted}

        def along(*inputs: torch.Tensor) -> torch.Tensor:
            declared = {**rest, **dict(zip(names, inputs, strict=True))}
            return self._simulate(declared, sequence, batch)

        columns = []
        signal = None
        for name in names:
            tangents = tuple(
                torch.ones_like(value) if other == name else torch.zeros_like(value)
                for other, value in zip(names, primals, strict=True)
            )
            signal, column = torch.func.jvp(along, primals, tangents)
            columns.append(column)
        return signal, columns[0] if single else torch.stack(columns, dim=-2)

    # -- what a subclass does not have to write -----------------------------

    def _fields(self) -> dict[str, str]:
        """Return the public-name to tissue-field map, however it was declared."""
        declared = self.properties
        pairs = (
            dict(declared)
            if isinstance(declared, Mapping)
            else {name: name for name in declared}
        )
        unknown = set(pairs.values()) - set(TISSUE_NAMES)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} declares unknown tissue: {sorted(unknown)}"
            )
        return pairs

    def _split(self, values: Mapping[str, Any]) -> tuple[dict, dict]:
        """Tell the tissue arguments from the sequence ones."""
        fields = self._fields()
        tissue = {name: values[name] for name in fields if name in values}
        sequence = {
            name: value for name, value in values.items() if name not in fields
        }
        return tissue, sequence

    def _batch(self, held: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        """Return the voxel shape the declared maps agree on.

        Empty when every property is a scalar, which is what makes a
        single-voxel call return one signal rather than a batch of one.
        """
        return tuple(
            torch.broadcast_shapes(*(value.shape for value in held.values()))
            if held
            else ()
        )

    def _tissue(self, tissue: Mapping[str, Any]) -> TissueProperties:
        """Build the tissue, leaving everything undeclared at its identity.

        Undeclared properties are never named here, and a declared one left at
        a scalar is passed as a scalar, so both reach the gate as the defaults
        :class:`TissueProperties` holds and the kernels leave their terms out.
        """
        fields = self._fields()
        return TissueProperties(
            **{fields[name]: value for name, value in tissue.items()}
        )

    def _simulate(
        self,
        tissue: Mapping[str, Any],
        sequence: Mapping[str, Any],
        batch: tuple[int, ...],
    ) -> torch.Tensor:
        """Run one simulation and give the voxel axis the caller's shape back."""
        described = self.describe(**sequence)
        signal = self.simulator.simulate(described, self._tissue(tissue)).signal
        voxels = 1
        for size in batch:
            voxels *= size
        if signal.shape[0] == voxels:
            return signal.reshape(*batch, *signal.shape[1:])
        return signal
