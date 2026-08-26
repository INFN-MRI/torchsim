"""A simulator with the tissue it is being asked about already in place."""

from __future__ import annotations

__all__ = ["Acquisition"]

from collections.abc import Mapping, Sequence
from copy import copy as shallow_copy
from typing import Any

import torch

from ..sequence._array import as_torch


class Acquisition:
    """A simulator with its tissue bound, ready to be asked about a design.

    Parameters
    ----------
    simulator:
        Any :class:`~torchsim.model.SignalModel` -- a state machine or a
        closed form, since both are called the same way.
    resolve:
        Whether to hold the protocol's structure fixed across calls. A design
        loop plays the same sequence with different numbers every iteration,
        so resolving it once and rebinding the values is worth roughly eight
        times the whole call; see
        :meth:`~torchsim.model.AbstractSimulator.resolved`. Turning this off
        rebuilds the event stream every iteration, which is slower and agrees
        to the last bit rather than to float32 round-off.
    properties:
        The tissue the sequence is being designed for, under the names the
        simulator exposes. Several design points are given as arrays, and the
        cost then sees one row per point.

    Examples
    --------
    .. code-block:: python

        shots = Acquisition(
            FSESimulator(ESP=5.0, TR=1800.0),
            T1=[900.0, 1400.0],
            T2=[50.0, 120.0],
        )
        signal = shots.simulate(flip=train)          # (shots, tissue, echo)
    """

    def __init__(
        self, simulator: Any, *, resolve: bool = True, **properties: Any
    ) -> None:
        resolving = resolve and hasattr(simulator, "resolved")
        self.simulator = simulator.resolved() if resolving else simulator
        # Read here rather than at the call: a property left as NumPy would
        # decide the answer's array library, and the gradient a design loop
        # needs does not survive the trip out of torch and back.
        self.properties = {
            name: as_torch(value) for name, value in properties.items()
        }

    @property
    def exposes(self) -> tuple[str, ...]:
        """The property names the simulator declares, in its own order."""
        return tuple(self.simulator.properties)

    def to(self, device: torch.device | str) -> Acquisition:
        """This acquisition, with everything it holds on ``device``.

        A simulator carries its protocol -- echo times, a flip train -- and an
        acquisition carries the tissue bound to it, and the two have to arrive
        on a card together: properties moved on their own would be multiplied
        against echo times still on the host.

        Parameters
        ----------
        device:
            Where to put it.

        Returns
        -------
        Acquisition
            A copy. This one is left where it was.
        """
        where = torch.device(device)
        moved = shallow_copy(self)
        moved.properties = _moved(self.properties, where)
        simulator = shallow_copy(self.simulator)
        if hasattr(simulator, "protocol"):
            simulator.protocol = _moved(simulator.protocol, where)
            # Whatever was resolved was resolved somewhere else, and holds
            # tensors that live there.
            simulator._described = None
            simulator._packing = None
        moved.simulator = simulator
        return moved

    def bound(self, **values: Any) -> Acquisition:
        """This acquisition with more tissue bound to it, or some of it changed.

        What a fit does with a property measured separately: the same sequence,
        with one more map held fixed on it.

        Parameters
        ----------
        values:
            Properties to add or replace, under the names the simulator
            exposes.

        Returns
        -------
        Acquisition
            A copy. This one is left as it was.
        """
        moved = shallow_copy(self)
        moved.properties = {
            **self.properties,
            **{name: as_torch(value) for name, value in values.items()},
        }
        return moved

    def simulate(self, **design: Any) -> torch.Tensor:
        """Return what this acquisition records for ``design``.

        A name the acquisition already holds may be given again, and the value
        given wins. That is what a fitting loop does: one sequence, a different
        tissue on every iteration.
        """
        return self.simulator.simulate(**{**self.properties, **design})

    def jacobian(
        self, diff: str | Sequence[str], **design: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the signal and its derivative with respect to ``diff``.

        Forward mode, one directional derivative per property named -- so the
        cost of a Fisher matrix is one pass per parameter, not one per voxel.
        A property the acquisition holds may be overridden here, as in
        :meth:`simulate`.
        """
        return self.simulator.jacobian(diff, **{**self.properties, **design})


# %% private module subroutines


def _moved(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """The mapping with every tensor in it on ``device``, the rest untouched."""
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in values.items()
    }
