"""A simulator with the tissue it is being asked about already in place."""

from __future__ import annotations

__all__ = ["Acquisition"]

from collections.abc import Sequence
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

    def simulate(self, **design: Any) -> torch.Tensor:
        """Return what this acquisition records for ``design``."""
        return self.simulator.simulate(**self.properties, **design)

    def jacobian(
        self, diff: str | Sequence[str], **design: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the signal and its derivative with respect to ``diff``.

        Forward mode, one directional derivative per property named -- so the
        cost of a Fisher matrix is one pass per parameter, not one per voxel.
        """
        return self.simulator.jacobian(diff, **self.properties, **design)
