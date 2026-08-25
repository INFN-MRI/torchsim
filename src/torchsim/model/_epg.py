"""Signal models over the fused state machine.

An :class:`EpgModel` declares which tissue properties it exposes and what
sequence it plays; the rest -- building the tissue, sizing the states, picking
the kernel -- follows from those two.

``properties`` maps the name a caller uses to the tissue field it fills, so a
model can keep the vocabulary its protocol is written in while the engine
keeps its own. A field the model does not name is never given to the tissue at
all, which is what leaves its term out of the kernel. A property mapped to
``None`` is the model's own -- a proton density a closed-form recovery applies
after the simulation rather than during it -- and reaches :meth:`evaluate`
without reaching the tissue.
"""

from __future__ import annotations

__all__ = ["EpgModel"]

from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from ..sequence import EpgEngine, SequenceDescription, TissueProperties
from ..sequence._parameters import TISSUE_NAMES
from ..sequence._simulation import RecordMode
from ._signal import SignalModel


class EpgModel(SignalModel):
    """A model whose signal comes from running a sequence description.

    Subclasses set :attr:`properties` -- ``{public name: tissue field}`` -- and
    implement :meth:`describe`.

    Attributes
    ----------
    simulator:
        The policy whose state matrix the sequence needs. The base policy reads
        what the description declares and serves anything.
    states:
        Configuration orders to carry, or ``None`` to size them from the
        winding the description asks for.
    repetitions:
        How many times the description is played into its own steady state.
    record:
        Which ADCs the signal holds -- every one, only those the description
        marks acquired, or only the echoes.
    fixed:
        Tissue fields the model pins rather than exposes, as ``{field: value}``.
        A spoiled train that samples at the pulse has no transverse
        magnetization for a T2 to relax, so it pins one instead of asking for
        it.
    """

    simulator: EpgEngine = EpgEngine()
    states: int | None = None
    repetitions: int = 1
    record: RecordMode = "all"
    fixed: Mapping[str, Any] = MappingProxyType({})

    @abstractmethod
    def describe(self, **sequence: Any) -> SequenceDescription:
        """Return the sequence this model plays.

        Public units are the caller's -- milliseconds and degrees -- and the
        conversion to the seconds and radians a description carries belongs
        here, where the sequence is written.
        """

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Run one simulation of the described sequence.

        ``nstates``, ``repetitions``, ``record`` and ``device`` describe the
        run and are taken here, each falling back to the attribute of the same
        name; everything else describes the sequence and reaches
        :meth:`describe`.
        """
        held = dict(sequence)
        settings = {
            "nstates": held.pop("nstates", self.states),
            "repetitions": held.pop("repetitions", self.repetitions),
            "record": held.pop("record", self.record),
            "device": held.pop("device", None),
        }
        return self.simulator.simulate(
            self.describe(**held), self.tissue(properties), **settings
        ).signal

    def tissue(self, properties: Mapping[str, Any]) -> TissueProperties:
        """Build the tissue, leaving everything undeclared at its identity.

        A property the model does not expose, and one it exposes that the
        caller left out, are both simply absent here -- so each reaches the
        gate as the scalar default :class:`TissueProperties` holds and the
        kernels leave its term out.

        Raises
        ------
        ValueError
            If :attr:`properties` names a field the tissue does not have.
        """
        fields = self._fields()
        values = dict(self.fixed)
        values.update(
            {
                fields[name]: value
                for name, value in properties.items()
                if fields.get(name) is not None
            }
        )
        return TissueProperties(**values)

    def _fields(self) -> dict[str, str]:
        """Return the public-name to tissue-field map this model declares."""
        declared = self.properties
        pairs = (
            dict(declared)
            if isinstance(declared, Mapping)
            else {name: name for name in declared}
        )
        unknown = {field for field in pairs.values() if field is not None}
        unknown |= set(self.fixed)
        unknown -= set(TISSUE_NAMES)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} declares unknown tissue: {sorted(unknown)}"
            )
        return pairs
