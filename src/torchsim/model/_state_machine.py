"""What the spins are, and what each kind of event does to them.

A signal model made of a state machine splits in two. A
:class:`StateMachineModel` says what a voxel holds -- which tissue properties
are exposed, and so which physics the kernels carry -- and what each kind of
event is realized as: whether an excitation is ideal or integrated from a
waveform, whether a readout is followed by an unbalanced gradient, by ideal
spoiling, or by nothing. An :class:`AbstractSimulator` says what order the
events are played in.

Splitting them is what lets one be changed without the other. The MRF timing
with a selective excitation, or a refocused train whose readout spoils rather
than winds, is an assignment rather than a new model.

**Triggers are resolved before a description exists.** A simulator binds each
slot when it is constructed; what a protocol then produces is an ordinary
:class:`~torchsim.sequence.SequenceDescription`, whose events carry their own
action word. From there the path is the fused one -- packing, the feature
mask, the real-subspace verdict, offload and sharding -- and nothing consults
a trigger again. There is no interpretation at run time and none per event.
"""

from __future__ import annotations

__all__ = [
    "AbstractSimulator",
    "BALANCED",
    "REFOCUSED",
    "SPOILED",
    "StateMachineModel",
    "Triggers",
    "UNBALANCED",
]

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from ..sequence import (
    EpgEngine,
    Operator,
    RfDefinition,
    SequenceDescription,
    TissueProperties,
    Delay,
    Excitation,
    FSEReadout,
    Inversion,
    Readout,
    Refocusing,
    SPGRReadout,
    SSFPFidReadout,
    Saturation,
    bSSFPReadout,
    compose,
    execution,
    ideal_rf_definition,
)
from ..sequence._array import brought, is_array, read
from ..sequence._parameters import TISSUE_NAMES
from ..sequence._simulation import RecordMode
from ._signal import SignalModel

_EMPTY: Mapping[str, Any] = MappingProxyType({})

# What a caller may name that describes the run rather than the sequence. Each
# has an attribute of the same name, set once at construction.
RUN_SETTINGS = ("nstates", "repetitions", "record", "device", "execution")


@dataclass(frozen=True)
class Triggers:
    """What each kind of event is realized as.

    Each field is an operator factory, called with the parameters the protocol
    has for that event. Assigning one is how a sequence says that its readouts
    wind the states on, or that its excitation is a shaped pulse rather than an
    ideal rotation.
    """

    excitation: Callable[..., Operator] = Excitation
    refocusing: Callable[..., Operator] = Refocusing
    inversion: Callable[..., Operator] = Inversion
    saturation: Callable[..., Operator] = Saturation
    readout: Callable[..., Operator] = Readout
    delay: Callable[..., Operator] = Delay


def _uncrushed(*args: Any, **kwargs: Any) -> Operator:
    """Return a refocusing pulse with no crushers, as a balanced sequence plays."""
    return Refocusing(*args, crushed=False, **kwargs)


#: A readout the repetition rewinds after, and refocusing pulses left uncrushed.
BALANCED = Triggers(readout=bSSFPReadout, refocusing=_uncrushed)
#: A readout followed by one unbalanced gradient.
UNBALANCED = Triggers(readout=SSFPFidReadout)
#: A readout followed by ideal transverse spoiling.
SPOILED = Triggers(readout=SPGRReadout)
#: A refocusing pulse between its crushers, and the sample at the echo centre.
REFOCUSED = Triggers(readout=FSEReadout)


@dataclass(frozen=True)
class StateMachineModel:
    """The physics: which properties a voxel has, and what an event does to it.

    Attributes
    ----------
    properties:
        ``{public name: tissue field}``. A field left unnamed is never given to
        the tissue, and the kernels leave its term out -- so this is how a
        model asks for off-resonance, diffusion, flow or a second pool. A name
        mapped to ``None`` is the model's own and reaches the signal without
        reaching the tissue.
    triggers:
        What each kind of event plays.
    fixed:
        Tissue fields the model pins rather than exposes, as
        ``{field: value}``.
    definitions:
        The RF resources the events name. The default is one ideal hard pulse
        at id 0; a model whose excitation is slice-selective supplies a shaped
        definition here instead.
    """

    properties: Mapping[str, str | None] = field(default_factory=lambda: _EMPTY)
    triggers: Triggers = Triggers()
    fixed: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    definitions: Mapping[int, RfDefinition] = field(
        default_factory=lambda: {0: ideal_rf_definition()}
    )

    def tissue(self, properties: Mapping[str, Any]) -> TissueProperties:
        """Build the tissue, leaving everything undeclared at its identity.

        A property the model does not expose, and one it exposes that the
        caller left out, are both simply absent -- so each reaches the gate as
        the scalar default :class:`TissueProperties` holds and the kernels
        leave its term out.

        Raises
        ------
        ValueError
            If a name is mapped to a field the tissue does not have.
        """
        values = dict(self.fixed)
        values.update(
            {
                self.fields[name]: value
                for name, value in properties.items()
                if self.fields.get(name) is not None
            }
        )
        return TissueProperties(**values)

    @property
    def fields(self) -> dict[str, str | None]:
        """The public-name to tissue-field map, checked against the tissue."""
        pairs = dict(self.properties)
        unknown = {field for field in pairs.values() if field is not None}
        unknown |= set(self.fixed)
        unknown -= set(TISSUE_NAMES)
        if unknown:
            raise ValueError(f"unknown tissue: {sorted(unknown)}")
        return pairs


class AbstractSimulator(SignalModel):
    """A protocol: what a sequence plays, and the physics behind it.

    Subclasses set :attr:`model` and implement :meth:`layout`, which returns
    the operators of one repetition in the order they are played. The protocol
    arguments are given to the constructor and may be overridden per call.

    A protocol with a closed form -- a steady state that needs no state
    machine -- overrides :meth:`evaluate` instead and never reaches
    :meth:`layout`. Its :attr:`model` then carries only the property
    declaration, since there are no events for triggers to realize. Both kinds
    are constructed and called the same way, which is what lets parameter
    inference and sequence optimization take either.

    Attributes
    ----------
    model:
        The physics behind the protocol.
    states:
        Configuration orders to carry, or ``None`` to size them from the
        winding the description asks for.
    """

    model: StateMachineModel = StateMachineModel()
    states: int | None = None

    def __init__(
        self,
        *,
        model: StateMachineModel | None = None,
        states: int | None = None,
        repetitions: int = 1,
        record: RecordMode = "all",
        execution: str | torch.device | Sequence[Any] | None = None,
        crusher_dephasing_rad: float = 0.0,
        voxel_size_m: float | None = None,
        **protocol: Any,
    ) -> None:
        """Bind the physics and the protocol this simulator plays.

        Parameters
        ----------
        model:
            The state-machine model, or ``None`` for the class's own.
        states:
            Configuration orders to carry.
        repetitions:
            How many times the description is played into its steady state.
        record:
            Which ADCs the signal holds.
        execution:
            Where to run -- ``"auto"`` to decide per call against what the
            devices have free, ``"cpu"``, or a device or list of devices.
            ``None`` follows whatever :func:`~torchsim.sequence.execution`
            block is in scope, which is what lets a caller decide instead.
        crusher_dephasing_rad, voxel_size_m:
            The unbalanced gradient the sequence plays, and the voxel it winds
            across. Their ratio is what diffusion is damped by and what flow
            turns each dephasing order through.
        protocol:
            The sequence arguments :meth:`layout` reads.
        """
        self.model = model if model is not None else type(self).model
        # Bound here rather than looked up per event: after this, what the
        # protocol produces is an ordinary description and no trigger is
        # consulted again.
        self.triggers = self.model.triggers
        self.properties = self.model.properties
        self.states = states if states is not None else type(self).states
        self.repetitions = repetitions
        self.record = record
        self.execution = execution
        self.crusher_dephasing_rad = crusher_dephasing_rad
        self.voxel_size_m = voxel_size_m
        # Read once, so a layout can be written in torch whatever the caller
        # brought, and so the answer knows where to go back to.
        self._brought = brought(protocol.values())
        self.protocol = read(
            {
                name: value
                for name, value in protocol.items()
                if name not in RUN_SETTINGS
            }
        )
        self._described: SequenceDescription | None = None

    def _backend(self, values: Mapping[str, Any]) -> Any:
        """Return the caller's array library, from the call or the constructor.

        A call carrying arrays of its own decides, even when they are torch --
        the tissue is what the answer is about. Only a call with no arrays at
        all falls back to what the protocol was written in.
        """
        if any(is_array(value) for value in values.values()):
            return super()._backend(values)
        return self._brought

    # -- what a protocol says -----------------------------------------------

    def layout(self, **protocol: Any) -> Sequence[Operator | tuple[Any, Operator]]:
        """Return the operators of one repetition, in the order they play.

        A bare operator starts where the one before it ended; one given as
        ``(offset_s, operator)`` starts that far into the repetition instead,
        which is how a sequence that times itself from an echo says so.

        Raises
        ------
        NotImplementedError
            If the subclass implements neither this nor :meth:`describe`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements neither layout() nor describe()"
        )

    def repetition_s(self, played_s: Any, **protocol: Any) -> Any:
        """Return how long one repetition lasts, given what the layout played.

        The default is the span the layout covers. A sequence whose TR is
        longer than what it plays -- a refocused train waiting out its
        recovery -- says so here, and only a run of more than one repetition
        can tell the difference.
        """
        del protocol
        return played_s

    def played(self, **sequence: Any) -> dict[str, Any]:
        """Return the protocol as it will be laid out.

        The constructor's arguments, with anything given at the call
        overriding them, every array read as torch. Anything naming a run
        setting is left out: those describe the run, and a layout has no use
        for them.
        """
        given = {
            name: value
            for name, value in sequence.items()
            if name not in RUN_SETTINGS
        }
        return {**self.protocol, **read(given)}

    def describe(self, **protocol: Any) -> SequenceDescription:
        """Return the description this protocol plays."""
        if self._described is not None:
            return self._described
        events, played_s = compose(*self.layout(**protocol))
        return SequenceDescription(
            subsequence_index=0,
            tr_duration_us=1e6 * self.repetition_s(played_s, **protocol),
            events=events,
            rf_definitions=dict(self.model.definitions),
            crusher_dephasing_rad=self.crusher_dephasing_rad,
            voxel_size_m=self.voxel_size_m,
        )

    @classmethod
    def from_description(
        cls,
        description: SequenceDescription,
        model: StateMachineModel,
        **settings: Any,
    ) -> AbstractSimulator:
        """Return a simulator over a stream someone else assembled.

        The events are already concrete -- they carry their own action word --
        so no layout is walked and no trigger is applied. This is the path a
        description arriving from a scanner takes.
        """
        simulator = _Described(model=model, **settings)
        simulator._described = description
        return simulator

    # -- what a signal model owes -------------------------------------------

    def evaluate(
        self, properties: Mapping[str, Any], **sequence: Any
    ) -> torch.Tensor:
        """Run one simulation of the described protocol.

        ``nstates``, ``repetitions``, ``record``, ``device`` and ``execution``
        describe the run and are taken here, each falling back to what the
        constructor was given; everything else overrides a protocol argument.
        """
        given = dict(sequence)
        states = given.pop("nstates", self.states)
        settings = {
            "repetitions": given.pop("repetitions", self.repetitions),
            "record": given.pop("record", self.record),
            "device": given.pop("device", None),
        }
        target = given.pop("execution", self.execution)
        described = self.describe(**self.played(**given))
        tissue = self.model.tissue(properties)
        block = nullcontext() if target is None else execution(target)
        with block:
            return EpgEngine().simulate(
                described, tissue, nstates=states, **settings
            ).signal


class _Described(AbstractSimulator):
    """A simulator whose description was handed to it whole."""
