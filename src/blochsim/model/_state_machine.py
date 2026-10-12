"""What the spins are, and what each kind of event does to them.

A signal model made of a state machine splits in two. A
:class:`SpinPhysics` says what a voxel holds -- which tissue properties
are exposed, and so which physics the kernels carry -- and what each kind of
event is realized as: whether an excitation is ideal or integrated from a
waveform, whether a readout is followed by an unbalanced gradient, by ideal
spoiling, or by nothing. A :class:`Simulator` says what order the
events are played in.

Splitting them is what lets one be changed without the other. The MRF timing
with a selective excitation, or a refocused train whose readout spoils rather
than winds, is an assignment rather than a new model.

**The operators are resolved before a description exists.** A simulator binds
each slot when it is constructed; what a protocol then produces is an ordinary
:class:`~blochsim.sequence.SequenceDescription`, whose events carry their own
action word. From there the path is the fused one -- packing, the feature
mask, the real-subspace verdict, offload and sharding -- and nothing consults
an operator slot again. There is no interpretation at run time and none per
event.

Three vocabularies name a pulse along that path, and they are not the same
vocabulary at three sizes. :class:`EventOperators` has one slot per role a
sequence is written in terms of, and its values are operator factories, read
only while a description is being assembled. Each factory emits events tagged
with an :class:`~blochsim.sequence.RfUse`, which is what a Pulseq file
carries, and with an :class:`~blochsim.sequence.EventAction`, which is the bit
field the kernels read. The three do not line up one to one and are not meant
to: the :attr:`~EventOperators.saturation` slot plays a pulse tagged
``RfUse.EXCITATION``, because what the scanner is told about a pulse and what
role the sequence gives it are separate questions.
"""

from __future__ import annotations

__all__ = [
    "Simulator",
    "BALANCED",
    "REFOCUSED",
    "SPOILED",
    "SpinPhysics",
    "EventOperators",
    "UNBALANCED",
]

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import copy as shallow_copy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch

from ..sequence import (
    Delay,
    EpgEngine,
    EventType,
    Excitation,
    FSEReadout,
    Inversion,
    Operator,
    Readout,
    Refocusing,
    RfDefinition,
    RfUse,
    Saturation,
    SequenceDescription,
    SequenceEvent,
    ShimDefinition,
    SPGRReadout,
    SSFPFidReadout,
    TissueProperties,
    bSSFPReadout,
    compose,
    execution,
    ideal_rf_definition,
)
from ..sequence._array import brought, is_array, read
from ..sequence._parameters import PROPERTY_NAMES, PUBLIC_PROPERTIES
from ..sequence._simulation import RecordMode, target_device
from ..sequence._transition import across_the_slice
from ._binding import Packing, bind, run_key
from ._signal import _moved, _SignalModel

_EMPTY: Mapping[str, Any] = MappingProxyType({})

# Stands for a setting the caller did not name, so that the class body is
# what decides -- None being a value several of them take.
_UNSET: Any = object()

# What a caller may name that describes the run rather than the sequence, at
# the constructor, on bind() or at the call, each spelled the same in all
# three. Two are not held under the name they arrive by: "device" belongs to
# one launch, and "pulse" is substituted into the physics.
RUN_SETTINGS = (
    "states",
    "repetitions",
    "record",
    "device",
    "execution",
    "pulse",
    "shims",
    "across_slice",
)

# The raster :class:`~blochsim.sequence.EpgEngine` reads a pulse's shape on,
# named here because a packing resolved against one is not valid against
# another.
_RF_RASTER_TIME_S = 1e-6


def realised(
    description: SequenceDescription, physics: SpinPhysics
) -> SequenceDescription:
    """An arriving event stream, re-emitted through a model's own handlers.

    The transport carries RF pulses and ADC windows and no gradients, so a
    description that arrives says what was played and not how the sequence
    dephased between one event and the next. That belongs to the sequence
    family rather than to the stream, and it is what the operators hold: a
    refocusing pulse brings its crusher pair, an unbalanced sample winds an
    order after it, a spoiled one discards the transverse states.

    So each event is played back through the operator its kind and its RF use
    name, at the timestamp it arrived with. A gradient hangs off a pulse or a
    sample, which is what the wire carries; one standing on its own -- the
    spoiler of a preparation written by hand -- has no handler to reinstate it
    and does not survive, because an arriving stream could not have held it
    either. A pulse whose use is one the
    handlers have no reading for -- a preparation, or an untagged one -- is
    emitted as it stands, with no gradient behaviour added, since guessing one
    is how a stream comes back as the wrong sequence.
    """
    parts: list[tuple[Any, Any]] = []
    for event in description.events:
        when = float(event.timestamp_us) * 1e-6
        if event.type is EventType.RF:
            parts.append((when, _rf_operator(event, physics.operators)))
        elif event.type is EventType.ADC:
            parts.append((when, _adc_operator(event, physics.operators)))
    if not parts:
        return description
    events, _played_s = compose(*parts)
    return replace(description, events=events)


#: The engine's spelling for a setting this layer takes under another name.
_RENAMED = MappingProxyType({"nstates": "states"})


def _no_renamed(values: Mapping[str, Any]) -> None:
    """Refuse a run setting under the name the engine takes it by.

    Held quietly instead, it would become a protocol argument and reach a
    layout that has no parameter for it -- or a closed form that ignores it,
    and answers with the wrong number of orders.
    """
    for given in _RENAMED.keys() & values.keys():
        raise TypeError(
            f"{given!r} is EpgEngine's name for this setting; a simulator "
            f"takes it as {_RENAMED[given]!r}, on the constructor, on bind() "
            f"or at the call"
        )


def _named(given: Any, declared: Any) -> Any:
    """The value the caller named, or the one the class body declares."""
    return declared if given is _UNSET else given


def _accepted(handler: Any, **offered: Any) -> dict[str, Any]:
    """The offered arguments this handler has somewhere to put.

    The handlers differ in what a stream can tell them: a readout that fixes
    the role it records at takes none, an inversion is a pulse whose flip is
    its own. Passing what a handler does not take is how a stream that carries
    more than one family of sequence stops working on the second one.
    """
    from inspect import signature

    takes = signature(handler).parameters
    return {name: value for name, value in offered.items() if name in takes}


def _adc_operator(event: SequenceEvent, operators: EventOperators) -> Any:
    """The sample this model's readout makes of an ADC window."""
    return operators.readout(
        event.adc_phase_rad,
        **_accepted(operators.readout, role=event.adc_role, is_echo=event.is_echo),
    )


def _rf_operator(event: SequenceEvent, operators: EventOperators) -> Any:
    """The operator a pulse's own ``use`` tag names."""
    handler = {
        RfUse.REFOCUSING: operators.refocusing,
        RfUse.INVERSION: operators.inversion,
        RfUse.SATURATION: operators.saturation,
    }.get(event.rf_use, operators.excitation)
    offered = _accepted(
        handler,
        flip_rad=event.rf_amplitude_hz,
        phase_rad=event.rf_phase_rad,
        definition_id=event.rf_definition_id,
        frequency_hz=event.rf_frequency_hz,
        offset_hz=event.rf_frequency_hz,
        shim_id=event.rf_shim_id,
    )
    return handler(**offered)


def replace_pulse(simulator: Simulator, pulse: RfDefinition) -> Simulator:
    """A copy of ``simulator`` whose events drive ``pulse``.

    The shipped operators name definition zero, so substituting there is what
    makes a shaped pulse the one they play.
    """
    held = shallow_copy(simulator)
    held.model = replace(simulator.model, definitions={0: replace(pulse, id=0)})
    held._packing = None
    held._described = None
    return held


#: The kinds of event a model says what to do with.
_HANDLER_SLOTS = (
    "excitation",
    "refocusing",
    "inversion",
    "saturation",
    "readout",
    "delay",
)


@dataclass(frozen=True)
class EventOperators:
    """Operator factories, one per kind of event.

    Each field is called with the protocol's parameters for that event and
    decides which :class:`~blochsim.sequence.RfUse` and
    :class:`~blochsim.sequence.EventAction` its events carry. Assigning one
    selects, for example, a readout followed by an unbalanced gradient or a
    shaped excitation. The presets :data:`BALANCED`, :data:`UNBALANCED`,
    :data:`SPOILED` and :data:`REFOCUSED` cover the standard sequence
    families.

    Parameters
    ----------
    excitation : callable, default=Excitation
        Plays a pulse that tips magnetization into the transverse plane.
    refocusing : callable, default=Refocusing
        Plays a refocusing pulse, with the crusher gradients it sits between.
    inversion : callable, default=Inversion
        Plays an inversion and holds the timeline for the recovery after it.
    saturation : callable, default=Saturation
        Plays a saturation pulse.
    readout : callable, default=Readout
        Plays an ADC sample and the rest of its repetition: whether the
        gradient after it is balanced, unbalanced (shifting every state by one
        order) or followed by spoiling.
    delay : callable, default=Delay
        Plays a wait.
    """

    #: What a pulse that tips magnetization into the transverse plane plays.
    excitation: Callable[..., Operator] = Excitation
    #: What a refocusing pulse plays, including the gradients it sits between.
    refocusing: Callable[..., Operator] = Refocusing
    #: What an inversion plays, and the recovery it holds the timeline for.
    inversion: Callable[..., Operator] = Inversion
    #: What a saturation pulse plays.
    saturation: Callable[..., Operator] = Saturation
    #: What a sample and the rest of its repetition play -- which is where a
    #: sequence says whether it winds the states on, spoils them, or rewinds
    #: them.
    readout: Callable[..., Operator] = Readout
    #: What a wait plays, which is nothing but time.
    delay: Callable[..., Operator] = Delay


def _uncrushed(*args: Any, **kwargs: Any) -> Operator:
    """Return a refocusing pulse with no crushers, as a balanced sequence plays."""
    return Refocusing(*args, crushed=False, **kwargs)


#: A readout the repetition rewinds after, and refocusing pulses left uncrushed.
BALANCED = EventOperators(readout=bSSFPReadout, refocusing=_uncrushed)
#: A readout followed by one unbalanced gradient.
UNBALANCED = EventOperators(readout=SSFPFidReadout)
#: A readout followed by ideal transverse spoiling.
SPOILED = EventOperators(readout=SPGRReadout)
#: A refocusing pulse between its crushers, and the sample at the echo centre.
REFOCUSED = EventOperators(readout=FSEReadout)


@dataclass(frozen=True)
class SpinPhysics:
    """The tissue properties a model exposes and the operators that realize its events.

    A :class:`Simulator` subclass assigns one as its ``model``. The properties
    are the keys a caller may pass to :meth:`Simulator.simulate`; the
    operators decide what each kind of event does to the spins (ideal or
    shaped excitation, balanced, unbalanced or spoiled readout).

    Parameters
    ----------
    properties : mapping, default={}
        ``{public name: tissue field}``. A name mapped to ``None`` reaches the
        signal without reaching the tissue, as for a closed form. Tissue
        fields not named here are still accepted under their public names.
    operators : EventOperators, default=EventOperators()
        Operator factory for each kind of event: one of :data:`BALANCED`,
        :data:`UNBALANCED`, :data:`SPOILED`, :data:`REFOCUSED`, or a custom
        :class:`EventOperators`.
    fixed : mapping, default={}
        ``{tissue field: value}`` pinned rather than exposed, such as the
        ``t2_ms`` a spoiled sequence does not depend on.
    definitions : mapping, default={0: ideal hard pulse}
        RF definitions by id, in :class:`~blochsim.sequence.RfDefinition`
        form. The shipped operators play definition 0.
    """

    properties: Mapping[str, str | None] = field(default_factory=lambda: _EMPTY)
    operators: EventOperators = field(default_factory=EventOperators)
    fixed: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    definitions: Mapping[int, RfDefinition] = field(
        default_factory=lambda: {0: ideal_rf_definition()}
    )

    def tissue(self, properties: Mapping[str, Any]) -> TissueProperties:
        """Return the :class:`~blochsim.sequence.TissueProperties` for ``properties``.

        Fields without a value stay at their scalar defaults, so the kernels
        omit their terms. Values in :attr:`fixed` are applied first and are
        overridden by a given property.

        Parameters
        ----------
        properties : mapping
            ``{public name: value}``, in public units (ms, degrees, Hz).

        Returns
        -------
        TissueProperties
            The values under the tissue's field names.

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
        """The ``{public name: tissue field}`` map, validated against the tissue.

        Every tissue field is nameable; ``properties`` adds model-specific
        names.

        Raises
        ------
        ValueError
            If a name maps to a field the tissue does not have.
        """
        pairs = {**PUBLIC_PROPERTIES, **self.properties}
        unknown = {field for field in pairs.values() if field is not None}
        unknown |= set(self.fixed)
        unknown -= set(PROPERTY_NAMES)
        if unknown:
            raise ValueError(f"unknown tissue: {sorted(unknown)}")
        return pairs


class Simulator(_SignalModel):
    """Base class of a pulse sequence protocol and its signal model.

    A subclass sets :attr:`model`, which says which tissue properties are
    exposed and how each kind of event is played, and implements
    :meth:`layout`, which returns the events of one repetition from the
    protocol parameters. A closed-form signal implements :meth:`evaluate`
    instead. :meth:`simulate` returns the complex signal of the ADC events
    recorded, and :meth:`jacobian` its derivative with respect to tissue
    properties.

    The constructor fixes the keywords :meth:`simulate` takes, as protocol
    parameters, tissue properties or run settings; a call overrides them.
    Subclasses document their own protocol parameters; the run settings below
    are common to all of them, and a subclass may change their class-level
    defaults.

    Parameters
    ----------
    **protocol : float or array-like
        Protocol parameters (the arguments of :meth:`layout`) and tissue
        properties to fix. Arrays may be NumPy, CuPy or torch; the signal is
        returned in the library they were given in.
    model : SpinPhysics, default=None
        Replaces the class's :attr:`model`, and with it the exposed tissue
        properties and the event operators.
    states : int, default=None
        Configuration orders carried by the extended-phase-graph simulation.
        ``None`` sizes them from the dephasing in the sequence.
    repetitions : int or "auto", default=None
        Playings of the sequence; the last is recorded. ``None`` takes the
        class's own (1 unless it declares otherwise); ``"auto"`` extrapolates
        the steady state from a few playings.
    record : {"all", "acquired", "echo"}, default="all"
        Which ADC events the signal holds: every one, those that acquire data,
        or those flagged as echoes.
    execution : str, torch.device or sequence of device, default=None
        ``"auto"``, ``"cpu"`` or CUDA device(s). ``None`` follows the
        enclosing :func:`~blochsim.execution` block.
    pulse : RfDefinition, default=None
        RF waveform replacing the ideal hard pulse of definition 0, which the
        shipped operators play.
    shims : mapping, default=None
        Transmit shims by ``shim_id``. By default one channel drives every
        pulse alike.
    across_slice : int or ExactSliceProfile, default=None
        Number of positions across the slice at which a shaped ``pulse`` is
        integrated, or an ``exact_slice_profile()``. ``None`` uses the slice
        centre only.
    resolve : bool, default=True
        Reuse the packed event structure across calls that change only
        values. ``False`` rebuilds it every call.
    crusher_dephasing_rad : float, default=0.0
        Dephasing of the unbalanced gradient across one voxel, in rad.
    voxel_size_m : float, default=None
        Voxel size along the unbalanced gradient, in m.

    Tissue properties
    -----------------
    T1, T2 : float or array-like
        Longitudinal and transverse relaxation times, in ms.
    M0 : float or array-like, default=1.0
        Equilibrium magnetization.
    B1 : float or array-like, default=1.0
        Transmit field, as a fraction of the nominal flip angle.
    B1phase : float or array-like, default=0.0
        Transmit phase, in rad.
    B0 : float or array-like, default=0.0
        Off-resonance, in Hz.
    T2prime : float or array-like, default=inf
        Reversible dephasing time, in ms.
    inv_efficiency : float or array-like, default=1.0
        Fraction of a perfect inversion achieved by an inversion pulse.
    D : float or array-like, default=0.0
        Apparent diffusion coefficient, in um^2/ms.
    v : float or array-like, default=0.0
        Velocity along the unbalanced gradient, in m/s.
    bound_fraction, bound_exchange, bound_T1 : float or array-like
        Semisolid pool for magnetization transfer: fraction, exchange rate
        in Hz and T1 in ms. A zero fraction leaves the pool out.
    poolB_fraction, poolB_exchange, poolB_T1, poolB_T2, poolB_shift : float or array-like
        A second exchanging water pool: fraction, exchange rate in Hz, T1 and
        T2 in ms, chemical shift in Hz. Pools C to E take ``poolC_*`` to ``poolE_*``.

    Attributes
    ----------
    model : SpinPhysics
        Tissue properties and event operators.
    protocol : mapping
        Protocol parameters fixed by the constructor, as torch tensors.
    """

    model: SpinPhysics = SpinPhysics()
    states: int | None = None
    # How many playings a sequence needs to reach the state a scanner plays it
    # in. One is the transient from equilibrium, which is what a scanner plays
    # once and never again; a sequence whose own physics says otherwise
    # overrides this.
    repetitions: int = 1
    record: RecordMode = "all"
    execution: str | torch.device | Sequence[Any] | None = None
    across_slice: Any = None
    crusher_dephasing_rad: float = 0.0
    voxel_size_m: float | None = None

    # Every attribute the constructor sets has a value here as well, so a
    # closed form that writes an __init__ of its own -- taking the two blocks
    # it concatenates, say -- still answers simulate() without chaining to
    # this one. What such a subclass declares in its class body is then the
    # whole of what it is.
    protocol: Mapping[str, Any] = _EMPTY
    shims: Mapping[int, ShimDefinition] = MappingProxyType({})
    _brought: Any = None
    _described: SequenceDescription | None = None
    _packing: Packing | None = None
    _refused: Sequence[Any] = ()
    _resolving: bool = True

    def __new__(cls, *args: Any, **kwargs: Any) -> Simulator:
        """Refuse to instantiate :class:`Simulator` itself."""
        if cls is Simulator:
            raise TypeError(
                "Simulator is what a sequence is written against, not a "
                "sequence: subclass it and implement layout() for a train of "
                "events, or evaluate() for a closed form"
            )
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Move operator slots named in the class body into ``model.operators``.

        Slots not named keep the base class's operators. A ``model`` defined in
        the class body takes precedence.
        """
        super().__init_subclass__(**kwargs)
        named = {
            slot: cls.__dict__[slot] for slot in _HANDLER_SLOTS if slot in cls.__dict__
        }
        if named and "model" not in cls.__dict__:
            cls.model = replace(
                cls.model, operators=replace(cls.model.operators, **named)
            )
            for slot in named:
                delattr(cls, slot)
        # The physics names what the model exposes, unless the class body did
        # -- which is what a model composing others, with no physics of its
        # own, does.
        if "properties" not in cls.__dict__:
            cls.properties = cls.model.properties

    def __init__(
        self,
        *,
        model: SpinPhysics | None = None,
        states: int | None = None,
        repetitions: int | str | None = None,
        record: RecordMode = _UNSET,
        execution: str | torch.device | Sequence[Any] | None = _UNSET,
        pulse: RfDefinition | None = None,
        shims: Mapping[int, ShimDefinition] | None = None,
        across_slice: Any = _UNSET,
        resolve: bool = True,
        crusher_dephasing_rad: float = _UNSET,
        voxel_size_m: float | None = _UNSET,
        **protocol: Any,
    ) -> None:
        """Fix the protocol, tissue properties and run settings; see the class."""
        _no_renamed(protocol)
        self.model = model if model is not None else type(self).model
        if pulse is not None:
            self.model = replace(self.model, definitions={0: replace(pulse, id=0)})
        if model is not None:
            # A physics given here names the properties; otherwise the class
            # body already said them, outright or through its own physics.
            self.properties = self.model.properties
        self.states = states if states is not None else type(self).states
        self.repetitions = (
            repetitions if repetitions is not None else type(self).repetitions
        )
        # Each falls back to what the class body says, which is where a
        # sequence that always records one shot or always spans one slice
        # states it once rather than in a constructor of its own.
        self.record = _named(record, type(self).record)
        self.execution = _named(execution, type(self).execution)
        self.crusher_dephasing_rad = _named(
            crusher_dephasing_rad, type(self).crusher_dephasing_rad
        )
        self.voxel_size_m = _named(voxel_size_m, type(self).voxel_size_m)
        self.shims = dict(shims) if shims else {}
        self.across_slice = across_the_slice(
            _named(across_slice, type(self).across_slice)
        )
        self._resolving = bool(resolve)
        self._packing = None
        self._refused = ()
        # Read once, so a layout can be written in torch whatever the caller
        # brought, and so the answer knows where to go back to.
        self._brought = brought(protocol.values())
        # Split the way a call is split, so the constructor takes exactly what
        # simulate() takes and fixes it.
        declared = set(self.accepts)
        self.bound = self._fix(
            {name: value for name, value in protocol.items() if name in declared}
        )
        self.protocol = read(
            {
                name: value
                for name, value in protocol.items()
                if name not in declared and name not in RUN_SETTINGS
            }
        )
        self._described = None

    def bind(self, **values: Any) -> Simulator:
        """Return a copy with further arguments fixed.

        Parameters
        ----------
        **values
            Tissue properties, protocol arguments and run settings (those of
            the constructor, e.g. ``pulse`` and ``states``), under the names
            :meth:`simulate` takes. Fixed values replace earlier ones.

        Returns
        -------
        Simulator
            The copy; ``self`` is unchanged.
        """
        _no_renamed(values)
        settings = {name: values.pop(name) for name in RUN_SETTINGS if name in values}
        held = super().bind(**values)
        pulse = settings.pop("pulse", None)
        if pulse is not None:
            held = replace_pulse(held, pulse)
        if "across_slice" in settings:
            held.across_slice = across_the_slice(settings.pop("across_slice"))
        for name, value in settings.items():
            setattr(held, name, value)
        return held

    @property
    def operators(self) -> EventOperators:
        """The :class:`EventOperators` of :attr:`model`, read by :meth:`layout`."""
        return self.model.operators

    @property
    def variables(self) -> tuple[str, ...]:
        """Names of the :meth:`layout` parameters: the protocol a call takes."""
        from inspect import Parameter, signature

        return tuple(
            name
            for name, parameter in signature(self.layout).parameters.items()
            if parameter.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
        )

    def to(self, device: torch.device | str) -> Simulator:
        """Return a copy with its protocol and fixed properties on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        Simulator
            The copy; ``self`` is unchanged.
        """
        moved = super().to(device)
        moved.protocol = _moved(moved.protocol, torch.device(device))
        # Whatever was resolved was resolved somewhere else, and holds tensors
        # that live there.
        moved._packing = None
        moved._refused = ()
        return moved

    def _structure(
        self,
        played: Mapping[str, Any],
        tissue: TissueProperties,
        *,
        repetitions: int | str,
        record: str,
        device: Any,
        across_slice: Any = None,
    ) -> tuple[SequenceDescription, Any]:
        """The description to run, and its events already packed if they are."""
        if not self._resolving or self._described is not None:
            return self.describe(**played), None
        if across_slice is not None:
            # A packing holds the event stream and not the table a pulse is
            # integrated over, so a profiled run walks the description instead
            # of rebinding onto a packing that has no table in it.
            return self.describe(**played), None
        if not isinstance(repetitions, int):
            # How many playings a settled run takes is decided against the
            # tissue it is given, so there is no one packing to hold fixed.
            return self.describe(**played), None
        where = target_device(tissue, device)
        settings = {
            "repetitions": repetitions,
            "record": record,
            "rf_raster_time_s": self.rf_raster_time_s,
        }
        key = run_key(played, device=where, **settings)
        if self._packing is not None and self._packing.matches(key):
            return self._packing.description, self._packing.pack(played)
        if any(key == refused for refused in self._refused):
            return self.describe(**played), None
        packing = bind(self, played, device=where, **settings)
        if packing is None:
            self._refused = (*self._refused, key)
            return self.describe(**played), None
        self._packing = packing
        return packing.description, packing.pack(played)

    def _split(self, values: Mapping[str, Any]) -> tuple[dict, dict]:
        """Tell the property arguments from the sequence ones.

        The protocol the constructor fixed joins the sequence arguments here,
        so a closed form reads everything it was written with from what it is
        handed, whichever side gave it. A call naming one again wins.
        """
        _no_renamed(values)
        held, sequence = super()._split(values)
        return held, {**self.protocol, **sequence}

    def _backend(self, values: Mapping[str, Any]) -> Any:
        """Return the caller's array library, from the call or the constructor.

        A call carrying arrays of its own decides, even when they are torch --
        the tissue is what the answer is about. Only a call with no arrays at
        all falls back to what the simulator was built from.
        """
        if any(is_array(value) for value in values.values()):
            return super()._backend(values)
        return self._brought

    # -- what a protocol says -----------------------------------------------

    def layout(self, **protocol: Any) -> Sequence[Operator | tuple[Any, Operator]]:
        """Return the operators of one repetition in playing order.

        A bare operator starts where the previous one ended; an
        ``(offset_s, operator)`` pair starts ``offset_s`` seconds into the
        repetition. A subclass overrides this with the protocol parameters as
        keyword-only arguments, in ms and degrees, and documents them on the
        class.

        Parameters
        ----------
        **protocol
            The arguments named by :attr:`variables`.

        Returns
        -------
        sequence of Operator or (float, Operator)
            Operators from :attr:`operators`.

        Raises
        ------
        NotImplementedError
            If the subclass implements neither this nor :meth:`describe`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} implements neither layout() nor describe()"
        )

    def repetition_s(self, played_s: Any, **protocol: Any) -> Any:
        """Return the duration of one repetition in seconds.

        Defaults to ``played_s``, the span the layout covers. Override when
        the repetition time exceeds it.

        Parameters
        ----------
        played_s : float or torch.Tensor
            Span covered by the layout, in seconds.
        **protocol
            The arguments of :meth:`layout`.
        """
        del protocol
        return played_s

    def played(self, **sequence: Any) -> dict[str, Any]:
        """Return the protocol arguments of :meth:`layout`, arrays as torch tensors.

        The constructor's protocol overridden by ``sequence``; run settings are
        dropped.

        Parameters
        ----------
        **sequence
            Protocol arguments of this call.
        """
        given = {
            name: value for name, value in sequence.items() if name not in RUN_SETTINGS
        }
        return {**self.protocol, **read(given)}

    @property
    def rf_raster_time_s(self) -> float:
        """RF raster time of the pulse shapes, in seconds."""
        if self._described is not None:
            return self._described.rf_raster_time_s
        return _RF_RASTER_TIME_S

    def describe(self, **protocol: Any) -> SequenceDescription:
        """Return the :class:`~blochsim.sequence.SequenceDescription` of a protocol.

        Parameters
        ----------
        **protocol
            Arguments of :meth:`layout`.

        Returns
        -------
        SequenceDescription
            The composed layout, or the stored description when the simulator
            was built by :meth:`from_description` or :meth:`from_pulseq`.
        """
        if self._described is not None:
            return self._described
        events, played_s = compose(*self.layout(**protocol))
        return SequenceDescription(
            subsequence_index=0,
            tr_duration_us=1e6 * self.repetition_s(played_s, **protocol),
            events=events,
            rf_definitions=dict(self.model.definitions),
            shim_definitions=dict(self.shims),
            crusher_dephasing_rad=self.crusher_dephasing_rad,
            voxel_size_m=self.voxel_size_m,
        )

    @classmethod
    def from_pulseq(
        cls,
        source: Any,
        *,
        tr_index: int | None = None,
        **settings: Any,
    ) -> Simulator:
        """Return a simulator over one repetition of a Pulseq sequence.

        The events are re-emitted through this class's operators, as in
        :meth:`from_description`.

        Parameters
        ----------
        source : str, pathlib.Path or object
            A ``.seq`` file, or a pypulseq or pypulseqpp ``Sequence``.
        tr_index : int, default=None
            Repetition to read, counted from 0. Defaults to the ``TRRef``
            definition, else the first repetition that acquires.
        **settings
            Run settings and tissue properties, as for the constructor.

        Returns
        -------
        Simulator
            Simulator playing that repetition.

        Raises
        ------
        ValueError
            If the file declares no ``TRSize``, ``tr_index`` is out of range,
            or the repetitions differ in their pulses and none was named.
        """
        from ..sequence._pulseq import read_pulseq_description

        return cls.from_description(
            read_pulseq_description(source, tr_index=tr_index), **settings
        )

    @classmethod
    def from_description(
        cls,
        description: SequenceDescription,
        model: SpinPhysics | None = None,
        **settings: Any,
    ) -> Simulator:
        """Return a simulator that plays an existing description.

        The description carries RF pulses and ADC windows but no gradients, so
        its events are re-emitted through this class's operators, which supply
        the dephasing (crushers, winding, spoiling). The class the method is
        called on therefore selects how the events are interpreted.

        Parameters
        ----------
        description : SequenceDescription
            Description decoded from an MRD stream or exported from a Pulseq
            design.
        model : SpinPhysics, default=None
            Physics to read it with. Defaults to the class's own.
        **settings
            Run settings and tissue properties, as for the constructor.
            ``repetitions`` defaults to the class's own.

        Returns
        -------
        Simulator
            Simulator playing the re-emitted description.

        Raises
        ------
        ValueError
            If called on :class:`Simulator` itself, which names no handlers and
            so says nothing about how the stream is to be read.
        """
        if cls is Simulator and model is None:
            raise ValueError(
                "from_description reads a stream through a simulator's own "
                "handlers, so call it on the simulator whose sequence it is -- "
                "FSESimulator.from_description(...) for a refocused train"
            )
        physics = model if model is not None else cls.model
        settings.setdefault("repetitions", cls.repetitions)
        simulator = _Described(model=physics, **settings)
        simulator._described = realised(description, physics)
        return simulator

    # -- what a signal model owes -------------------------------------------

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the signal of one simulation of the protocol.

        Parameters
        ----------
        properties : mapping
            Tissue property values by public name, as torch tensors in public
            units (ms, degrees, Hz).
        **sequence
            Run settings (``states``, ``repetitions``, ``record``,
            ``device``, ``execution``, ``across_slice``), each defaulting to
            the constructor's, and protocol arguments overriding the
            constructor's.

        Returns
        -------
        torch.Tensor
            Complex signal of the recorded ADC events, of shape
            ``(*voxel_shape, samples)``; with scalar properties there is no
            voxel axis.
        """
        given = dict(sequence)
        states = given.pop("states", self.states)
        settings = {
            "repetitions": given.pop("repetitions", self.repetitions),
            "record": given.pop("record", self.record),
            "device": given.pop("device", None),
            "rf_raster_time_s": self.rf_raster_time_s,
        }
        target = given.pop("execution", self.execution)
        profile = across_the_slice(given.pop("across_slice", None)) or self.across_slice
        played = self.played(**given)
        tissue = self.model.tissue(properties)
        described, events = self._structure(
            played,
            tissue,
            repetitions=settings["repetitions"],
            record=settings["record"],
            device=settings["device"],
            across_slice=profile,
        )
        block = nullcontext() if target is None else execution(target)
        with block:
            return (
                EpgEngine()
                .simulate(
                    described,
                    tissue,
                    nstates=states,
                    events=events,
                    across_slice=profile,
                    **settings,
                )
                .signal
            )

    def _shaped(self, signal: torch.Tensor, batch: tuple[int, ...]) -> torch.Tensor:
        # The engine gives a tissue of scalar properties no voxel axis, so a
        # leading axis of one is then a train or a readout, never a voxel.
        return super()._shaped(signal, batch) if batch else signal


class _Described(Simulator):
    """A simulator whose description was handed to it whole."""
