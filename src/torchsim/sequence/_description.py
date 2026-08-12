"""Backend-independent description of an MR sequence state machine."""

from __future__ import annotations

__all__ = [
    "AdcRole",
    "EventType",
    "RfDefinition",
    "RfShape",
    "RfUse",
    "SequenceDescription",
    "SequenceEvent",
    "ShimDefinition",
    "decompress_shape",
]

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class EventType(IntEnum):
    """State-machine event type used by Pulserver SEQDESC streams."""

    WAIT = 0
    RF = 1
    ADC = 2


class RfUse(IntEnum):
    """Pulseq RF-use tag."""

    UNKNOWN = 0
    EXCITATION = 1
    REFOCUSING = 2
    INVERSION = 3
    SATURATION = 4
    PREPARATION = 5
    OTHER = 6


class AdcRole(IntEnum):
    """Relationship between ADC events in a canonical event stream."""

    NON_ACQUIRED = 0
    SINGLE = 1
    ECHO_CENTER = 2
    NON_CENTER = 3


@dataclass(frozen=True)
class SequenceEvent:
    """One WAIT, RF, or ADC state-machine event.

    ``params`` deliberately accepts tensors. This lets a hard-coded sequence
    description carry differentiable flip angles, phases, and timings without
    introducing a second representation for sequence optimization.
    """

    type: EventType
    timestamp_us: Any
    params: tuple[Any, ...] = ()

    @classmethod
    def wait(cls, timestamp_us: Any) -> SequenceEvent:
        """Create a timing-only event."""
        return cls(EventType.WAIT, timestamp_us)

    @classmethod
    def rf(
        cls,
        timestamp_us: Any,
        definition_id: int,
        use: RfUse,
        amplitude_hz: Any,
        phase_rad: Any = 0.0,
        frequency_hz: Any = 0.0,
        shim_id: int = 0,
        slice_select_gradient_hz_per_m: Any = 0.0,
    ) -> SequenceEvent:
        """Create an RF event using the Pulserver SEQDESC field order."""
        return cls(
            EventType.RF,
            timestamp_us,
            (
                definition_id,
                use,
                amplitude_hz,
                phase_rad,
                frequency_hz,
                shim_id,
                slice_select_gradient_hz_per_m,
            ),
        )

    @classmethod
    def adc(
        cls,
        timestamp_us: Any,
        role: AdcRole = AdcRole.SINGLE,
        phase_rad: Any = 0.0,
        *,
        is_echo: bool = True,
    ) -> SequenceEvent:
        """Create an ADC event using the Pulserver SEQDESC field order."""
        return cls(EventType.ADC, timestamp_us, (role, phase_rad, int(is_echo)))

    @property
    def rf_definition_id(self) -> int:
        self._require(EventType.RF)
        return int(self.params[0])

    @property
    def rf_use(self) -> RfUse:
        self._require(EventType.RF)
        return RfUse(int(self.params[1]))

    @property
    def rf_amplitude_hz(self) -> Any:
        self._require(EventType.RF)
        return self.params[2]

    @property
    def rf_phase_rad(self) -> Any:
        self._require(EventType.RF)
        return self.params[3]

    @property
    def rf_frequency_hz(self) -> Any:
        self._require(EventType.RF)
        return self.params[4]

    @property
    def rf_shim_id(self) -> int:
        self._require(EventType.RF)
        return int(self.params[5])

    @property
    def slice_select_gradient_hz_per_m(self) -> Any:
        self._require(EventType.RF)
        return self.params[6]

    @property
    def adc_role(self) -> AdcRole:
        self._require(EventType.ADC)
        return AdcRole(int(self.params[0]))

    @property
    def adc_phase_rad(self) -> Any:
        self._require(EventType.ADC)
        return self.params[1]

    @property
    def is_echo(self) -> bool:
        """Whether this ADC position reaches the k-space origin."""
        self._require(EventType.ADC)
        return bool(int(self.params[2]))

    def _require(self, expected: EventType) -> None:
        if self.type is not expected:
            raise AttributeError(
                f"{expected.name} payload requested from {self.type.name} event"
            )


@dataclass(frozen=True)
class RfShape:
    """One Pulseq derivative/run-length encoded shape."""

    num_uncompressed: int
    samples: Any = field(repr=False, compare=False)

    def decompress(self, scale: float = 1.0) -> np.ndarray:
        """Return the decoded shape."""
        return decompress_shape(self.samples, self.num_uncompressed, scale=scale)


@dataclass(frozen=True)
class RfDefinition:
    """RF definition referenced by RF events."""

    id: int
    bandwidth_hz: float
    num_bands: int
    band_frequency_offsets_hz: tuple[float, ...]
    band_bandwidth_hz: float
    total_b1sq_power: float
    magnitude: RfShape
    phase: RfShape | None = None
    time: RfShape | None = None

    def complex_envelope(self) -> np.ndarray:
        """Return the normalized complex RF envelope."""
        magnitude = self.magnitude.decompress()
        if self.phase is None:
            return magnitude.astype(np.complex64)
        phase = self.phase.decompress(scale=2.0 * np.pi)
        if phase.size != magnitude.size:
            raise ValueError(f"RF definition {self.id}: magnitude/phase size mismatch")
        return magnitude * np.exp(1j * phase)

    def integral(
        self,
        *,
        rf_raster_time_s: float = 1e-6,
    ) -> complex:
        """Return the complex time integral of the normalized envelope.

        The envelope is normalized, so this depends only on the pulse shape and
        the raster, never on the amplitude of a particular occurrence. A
        sequence typically reuses one definition for every refocusing pulse and
        re-evaluates it on every simulation, so the result is memoized on the
        (immutable) definition.
        """
        cache = getattr(self, "_integral_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_integral_cache", cache)
        elif rf_raster_time_s in cache:
            return cache[rf_raster_time_s]

        envelope = self.complex_envelope()
        if envelope.size < 2:
            cache[rf_raster_time_s] = 0.0j
            return 0.0j
        if self.time is None:
            time_s = (
                np.arange(envelope.size, dtype=np.float64) + 0.5
            ) * rf_raster_time_s
        else:
            time_s = self.time.decompress(scale=rf_raster_time_s).astype(np.float64)
            if time_s.size != envelope.size:
                raise ValueError(
                    f"RF definition {self.id}: envelope/time size mismatch"
                )
        area = complex(_trapezoid(envelope, time_s))
        cache[rf_raster_time_s] = area
        return area

    def flip_angle(
        self,
        amplitude_hz: Any,
        *,
        rf_raster_time_s: float = 1e-6,
    ) -> tuple[Any, Any]:
        """Return ``(flip_rad, integral_phase_rad)`` for one RF occurrence.

        Tensor amplitudes remain tensors, so both reverse- and forward-mode
        automatic differentiation pass through a description unchanged.
        """
        area = self.integral(rf_raster_time_s=rf_raster_time_s)
        try:
            import torch
        except ImportError:  # pragma: no cover - Torch is a required dependency
            torch = None
        if torch is not None and isinstance(amplitude_hz, torch.Tensor):
            area_tensor = torch.as_tensor(area, device=amplitude_hz.device)
            signed_area = amplitude_hz * area_tensor
            return 2.0 * torch.pi * torch.abs(signed_area), torch.angle(signed_area)
        signed_area = amplitude_hz * area
        return 2.0 * np.pi * abs(signed_area), np.angle(signed_area)


@dataclass(frozen=True)
class ShimDefinition:
    """One transmit-shim definition."""

    id: int
    magnitudes: tuple[float, ...]
    phases_rad: tuple[float, ...]


@dataclass(frozen=True)
class SequenceDescription:
    """Event stream and RF resources for one sequence or subsequence."""

    subsequence_index: int
    tr_duration_us: Any
    events: tuple[SequenceEvent, ...]
    rf_definitions: dict[int, RfDefinition]
    shim_definitions: dict[int, ShimDefinition] = field(default_factory=dict)

    @property
    def adc_events(self) -> tuple[SequenceEvent, ...]:
        """Return ADC events in state-machine order."""
        return tuple(event for event in self.events if event.type is EventType.ADC)


def decompress_shape(
    packed: Any,
    num_uncompressed: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Decode Pulseq derivative/run-length shape compression."""
    packed = np.asarray(packed, dtype=np.float32).reshape(-1)
    if num_uncompressed < 0:
        raise ValueError("negative uncompressed shape length")
    if packed.size == num_uncompressed:
        return (packed * scale).astype(np.float32, copy=True)
    if num_uncompressed == 0 and packed.size == 0:
        return np.empty(0, dtype=np.float32)

    delta = np.empty(num_uncompressed, dtype=np.float32)
    source = 0
    target = 0
    while source < packed.size:
        value = packed[source]
        if source + 1 < packed.size and packed[source + 1] == value:
            if source + 2 >= packed.size:
                raise ValueError("truncated Pulseq RLE triplet")
            count_value = float(packed[source + 2]) + 2.0
            count = round(count_value)
            if abs(count_value - count) > 1e-6 or count < 2:
                raise ValueError("invalid Pulseq RLE repeat count")
            if target + count > num_uncompressed:
                raise ValueError("Pulseq RLE shape expands past declared length")
            delta[target : target + count] = value
            target += count
            source += 3
        else:
            if target >= num_uncompressed:
                raise ValueError("Pulseq shape expands past declared length")
            delta[target] = value
            target += 1
            source += 1
    if target != num_uncompressed:
        raise ValueError(
            f"Pulseq shape expanded to {target}, expected {num_uncompressed}"
        )
    return (np.cumsum(delta, dtype=np.float32) * scale).astype(np.float32)

