"""Differentiable EPG interpreter for sequence descriptions."""

from __future__ import annotations

__all__ = [
    "BSSFP",
    "FSE",
    "SPGR",
    "SSFPFID",
    "EpgSimulator",
    "SSFPEcho",
    "SimulationResult",
    "SubspaceBasis",
    "TissueProperties",
    "make_simulator",
    "simulate_subspace",
]

from dataclasses import dataclass
from typing import Any, Literal

import torch

from .. import epg
from ._accelerators import simulate_native
from ._description import AdcRole, EventType, RfUse, SequenceDescription, SequenceEvent

RecordMode = Literal["all", "acquired", "echo"]

# Relaxation times enter every backend as rates ``1000 / T``. A non-positive
# time therefore yields an infinite rate, and zero-duration events (an RF pulse
# at the same timestamp as its neighbour) turn ``inf * 0`` into NaN, which the
# state matrix then propagates to every echo. Air voxels in a measured map are
# routinely exactly zero, so clamp the times to a small positive value: the
# resulting rate is large enough to null the signal over any real interval
# while keeping ``exp(-R * 0) == 1``. Clamping (rather than ``1 / (T + eps)``)
# also keeps the gradient finite, since clamped entries simply stop
# contributing to the backward pass.
MINIMUM_RELAXATION_TIME_MS = 1e-6


@dataclass(frozen=True)
class TissueProperties:
    """Single-pool tissue and transmit-field properties.

    Values may be scalars or broadcastable tensors. Relaxation times use
    milliseconds; off-resonance uses Hz.
    """

    t1_ms: Any
    t2_ms: Any
    m0: Any = 1.0
    b1: Any = 1.0
    b1_phase_rad: Any = 0.0
    b0_hz: Any = 0.0
    inversion_efficiency: Any = 1.0


@dataclass(frozen=True)
class SimulationResult:
    """Signals and labels recorded by one state-machine simulation."""

    signal: torch.Tensor
    time_us: torch.Tensor
    event_index: torch.Tensor
    repetition: torch.Tensor
    echo: torch.Tensor


@dataclass(frozen=True)
class SubspaceBasis:
    """Low-rank temporal basis and its simulated dictionary."""

    basis: torch.Tensor
    singular_values: torch.Tensor
    dictionary: torch.Tensor
    simulation: SimulationResult


class EpgSimulator:
    """Translate RF/ADC events into differentiable TorchSim EPG operations."""

    name = "base"

    def before_rf(
        self,
        states: Any,
        _event: SequenceEvent,
    ) -> Any:
        """Apply sequence-specific operations immediately before RF."""
        return states

    def after_rf(
        self,
        states: Any,
        _event: SequenceEvent,
    ) -> Any:
        """Apply sequence-specific operations immediately after RF."""
        return states

    def before_adc(
        self,
        states: Any,
        _event: SequenceEvent,
    ) -> Any:
        """Apply sequence-specific operations immediately before ADC."""
        return states

    def after_adc(
        self,
        states: Any,
        _event: SequenceEvent,
    ) -> Any:
        """Apply sequence-specific operations immediately after ADC."""
        return states

    def shifts_per_repetition(self, _description: SequenceDescription) -> int:
        """Return an upper bound used to size the EPG state matrix."""
        return 0

    def simulate(
        self,
        description: SequenceDescription,
        tissue: TissueProperties,
        *,
        repetitions: int = 1,
        record: RecordMode = "all",
        nstates: int | None = None,
        slice_profile: Any = 1.0,
        rf_raster_time_s: float = 1e-6,
        device: torch.device | str | None = None,
        backend: Literal["auto", "torch", "native"] = "auto",
    ) -> SimulationResult:
        """Walk an event stream and record selected ADC signals.

        The Torch implementation is intentionally fully functional with
        respect to its numerical inputs. It is therefore the path used by
        forward-mode AD and sequence optimization. Inference-only fused CPU
        and CUDA kernels dispatch above this layer when available.
        """
        repetitions = _as_integer(repetitions, "repetitions")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        if record not in {"all", "acquired", "echo"}:
            raise ValueError("record must be 'all', 'acquired', or 'echo'")

        prepared, output_shape, target_device = _prepare_tissue(tissue, device)
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency = prepared

        minimum_states = 1 + repetitions * self.shifts_per_repetition(description)
        if nstates is None:
            nstates = max(8, min(64, minimum_states))
        else:
            nstates = _as_integer(nstates, "nstates")
        if nstates < 1:
            raise ValueError("nstates must be positive")

        profile = _as_float_tensor(slice_profile, target_device).reshape(-1)
        if backend not in {"auto", "torch", "native"}:
            raise ValueError("backend must be 'auto', 'torch', or 'native'")
        if backend != "torch":
            accelerated = simulate_native(
                self.name,
                description,
                prepared,
                output_shape,
                repetitions=repetitions,
                record=record,
                nstates=nstates,
                slice_profile=profile,
                rf_raster_time_s=rf_raster_time_s,
            )
            if accelerated is not None:
                return SimulationResult(*accelerated)
            if backend == "native":
                raise RuntimeError(
                    "native EPG backend is unavailable for this device or AD context"
                )
        states = epg.states_matrix(
            device=target_device,
            nstates=nstates,
            nlocs=t1.numel() * profile.numel(),
        )
        atom_count = t1.numel()
        location_count = profile.numel()
        states = _reshape_states(states, atom_count, location_count)

        signals: list[torch.Tensor] = []
        times: list[Any] = []
        event_indices: list[int] = []
        repetition_indices: list[int] = []
        echo_flags: list[bool] = []
        absolute_offset_us: Any = 0.0

        for repetition in range(repetitions):
            current_us: Any = 0.0
            for event_index, event in enumerate(description.events):
                states = _free_precess(
                    states,
                    t1,
                    t2,
                    b0,
                    (event.timestamp_us - current_us) * 1e-6,
                )
                current_us = event.timestamp_us

                if event.type is EventType.RF:
                    states = self.before_rf(states, event)
                    if event.rf_use is RfUse.INVERSION:
                        states = epg.adiabatic_inversion(
                            states, inversion_efficiency[None, :, None, None]
                        )
                    else:
                        definition = description.rf_definitions[event.rf_definition_id]
                        flip, integral_phase = definition.flip_angle(
                            event.rf_amplitude_hz,
                            rf_raster_time_s=rf_raster_time_s,
                        )
                        flip = _as_float_tensor(flip, target_device)
                        phase = _as_float_tensor(
                            event.rf_phase_rad + integral_phase, target_device
                        )
                        operator = epg.phased_rf_pulse_op(
                            flip,
                            phase,
                            slice_prof=profile[None, :],
                            B1=b1[:, None],
                            B1phase=b1_phase[:, None],
                        )
                        states = epg.rf_pulse(states, operator)
                    states = self.after_rf(states, event)
                elif event.type is EventType.ADC:
                    states = self.before_adc(states, event)
                    if _record_event(event, record):
                        phase = _as_float_tensor(event.adc_phase_rad, target_device)
                        signal = states.Fplus[0, :, :, 0].mean(dim=-1)
                        signals.append(signal * torch.exp(-1j * phase))
                        times.append(absolute_offset_us + event.timestamp_us)
                        event_indices.append(event_index)
                        repetition_indices.append(repetition)
                        echo_flags.append(event.is_echo)
                    states = self.after_adc(states, event)

            states = _free_precess(
                states,
                t1,
                t2,
                b0,
                (description.tr_duration_us - current_us) * 1e-6,
            )
            absolute_offset_us = absolute_offset_us + description.tr_duration_us

        if signals:
            signal = torch.stack(signals, dim=-1) * m0[:, None]
        else:
            signal = torch.empty(
                (t1.numel(), 0), dtype=torch.complex64, device=target_device
            )
        signal = signal.reshape(*output_shape, signal.shape[-1])
        return SimulationResult(
            signal=signal,
            time_us=_stack_scalars(times, target_device),
            event_index=torch.as_tensor(
                event_indices, dtype=torch.int64, device=target_device
            ),
            repetition=torch.as_tensor(
                repetition_indices, dtype=torch.int64, device=target_device
            ),
            echo=torch.as_tensor(echo_flags, dtype=torch.bool, device=target_device),
        )


class FSE(EpgSimulator):
    """Fast spin echo policy: crusher, refocusing RF, crusher."""

    name = "fse"

    def before_rf(self, states: Any, event: SequenceEvent) -> Any:
        if event.rf_use is RfUse.REFOCUSING:
            return _shift(states)
        return states

    def after_rf(self, states: Any, event: SequenceEvent) -> Any:
        if event.rf_use is RfUse.REFOCUSING:
            return _shift(states)
        return states

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        return 2 * sum(
            event.type is EventType.RF and event.rf_use is RfUse.REFOCUSING
            for event in description.events
        )


class SPGR(EpgSimulator):
    """Spoiled GRE policy using ideal transverse spoiling after every ADC."""

    name = "spgr"

    def after_adc(self, states: Any, _event: SequenceEvent) -> Any:
        return _spoil(states)


class SSFPFID(EpgSimulator):
    """Unbalanced SSFP-FID policy using one crusher after every ADC."""

    name = "ssfp-fid"

    def after_adc(self, states: Any, _event: SequenceEvent) -> Any:
        return _shift(states)

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        return sum(event.type is EventType.ADC for event in description.events)


class SSFPEcho(EpgSimulator):
    """SSFP-Echo policy using one dephasing crusher before every ADC."""

    name = "ssfp-echo"

    def before_adc(self, states: Any, _event: SequenceEvent) -> Any:
        return _shift(states)

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        return sum(event.type is EventType.ADC for event in description.events)


class BSSFP(EpgSimulator):
    """Balanced SSFP policy without crushers or ideal spoiling."""

    name = "bssfp"


def make_simulator(name: str) -> EpgSimulator:
    """Construct a built-in sequence policy by name."""
    normalized = name.lower().replace("_", "-")
    policies = {
        "fse": FSE,
        "spgr": SPGR,
        "ssfp-fid": SSFPFID,
        "ssfp-echo": SSFPEcho,
        "bssfp": BSSFP,
    }
    try:
        return policies[normalized]()
    except KeyError as error:
        raise ValueError(f"unknown EPG simulator {name!r}") from error


def simulate_subspace(
    description: SequenceDescription,
    simulator: EpgSimulator | str,
    tissue: TissueProperties,
    *,
    rank: int,
    repetitions: int = 1,
    record: RecordMode = "all",
    nstates: int | None = None,
    slice_profile: Any = 1.0,
    rf_raster_time_s: float = 1e-6,
    device: torch.device | str | None = None,
) -> SubspaceBasis:
    """Simulate a dictionary and return its leading temporal basis."""
    if rank < 1:
        raise ValueError("rank must be positive")
    if isinstance(simulator, str):
        simulator = make_simulator(simulator)
    result = simulator.simulate(
        description,
        tissue,
        repetitions=repetitions,
        record=record,
        nstates=nstates,
        slice_profile=slice_profile,
        rf_raster_time_s=rf_raster_time_s,
        device=device,
    )
    dictionary = result.signal.reshape(-1, result.signal.shape[-1])
    if rank > min(dictionary.shape):
        raise ValueError(
            f"rank={rank} exceeds dictionary dimensions {tuple(dictionary.shape)}"
        )
    basis, singular_values, _ = torch.linalg.svd(
        dictionary.mT, full_matrices=False
    )
    return SubspaceBasis(
        basis=basis[:, :rank],
        singular_values=singular_values,
        dictionary=result.signal,
        simulation=result,
    )


# %% private module subroutines


def _prepare_tissue(
    tissue: TissueProperties,
    device: torch.device | str | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Size, torch.device]:
    values = (
        tissue.t1_ms,
        tissue.t2_ms,
        tissue.m0,
        tissue.b1,
        tissue.b1_phase_rad,
        tissue.b0_hz,
        tissue.inversion_efficiency,
    )
    if device is None:
        device = next(
            (value.device for value in values if isinstance(value, torch.Tensor)),
            torch.device("cpu"),
        )
    device = torch.device(device)
    tensors = torch.broadcast_tensors(
        *(_as_float_tensor(value, device) for value in values)
    )
    shape = tensors[0].shape
    flat = [value.reshape(-1) for value in tensors]
    # t1_ms and t2_ms are the two entries used as denominators downstream.
    flat[0] = flat[0].clamp_min(MINIMUM_RELAXATION_TIME_MS)
    flat[1] = flat[1].clamp_min(MINIMUM_RELAXATION_TIME_MS)
    return tuple(flat), shape, device


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_integer(value: Any, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def _reshape_states(states: Any, atoms: int, locations: int) -> Any:
    states.Fplus = states.Fplus.reshape(states.Fplus.shape[0], atoms, locations, -1)
    states.Fminus = states.Fminus.reshape(
        states.Fminus.shape[0], atoms, locations, -1
    )
    states.Z = states.Z.reshape(states.Z.shape[0], atoms, locations, -1)
    return states


def _free_precess(
    states: Any,
    t1_ms: torch.Tensor,
    t2_ms: torch.Tensor,
    b0_hz: torch.Tensor,
    duration_s: Any,
) -> Any:
    duration = _as_float_tensor(duration_s, t1_ms.device)
    e1, recovery = epg.longitudinal_relaxation_op(
        1e3 / t1_ms[None, :, None, None], duration
    )
    e2 = epg.transverse_relaxation_op(
        1e3 / t2_ms[None, :, None, None], duration
    )
    states = epg.longitudinal_relaxation(states, e1, recovery)
    states = epg.transverse_relaxation(states, e2)
    phase = torch.exp(
        -1j * 2.0 * torch.pi * b0_hz[None, :, None, None] * duration
    )
    states.Fplus = states.Fplus * phase
    states.Fminus = states.Fminus * phase.conj()
    return states


def _record_event(event: SequenceEvent, mode: RecordMode) -> bool:
    if mode == "all":
        return True
    if mode == "acquired":
        return event.adc_role is not AdcRole.NON_ACQUIRED
    return event.is_echo


def _stack_scalars(values: list[Any], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack([_as_float_tensor(value, device) for value in values])


def _shift(states: Any, delta: int = 1) -> Any:
    fminus = torch.roll(states.Fminus, -delta, dims=0)
    fplus = torch.roll(states.Fplus, delta, dims=0)
    zero = torch.zeros_like(fminus[:delta])
    fminus = torch.cat((fminus[:-delta], zero), dim=0)
    fplus = torch.cat((fminus[:1].conj(), fplus[1:]), dim=0)
    states.Fplus = fplus
    states.Fminus = fminus
    return states


def _spoil(states: Any) -> Any:
    states.Fplus = torch.zeros_like(states.Fplus)
    states.Fminus = torch.zeros_like(states.Fminus)
    return states
