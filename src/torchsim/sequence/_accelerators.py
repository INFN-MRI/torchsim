"""Inference-only dispatch to fused EPG state-machine kernels."""

from __future__ import annotations

__all__: list[str] = []

import os
from dataclasses import dataclass
from typing import Any

import torch

from ._description import EventType, RfUse, SequenceDescription, SequenceEvent

_PRE_SHIFT = 1
_POST_SHIFT = 2
_INVERSION = 4
_SPOIL_AFTER = 8
_SHIFT_AFTER = 16
_RECORD = 32


@dataclass(frozen=True)
class _PackedEvents:
    duration: torch.Tensor
    kind: torch.Tensor
    flip: torch.Tensor
    phase: torch.Tensor
    action: torch.Tensor
    output_index: torch.Tensor
    time_us: torch.Tensor
    event_index: torch.Tensor
    repetition: torch.Tensor
    echo: torch.Tensor

    @property
    def output_count(self) -> int:
        return int(self.time_us.numel())


def simulate_native(
    policy_name: str,
    description: SequenceDescription,
    prepared_tissue: tuple[torch.Tensor, ...],
    output_shape: torch.Size,
    *,
    repetitions: int,
    record: str,
    nstates: int,
    slice_profile: torch.Tensor,
    rf_raster_time_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run a fused CPU/CUDA state machine with explicit AD rules."""
    if slice_profile.numel() != 1:
        return None
    device = prepared_tissue[0].device
    if device.type not in {"cpu", "cuda"} or not _backend_available(device):
        return None

    packed = _pack_events(
        policy_name,
        description,
        repetitions=repetitions,
        record=record,
        device=prepared_tissue[0].device,
        rf_raster_time_s=rf_raster_time_s,
    )
    tissue = tuple(
        value.to(dtype=torch.float32).contiguous() for value in prepared_tissue
    )
    tissue = (
        tissue[0],
        tissue[1],
        tissue[2],
        (tissue[3] * slice_profile.reshape(())).contiguous(),
        *tissue[4:],
    )
    threads = int(os.environ.get("TORCHSIM_NUM_THREADS", str(torch.get_num_threads())))
    signal = _NativeEpg.apply(
        *tissue,
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
        nstates,
        packed.output_count,
        threads,
    )
    signal = signal.reshape(*output_shape, packed.output_count)
    return (
        signal,
        packed.time_us,
        packed.event_index,
        packed.repetition,
        packed.echo,
    )


# %% private module subroutines


def _pack_events(
    policy_name: str,
    description: SequenceDescription,
    *,
    repetitions: int,
    record: str,
    device: torch.device,
    rf_raster_time_s: float,
) -> _PackedEvents:
    durations: list[torch.Tensor] = []
    kinds: list[int] = []
    flips: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    actions: list[int] = []
    output_indices: list[int] = []
    times: list[torch.Tensor] = []
    event_indices: list[int] = []
    repetition_indices: list[int] = []
    echo_flags: list[bool] = []
    previous_absolute = _scalar(0.0, device)
    output_index = 0

    for repetition in range(repetitions):
        repetition_offset = _scalar(description.tr_duration_us, device) * repetition
        for event_index, event in enumerate(description.events):
            absolute = repetition_offset + _scalar(event.timestamp_us, device)
            durations.append((absolute - previous_absolute) * 1e-6)
            previous_absolute = absolute
            kinds.append(int(event.type))
            action = _action(policy_name, event)
            flip = _scalar(0.0, device)
            phase = _scalar(0.0, device)
            event_output_index = -1
            if event.type is EventType.RF:
                phase = _scalar(event.rf_phase_rad, device)
                if event.rf_use is RfUse.INVERSION:
                    action |= _INVERSION
                else:
                    definition = description.rf_definitions[event.rf_definition_id]
                    flip_value, integral_phase = definition.flip_angle(
                        event.rf_amplitude_hz,
                        rf_raster_time_s=rf_raster_time_s,
                    )
                    flip = _scalar(flip_value, device)
                    phase = phase + _scalar(integral_phase, device)
            elif event.type is EventType.ADC:
                phase = _scalar(event.adc_phase_rad, device)
                if _record_event(event, record):
                    action |= _RECORD
                    event_output_index = output_index
                    times.append(absolute)
                    event_indices.append(event_index)
                    repetition_indices.append(repetition)
                    echo_flags.append(event.is_echo)
                    output_index += 1
            output_indices.append(event_output_index)
            flips.append(flip)
            phases.append(phase)
            actions.append(action)

    return _PackedEvents(
        duration=torch.stack(durations).to(torch.float32).contiguous(),
        kind=torch.as_tensor(kinds, dtype=torch.int32, device=device).contiguous(),
        flip=torch.stack(flips).to(torch.float32).contiguous(),
        phase=torch.stack(phases).to(torch.float32).contiguous(),
        action=torch.as_tensor(actions, dtype=torch.uint8, device=device).contiguous(),
        output_index=torch.as_tensor(
            output_indices, dtype=torch.int32, device=device
        ).contiguous(),
        time_us=_stack(times, torch.float32, device),
        event_index=torch.as_tensor(event_indices, dtype=torch.int64, device=device),
        repetition=torch.as_tensor(
            repetition_indices, dtype=torch.int64, device=device
        ),
        echo=torch.as_tensor(echo_flags, dtype=torch.bool, device=device),
    )


def _action(policy_name: str, event: SequenceEvent) -> int:
    if event.type is EventType.RF and event.rf_use is RfUse.REFOCUSING:
        return _PRE_SHIFT | _POST_SHIFT if policy_name == "fse" else 0
    if event.type is not EventType.ADC:
        return 0
    if policy_name == "spgr":
        return _SPOIL_AFTER
    if policy_name == "ssfp-fid":
        return _SHIFT_AFTER
    if policy_name == "ssfp-echo":
        return _PRE_SHIFT
    return 0


def _record_event(event: SequenceEvent, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "acquired":
        return event.adc_role.value != 0
    return event.is_echo


def _scalar(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(())
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(())


def _stack(
    values: list[torch.Tensor], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.stack(values).to(dtype=dtype)


class _NativeEpg(torch.autograd.Function):
    @staticmethod
    def forward(
        t1: torch.Tensor,
        t2: torch.Tensor,
        m0: torch.Tensor,
        b1: torch.Tensor,
        b1_phase: torch.Tensor,
        b0: torch.Tensor,
        inversion_efficiency: torch.Tensor,
        duration: torch.Tensor,
        kind: torch.Tensor,
        flip: torch.Tensor,
        phase: torch.Tensor,
        action: torch.Tensor,
        output_index: torch.Tensor,
        state_count: int,
        output_count: int,
        threads: int,
    ) -> torch.Tensor:
        tissue = (t1, t2, m0, b1, b1_phase, b0, inversion_efficiency)
        events = (duration, kind, flip, phase, action, output_index)
        return _run_packed(tissue, events, state_count, output_count, threads)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        tensors = inputs[:13]
        ctx.save_for_backward(*tensors)
        ctx.save_for_forward(*tensors)
        ctx.state_count = inputs[13]
        ctx.output_count = inputs[14]
        ctx.threads = inputs[15]

    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor | None) -> torch.Tensor:
        saved = ctx.saved_tensors
        float_indices = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
        float_tangents = tuple(
            torch.zeros_like(saved[index])
            if tangents[index] is None
            else tangents[index].to(saved[index]).contiguous()
            for index in float_indices
        )
        return _NativeEpgJvp.apply(
            *saved,
            *float_tangents,
            ctx.state_count,
            ctx.output_count,
            ctx.threads,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        with torch.enable_grad():
            output = _simulate_packed_torch(
                saved[:7],
                saved[7:13],
                state_count=ctx.state_count,
                output_count=ctx.output_count,
            )
            differentiable = [
                tensor
                for index, tensor in enumerate(saved)
                if index in {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
                and ctx.needs_input_grad[index]
            ]
            gradients = torch.autograd.grad(
                output,
                differentiable,
                grad_output,
                allow_unused=True,
                create_graph=torch.is_grad_enabled(),
            ) if differentiable else ()
        result: list[torch.Tensor | None] = []
        gradient_index = 0
        for index in range(13):
            if index in {0, 1, 2, 3, 4, 5, 6, 7, 9, 10} and ctx.needs_input_grad[index]:
                result.append(gradients[gradient_index])
                gradient_index += 1
            else:
                result.append(None)
        return (*result, None, None, None)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *inputs: Any,
    ) -> tuple[torch.Tensor, int]:
        return _loop_vmap(_NativeEpg, info.batch_size, in_dims, inputs)


class _NativeEpgJvp(torch.autograd.Function):
    @staticmethod
    def forward(*inputs: Any) -> torch.Tensor:
        saved = inputs[:13]
        tangents = inputs[13:23]
        tissue_tangents = tangents[:7]
        event_tangents = tangents[7:]
        return _run_packed_jvp(
            saved[:7],
            saved[7:13],
            tissue_tangents,
            event_tangents,
            inputs[23],
            inputs[24],
            inputs[25],
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        ctx.save_for_backward(*inputs[:23])
        ctx.state_count = inputs[23]
        ctx.output_count = inputs[24]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        primal = saved[:13]
        tangent = saved[13:23]
        float_indices = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
        float_primal = tuple(primal[index] for index in float_indices)

        def function(*values: torch.Tensor) -> torch.Tensor:
            rebuilt = list(primal)
            for index, value in zip(float_indices, values, strict=True):
                rebuilt[index] = value
            return _simulate_packed_torch(
                tuple(rebuilt[:7]),
                tuple(rebuilt[7:13]),
                state_count=ctx.state_count,
                output_count=ctx.output_count,
            )

        with torch.enable_grad():
            _, directional = torch.func.jvp(function, float_primal, tangent)
            requested = [
                tensor
                for index, tensor in enumerate((*primal, *tangent))
                if index < len(ctx.needs_input_grad) and ctx.needs_input_grad[index]
            ]
            gradients = torch.autograd.grad(
                directional,
                requested,
                grad_output,
                allow_unused=True,
                create_graph=torch.is_grad_enabled(),
            ) if requested else ()
        result: list[torch.Tensor | None] = []
        gradient_index = 0
        for index in range(23):
            if ctx.needs_input_grad[index]:
                result.append(gradients[gradient_index])
                gradient_index += 1
            else:
                result.append(None)
        return (*result, None, None, None)

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *inputs: Any,
    ) -> tuple[torch.Tensor, int]:
        return _loop_vmap(_NativeEpgJvp, info.batch_size, in_dims, inputs)


def _backend_available(device: torch.device) -> bool:
    try:
        if device.type == "cpu":
            from torchsim import _epg_cpu  # noqa: F401
        else:
            from . import _epg_triton  # noqa: F401
    except ImportError:
        return False
    return True


def _loop_vmap(
    function: type[torch.autograd.Function],
    batch_size: int,
    in_dims: tuple[int | None, ...],
    inputs: tuple[Any, ...],
) -> tuple[torch.Tensor, int]:
    outputs = []
    for batch in range(batch_size):
        sliced = tuple(
            value if dimension is None else value.select(dimension, batch)
            for value, dimension in zip(inputs, in_dims, strict=True)
        )
        outputs.append(function.apply(*sliced))
    return torch.stack(outputs), 0


def _run_packed(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    state_count: int,
    output_count: int,
    threads: int,
) -> torch.Tensor:
    if tissue[0].device.type == "cuda":
        from ._epg_triton import simulate

        return simulate(
            tissue,
            events,
            state_count=state_count,
            output_count=output_count,
        )
    from torchsim import _epg_cpu

    output_real = torch.empty((tissue[0].numel(), output_count), dtype=torch.float32)
    output_imag = torch.empty_like(output_real)
    pointers = (*tissue, *events, output_real, output_imag)
    _epg_cpu.simulate(
        tuple(value.data_ptr() for value in pointers),
        tissue[0].numel(),
        events[1].numel(),
        state_count,
        output_count,
        threads,
    )
    return torch.complex(output_real, output_imag)


def _run_packed_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, ...],
    state_count: int,
    output_count: int,
    threads: int,
) -> torch.Tensor:
    if tissue[0].device.type == "cuda":
        from ._epg_triton import simulate_jvp

        return simulate_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
            state_count=state_count,
            output_count=output_count,
        )
    from torchsim import _epg_cpu

    output_real = torch.empty((tissue[0].numel(), output_count), dtype=torch.float32)
    output_imag = torch.empty_like(output_real)
    pointers = (
        *tissue,
        *events,
        *tissue_tangents,
        *event_tangents,
        output_real,
        output_imag,
    )
    _epg_cpu.simulate_jvp(
        tuple(value.data_ptr() for value in pointers),
        tissue[0].numel(),
        events[1].numel(),
        state_count,
        output_count,
        threads,
    )
    return torch.complex(output_real, output_imag)


def _simulate_packed_torch(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
) -> torch.Tensor:
    t1, t2, m0, b1, b1_phase, b0, inversion_efficiency = tissue
    duration, kind, flip, phase, action, _output_index = events
    atom_count = t1.numel()
    shape = (atom_count, state_count)
    fplus = torch.zeros(shape, dtype=torch.complex64, device=t1.device)
    fminus = torch.zeros_like(fplus)
    longitudinal = torch.zeros_like(fplus)
    longitudinal[:, 0] = 1.0
    signals = []

    for event in range(kind.numel()):
        dt = duration[event]
        e1 = torch.exp(-(1000.0 / t1) * dt)
        e2 = torch.exp(-(1000.0 / t2) * dt)
        off = e2 * torch.exp(-2j * torch.pi * b0 * dt)
        fplus = fplus * off[:, None]
        fminus = fminus * off.conj()[:, None]
        longitudinal = longitudinal * e1[:, None]
        recovery = torch.zeros_like(longitudinal)
        recovery[:, 0] = 1.0 - e1
        longitudinal = longitudinal + recovery

        event_action = int(action[event])
        if event_action & _PRE_SHIFT:
            fplus, fminus = _shift_tensors(fplus, fminus)
        event_kind = int(kind[event])
        if event_kind == 1:
            if event_action & _INVERSION:
                longitudinal = -inversion_efficiency[:, None] * longitudinal
            else:
                alpha = flip[event] * b1
                phi = phase[event] + b1_phase
                fplus, fminus, longitudinal = _rotate_tensors(
                    fplus, fminus, longitudinal, alpha, phi
                )
        elif event_kind == 2 and event_action & _RECORD:
            signals.append(m0 * fplus[:, 0] * torch.exp(-1j * phase[event]))
        if event_action & _POST_SHIFT:
            fplus, fminus = _shift_tensors(fplus, fminus)
        if event_action & _SPOIL_AFTER:
            fplus = torch.zeros_like(fplus)
            fminus = torch.zeros_like(fminus)
        elif event_action & _SHIFT_AFTER:
            fplus, fminus = _shift_tensors(fplus, fminus)

    if not signals:
        return torch.empty(
            (atom_count, output_count),
            dtype=torch.complex64,
            device=t1.device,
        )
    return torch.stack(signals, dim=-1)


def _shift_tensors(
    fplus: torch.Tensor,
    fminus: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(fplus[:, :1])
    shifted_minus = torch.cat((fminus[:, 1:], zero), dim=-1)
    shifted_plus = torch.cat((shifted_minus[:, :1].conj(), fplus[:, :-1]), dim=-1)
    return shifted_plus, shifted_minus


def _rotate_tensors(
    fplus: torch.Tensor,
    fminus: torch.Tensor,
    longitudinal: torch.Tensor,
    alpha: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine = torch.cos(alpha)[:, None]
    sine = torch.sin(alpha)[:, None]
    phase_one = torch.exp(1j * phi)[:, None]
    phase_two = phase_one.square()
    t00 = 0.5 * (1.0 + cosine)
    t01 = 0.5 * (1.0 - cosine) * phase_two
    t02 = -1j * sine * phase_one
    t10 = t01.conj()
    t12 = 1j * sine * phase_one.conj()
    t20 = -0.5j * sine * phase_one.conj()
    t21 = 0.5j * sine * phase_one
    old_plus, old_minus, old_z = fplus, fminus, longitudinal
    return (
        t00 * old_plus + t01 * old_minus + t02 * old_z,
        t10 * old_plus + t00 * old_minus + t12 * old_z,
        t20 * old_plus + t21 * old_minus + cosine * old_z,
    )
