"""Differentiable builders for the built-in sequence state machines."""

from __future__ import annotations

__all__ = [
    "fse_description",
    "mpnrage_description",
    "mprage_description",
    "mrf_description",
    "spgr_description",
]

from typing import Any

import numpy as np
import torch

from ._description import (
    AdcRole,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceEvent,
)


def fse_description(
    flip_rad: Any,
    echo_spacing_s: Any,
    *,
    phases_rad: Any = 0.0,
    excitation_flip_rad: Any = torch.pi / 2,
    excitation_phase_rad: Any = torch.pi / 2,
    repetition_time_s: Any | None = None,
    crusher_dephasing_rad: float = 0.0,
    voxel_size_m: float | None = None,
) -> SequenceDescription:
    """Build an ideal FSE echo-train description.

    ``flip_rad`` is ``(echo_train_length,)`` for one train, or ``(n_trains,
    echo_train_length)`` to describe a batch of trains that share the event
    structure and differ only in their refocusing schedule.
    """
    flip = torch.atleast_1d(flip_rad)
    if flip.dim() > 2:
        raise ValueError("flip angles must be 1- or 2-dimensional")
    phases = torch.as_tensor(phases_rad, device=flip.device, dtype=flip.dtype)
    phases = phases.expand_as(flip) if phases.numel() == 1 else phases
    if phases.shape[-1] != flip.shape[-1]:
        raise ValueError("refocusing phases must be scalar or match flip angles")

    events = [
        SequenceEvent.rf(
            0.0,
            0,
            RfUse.EXCITATION,
            excitation_flip_rad,
            excitation_phase_rad,
        )
    ]
    echo_train_length = flip.shape[-1]
    for index in range(echo_train_length):
        echo_time_s = (index + 1) * echo_spacing_s
        events.append(
            SequenceEvent.rf(
                1e6 * (echo_time_s - 0.5 * echo_spacing_s),
                0,
                RfUse.REFOCUSING,
                flip[..., index],
                phases[..., index],
            )
        )
        events.append(
            SequenceEvent.adc(
                1e6 * echo_time_s,
                AdcRole.ECHO_CENTER,
                phases[..., index],
                is_echo=True,
            )
        )

    train_duration = echo_train_length * echo_spacing_s
    duration = train_duration if repetition_time_s is None else repetition_time_s
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration,
        events=tuple(events),
        rf_definitions={0: _unit_flip_definition()},
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
    )


def mrf_description(
    flip_rad: Any,
    repetition_time_s: Any,
    *,
    inversion_time_s: Any = 0.0,
    phases_rad: Any = 0.0,
    crusher_dephasing_rad: float = 0.0,
    voxel_size_m: float | None = None,
) -> SequenceDescription:
    """Build an inversion-prepared unbalanced SSFP fingerprint description."""
    flip = torch.atleast_1d(flip_rad)
    repetition_time = torch.as_tensor(
        repetition_time_s, device=flip.device, dtype=flip.dtype
    )
    repetition_time = (
        repetition_time.expand_as(flip)
        if repetition_time.numel() == 1
        else repetition_time
    )
    phases = torch.as_tensor(phases_rad, device=flip.device, dtype=flip.dtype)
    phases = phases.expand_as(flip) if phases.numel() == 1 else phases
    if repetition_time.shape != flip.shape or phases.shape != flip.shape:
        raise ValueError("TR and phase must be scalar or match flip angles")

    events = [SequenceEvent.rf(0.0, 0, RfUse.INVERSION, torch.pi, 0.0)]
    current_s = inversion_time_s
    for index in range(flip.numel()):
        events.append(
            SequenceEvent.rf(
                1e6 * current_s,
                0,
                RfUse.EXCITATION,
                flip[index],
                phases[index],
            )
        )
        events.append(
            SequenceEvent.adc(
                1e6 * current_s,
                AdcRole.SINGLE,
                phases[index],
                is_echo=True,
            )
        )
        current_s = current_s + repetition_time[index]

    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * current_s,
        events=tuple(events),
        rf_definitions={0: _unit_flip_definition()},
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
    )


def mpnrage_description(
    nshots: int,
    flip_rad: Any,
    repetition_time_s: Any,
    *,
    inversion_time_s: Any = 0.0,
    phases_rad: Any = 0.0,
    crusher_dephasing_rad: float = 0.0,
    voxel_size_m: float | None = None,
) -> SequenceDescription:
    """Build an inversion-prepared spoiled GRE echo train."""
    if nshots < 1:
        raise ValueError("nshots must be positive")
    flip = torch.atleast_1d(flip_rad)
    if flip.numel() == 1:
        flip = flip.expand(nshots)
    elif flip.numel() != nshots:
        raise ValueError("flip must be scalar or contain nshots values")
    return _inversion_prepared_gre_description(
        flip,
        repetition_time_s,
        inversion_time_s=inversion_time_s,
        phases_rad=phases_rad,
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
        center_index=None,
    )


def mprage_description(
    nshots_before: int,
    nshots_after: int,
    flip_rad: Any,
    repetition_time_s: Any,
    inversion_time_s: Any,
    *,
    phases_rad: Any = 0.0,
    crusher_dephasing_rad: float = 0.0,
    voxel_size_m: float | None = None,
) -> SequenceDescription:
    """Build an MPRAGE train whose sole acquired ADC is the k-space center."""
    if nshots_before < 0 or nshots_after < 0:
        raise ValueError("shot counts must be nonnegative")
    shot_count = nshots_before + nshots_after + 1
    flip = torch.atleast_1d(flip_rad)
    if flip.numel() == 1:
        flip = flip.expand(shot_count)
    elif flip.numel() != shot_count:
        raise ValueError("flip must be scalar or contain one value per shot")
    repetition_time = _expand_like(repetition_time_s, flip, "TR")
    time_before_center = repetition_time[:nshots_before].sum()
    return _inversion_prepared_gre_description(
        flip,
        repetition_time,
        inversion_time_s=inversion_time_s - time_before_center,
        phases_rad=phases_rad,
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
        center_index=nshots_before,
    )


def spgr_description(
    flip_rad: Any,
    repetition_time_s: Any,
    echo_time_s: Any,
    *,
    phases_rad: Any = 0.0,
    crusher_dephasing_rad: float = 0.0,
    voxel_size_m: float | None = None,
) -> SequenceDescription:
    """Build a spoiled GRE train description."""
    flip = torch.atleast_1d(flip_rad)
    repetition_time = _expand_like(repetition_time_s, flip, "TR")
    echo_time = _expand_like(echo_time_s, flip, "TE")
    phases = _expand_like(phases_rad, flip, "phase")
    if torch.any(echo_time < 0) or torch.any(echo_time > repetition_time):
        raise ValueError("SPGR requires 0 <= TE <= TR")

    events = []
    current_s = torch.zeros((), device=flip.device, dtype=flip.dtype)
    for index in range(flip.numel()):
        events.append(
            SequenceEvent.rf(
                1e6 * current_s,
                0,
                RfUse.EXCITATION,
                flip[index],
                phases[index],
            )
        )
        events.append(
            SequenceEvent.adc(
                1e6 * (current_s + echo_time[index]),
                AdcRole.SINGLE,
                phases[index],
                is_echo=True,
            )
        )
        current_s = current_s + repetition_time[index]
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * current_s,
        events=tuple(events),
        rf_definitions={0: _unit_flip_definition()},
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
    )


# %% private module subroutines


def _unit_flip_definition() -> RfDefinition:
    # The envelope integral is exactly 1 / (2*pi) seconds. Consequently an
    # event amplitude expressed in radians is returned unchanged as its flip.
    duration_us = 1e6 / (2.0 * np.pi)
    return RfDefinition(
        id=0,
        bandwidth_hz=0.0,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=0.0,
        total_b1sq_power=0.0,
        magnitude=RfShape(2, np.ones(2, dtype=np.float32)),
        time=RfShape(2, np.asarray([0.0, duration_us], dtype=np.float32)),
    )


def _inversion_prepared_gre_description(
    flip: torch.Tensor,
    repetition_time_s: Any,
    *,
    inversion_time_s: Any,
    phases_rad: Any,
    center_index: int | None,
    crusher_dephasing_rad: float,
    voxel_size_m: float | None,
) -> SequenceDescription:
    repetition_time = _expand_like(repetition_time_s, flip, "TR")
    phases = _expand_like(phases_rad, flip, "phase")
    events = [SequenceEvent.rf(0.0, 0, RfUse.INVERSION, torch.pi, 0.0)]
    current_s = inversion_time_s
    for index in range(flip.numel()):
        events.append(
            SequenceEvent.rf(
                1e6 * current_s,
                0,
                RfUse.EXCITATION,
                flip[index],
                phases[index],
            )
        )
        acquired = center_index is None or index == center_index
        events.append(
            SequenceEvent.adc(
                1e6 * current_s,
                AdcRole.SINGLE if acquired else AdcRole.NON_ACQUIRED,
                phases[index],
                is_echo=acquired,
            )
        )
        current_s = current_s + repetition_time[index]
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * current_s,
        events=tuple(events),
        rf_definitions={0: _unit_flip_definition()},
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
    )


def _expand_like(value: Any, reference: torch.Tensor, name: str) -> torch.Tensor:
    output = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    output = output.expand_as(reference) if output.numel() == 1 else output
    if output.shape != reference.shape:
        raise ValueError(f"{name} must be scalar or match flip angles")
    return output
