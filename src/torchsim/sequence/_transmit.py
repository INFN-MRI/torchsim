"""Reducing a transmit array to the field it produces.

A pulse driven on several channels at once excites a voxel through the sum of
what each channel puts there:

    B1(v, s) = sum_c  S_c(v) * w_c(s)

for per-channel complex sensitivities ``S_c`` and the complex weights ``w_c``
of shim setting ``s``. The state machine's RF operator never sees the channels
themselves -- it takes one flip angle and one phase -- so the whole array
reduces to ``|B1|`` and ``arg(B1)``, which is precisely the pair of per-voxel
buffers the kernels already carry.

That is why transmit arrays are resolved here rather than in the kernels:
combining is a sum and two real functions of a complex number, all of which
autograd differentiates, so a gradient reaches the per-channel maps and the
shim weights through this without any kernel knowing they exist.

Summing the magnitudes and the phases separately is a different and wrong
operation: two channels driven in antiphase must leave the voxel untouched,
which only the complex sum reproduces.
"""

from __future__ import annotations

__all__ = ["channel_count", "transmit_field"]

from typing import Any

import torch

from ._description import EventType, SequenceDescription


def channel_count(description: SequenceDescription) -> int:
    """How many transmit channels the sequence's shims drive.

    One when the description declares no shim, which is the single-channel
    sequence every builder produces.

    Raises:
        ValueError: if the shim definitions disagree on their width.
    """
    widths = {
        len(shim.magnitudes) for shim in description.shim_definitions.values()
    } | {len(shim.phases_rad) for shim in description.shim_definitions.values()}
    if not widths:
        return 1
    if len(widths) != 1:
        raise ValueError(
            f"shim definitions disagree on channel count: {sorted(widths)}"
        )
    return widths.pop()


def transmit_field(
    description: SequenceDescription,
    b1: Any,
    b1_phase_rad: Any,
    device: torch.device,
) -> tuple[Any, Any]:
    """The field the array produces, as a per-voxel magnitude and phase.

    ``b1`` and ``b1_phase_rad`` describe one channel each along a leading axis
    when the sequence declares a multi-channel shim; a value with no such axis
    is taken to be the same sensitivity on every channel. A sequence with no
    shim passes them straight back.

    Returns the pair the kernels take, differentiable in both arguments and in
    the shim weights.

    Raises:
        ValueError: if the leading axis disagrees with the shim width.
        NotImplementedError: if the RF events reference more than one shim.
    """
    channels = channel_count(description)
    if channels == 1 and not description.shim_definitions:
        return b1, b1_phase_rad

    shim = _single_shim(description)
    weights = torch.polar(
        _as_tensor(shim.magnitudes, device), _as_tensor(shim.phases_rad, device)
    )
    sensitivity = torch.polar(
        *_aligned(
            _per_channel(b1, channels, device, "b1", 1.0),
            _per_channel(b1_phase_rad, channels, device, "b1_phase_rad", 0.0),
        )
    )
    field = (sensitivity * weights.reshape(-1, *(1,) * (sensitivity.dim() - 1))).sum(
        dim=0
    )
    return field.abs(), field.angle()


def _single_shim(description: SequenceDescription) -> Any:
    """The one shim every RF event in the sequence drives.

    Raises:
        NotImplementedError: if the events reference more than one, which needs
            a shim axis on the transmit buffers rather than one field.
        KeyError: if an event names a shim the description does not define.
    """
    referenced = {
        event.rf_shim_id
        for event in description.events
        if event.type is EventType.RF
    }
    if not referenced:
        referenced = set(description.shim_definitions)
    if len(referenced) > 1:
        raise NotImplementedError(
            "the transmit array is resolved into one field per voxel, so every "
            f"pulse must drive the same shim; this sequence drives "
            f"{sorted(referenced)}"
        )
    identifier = referenced.pop()
    if identifier not in description.shim_definitions:
        raise KeyError(f"no shim definition with id {identifier}")
    return description.shim_definitions[identifier]


def _as_tensor(values: Any, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _aligned(
    magnitude: torch.Tensor, phase: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both laid out over the same voxel axes, the channel axis kept leading.

    Broadcasting aligns to the right, which would put a bare per-channel value
    against the voxels rather than the channels, so the padding goes in behind
    the channel axis rather than in front of it.
    """
    rank = max(magnitude.dim(), phase.dim())
    return _padded(magnitude, rank), _padded(phase, rank)


def _padded(value: torch.Tensor, rank: int) -> torch.Tensor:
    if value.dim() == rank:
        return value
    return value.reshape(
        value.shape[0], *(1,) * (rank - value.dim()), *value.shape[1:]
    )


def _per_channel(
    value: Any, channels: int, device: torch.device, name: str, identity: float
) -> torch.Tensor:
    """``value`` laid out with a leading channel axis of the declared width."""
    if value is None:
        value = identity
    tensor = _as_tensor(value, device)
    if tensor.dim() == 0:
        return tensor.reshape(1).expand(channels)
    if tensor.shape[0] == channels:
        return tensor
    if channels == 1:
        return tensor.reshape(1, *tensor.shape)
    raise ValueError(
        f"{name} must lead with the {channels} transmit channels the shim "
        f"declares, got shape {tuple(tensor.shape)}"
    )
