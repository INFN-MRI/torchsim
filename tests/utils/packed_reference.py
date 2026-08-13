"""The packed state machine written out in torch, as an oracle for the kernels.

Every tensor operation here is one autograd knows how to differentiate, to any
order, so this doubles as the reference for the analytic first- and
second-order adjoints. It takes the same packed buffers the kernels do --
``_pack_events`` output, and prepared tissue -- so a parity test can feed both
from one place.

It is deliberately slow and deliberately naive: it composes whole-array
operations per event with no fusion and no state truncation, which is what
makes it independent of the kernels it checks.
"""

from __future__ import annotations

import torch

from torchsim.sequence._accelerators import (
    _INVERSION,
    _POST_SHIFT,
    _PRE_SHIFT,
    _RECORD,
    _SHIFT_AFTER,
    _SPOIL_AFTER,
)

__all__ = ["simulate_packed"]


def simulate_packed(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
) -> torch.Tensor:
    """Run one echo train and return its recorded signal.

    Parameters
    ----------
    tissue
        ``(t1, t2, m0, b1, b1_phase, b0, inversion_efficiency)``, one entry per
        voxel.
    events
        ``(duration, kind, flip, phase, action, output_index)``, one entry per
        event, for a single train.
    state_count
        Configuration orders to carry.
    output_count
        Recorded echoes, used only for the shape of an empty result.

    Returns
    -------
    torch.Tensor
        Complex signal of shape ``(voxels, recorded echoes)``.
    """
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
            fplus, fminus = _shift(fplus, fminus)
        event_kind = int(kind[event])
        if event_kind == 1:
            if event_action & _INVERSION:
                longitudinal = -inversion_efficiency[:, None] * longitudinal
            else:
                alpha = flip[event] * b1
                phi = phase[event] + b1_phase
                fplus, fminus, longitudinal = _rotate(
                    fplus, fminus, longitudinal, alpha, phi
                )
        elif event_kind == 2 and event_action & _RECORD:
            signals.append(m0 * fplus[:, 0] * torch.exp(-1j * phase[event]))
        if event_action & _POST_SHIFT:
            fplus, fminus = _shift(fplus, fminus)
        if event_action & _SPOIL_AFTER:
            fplus = torch.zeros_like(fplus)
            fminus = torch.zeros_like(fminus)
        elif event_action & _SHIFT_AFTER:
            fplus, fminus = _shift(fplus, fminus)

    if not signals:
        return torch.empty(
            (atom_count, output_count), dtype=torch.complex64, device=t1.device
        )
    return torch.stack(signals, dim=-1)


def _shift(
    fplus: torch.Tensor, fminus: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(fplus[:, :1])
    shifted_minus = torch.cat((fminus[:, 1:], zero), dim=-1)
    shifted_plus = torch.cat((shifted_minus[:, :1].conj(), fplus[:, :-1]), dim=-1)
    return shifted_plus, shifted_minus


def _rotate(
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
