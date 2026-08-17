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

from typing import Any

import torch

from torchsim.sequence._accelerators import (
    _INVERSION,
    _POST_SHIFT,
    _PRE_SHIFT,
    _RECORD,
    _SHIFT_AFTER,
    _SPOIL_AFTER,
)
from torchsim.sequence._parameters import NO_GEOMETRY, Geometry

__all__ = ["simulate_packed"]


def simulate_packed(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    locations: int = 1,
) -> torch.Tensor:
    """Run one echo train and return its recorded signal.

    Parameters
    ----------
    tissue
        The prepared tissue properties in packing order, one entry per voxel.
    events
        ``(duration, kind, flip, phase, action, output_index)``, one entry per
        event, for a single train.
    state_count
        Configuration orders to carry.
    output_count
        Recorded echoes, used only for the shape of an empty result.
    geometry
        The two scales the velocity is read through, exactly as the kernels
        take them.
    profile
        A :class:`~torchsim.sequence._transition.TransitionTable`, or ``None``
        for the instantaneous pulse. Given one, a pulse turns through the
        rotation the table holds at its effective flip rather than through a
        flip and a phase.
    locations
        Slice positions each voxel was spread over. The prepared tissue runs
        voxel-major, so a voxel's position along the slice is its index modulo
        this, which is the row of the table it reads.

    Returns
    -------
    torch.Tensor
        Complex signal of shape ``(voxels, recorded echoes)``.
    """
    (
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, damping, velocity,
        _bound_fraction, _exchange_rate, _t1_bound,
    ) = tissue
    # ``b1`` and ``b1_phase`` hold one row per shim the sequence drives, so a
    # pulse reads the field of the shim its event names.
    transmit = b1.reshape(-1, t1.numel())
    transmit_phase = b1_phase.reshape(-1, t1.numel())
    flow = velocity * geometry.flow_scale
    washout = velocity.abs() * geometry.washout_scale
    (
        duration, kind, flip, phase, action, _output_index, shim_index,
        _saturation, _rf_frequency,
    ) = events
    atom_count = t1.numel()
    shape = (atom_count, state_count)
    # The dephasing order each state sits at, and the b-factor weights that
    # follow from it: a state travelling from order l to l+1 over the interval
    # accumulates the transverse weight, while a longitudinal state stays put.
    order = torch.arange(state_count, dtype=torch.float32, device=t1.device)
    longitudinal_weight = order.square()
    transverse_weight = order.square() + order + 1.0 / 3.0
    # Flow turns each order through a phase instead of damping it, and the
    # transverse states sit half an order further along the gradient.
    longitudinal_turn = order
    transverse_turn = order + 0.5
    fplus = torch.zeros(shape, dtype=torch.complex64, device=t1.device)
    fminus = torch.zeros_like(fplus)
    longitudinal = torch.zeros_like(fplus)
    longitudinal[:, 0] = 1.0
    signals = []
    slice_index = (
        torch.arange(atom_count, device=t1.device) % locations
        if profile is not None
        else None
    )

    for event in range(kind.numel()):
        dt = duration[event]
        # Inflowing spins are fully relaxed and unexcited, which makes washout
        # a scaling of both relaxation factors and nothing more: the affine
        # recovery term ``1 - e1`` already carries the magnetization they bring.
        wout = 1.0 - (washout * dt).clamp(max=1.0)
        e1 = torch.exp(-(1000.0 / t1) * dt) * wout
        e2 = torch.exp(-(1000.0 / t2) * dt) * wout
        off = e2 * torch.exp(-2j * torch.pi * b0 * dt)
        b_factor = (damping * dt)[:, None]
        transverse_damping = torch.exp(-b_factor * transverse_weight[None, :])
        longitudinal_damping = torch.exp(-b_factor * longitudinal_weight[None, :])
        turn = (flow * dt)[:, None]
        transverse_phase = torch.exp(-1j * turn * transverse_turn[None, :])
        longitudinal_phase = torch.exp(-1j * turn * longitudinal_turn[None, :])
        fplus = fplus * off[:, None] * transverse_damping * transverse_phase
        fminus = (
            fminus * off.conj()[:, None] * transverse_damping
            * transverse_phase.conj()
        )
        longitudinal = (
            longitudinal * e1[:, None] * longitudinal_damping * longitudinal_phase
        )
        # Order zero is undamped, so recovery is unaffected by diffusion.
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
                row = int(shim_index[event])
                alpha = flip[event] * transmit[row]
                phi = phase[event] + transmit_phase[row]
                if profile is None:
                    fplus, fminus, longitudinal = _rotate(
                        fplus, fminus, longitudinal, alpha, phi
                    )
                else:
                    # The table is built at zero RF phase, which turns the
                    # rotation axis and so multiplies ``b`` alone.
                    spinor_a, spinor_b = profile.at(slice_index, alpha)
                    fplus, fminus, longitudinal = _rotate_spinor(
                        fplus,
                        fminus,
                        longitudinal,
                        spinor_a,
                        spinor_b * torch.exp(-1j * phi),
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


def _rotate_spinor(
    fplus: torch.Tensor,
    fminus: torch.Tensor,
    longitudinal: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The rotation named by its Cayley-Klein pair rather than by flip angle."""
    a = a[:, None]
    b = b[:, None]
    t00 = a.conj().square()
    t01 = -b.conj().square()
    t02 = -2.0 * (a * b).conj()
    t10 = -b.square()
    t11 = a.square()
    t12 = -2.0 * a * b
    t20 = a.conj() * b
    t21 = a * b.conj()
    t22 = (a.abs().square() - b.abs().square()).to(a.dtype)
    old_plus, old_minus, old_z = fplus, fminus, longitudinal
    return (
        t00 * old_plus + t01 * old_minus + t02 * old_z,
        t10 * old_plus + t11 * old_minus + t12 * old_z,
        t20 * old_plus + t21 * old_minus + t22 * old_z,
    )


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
