"""The packed state machine written out in torch, as an oracle for the kernels.

Every tensor operation here is one autograd knows how to differentiate, to any
order, so this doubles as the reference for the analytic first- and
second-order adjoints. It takes the same packed buffers the kernels do --
``_pack_events`` output, and prepared tissue -- so a parity test can feed both
from one place.

It is deliberately slow and deliberately naive: it composes whole-array
operations per event with no fusion and no state truncation, which is what
makes it independent of the kernels it checks. Both two-pool steps go further
and reach their exponential through ``torch.matrix_exp``, so the closed form
the kernels evaluate is checked against a different algorithm and not against
a second copy of itself.
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
    lineshape: Any = None,
    exchanging: bool = False,
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
    lineshape
        A :class:`~torchsim.sequence._lineshape.LineshapeTable`, or ``None``
        for a single pool. Given one, the longitudinal step carries a bound
        pool alongside the free water and each pulse saturates it.
    exchanging
        Whether to carry a chemically exchanging pool: a second pool with a
        transverse pair of its own, its own T2 and chemical shift, which a
        pulse rotates rather than saturates. Given beside a ``lineshape`` this
        is the three-pool system, whose longitudinal step is a 3x3 while its
        transverse one stays the 2x2 the exchanging pool alone needs.

    Returns
    -------
    torch.Tensor
        Complex signal of shape ``(voxels, recorded echoes)``.
    """
    (
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, damping, velocity,
        bound_fraction, exchange_rate, t1_bound,
        pool_b_fraction, pool_b_exchange, t1_pool_b, t2_pool_b, pool_b_shift,
    ) = tissue
    # Free water is pool a, the chemically exchanging pool b and the semisolid
    # pool c. ``bound`` names whichever second pool the longitudinal step pairs
    # the free water with when there is only one of them.
    three = lineshape is not None and exchanging
    semisolid_fraction = bound_fraction if three else None
    semisolid_exchange = exchange_rate if three else None
    t1_semisolid = t1_bound if three else None
    if exchanging:
        bound_fraction = pool_b_fraction
        exchange_rate = pool_b_exchange
        t1_bound = t1_pool_b
    free_fraction = (
        1.0 - bound_fraction - semisolid_fraction if three else 1.0 - bound_fraction
    )
    # ``b1`` and ``b1_phase`` hold one row per shim the sequence drives, so a
    # pulse reads the field of the shim its event names.
    transmit = b1.reshape(-1, t1.numel())
    transmit_phase = b1_phase.reshape(-1, t1.numel())
    flow = velocity * geometry.flow_scale
    washout = velocity.abs() * geometry.washout_scale
    (
        duration, kind, flip, phase, action, _output_index, shim_index,
        saturation, rf_frequency,
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
    bound = torch.zeros_like(fplus)
    bound_plus = torch.zeros_like(fplus)
    bound_minus = torch.zeros_like(fplus)
    semisolid = torch.zeros_like(fplus)
    across_generator = (
        _transverse_generator(
            t2, t2_pool_b, exchange_rate, bound_fraction, free_fraction,
            pool_b_shift,
        )
        if exchanging
        else None
    )
    triple = (
        _three_pool_generator(
            t1, t1_pool_b, t1_semisolid, exchange_rate, semisolid_exchange,
            bound_fraction, semisolid_fraction,
        )
        if three
        else None
    )
    if lineshape is None and not exchanging:
        longitudinal = longitudinal + _at_order_zero(longitudinal, 1.0)
        generator = None
    else:
        # Equilibrium is split between the pools, so the free water starts at
        # what the bound pool leaves it.
        longitudinal = longitudinal + _at_order_zero(longitudinal, free_fraction)
        bound = bound + _at_order_zero(bound, bound_fraction)
        if three:
            semisolid = semisolid + _at_order_zero(semisolid, semisolid_fraction)
        generator = _exchange_generator(
            t1, t1_bound, exchange_rate, bound_fraction
        )
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
        if across_generator is None:
            fplus = fplus * off[:, None] * transverse_damping * transverse_phase
            fminus = (
                fminus * off.conj()[:, None] * transverse_damping
                * transverse_phase.conj()
            )
        else:
            # The relaxation of both pools sits inside the operator; what stays
            # outside is the off-resonance the whole voxel takes and the
            # per-order damping and turn.
            across = torch.matrix_exp(across_generator * dt) * wout[:, None, None]
            shared = (
                torch.exp(-2j * torch.pi * b0 * dt)[:, None]
                * transverse_damping * transverse_phase
            )
            free_plus = (
                across[:, 0, 0, None] * fplus + across[:, 0, 1, None] * bound_plus
            )
            pool_plus = (
                across[:, 1, 0, None] * fplus + across[:, 1, 1, None] * bound_plus
            )
            # ``F-`` follows the conjugate of the operator entry by entry, not
            # its transpose: it is the conjugate state.
            conjugated = across.conj()
            free_minus = (
                conjugated[:, 0, 0, None] * fminus
                + conjugated[:, 0, 1, None] * bound_minus
            )
            pool_minus = (
                conjugated[:, 1, 0, None] * fminus
                + conjugated[:, 1, 1, None] * bound_minus
            )
            fplus = free_plus * shared
            bound_plus = pool_plus * shared
            fminus = free_minus * shared.conj()
            bound_minus = pool_minus * shared.conj()
        carried = longitudinal_damping * longitudinal_phase
        if generator is None:
            longitudinal = longitudinal * e1[:, None] * carried
            # Order zero is undamped, so recovery is unaffected by diffusion.
            longitudinal = longitudinal + _at_order_zero(longitudinal, 1.0 - e1)
        elif three:
            # All three pools mix over the interval, which is one 3x3
            # exponential for the whole event; the per-order damping and turn
            # multiply what it leaves.
            operator, restored = _three_pool_step(
                triple,
                torch.stack(
                    (free_fraction, bound_fraction, semisolid_fraction), dim=-1
                ),
                torch.stack(
                    (1000.0 / t1, 1000.0 / t1_pool_b, 1000.0 / t1_semisolid),
                    dim=-1,
                ),
                dt,
                wout,
            )
            pools = torch.stack((longitudinal, bound, semisolid), dim=-2)
            mixed = torch.einsum("vij,vjs->vis", operator.to(pools.dtype), pools)
            longitudinal = mixed[:, 0] * carried + _at_order_zero(
                longitudinal, restored[:, 0]
            )
            bound = mixed[:, 1] * carried + _at_order_zero(bound, restored[:, 1])
            semisolid = mixed[:, 2] * carried + _at_order_zero(
                semisolid, restored[:, 2]
            )
        else:
            # Exchange mixes the two pools over the interval, which is one 2x2
            # exponential for the whole event; the per-order damping and turn
            # multiply what it leaves.
            operator, restored = _two_pool_step(
                generator, bound_fraction, t1, t1_bound, dt, wout
            )
            free_row = (
                operator[:, 0, 0, None] * longitudinal
                + operator[:, 0, 1, None] * bound
            )
            bound_row = (
                operator[:, 1, 0, None] * longitudinal
                + operator[:, 1, 1, None] * bound
            )
            longitudinal = free_row * carried + _at_order_zero(
                longitudinal, restored[:, 0]
            )
            bound = bound_row * carried + _at_order_zero(bound, restored[:, 1])

        event_action = int(action[event])
        if event_action & _PRE_SHIFT:
            fplus, fminus = _shift(fplus, fminus)
            if exchanging:
                bound_plus, bound_minus = _shift(bound_plus, bound_minus)
        event_kind = int(kind[event])
        if event_kind == 1:
            if event_action & _INVERSION:
                longitudinal = -inversion_efficiency[:, None] * longitudinal
                if exchanging:
                    # A chemically exchanging pool is free water, so a sweep
                    # turns it over rather than saturating it.
                    bound = -inversion_efficiency[:, None] * bound
            else:
                row = int(shim_index[event])
                alpha = flip[event] * transmit[row]
                phi = phase[event] + transmit_phase[row]
                if lineshape is not None:
                    # The bound pool absorbs the power the pulse deposits, so
                    # it reads the bare flip the transmit field gives the
                    # voxel rather than the slice-shaped rotation below.
                    absorbed = torch.exp(
                        saturation[event] * alpha.square()
                        * lineshape.at(rf_frequency[event] - b0)
                    )
                    if three:
                        semisolid = semisolid * absorbed[:, None]
                    else:
                        bound = bound * absorbed[:, None]
                if profile is None:
                    fplus, fminus, longitudinal = _rotate(
                        fplus, fminus, longitudinal, alpha, phi
                    )
                    if exchanging:
                        bound_plus, bound_minus, bound = _rotate(
                            bound_plus, bound_minus, bound, alpha, phi
                        )
                else:
                    # The table is built at zero RF phase, which turns the
                    # rotation axis and so multiplies ``b`` alone.
                    spinor_a, spinor_b = profile.at(slice_index, alpha)
                    spun = spinor_b * torch.exp(-1j * phi)
                    fplus, fminus, longitudinal = _rotate_spinor(
                        fplus, fminus, longitudinal, spinor_a, spun
                    )
                    if exchanging:
                        bound_plus, bound_minus, bound = _rotate_spinor(
                            bound_plus, bound_minus, bound, spinor_a, spun
                        )
        elif event_kind == 2 and event_action & _RECORD:
            # A coil sees the whole voxel, so what it records is the sum over
            # pools; each pool's share is already in its own state.
            recorded = fplus[:, 0] + bound_plus[:, 0] if exchanging else fplus[:, 0]
            signals.append(m0 * recorded * torch.exp(-1j * phase[event]))
        if event_action & _POST_SHIFT:
            fplus, fminus = _shift(fplus, fminus)
            if exchanging:
                bound_plus, bound_minus = _shift(bound_plus, bound_minus)
        if event_action & _SPOIL_AFTER:
            fplus = torch.zeros_like(fplus)
            fminus = torch.zeros_like(fminus)
            if exchanging:
                bound_plus = torch.zeros_like(bound_plus)
                bound_minus = torch.zeros_like(bound_minus)
        elif event_action & _SHIFT_AFTER:
            fplus, fminus = _shift(fplus, fminus)
            if exchanging:
                bound_plus, bound_minus = _shift(bound_plus, bound_minus)

    if not signals:
        return torch.empty(
            (atom_count, output_count), dtype=torch.complex64, device=t1.device
        )
    return torch.stack(signals, dim=-1)


def _at_order_zero(like: torch.Tensor, value: Any) -> torch.Tensor:
    """A state vector carrying ``value`` at order zero and nothing elsewhere."""
    orders = torch.zeros(like.shape[-1], dtype=like.dtype, device=like.device)
    orders[0] = 1.0
    carried = torch.as_tensor(value, device=like.device).to(like.dtype)
    return carried.reshape(-1, 1) * orders


def _exchange_generator(
    t1: torch.Tensor,
    t1_bound: torch.Tensor,
    exchange_rate: torch.Tensor,
    bound_fraction: torch.Tensor,
) -> torch.Tensor:
    """``K - diag(R1)``, the generator of the two-pool longitudinal step.

    A pool leaves at the rate scaled by the *other* pool's fraction, so the
    exchange part conserves the total magnetization on its own.
    """
    free = 1.0 - bound_fraction
    kab = exchange_rate * bound_fraction
    kba = exchange_rate * free
    return torch.stack(
        (
            torch.stack((-kab - 1000.0 / t1, kba), dim=-1),
            torch.stack((kab, -kba - 1000.0 / t1_bound), dim=-1),
        ),
        dim=-2,
    )


def _transverse_generator(
    t2: torch.Tensor,
    t2_pool_b: torch.Tensor,
    exchange_rate: torch.Tensor,
    fraction: torch.Tensor,
    free: torch.Tensor,
    shift_hz: torch.Tensor,
) -> torch.Tensor:
    """``K - diag(R2) - 2 pi i diag(df)``, the transverse step's generator.

    Only pool b's offset appears: pool a sits at whatever off-resonance the
    free precession already carries the whole voxel through.
    """
    kab = exchange_rate * fraction
    kba = exchange_rate * free
    rates = torch.stack(
        (
            torch.stack((-kab - 1000.0 / t2, kba), dim=-1),
            torch.stack((kab, -kba - 1000.0 / t2_pool_b), dim=-1),
        ),
        dim=-2,
    ).to(torch.complex64)
    offsets = torch.stack((torch.zeros_like(shift_hz), shift_hz), dim=-1)
    return rates + torch.diag_embed(-2j * torch.pi * offsets)


def _three_pool_generator(
    t1_ms,
    t1_pool_b_ms,
    t1_semisolid_ms,
    exchange_b,
    exchange_c,
    fraction_b,
    fraction_c,
) -> torch.Tensor:
    """``K - diag(R1)`` for free water beside both second pools.

    Each second pool exchanges with the free water and not with the other, so
    a pool leaves at the rate scaled by the *other* pool's fraction and the
    exchange part conserves the total on its own.
    """
    free = 1.0 - fraction_b - fraction_c
    kab, kba = exchange_b * fraction_b, exchange_b * free
    kac, kca = exchange_c * fraction_c, exchange_c * free
    zero = torch.zeros_like(free)
    return torch.stack(
        (
            torch.stack((-kab - kac - 1000.0 / t1_ms, kba, kca), dim=-1),
            torch.stack((kab, -kba - 1000.0 / t1_pool_b_ms, zero), dim=-1),
            torch.stack((kac, zero, -kca - 1000.0 / t1_semisolid_ms), dim=-1),
        ),
        dim=-2,
    )


def _three_pool_step(
    generator: torch.Tensor,
    equilibrium: torch.Tensor,
    rates: torch.Tensor,
    dt: Any,
    wout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The interval's three-pool operator and the recovery beside it.

    Reached through ``torch.matrix_exp`` and a linear solve, so neither the
    closed form the kernels evaluate nor the identity that saves them the
    solve is assumed here.
    """
    exponential = torch.matrix_exp(generator.double() * dt)
    settled = torch.linalg.solve(generator.double(), (equilibrium * rates).double())
    identity = torch.eye(3, dtype=torch.float64, device=generator.device)
    restored = ((exponential - identity) @ settled[..., None])[..., 0]
    attenuation = wout[:, None].double()
    return (
        (exponential * attenuation[..., None]).to(torch.complex64),
        (attenuation * restored + (1.0 - attenuation) * equilibrium.double()).to(
            torch.complex64
        ),
    )


def _two_pool_step(
    generator: torch.Tensor,
    bound_fraction: torch.Tensor,
    t1: torch.Tensor,
    t1_bound: torch.Tensor,
    dt: Any,
    wout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The interval's exchange operator and the recovery that goes with it.

    Reached through ``torch.matrix_exp`` and a linear solve -- Pade with
    scaling and squaring, and the textbook affine solution ``(e^{Lt} - I)
    L^-1 C`` -- so neither the closed form the kernels evaluate nor the
    identity that saves them the solve is assumed here.

    Washout replaces a share of the voxel with fully relaxed spins, which
    scales the operator and leaves the rest at equilibrium.
    """
    free = 1.0 - bound_fraction
    equilibrium = torch.stack((free, bound_fraction), dim=-1)
    source = torch.stack(
        (free * 1000.0 / t1, bound_fraction * 1000.0 / t1_bound), dim=-1
    )
    exponential = torch.matrix_exp(generator * dt)
    settled = torch.linalg.solve(generator, source)
    identity = torch.eye(2, dtype=generator.dtype, device=generator.device)
    restored = ((exponential - identity) @ settled[..., None])[..., 0]
    attenuation = wout[:, None]
    return (
        exponential * attenuation[..., None],
        attenuation * restored + (1.0 - attenuation) * equilibrium,
    )


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
