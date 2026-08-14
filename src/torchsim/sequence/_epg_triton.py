"""Fused Triton kernel for inference-only EPG state machines."""

from __future__ import annotations

__all__: list[str] = []

import torch

from ._accelerators import _shim_count, _train_count
import triton
import triton.language as tl

from ._parameters import (
    NO_GEOMETRY,
    Geometry,
    tissue_gradient_bases,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from ._parameters import TISSUE_COUNT as _TISSUE_PARAMETERS
from ._parameters import TRANSMIT_INPUTS as _TRANSMIT_INPUTS

# Triton reads globals only through its own constexpr wrapper.
_TISSUE_COUNT = tl.constexpr(_TISSUE_PARAMETERS)

# The gradient plane holds a row of voxels per tissue parameter, except that
# the transmit pair holds one per shim. Both sit ahead of everything that
# widens, so a plane's row is its parameter index shifted by the rows the pair
# added ahead of it.
_B1_ROW = tl.constexpr(_TRANSMIT_INPUTS[0])
_B1_PHASE_ROW = tl.constexpr(_TRANSMIT_INPUTS[1])


@triton.jit
def _shift(
    fplus_real,
    fplus_imag,
    fminus_real,
    fminus_imag,
    scratch_fplus_real,
    scratch_fplus_imag,
    scratch_fminus_real,
    scratch_fminus_imag,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    tl.store(scratch_fplus_real + state, fplus_real, mask=state_mask)
    tl.store(scratch_fplus_imag + state, fplus_imag, mask=state_mask)
    tl.store(scratch_fminus_real + state, fminus_real, mask=state_mask)
    tl.store(scratch_fminus_imag + state, fminus_imag, mask=state_mask)
    tl.debug_barrier()

    plus_real = tl.load(
        scratch_fplus_real + state - 1,
        mask=(state > 0) & state_mask,
        other=0.0,
    )
    plus_imag = tl.load(
        scratch_fplus_imag + state - 1,
        mask=(state > 0) & state_mask,
        other=0.0,
    )
    minus_real = tl.load(
        scratch_fminus_real + state + 1,
        mask=(state + 1 < state_count) & state_mask,
        other=0.0,
    )
    minus_imag = tl.load(
        scratch_fminus_imag + state + 1,
        mask=(state + 1 < state_count) & state_mask,
        other=0.0,
    )
    plus_real = tl.where(state == 0, minus_real, plus_real)
    plus_imag = tl.where(state == 0, -minus_imag, plus_imag)
    return plus_real, plus_imag, minus_real, minus_imag


@triton.jit
def _shift_real(
    plus,
    minus,
    scratch_plus,
    scratch_minus,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    tl.store(scratch_plus + state, plus, mask=state_mask)
    tl.store(scratch_minus + state, minus, mask=state_mask)
    tl.debug_barrier()

    shifted_plus = tl.load(
        scratch_plus + state - 1,
        mask=(state > 0) & state_mask,
        other=0.0,
    )
    shifted_minus = tl.load(
        scratch_minus + state + 1,
        mask=(state + 1 < state_count) & state_mask,
        other=0.0,
    )
    return tl.where(state == 0, -shifted_minus, shifted_plus), shifted_minus


# ---------------------------------------------------------------------------
# Dual complex arithmetic.
#
# A dual complex number is four planes: the real and imaginary parts of the
# value, then of the tangent. Triton has no structs, so every quantity travels
# as four separate registers and these helpers keep the bookkeeping in one
# place rather than spread through the kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _damping(rate, dt, order):
    """Longitudinal and transverse diffusion damping for one interval.

    ``rate`` already carries the sequence's gradient geometry, so an interval's
    b-factor is that rate times its duration. Order zero has no longitudinal
    weight, which is what keeps the recovery term undamped.
    """
    b_factor = rate * dt
    squared = order * order
    return (
        tl.exp(-b_factor * squared),
        tl.exp(-b_factor * (squared + order + 0.3333333333333333)),
    )


@triton.jit
def _flow(rate, dt, order):
    """Phase each dephasing order turns through over one interval.

    ``rate`` already carries the sequence's gradient geometry, so it is the
    winding per unit order per second: a longitudinal state at order l turns
    through ``l * rate * dt``. The transverse states sit half an order further
    along the gradient, which is where the extra half turn comes from. Order
    zero is left alone while longitudinal, so the recovery term is unaffected.
    """
    turn = rate * dt
    return -order * turn, -(order + 0.5) * turn


@triton.jit
def _washout(rate, dt):
    """The fraction of a voxel's spins that stay put over one interval.

    Inflowing spins are taken to be fully relaxed and unexcited, which makes
    washout an affine map of the shape longitudinal recovery already has:

        wout * (Z * e1 + (1 - e1)) + win  ==  Z * (e1 * wout) + (1 - e1 * wout)

    so scaling both relaxation factors by it carries the whole term. Clamped at
    one, past which the interval has replaced the voxel outright.
    """
    return 1.0 - tl.minimum(rate * dt, 1.0)


@triton.jit
def _washout_jvp(rate, rate_tangent, dt, dt_tangent):
    """The same fraction and its directional derivative."""
    fraction = rate * dt
    live = fraction < 1.0
    return (
        tl.where(live, 1.0 - fraction, 0.0),
        tl.where(live, -(rate_tangent * dt + rate * dt_tangent), 0.0),
    )


@triton.jit
def _damping_jvp(rate, rate_tangent, dt, dt_tangent, order):
    """Diffusion damping and its directional derivative, per state order."""
    b_factor = rate * dt
    b_tangent = rate_tangent * dt + rate * dt_tangent
    squared = order * order
    transverse_weight = squared + order + 0.3333333333333333
    damp_z = tl.exp(-b_factor * squared)
    damp_t = tl.exp(-b_factor * transverse_weight)
    return (
        damp_z,
        damp_z * (-b_tangent * squared),
        damp_t,
        damp_t * (-b_tangent * transverse_weight),
    )


@triton.jit
def _complex_mul(a_real, a_imag, b_real, b_imag):
    return a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real


@triton.jit
def _dual_mul(a_vr, a_vi, a_tr, a_ti, b_vr, b_vi, b_tr, b_ti):
    """Product of two dual complex numbers."""
    value_real, value_imag = _complex_mul(a_vr, a_vi, b_vr, b_vi)
    left_real, left_imag = _complex_mul(a_tr, a_ti, b_vr, b_vi)
    right_real, right_imag = _complex_mul(a_vr, a_vi, b_tr, b_ti)
    return value_real, value_imag, left_real + right_real, left_imag + right_imag


@triton.jit
def _dual_scale(scale_value, scale_tangent, vr, vi, tr, ti):
    """A real dual number times a complex one."""
    return (
        scale_value * vr,
        scale_value * vi,
        scale_tangent * vr + scale_value * tr,
        scale_tangent * vi + scale_value * ti,
    )


@triton.jit
def _dual_times_i(vr, vi, tr, ti):
    return -vi, vr, -ti, tr


@triton.jit
def _dual_real_conj_mul(a_vr, a_vi, a_tr, a_ti, b_vr, b_vi, b_tr, b_ti):
    """``real_part(conj(a) * b)``, the contraction an adjoint asks for."""
    value = a_vr * b_vr + a_vi * b_vi
    tangent = a_tr * b_vr + a_ti * b_vi + a_vr * b_tr + a_vi * b_ti
    return value, tangent


@triton.jit
def _dual_polar(angle_value, angle_tangent):
    """``exp(i * angle)`` for a real dual angle."""
    cosine = tl.cos(angle_value)
    sine = tl.sin(angle_value)
    return cosine, sine, -sine * angle_tangent, cosine * angle_tangent


@triton.jit
def _shift_adjoint(
    plus_bar_real,
    plus_bar_imag,
    minus_bar_real,
    minus_bar_imag,
    scratch_pr,
    scratch_pi,
    scratch_mr,
    scratch_mi,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    """Transpose of ``_shift``.

    The conjugate refill at order zero sends the incoming plus adjoint back
    onto minus, conjugated, at the index the minus shift moves it to.
    """
    tl.store(scratch_pr + state, plus_bar_real, mask=state_mask)
    tl.store(scratch_pi + state, plus_bar_imag, mask=state_mask)
    tl.store(scratch_mr + state, minus_bar_real, mask=state_mask)
    tl.store(scratch_mi + state, minus_bar_imag, mask=state_mask)
    tl.debug_barrier()

    carry_real = tl.load(scratch_pr, mask=state_mask, other=0.0)
    carry_imag = -tl.load(scratch_pi, mask=state_mask, other=0.0)
    forward = (state + 1 < state_count) & state_mask
    backward = (state > 0) & state_mask
    shifted_pr = tl.load(scratch_pr + state + 1, mask=forward, other=0.0)
    shifted_pi = tl.load(scratch_pi + state + 1, mask=forward, other=0.0)
    shifted_mr = tl.load(scratch_mr + state - 1, mask=backward, other=0.0)
    shifted_mi = tl.load(scratch_mi + state - 1, mask=backward, other=0.0)
    shifted_mr = tl.where(state == 1, shifted_mr + carry_real, shifted_mr)
    shifted_mi = tl.where(state == 1, shifted_mi + carry_imag, shifted_mi)
    return shifted_pr, shifted_pi, shifted_mr, shifted_mi


@triton.jit
def _shift_real_adjoint(
    plus_bar,
    minus_bar,
    scratch_plus,
    scratch_minus,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    """Transpose of ``_shift_real``.

    The ``a0 = -b0`` coupling sends the incoming plus adjoint back onto minus,
    at the index the minus shift moves it to.
    """
    tl.store(scratch_plus + state, plus_bar, mask=state_mask)
    tl.store(scratch_minus + state, minus_bar, mask=state_mask)
    tl.debug_barrier()

    carry = -tl.load(scratch_plus, mask=state_mask, other=0.0)
    shifted_plus = tl.load(
        scratch_plus + state + 1,
        mask=(state + 1 < state_count) & state_mask,
        other=0.0,
    )
    shifted_minus = tl.load(
        scratch_minus + state - 1,
        mask=(state > 0) & state_mask,
        other=0.0,
    )
    shifted_minus = tl.where(state == 1, shifted_minus + carry, shifted_minus)
    return shifted_plus, shifted_minus


@triton.jit
def _rotation_block(
    a_value, a_tangent,
    b_value, b_tangent,
    c_value, c_tangent,
    d_value, d_tangent,
    p1r, p1i, p1tr, p1ti,
    p2r, p2i, p2tr, p2ti,
    pcr, pci, pctr, pcti,
):
    """Seven of the nine rotation coefficients; the rest follow by symmetry.

    ``t11`` repeats ``t00`` and ``t10`` is the conjugate of ``t01``, so the
    caller derives those. Feeding ``(cos, sin)`` gives the rotation itself and
    ``(sin, cos)`` rearranged gives its derivative in the flip angle, which is
    why this is one routine rather than two.
    """
    t00 = (a_value, 0.0 * a_value, a_tangent, 0.0 * a_tangent)
    t01 = _dual_scale(b_value, b_tangent, p2r, p2i, p2tr, p2ti)
    t02 = _dual_mul(
        0.0 * c_value, -c_value, 0.0 * c_tangent, -c_tangent, p1r, p1i, p1tr, p1ti
    )
    t12 = _dual_mul(
        0.0 * c_value, c_value, 0.0 * c_tangent, c_tangent, pcr, pci, pctr, pcti
    )
    t20 = _dual_mul(
        0.0 * c_value, -0.5 * c_value, 0.0 * c_tangent, -0.5 * c_tangent,
        pcr, pci, pctr, pcti,
    )
    t21 = _dual_mul(
        0.0 * c_value, 0.5 * c_value, 0.0 * c_tangent, 0.5 * c_tangent,
        p1r, p1i, p1tr, p1ti,
    )
    t22 = (d_value, 0.0 * d_value, d_tangent, 0.0 * d_tangent)
    return t00, t01, t02, t12, t20, t21, t22


@triton.jit
def _epg_vjp_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    b1_phase,
    b0,
    inversion_efficiency,
    diffusion,
    velocity,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    dot_t1,
    dot_t2,
    dot_m0,
    dot_b1,
    dot_b1_phase,
    dot_b0,
    dot_inversion_efficiency,
    dot_diffusion,
    dot_velocity,
    dot_duration,
    dot_flip,
    dot_phase,
    grad_output_real,
    grad_output_imag,
    grad_tissue_value,
    grad_tissue_tangent,
    grad_flip_value,
    grad_flip_tangent,
    grad_phase_value,
    grad_phase_tangent,
    grad_duration_value,
    grad_duration_tangent,
    trajectory_vr,
    trajectory_vi,
    trajectory_tr,
    trajectory_ti,
    scratch_pr,
    scratch_pi,
    scratch_mr,
    scratch_mi,
    problem_base,
    problem_end,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    state_count: tl.constexpr,
    shims: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = problem_base + tl.program_id(0) * problems
    problem = problem + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < problem_end
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    train = problem // atom_count
    local = problem - problem_base
    scratch_offset = local * state_count
    sp_r = scratch_pr + scratch_offset
    sp_i = scratch_pi + scratch_offset
    sm_r = scratch_mr + scratch_offset
    sm_i = scratch_mi + scratch_offset
    record_stride = 3 * state_count
    trajectory = local * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    pvr = empty
    pvi = empty
    ptr = empty
    pti = empty
    mvr = empty
    mvi = empty
    mtr = empty
    mti = empty
    zvr = empty + tl.where(state == 0, 1.0, 0.0)
    zvi = empty
    ztr = empty
    zti = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = tl.load(m0 + atom, mask=active_atom, other=0.0)
    atom_b1 = tl.load(b1 + atom, mask=active_atom, other=1.0)
    atom_b1_phase = tl.load(b1_phase + atom, mask=active_atom, other=0.0)
    atom_b0 = tl.load(b0 + atom, mask=active_atom, other=0.0)
    atom_inv = tl.load(inversion_efficiency + atom, mask=active_atom, other=1.0)
    d_t1 = tl.load(dot_t1 + atom, mask=active_atom, other=0.0)
    d_t2 = tl.load(dot_t2 + atom, mask=active_atom, other=0.0)
    d_m0 = tl.load(dot_m0 + atom, mask=active_atom, other=0.0)
    d_b1 = tl.load(dot_b1 + atom, mask=active_atom, other=0.0)
    d_b1_phase = tl.load(dot_b1_phase + atom, mask=active_atom, other=0.0)
    d_b0 = tl.load(dot_b0 + atom, mask=active_atom, other=0.0)
    d_inv = tl.load(
        dot_inversion_efficiency + atom, mask=active_atom, other=0.0
    )
    atom_damping = tl.load(diffusion + atom, mask=active_atom, other=0.0)
    d_damping = tl.load(dot_diffusion + atom, mask=active_atom, other=0.0)
    atom_velocity = tl.load(velocity + atom, mask=active_atom, other=0.0)
    d_velocity = tl.load(dot_velocity + atom, mask=active_atom, other=0.0)
    atom_flow = atom_velocity * flow_scale
    d_flow = d_velocity * flow_scale
    # |v| has no derivative at the origin, so a still voxel contributes none.
    direction = (atom_velocity > 0.0).to(tl.float32) - (atom_velocity < 0.0).to(
        tl.float32
    )
    atom_washout = tl.abs(atom_velocity) * washout_scale
    d_washout = direction * d_velocity * washout_scale
    order = state.to(tl.float32)
    longitudinal_weight = order * order
    transverse_weight = longitudinal_weight + order + 0.3333333333333333
    r1_value = 1000.0 / atom_t1
    r1_tangent = -1000.0 * d_t1 / (atom_t1 * atom_t1)
    r2_value = 1000.0 / atom_t2
    r2_tangent = -1000.0 * d_t2 / (atom_t2 * atom_t2)

    event_base = train * event_count
    for event in range(0, event_count):
        slot = trajectory + event * record_stride
        tl.store(trajectory_vr + slot, pvr, mask=state_mask)
        tl.store(trajectory_vi + slot, pvi, mask=state_mask)
        tl.store(trajectory_tr + slot, ptr, mask=state_mask)
        tl.store(trajectory_ti + slot, pti, mask=state_mask)
        tl.store(trajectory_vr + slot + minus_plane, mvr, mask=state_mask)
        tl.store(trajectory_vi + slot + minus_plane, mvi, mask=state_mask)
        tl.store(trajectory_tr + slot + minus_plane, mtr, mask=state_mask)
        tl.store(trajectory_ti + slot + minus_plane, mti, mask=state_mask)
        tl.store(trajectory_vr + slot + long_plane, zvr, mask=state_mask)
        tl.store(trajectory_vi + slot + long_plane, zvi, mask=state_mask)
        tl.store(trajectory_tr + slot + long_plane, ztr, mask=state_mask)
        tl.store(trajectory_ti + slot + long_plane, zti, mask=state_mask)

        dt_value = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        dt_tangent = tl.load(
            dot_duration + event_base + event, mask=active_atom, other=0.0
        )
        wout_value, wout_tangent = _washout_jvp(
            atom_washout, d_washout, dt_value, dt_tangent
        )
        dry1_value = tl.exp(-r1_value * dt_value)
        dry1_tangent = -dry1_value * (
            r1_value * dt_tangent + r1_tangent * dt_value
        )
        dry2_value = tl.exp(-r2_value * dt_value)
        dry2_tangent = -dry2_value * (
            r2_value * dt_tangent + r2_tangent * dt_value
        )
        e1_value = dry1_value * wout_value
        e1_tangent = dry1_tangent * wout_value + dry1_value * wout_tangent
        e2_value = dry2_value * wout_value
        e2_tangent = dry2_tangent * wout_value + dry2_value * wout_tangent
        damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
            atom_damping, d_damping, dt_value, dt_tangent, order
        )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t
        turn_z, turn_t = _flow(atom_flow, dt_value, order)
        d_turn = d_flow * dt_value + atom_flow * dt_tangent
        dturn_z = -order * d_turn
        dturn_t = -(order + 0.5) * d_turn
        szr, szi, sztr, szti = _dual_polar(turn_z, dturn_z)
        angle_value = (
            -2.0 * 3.141592653589793 * (atom_b0 * dt_value) + turn_t
        )
        angle_tangent = -2.0 * 3.141592653589793 * (
            d_b0 * dt_value + atom_b0 * dt_tangent
        ) + dturn_t
        qr, qi, qtr, qti = _dual_polar(angle_value, angle_tangent)
        ovr, ovi, otr, oti = _dual_scale(e2_value, e2_tangent, qr, qi, qtr, qti)
        lvr, lvi, ltr, lti = _dual_scale(e1_value, e1_tangent, szr, szi, sztr, szti)

        pvr, pvi, ptr, pti = _dual_mul(ovr, ovi, otr, oti, pvr, pvi, ptr, pti)
        mvr, mvi, mtr, mti = _dual_mul(ovr, -ovi, otr, -oti, mvr, mvi, mtr, mti)
        zvr, zvi, ztr, zti = _dual_mul(lvr, lvi, ltr, lti, zvr, zvi, ztr, zti)
        zvr += tl.where(state == 0, recovery_value, 0.0)
        ztr += tl.where(state == 0, recovery_tangent, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        svr, svi, wvr, wvi = _shift(
            pvr, pvi, mvr, mvi, sp_r, sp_i, sm_r, sm_i, state, state_mask, state_count
        )
        str_, sti, wtr, wti = _shift(
            ptr, pti, mtr, mti, sp_r, sp_i, sm_r, sm_i, state, state_mask, state_count
        )
        pvr = tl.where(pre_shift, svr, pvr)
        pvi = tl.where(pre_shift, svi, pvi)
        ptr = tl.where(pre_shift, str_, ptr)
        pti = tl.where(pre_shift, sti, pti)
        mvr = tl.where(pre_shift, wvr, mvr)
        mvi = tl.where(pre_shift, wvi, mvi)
        mtr = tl.where(pre_shift, wtr, mtr)
        mti = tl.where(pre_shift, wti, mti)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        ivr, ivi, itr, iti = _dual_scale(-atom_inv, -d_inv, zvr, zvi, ztr, zti)
        zvr = tl.where(invert, ivr, zvr)
        zvi = tl.where(invert, ivi, zvi)
        ztr = tl.where(invert, itr, ztr)
        zti = tl.where(invert, iti, zti)

        event_flip = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        event_dot_flip = tl.load(
            dot_flip + event_base + event, mask=active_atom, other=0.0
        )
        event_phase = tl.load(phase + event_base + event, mask=active_atom, other=0.0)
        event_dot_phase = tl.load(
            dot_phase + event_base + event, mask=active_atom, other=0.0
        )
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shims > 1:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            atom_b1_phase = tl.load(
                b1_phase + row + atom, mask=active_atom, other=0.0
            )
            d_b1 = tl.load(dot_b1 + row + atom, mask=active_atom, other=0.0)
            d_b1_phase = tl.load(
                dot_b1_phase + row + atom, mask=active_atom, other=0.0
            )
        alpha_value = event_flip * atom_b1
        alpha_tangent = event_dot_flip * atom_b1 + event_flip * d_b1
        phi_value = event_phase + atom_b1_phase
        phi_tangent = event_dot_phase + d_b1_phase
        cos_value = tl.cos(alpha_value)
        sin_value = tl.sin(alpha_value)
        cos_tangent = -sin_value * alpha_tangent
        sin_tangent = cos_value * alpha_tangent
        p1r, p1i, p1tr, p1ti = _dual_polar(phi_value, phi_tangent)
        p2r, p2i, p2tr, p2ti = _dual_mul(p1r, p1i, p1tr, p1ti, p1r, p1i, p1tr, p1ti)
        t00, t01, t02, t12, t20, t21, t22 = _rotation_block(
            0.5 * (1.0 + cos_value), 0.5 * cos_tangent,
            0.5 * (1.0 - cos_value), -0.5 * cos_tangent,
            sin_value, sin_tangent,
            cos_value, cos_tangent,
            p1r, p1i, p1tr, p1ti,
            p2r, p2i, p2tr, p2ti,
            p1r, -p1i, p1tr, -p1ti,
        )
        a0 = _dual_mul(t00[0], t00[1], t00[2], t00[3], pvr, pvi, ptr, pti)
        a1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], mvr, mvi, mtr, mti)
        a2 = _dual_mul(t02[0], t02[1], t02[2], t02[3], zvr, zvi, ztr, zti)
        b0_ = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], pvr, pvi, ptr, pti)
        b1_ = _dual_mul(t00[0], t00[1], t00[2], t00[3], mvr, mvi, mtr, mti)
        b2 = _dual_mul(t12[0], t12[1], t12[2], t12[3], zvr, zvi, ztr, zti)
        c0 = _dual_mul(t20[0], t20[1], t20[2], t20[3], pvr, pvi, ptr, pti)
        c1 = _dual_mul(t21[0], t21[1], t21[2], t21[3], mvr, mvi, mtr, mti)
        c2 = _dual_mul(t22[0], t22[1], t22[2], t22[3], zvr, zvi, ztr, zti)

        rotate = is_rf & ~is_inversion
        pvr = tl.where(rotate, a0[0] + a1[0] + a2[0], pvr)
        pvi = tl.where(rotate, a0[1] + a1[1] + a2[1], pvi)
        ptr = tl.where(rotate, a0[2] + a1[2] + a2[2], ptr)
        pti = tl.where(rotate, a0[3] + a1[3] + a2[3], pti)
        mvr = tl.where(rotate, b0_[0] + b1_[0] + b2[0], mvr)
        mvi = tl.where(rotate, b0_[1] + b1_[1] + b2[1], mvi)
        mtr = tl.where(rotate, b0_[2] + b1_[2] + b2[2], mtr)
        mti = tl.where(rotate, b0_[3] + b1_[3] + b2[3], mti)
        zvr = tl.where(rotate, c0[0] + c1[0] + c2[0], zvr)
        zvi = tl.where(rotate, c0[1] + c1[1] + c2[1], zvi)
        ztr = tl.where(rotate, c0[2] + c1[2] + c2[2], ztr)
        zti = tl.where(rotate, c0[3] + c1[3] + c2[3], zti)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        svr, svi, wvr, wvi = _shift(
            pvr, pvi, mvr, mvi, sp_r, sp_i, sm_r, sm_i, state, state_mask, state_count
        )
        str_, sti, wtr, wti = _shift(
            ptr, pti, mtr, mti, sp_r, sp_i, sm_r, sm_i, state, state_mask, state_count
        )
        pvr = tl.where(do_shift, svr, pvr)
        pvi = tl.where(do_shift, svi, pvi)
        ptr = tl.where(do_shift, str_, ptr)
        pti = tl.where(do_shift, sti, pti)
        mvr = tl.where(do_shift, wvr, mvr)
        mvi = tl.where(do_shift, wvi, mvi)
        mtr = tl.where(do_shift, wtr, mtr)
        mti = tl.where(do_shift, wti, mti)
        spoil = (event_action & 8) != 0
        pvr = tl.where(spoil, 0.0, pvr)
        pvi = tl.where(spoil, 0.0, pvi)
        ptr = tl.where(spoil, 0.0, ptr)
        pti = tl.where(spoil, 0.0, pti)
        mvr = tl.where(spoil, 0.0, mvr)
        mvi = tl.where(spoil, 0.0, mvi)
        mtr = tl.where(spoil, 0.0, mtr)
        mti = tl.where(spoil, 0.0, mti)

    # ---- reverse ----
    pbvr = empty
    pbvi = empty
    pbtr = empty
    pbti = empty
    mbvr = empty
    mbvi = empty
    mbtr = empty
    mbti = empty
    zbvr = empty
    zbvi = empty
    zbtr = empty
    zbti = empty
    zero = tl.zeros((problems, 1), tl.float32)
    g_diffv = zero
    g_difft = zero
    g_flowv = zero
    g_flowt = zero
    g_washv = zero
    g_washt = zero
    g_t1v = zero
    g_t1t = zero
    g_t2v = zero
    g_t2t = zero
    g_m0v = zero
    g_m0t = zero
    g_b1v = zero
    g_b1t = zero
    g_b1pv = zero
    g_b1pt = zero
    g_b0v = zero
    g_b0t = zero
    g_invv = zero
    g_invt = zero

    for reverse in range(0, event_count):
        event = event_count - 1 - reverse
        slot = trajectory + event * record_stride
        xpvr = tl.load(trajectory_vr + slot, mask=state_mask, other=0.0)
        xpvi = tl.load(trajectory_vi + slot, mask=state_mask, other=0.0)
        xptr = tl.load(trajectory_tr + slot, mask=state_mask, other=0.0)
        xpti = tl.load(trajectory_ti + slot, mask=state_mask, other=0.0)
        xmvr = tl.load(trajectory_vr + slot + minus_plane, mask=state_mask, other=0.0)
        xmvi = tl.load(trajectory_vi + slot + minus_plane, mask=state_mask, other=0.0)
        xmtr = tl.load(trajectory_tr + slot + minus_plane, mask=state_mask, other=0.0)
        xmti = tl.load(trajectory_ti + slot + minus_plane, mask=state_mask, other=0.0)
        xzvr = tl.load(trajectory_vr + slot + long_plane, mask=state_mask, other=0.0)
        xzvi = tl.load(trajectory_vi + slot + long_plane, mask=state_mask, other=0.0)
        xztr = tl.load(trajectory_tr + slot + long_plane, mask=state_mask, other=0.0)
        xzti = tl.load(trajectory_ti + slot + long_plane, mask=state_mask, other=0.0)

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event)
        dt_value = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        dt_tangent = tl.load(
            dot_duration + event_base + event, mask=active_atom, other=0.0
        )
        wout_value, wout_tangent = _washout_jvp(
            atom_washout, d_washout, dt_value, dt_tangent
        )
        dry1_value = tl.exp(-r1_value * dt_value)
        dry1_tangent = -dry1_value * (
            r1_value * dt_tangent + r1_tangent * dt_value
        )
        dry2_value = tl.exp(-r2_value * dt_value)
        dry2_tangent = -dry2_value * (
            r2_value * dt_tangent + r2_tangent * dt_value
        )
        e1_value = dry1_value * wout_value
        e1_tangent = dry1_tangent * wout_value + dry1_value * wout_tangent
        e2_value = dry2_value * wout_value
        e2_tangent = dry2_tangent * wout_value + dry2_value * wout_tangent
        damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
            atom_damping, d_damping, dt_value, dt_tangent, order
        )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t
        turn_z, turn_t = _flow(atom_flow, dt_value, order)
        d_turn = d_flow * dt_value + atom_flow * dt_tangent
        dturn_z = -order * d_turn
        dturn_t = -(order + 0.5) * d_turn
        szr, szi, sztr, szti = _dual_polar(turn_z, dturn_z)
        angle_value = (
            -2.0 * 3.141592653589793 * (atom_b0 * dt_value) + turn_t
        )
        angle_tangent = -2.0 * 3.141592653589793 * (
            d_b0 * dt_value + atom_b0 * dt_tangent
        ) + dturn_t
        qr, qi, qtr, qti = _dual_polar(angle_value, angle_tangent)
        ovr, ovi, otr, oti = _dual_scale(e2_value, e2_tangent, qr, qi, qtr, qti)
        lvr, lvi, ltr, lti = _dual_scale(e1_value, e1_tangent, szr, szi, sztr, szti)

        # Replay the intra-event stages from the recorded entry state.
        rpvr, rpvi, rptr, rpti = _dual_mul(
            ovr, ovi, otr, oti, xpvr, xpvi, xptr, xpti
        )
        rmvr, rmvi, rmtr, rmti = _dual_mul(
            ovr, -ovi, otr, -oti, xmvr, xmvi, xmtr, xmti
        )
        rzvr, rzvi, rztr, rzti = _dual_mul(
            lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti
        )
        rzvr += tl.where(state == 0, recovery_value, 0.0)
        rztr += tl.where(state == 0, recovery_tangent, 0.0)

        pre_shift = (event_action & 1) != 0
        svr, svi, wvr, wvi = _shift(
            rpvr, rpvi, rmvr, rmvi, sp_r, sp_i, sm_r, sm_i, state, state_mask,
            state_count,
        )
        str_, sti, wtr, wti = _shift(
            rptr, rpti, rmtr, rmti, sp_r, sp_i, sm_r, sm_i, state, state_mask,
            state_count,
        )
        spvr = tl.where(pre_shift, svr, rpvr)
        spvi = tl.where(pre_shift, svi, rpvi)
        sptr = tl.where(pre_shift, str_, rptr)
        spti = tl.where(pre_shift, sti, rpti)
        smvr = tl.where(pre_shift, wvr, rmvr)
        smvi = tl.where(pre_shift, wvi, rmvi)
        smtr = tl.where(pre_shift, wtr, rmtr)
        smti = tl.where(pre_shift, wti, rmti)

        # Undo the trailing spoil or shift.
        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        spoil = (event_action & 8) != 0
        avr, avi, bvr, bvi = _shift_adjoint(
            pbvr, pbvi, mbvr, mbvi, sp_r, sp_i, sm_r, sm_i, state, state_mask,
            state_count,
        )
        atr, ati, btr, bti = _shift_adjoint(
            pbtr, pbti, mbtr, mbti, sp_r, sp_i, sm_r, sm_i, state, state_mask,
            state_count,
        )
        trailing = do_shift & ~spoil
        pbvr = tl.where(spoil, 0.0, tl.where(trailing, avr, pbvr))
        pbvi = tl.where(spoil, 0.0, tl.where(trailing, avi, pbvi))
        pbtr = tl.where(spoil, 0.0, tl.where(trailing, atr, pbtr))
        pbti = tl.where(spoil, 0.0, tl.where(trailing, ati, pbti))
        mbvr = tl.where(spoil, 0.0, tl.where(trailing, bvr, mbvr))
        mbvi = tl.where(spoil, 0.0, tl.where(trailing, bvi, mbvi))
        mbtr = tl.where(spoil, 0.0, tl.where(trailing, btr, mbtr))
        mbti = tl.where(spoil, 0.0, tl.where(trailing, bti, mbti))

        event_flip = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        event_dot_flip = tl.load(
            dot_flip + event_base + event, mask=active_atom, other=0.0
        )
        event_phase = tl.load(phase + event_base + event, mask=active_atom, other=0.0)
        event_dot_phase = tl.load(
            dot_phase + event_base + event, mask=active_atom, other=0.0
        )

        # ---- recorded sample ----
        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        seed_mask = active_atom & record & (out >= 0)
        seed_real = tl.load(
            grad_output_real + problem * output_count + out, mask=seed_mask, other=0.0
        )
        seed_imag = tl.load(
            grad_output_imag + problem * output_count + out, mask=seed_mask, other=0.0
        )
        dvr, dvi, dtr, dti = _dual_polar(-event_phase, -event_dot_phase)
        # grad_m0 = Re(conj(seed) * recorded * demodulation)
        wr, wi, wtr_, wti_ = _dual_mul(spvr, spvi, sptr, spti, dvr, dvi, dtr, dti)
        m0_value, m0_tangent = _dual_real_conj_mul(
            seed_real, seed_imag, 0.0 * seed_real, 0.0 * seed_imag,
            wr, wi, wtr_, wti_,
        )
        g_m0v += tl.sum(tl.where(state == 0, m0_value, 0.0), axis=1)[:, None]
        g_m0t += tl.sum(tl.where(state == 0, m0_tangent, 0.0), axis=1)[:, None]
        # grad_phase = Re(conj(seed) * m0 * recorded * (-i) * demodulation)
        yr, yi, ytr, yti = _dual_scale(atom_m0, d_m0, spvr, spvi, sptr, spti)
        yr, yi, ytr, yti = _dual_times_i(yr, yi, ytr, yti)
        yr, yi, ytr, yti = -yr, -yi, -ytr, -yti
        yr, yi, ytr, yti = _dual_mul(yr, yi, ytr, yti, dvr, dvi, dtr, dti)
        phase_value, phase_tangent = _dual_real_conj_mul(
            seed_real, seed_imag, 0.0 * seed_real, 0.0 * seed_imag, yr, yi, ytr, yti
        )
        tl.atomic_add(
            grad_phase_value + event_base + event,
            tl.sum(tl.where(state == 0, phase_value, 0.0), axis=1)[:, None],
            mask=seed_mask,
        )
        tl.atomic_add(
            grad_phase_tangent + event_base + event,
            tl.sum(tl.where(state == 0, phase_tangent, 0.0), axis=1)[:, None],
            mask=seed_mask,
        )
        # fplus_bar[0] += conj(m0 * demodulation) * seed
        kr, ki, ktr, kti = _dual_scale(atom_m0, d_m0, dvr, dvi, dtr, dti)
        sr, si, stg_r, stg_i = _dual_mul(
            kr, -ki, ktr, -kti,
            seed_real, seed_imag, 0.0 * seed_real, 0.0 * seed_imag,
        )
        pbvr += tl.where(state == 0, sr, 0.0)
        pbvi += tl.where(state == 0, si, 0.0)
        pbtr += tl.where(state == 0, stg_r, 0.0)
        pbti += tl.where(state == 0, stg_i, 0.0)

        # ---- RF adjoint ----
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        inv_value, inv_tangent = _dual_real_conj_mul(
            zbvr, zbvi, zbtr, zbti, -rzvr, -rzvi, -rztr, -rzti
        )
        g_invv += tl.sum(tl.where(invert, inv_value, 0.0), axis=1)[:, None]
        g_invt += tl.sum(tl.where(invert, inv_tangent, 0.0), axis=1)[:, None]
        ivr, ivi, itr, iti = _dual_scale(-atom_inv, -d_inv, zbvr, zbvi, zbtr, zbti)
        zbvr = tl.where(invert, ivr, zbvr)
        zbvi = tl.where(invert, ivi, zbvi)
        zbtr = tl.where(invert, itr, zbtr)
        zbti = tl.where(invert, iti, zbti)

        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shims > 1:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            atom_b1_phase = tl.load(
                b1_phase + row + atom, mask=active_atom, other=0.0
            )
            d_b1 = tl.load(dot_b1 + row + atom, mask=active_atom, other=0.0)
            d_b1_phase = tl.load(
                dot_b1_phase + row + atom, mask=active_atom, other=0.0
            )
        alpha_value = event_flip * atom_b1
        alpha_tangent = event_dot_flip * atom_b1 + event_flip * d_b1
        phi_value = event_phase + atom_b1_phase
        phi_tangent = event_dot_phase + d_b1_phase
        cos_value = tl.cos(alpha_value)
        sin_value = tl.sin(alpha_value)
        cos_tangent = -sin_value * alpha_tangent
        sin_tangent = cos_value * alpha_tangent
        p1r, p1i, p1tr, p1ti = _dual_polar(phi_value, phi_tangent)
        p2r, p2i, p2tr, p2ti = _dual_mul(p1r, p1i, p1tr, p1ti, p1r, p1i, p1tr, p1ti)
        t00, t01, t02, t12, t20, t21, t22 = _rotation_block(
            0.5 * (1.0 + cos_value), 0.5 * cos_tangent,
            0.5 * (1.0 - cos_value), -0.5 * cos_tangent,
            sin_value, sin_tangent,
            cos_value, cos_tangent,
            p1r, p1i, p1tr, p1ti,
            p2r, p2i, p2tr, p2ti,
            p1r, -p1i, p1tr, -p1ti,
        )
        d00, d01, d02, d12, d20, d21, d22 = _rotation_block(
            -0.5 * sin_value, -0.5 * sin_tangent,
            0.5 * sin_value, 0.5 * sin_tangent,
            cos_value, cos_tangent,
            -sin_value, -sin_tangent,
            p1r, p1i, p1tr, p1ti,
            p2r, p2i, p2tr, p2ti,
            p1r, -p1i, p1tr, -p1ti,
        )

        # d/dalpha, contracted with the adjoint.
        row0 = _dual_mul(d00[0], d00[1], d00[2], d00[3], spvr, spvi, sptr, spti)
        add1 = _dual_mul(d01[0], d01[1], d01[2], d01[3], smvr, smvi, smtr, smti)
        add2 = _dual_mul(d02[0], d02[1], d02[2], d02[3], rzvr, rzvi, rztr, rzti)
        alpha_v, alpha_t = _dual_real_conj_mul(
            pbvr, pbvi, pbtr, pbti,
            row0[0] + add1[0] + add2[0], row0[1] + add1[1] + add2[1],
            row0[2] + add1[2] + add2[2], row0[3] + add1[3] + add2[3],
        )
        row0 = _dual_mul(d01[0], -d01[1], d01[2], -d01[3], spvr, spvi, sptr, spti)
        add1 = _dual_mul(d00[0], d00[1], d00[2], d00[3], smvr, smvi, smtr, smti)
        add2 = _dual_mul(d12[0], d12[1], d12[2], d12[3], rzvr, rzvi, rztr, rzti)
        part_v, part_t = _dual_real_conj_mul(
            mbvr, mbvi, mbtr, mbti,
            row0[0] + add1[0] + add2[0], row0[1] + add1[1] + add2[1],
            row0[2] + add1[2] + add2[2], row0[3] + add1[3] + add2[3],
        )
        alpha_v += part_v
        alpha_t += part_t
        row0 = _dual_mul(d20[0], d20[1], d20[2], d20[3], spvr, spvi, sptr, spti)
        add1 = _dual_mul(d21[0], d21[1], d21[2], d21[3], smvr, smvi, smtr, smti)
        add2 = _dual_mul(d22[0], d22[1], d22[2], d22[3], rzvr, rzvi, rztr, rzti)
        part_v, part_t = _dual_real_conj_mul(
            zbvr, zbvi, zbtr, zbti,
            row0[0] + add1[0] + add2[0], row0[1] + add1[1] + add2[1],
            row0[2] + add1[2] + add2[2], row0[3] + add1[3] + add2[3],
        )
        alpha_v += part_v
        alpha_t += part_t

        # d/dphi, where only the phase factors carry the dependence.
        u1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], smvr, smvi, smtr, smti)
        u2 = _dual_mul(t02[0], t02[1], t02[2], t02[3], rzvr, rzvi, rztr, rzti)
        ur, ui, utr, uti = _dual_times_i(
            2.0 * u1[0] + u2[0], 2.0 * u1[1] + u2[1],
            2.0 * u1[2] + u2[2], 2.0 * u1[3] + u2[3],
        )
        phi_v, phi_t = _dual_real_conj_mul(pbvr, pbvi, pbtr, pbti, ur, ui, utr, uti)
        u1 = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], spvr, spvi, sptr, spti)
        u2 = _dual_mul(t12[0], t12[1], t12[2], t12[3], rzvr, rzvi, rztr, rzti)
        ur, ui, utr, uti = _dual_times_i(
            -2.0 * u1[0] - u2[0], -2.0 * u1[1] - u2[1],
            -2.0 * u1[2] - u2[2], -2.0 * u1[3] - u2[3],
        )
        part_v, part_t = _dual_real_conj_mul(
            mbvr, mbvi, mbtr, mbti, ur, ui, utr, uti
        )
        phi_v += part_v
        phi_t += part_t
        u1 = _dual_mul(t20[0], t20[1], t20[2], t20[3], spvr, spvi, sptr, spti)
        u2 = _dual_mul(t21[0], t21[1], t21[2], t21[3], smvr, smvi, smtr, smti)
        ur, ui, utr, uti = _dual_times_i(
            u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2], u2[3] - u1[3]
        )
        part_v, part_t = _dual_real_conj_mul(
            zbvr, zbvi, zbtr, zbti, ur, ui, utr, uti
        )
        phi_v += part_v
        phi_t += part_t

        rotate = is_rf & ~is_inversion
        grad_alpha_v = tl.sum(tl.where(rotate, alpha_v, 0.0), axis=1)[:, None]
        grad_alpha_t = tl.sum(tl.where(rotate, alpha_t, 0.0), axis=1)[:, None]
        grad_phi_v = tl.sum(tl.where(rotate, phi_v, 0.0), axis=1)[:, None]
        grad_phi_t = tl.sum(tl.where(rotate, phi_t, 0.0), axis=1)[:, None]

        # Conjugate transpose of the rotation.
        n0 = _dual_mul(t00[0], -t00[1], t00[2], -t00[3], pbvr, pbvi, pbtr, pbti)
        n1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], mbvr, mbvi, mbtr, mbti)
        n2 = _dual_mul(t20[0], -t20[1], t20[2], -t20[3], zbvr, zbvi, zbtr, zbti)
        q0 = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], pbvr, pbvi, pbtr, pbti)
        q1 = _dual_mul(t00[0], -t00[1], t00[2], -t00[3], mbvr, mbvi, mbtr, mbti)
        q2 = _dual_mul(t21[0], -t21[1], t21[2], -t21[3], zbvr, zbvi, zbtr, zbti)
        w0 = _dual_mul(t02[0], -t02[1], t02[2], -t02[3], pbvr, pbvi, pbtr, pbti)
        w1 = _dual_mul(t12[0], -t12[1], t12[2], -t12[3], mbvr, mbvi, mbtr, mbti)
        w2 = _dual_mul(t22[0], -t22[1], t22[2], -t22[3], zbvr, zbvi, zbtr, zbti)

        pbvr = tl.where(rotate, n0[0] + n1[0] + n2[0], pbvr)
        pbvi = tl.where(rotate, n0[1] + n1[1] + n2[1], pbvi)
        pbtr = tl.where(rotate, n0[2] + n1[2] + n2[2], pbtr)
        pbti = tl.where(rotate, n0[3] + n1[3] + n2[3], pbti)
        mbvr = tl.where(rotate, q0[0] + q1[0] + q2[0], mbvr)
        mbvi = tl.where(rotate, q0[1] + q1[1] + q2[1], mbvi)
        mbtr = tl.where(rotate, q0[2] + q1[2] + q2[2], mbtr)
        mbti = tl.where(rotate, q0[3] + q1[3] + q2[3], mbti)
        zbvr = tl.where(rotate, w0[0] + w1[0] + w2[0], zbvr)
        zbvi = tl.where(rotate, w0[1] + w1[1] + w2[1], zbvi)
        zbtr = tl.where(rotate, w0[2] + w1[2] + w2[2], zbtr)
        zbti = tl.where(rotate, w0[3] + w1[3] + w2[3], zbti)

        writes_flip = active_atom & rotate
        tl.atomic_add(
            grad_flip_value + event_base + event,
            grad_alpha_v * atom_b1,
            mask=writes_flip,
        )
        tl.atomic_add(
            grad_flip_tangent + event_base + event,
            grad_alpha_t * atom_b1 + grad_alpha_v * d_b1,
            mask=writes_flip,
        )
        tl.atomic_add(
            grad_phase_value + event_base + event, grad_phi_v, mask=writes_flip
        )
        tl.atomic_add(
            grad_phase_tangent + event_base + event, grad_phi_t, mask=writes_flip
        )
        # A pulse's transmit gradient belongs to the shim it drives, so with
        # several it lands in that shim's row here rather than in a register
        # summed over the whole train. ``row`` is the offset of the row the
        # replay above read.
        if shims > 1:
            tl.atomic_add(
                grad_tissue_value + _B1_ROW * atom_count + row + atom,
                grad_alpha_v * event_flip,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_tangent + _B1_ROW * atom_count + row + atom,
                grad_alpha_t * event_flip + grad_alpha_v * event_dot_flip,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_value
                + (_B1_PHASE_ROW + shims - 1) * atom_count + row + atom,
                grad_phi_v,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_tangent
                + (_B1_PHASE_ROW + shims - 1) * atom_count + row + atom,
                grad_phi_t,
                mask=writes_flip,
            )
        else:
            g_b1v += grad_alpha_v * event_flip
            g_b1t += grad_alpha_t * event_flip + grad_alpha_v * event_dot_flip
            g_b1pv += grad_phi_v
            g_b1pt += grad_phi_t

        avr, avi, bvr, bvi = _shift_adjoint(
            pbvr, pbvi, mbvr, mbvi, sp_r, sp_i, sm_r, sm_i, state, state_mask,
            state_count,
        )
        atr, ati, btr, bti = _shift_adjoint(
            pbtr, pbti, mbtr, mbti, sp_r, sp_i, sm_r, sm_i, state, state_mask,
            state_count,
        )
        pbvr = tl.where(pre_shift, avr, pbvr)
        pbvi = tl.where(pre_shift, avi, pbvi)
        pbtr = tl.where(pre_shift, atr, pbtr)
        pbti = tl.where(pre_shift, ati, pbti)
        mbvr = tl.where(pre_shift, bvr, mbvr)
        mbvi = tl.where(pre_shift, bvi, mbvi)
        mbtr = tl.where(pre_shift, btr, mbtr)
        mbti = tl.where(pre_shift, bti, mbti)

        # ---- relaxation and off-resonance adjoint ----
        pq = _dual_mul(qr, qi, qtr, qti, xpvr, xpvi, xptr, xpti)
        mq = _dual_mul(qr, -qi, qtr, -qti, xmvr, xmvi, xmtr, xmti)
        e2_v, e2_t = _dual_real_conj_mul(
            pbvr, pbvi, pbtr, pbti, pq[0], pq[1], pq[2], pq[3]
        )
        part_v, part_t = _dual_real_conj_mul(
            mbvr, mbvi, mbtr, mbti, mq[0], mq[1], mq[2], mq[3]
        )
        cot2_v = e2_v + part_v
        cot2_t = e2_t + part_t
        grad_e2_v = tl.sum(cot2_v * damp_t, axis=1)[:, None]
        grad_e2_t = tl.sum(cot2_v * damp_t_tangent + cot2_t * damp_t, axis=1)[
            :, None
        ]

        po = _dual_mul(ovr, ovi, otr, oti, xpvr, xpvi, xptr, xpti)
        po = _dual_times_i(po[0], po[1], po[2], po[3])
        mo = _dual_mul(ovr, -ovi, otr, -oti, xmvr, xmvi, xmtr, xmti)
        mo = _dual_times_i(mo[0], mo[1], mo[2], mo[3])
        angle_v, angle_t = _dual_real_conj_mul(
            pbvr, pbvi, pbtr, pbti, po[0], po[1], po[2], po[3]
        )
        part_v, part_t = _dual_real_conj_mul(
            mbvr, mbvi, mbtr, mbti, mo[0], mo[1], mo[2], mo[3]
        )
        # A turn of the transverse states and the off-resonance angle are the
        # same derivative; only the weight each order carries differs.
        per_angle_v = angle_v - part_v
        per_angle_t = angle_t - part_t
        grad_angle_v = tl.sum(per_angle_v, axis=1)[:, None]
        grad_angle_t = tl.sum(per_angle_t, axis=1)[:, None]

        spun = _dual_mul(szr, szi, sztr, szti, xzvr, xzvi, xztr, xzti)
        e1_v, e1_t = _dual_real_conj_mul(
            zbvr, zbvi, zbtr, zbti, spun[0], spun[1], spun[2], spun[3]
        )
        grad_e1_v = tl.sum(e1_v * damp_z, axis=1)[:, None]
        grad_e1_t = tl.sum(e1_v * damp_z_tangent + e1_t * damp_z, axis=1)[:, None]
        grad_e1_v -= tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
        grad_e1_t -= tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None]

        # The rate and the interval multiply every order's b-weight, so both
        # take a weighted sum rather than one scalar. Order zero carries no
        # longitudinal weight, which keeps recovery out of this.
        weighted_v = (
            e1_v * bare1_value * damp_z * longitudinal_weight
            + cot2_v * bare2_value * damp_t * transverse_weight
        )
        weighted_t = (
            e1_t * bare1_value * damp_z
            + e1_v * bare1_tangent * damp_z
            + e1_v * bare1_value * damp_z_tangent
        ) * longitudinal_weight + (
            cot2_t * bare2_value * damp_t
            + cot2_v * bare2_tangent * damp_t
            + cot2_v * bare2_value * damp_t_tangent
        ) * transverse_weight
        spread_v = tl.sum(weighted_v, axis=1)[:, None]
        spread_t = tl.sum(weighted_t, axis=1)[:, None]
        g_diffv += -spread_v * dt_value
        g_difft += -(spread_v * dt_tangent + spread_t * dt_value)

        # The longitudinal states turn too, and by a whole order rather than
        # the transverse half-order more.
        zo = _dual_mul(lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti)
        zo = _dual_times_i(zo[0], zo[1], zo[2], zo[3])
        zangle_v, zangle_t = _dual_real_conj_mul(
            zbvr, zbvi, zbtr, zbti, zo[0], zo[1], zo[2], zo[3]
        )
        wound_v = tl.sum(
            per_angle_v * (order + 0.5) + zangle_v * order, axis=1
        )[:, None]
        wound_t = tl.sum(
            per_angle_t * (order + 0.5) + zangle_t * order, axis=1
        )[:, None]
        g_flowv += -wound_v * dt_value
        g_flowt += -(wound_v * dt_tangent + wound_t * dt_value)

        # Washout scales both relaxation factors, so its gradient is the one
        # they already carry, taken against the factors before that scaling.
        # Past the clamp the interval has replaced the voxel outright and
        # nothing further depends on the rate.
        washing = (atom_washout * dt_value < 1.0).to(tl.float32)
        wash_v = -washing * (grad_e1_v * dry1_value + grad_e2_v * dry2_value)
        wash_t = -washing * (
            grad_e1_v * dry1_tangent + grad_e1_t * dry1_value
            + grad_e2_v * dry2_tangent + grad_e2_t * dry2_value
        )
        g_washv += wash_v * dt_value
        g_washt += wash_v * dt_tangent + wash_t * dt_value

        pbvr, pbvi, pbtr, pbti = _dual_mul(
            ovr, -ovi, otr, -oti, pbvr, pbvi, pbtr, pbti
        )
        mbvr, mbvi, mbtr, mbti = _dual_mul(
            ovr, ovi, otr, oti, mbvr, mbvi, mbtr, mbti
        )
        zbvr, zbvi, zbtr, zbti = _dual_mul(
            lvr, -lvi, ltr, -lti, zbvr, zbvi, zbtr, zbti
        )

        inverse1_value = 1000.0 / (atom_t1 * atom_t1)
        inverse1_tangent = -2000.0 * d_t1 / (atom_t1 * atom_t1 * atom_t1)
        inverse2_value = 1000.0 / (atom_t2 * atom_t2)
        inverse2_tangent = -2000.0 * d_t2 / (atom_t2 * atom_t2 * atom_t2)
        scale1_value = bare1_value * dt_value * inverse1_value
        scale1_tangent = bare1_tangent * dt_value * inverse1_value
        scale1_tangent += bare1_value * dt_tangent * inverse1_value
        scale1_tangent += bare1_value * dt_value * inverse1_tangent
        scale2_value = bare2_value * dt_value * inverse2_value
        scale2_tangent = bare2_tangent * dt_value * inverse2_value
        scale2_tangent += bare2_value * dt_tangent * inverse2_value
        scale2_tangent += bare2_value * dt_value * inverse2_tangent
        g_t1v += grad_e1_v * scale1_value
        g_t1t += grad_e1_v * scale1_tangent + grad_e1_t * scale1_value
        g_t2v += grad_e2_v * scale2_value
        g_t2t += grad_e2_v * scale2_tangent + grad_e2_t * scale2_value

        turn = -2.0 * 3.141592653589793
        g_b0v += grad_angle_v * (turn * dt_value)
        g_b0t += grad_angle_v * (turn * dt_tangent) + grad_angle_t * (turn * dt_value)

        decay1_value = r1_value * bare1_value
        decay1_tangent = (
            r1_value * bare1_tangent + r1_tangent * bare1_value
        )
        decay2_value = r2_value * bare2_value
        decay2_tangent = (
            r2_value * bare2_tangent + r2_tangent * bare2_value
        )
        duration_v = -grad_e1_v * decay1_value - grad_e2_v * decay2_value
        duration_v += grad_angle_v * (turn * atom_b0)
        duration_t = -(grad_e1_v * decay1_tangent + grad_e1_t * decay1_value)
        duration_t -= grad_e2_v * decay2_tangent + grad_e2_t * decay2_value
        duration_t += grad_angle_v * (turn * d_b0) + grad_angle_t * (turn * atom_b0)
        duration_v += -spread_v * atom_damping - wound_v * atom_flow
        duration_t += -(spread_v * d_damping + spread_t * atom_damping)
        duration_t += -(wound_v * d_flow + wound_t * atom_flow)
        duration_v += wash_v * atom_washout
        duration_t += wash_v * d_washout + wash_t * atom_washout
        tl.atomic_add(
            grad_duration_value + event_base + event, duration_v, mask=active_atom
        )
        tl.atomic_add(
            grad_duration_tangent + event_base + event, duration_t, mask=active_atom
        )

    velocity_v = g_flowv * flow_scale + g_washv * direction * washout_scale
    velocity_t = g_flowt * flow_scale + g_washt * direction * washout_scale
    values = (
        g_t1v, g_t2v, g_m0v, g_b1v, g_b1pv, g_b0v, g_invv, g_diffv, velocity_v,
    )
    tangents = (
        g_t1t, g_t2t, g_m0t, g_b1t, g_b1pt, g_b0t, g_invt, g_difft, velocity_t,
    )
    for parameter in tl.static_range(_TISSUE_COUNT):
        # The transmit pair went to its shim's row above when there is more
        # than one; the rest sit past whatever rows that pair took.
        if shims == 1 or (parameter != _B1_ROW and parameter != _B1_PHASE_ROW):
            plane = (
                parameter if parameter < _B1_ROW
                else parameter + 2 * (shims - 1)
            )
            tl.atomic_add(
                grad_tissue_value + plane * atom_count + atom,
                values[parameter],
                mask=active_atom,
            )
            tl.atomic_add(
                grad_tissue_tangent + plane * atom_count + atom,
                tangents[parameter],
                mask=active_atom,
            )


@triton.jit
def _epg_real_vjp_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    dot_t1,
    dot_t2,
    dot_m0,
    dot_b1,
    dot_inversion_efficiency,
    dot_diffusion,
    dot_duration,
    dot_flip,
    grad_output_imag,
    grad_tissue_value,
    grad_tissue_tangent,
    grad_flip_value,
    grad_flip_tangent,
    grad_duration_value,
    grad_duration_tangent,
    trajectory_value,
    trajectory_tangent,
    scratch_plus,
    scratch_minus,
    problem_base,
    problem_end,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = problem_base + tl.program_id(0) * problems
    problem = problem + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    # The grid rounds up to whole tiles, so the last program of a wave reaches
    # past it. Those problems are real, but their trajectory rows belong to a
    # later launch and do not exist yet.
    active_atom = problem < problem_end
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    train = problem // atom_count
    scratch_offset = (problem - problem_base) * state_count
    scratch_p = scratch_plus + scratch_offset
    scratch_m = scratch_minus + scratch_offset
    # The trajectory holds the state entering every event: three planes of
    # configuration orders, for the value and the tangent alike.
    record_stride = 3 * state_count
    trajectory = (problem - problem_base) * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    plus_value = empty
    plus_tangent = empty
    minus_value = empty
    minus_tangent = empty
    long_value = empty + tl.where(state == 0, 1.0, 0.0)
    long_tangent = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = tl.load(m0 + atom, mask=active_atom, other=0.0)
    atom_b1 = tl.load(b1 + atom, mask=active_atom, other=1.0)
    atom_inversion = tl.load(
        inversion_efficiency + atom, mask=active_atom, other=1.0
    )
    atom_dot_t1 = tl.load(dot_t1 + atom, mask=active_atom, other=0.0)
    atom_dot_t2 = tl.load(dot_t2 + atom, mask=active_atom, other=0.0)
    atom_dot_m0 = tl.load(dot_m0 + atom, mask=active_atom, other=0.0)
    atom_dot_b1 = tl.load(dot_b1 + atom, mask=active_atom, other=0.0)
    atom_dot_inversion = tl.load(
        dot_inversion_efficiency + atom, mask=active_atom, other=0.0
    )
    atom_damping = tl.load(diffusion + atom, mask=active_atom, other=0.0)
    atom_dot_damping = tl.load(dot_diffusion + atom, mask=active_atom, other=0.0)
    order = state.to(tl.float32)
    longitudinal_weight = order * order
    transverse_weight = longitudinal_weight + order + 0.3333333333333333
    rate1_value = 1000.0 / atom_t1
    rate1_tangent = -1000.0 * atom_dot_t1 / (atom_t1 * atom_t1)
    rate2_value = 1000.0 / atom_t2
    rate2_tangent = -1000.0 * atom_dot_t2 / (atom_t2 * atom_t2)

    event_base = train * event_count
    for event in range(0, event_count):
        slot = trajectory + event * record_stride
        tl.store(trajectory_value + slot, plus_value, mask=state_mask)
        tl.store(trajectory_value + slot + minus_plane, minus_value, mask=state_mask)
        tl.store(trajectory_value + slot + long_plane, long_value, mask=state_mask)
        tl.store(trajectory_tangent + slot, plus_tangent, mask=state_mask)
        tl.store(
            trajectory_tangent + slot + minus_plane, minus_tangent, mask=state_mask
        )
        tl.store(trajectory_tangent + slot + long_plane, long_tangent, mask=state_mask)

        dt_value = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        dt_tangent = tl.load(
            dot_duration + event_base + event, mask=active_atom, other=0.0
        )
        e1_value = tl.exp(-rate1_value * dt_value)
        e1_tangent = -e1_value * (rate1_value * dt_tangent + rate1_tangent * dt_value)
        e2_value = tl.exp(-rate2_value * dt_value)
        e2_tangent = -e2_value * (rate2_value * dt_tangent + rate2_tangent * dt_value)
        damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
            atom_damping, atom_dot_damping, dt_value, dt_tangent, order
        )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t

        plus_tangent = plus_value * e2_tangent + plus_tangent * e2_value
        plus_value = plus_value * e2_value
        minus_tangent = minus_value * e2_tangent + minus_tangent * e2_value
        minus_value = minus_value * e2_value
        long_tangent = long_value * e1_tangent + long_tangent * e1_value
        long_value = long_value * e1_value
        long_value += tl.where(state == 0, recovery_value, 0.0)
        long_tangent += tl.where(state == 0, recovery_tangent, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pv, shifted_mv = _shift_real(
            plus_value, minus_value, scratch_p, scratch_m, state, state_mask,
            state_count,
        )
        shifted_pt, shifted_mt = _shift_real(
            plus_tangent, minus_tangent, scratch_p, scratch_m, state, state_mask,
            state_count,
        )
        plus_value = tl.where(pre_shift, shifted_pv, plus_value)
        minus_value = tl.where(pre_shift, shifted_mv, minus_value)
        plus_tangent = tl.where(pre_shift, shifted_pt, plus_tangent)
        minus_tangent = tl.where(pre_shift, shifted_mt, minus_tangent)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        inverted_value = -atom_inversion * long_value
        inverted_tangent = -atom_inversion * long_tangent
        inverted_tangent -= atom_dot_inversion * long_value
        long_value = tl.where(invert, inverted_value, long_value)
        long_tangent = tl.where(invert, inverted_tangent, long_tangent)

        event_flip = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        event_dot_flip = tl.load(
            dot_flip + event_base + event, mask=active_atom, other=0.0
        )
        alpha_value = event_flip * atom_b1
        alpha_tangent = event_dot_flip * atom_b1 + event_flip * atom_dot_b1
        cosine_value = tl.cos(alpha_value)
        sine_value = tl.sin(alpha_value)
        cosine_tangent = -sine_value * alpha_tangent
        sine_tangent = cosine_value * alpha_tangent
        chs_value = 0.5 * (1.0 + cosine_value)
        chs_tangent = 0.5 * cosine_tangent
        shs_value = 0.5 * (1.0 - cosine_value)
        shs_tangent = -0.5 * cosine_tangent
        half_sine_value = 0.5 * sine_value
        half_sine_tangent = 0.5 * sine_tangent

        rotated_pv = chs_value * plus_value + shs_value * minus_value
        rotated_pv -= sine_value * long_value
        rotated_pt = chs_value * plus_tangent + chs_tangent * plus_value
        rotated_pt += shs_value * minus_tangent + shs_tangent * minus_value
        rotated_pt -= sine_value * long_tangent + sine_tangent * long_value
        rotated_mv = shs_value * plus_value + chs_value * minus_value
        rotated_mv += sine_value * long_value
        rotated_mt = shs_value * plus_tangent + shs_tangent * plus_value
        rotated_mt += chs_value * minus_tangent + chs_tangent * minus_value
        rotated_mt += sine_value * long_tangent + sine_tangent * long_value
        rotated_zv = half_sine_value * plus_value - half_sine_value * minus_value
        rotated_zv += cosine_value * long_value
        rotated_zt = half_sine_value * plus_tangent + half_sine_tangent * plus_value
        rotated_zt -= half_sine_value * minus_tangent + half_sine_tangent * minus_value
        rotated_zt += cosine_value * long_tangent + cosine_tangent * long_value

        rotate = is_rf & ~is_inversion
        plus_value = tl.where(rotate, rotated_pv, plus_value)
        plus_tangent = tl.where(rotate, rotated_pt, plus_tangent)
        minus_value = tl.where(rotate, rotated_mv, minus_value)
        minus_tangent = tl.where(rotate, rotated_mt, minus_tangent)
        long_value = tl.where(rotate, rotated_zv, long_value)
        long_tangent = tl.where(rotate, rotated_zt, long_tangent)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pv, shifted_mv = _shift_real(
            plus_value, minus_value, scratch_p, scratch_m, state, state_mask,
            state_count,
        )
        shifted_pt, shifted_mt = _shift_real(
            plus_tangent, minus_tangent, scratch_p, scratch_m, state, state_mask,
            state_count,
        )
        plus_value = tl.where(do_shift, shifted_pv, plus_value)
        minus_value = tl.where(do_shift, shifted_mv, minus_value)
        plus_tangent = tl.where(do_shift, shifted_pt, plus_tangent)
        minus_tangent = tl.where(do_shift, shifted_mt, minus_tangent)
        spoil = (event_action & 8) != 0
        plus_value = tl.where(spoil, 0.0, plus_value)
        minus_value = tl.where(spoil, 0.0, minus_value)
        plus_tangent = tl.where(spoil, 0.0, plus_tangent)
        minus_tangent = tl.where(spoil, 0.0, minus_tangent)

    plus_bar_value = empty
    plus_bar_tangent = empty
    minus_bar_value = empty
    minus_bar_tangent = empty
    long_bar_value = empty
    long_bar_tangent = empty
    zero = tl.zeros((problems, 1), tl.float32)
    grad_t1_value = zero
    grad_t1_tangent = zero
    grad_t2_value = zero
    grad_t2_tangent = zero
    grad_m0_value = zero
    grad_m0_tangent = zero
    grad_b1_value = zero
    grad_b1_tangent = zero
    grad_inversion_value = zero
    grad_inversion_tangent = zero
    grad_damping_value = zero
    grad_damping_tangent = zero

    for reverse in range(0, event_count):
        event = event_count - 1 - reverse
        slot = trajectory + event * record_stride
        entry_pv = tl.load(trajectory_value + slot, mask=state_mask, other=0.0)
        entry_mv = tl.load(
            trajectory_value + slot + minus_plane, mask=state_mask, other=0.0
        )
        entry_zv = tl.load(
            trajectory_value + slot + long_plane, mask=state_mask, other=0.0
        )
        entry_pt = tl.load(trajectory_tangent + slot, mask=state_mask, other=0.0)
        entry_mt = tl.load(
            trajectory_tangent + slot + minus_plane, mask=state_mask, other=0.0
        )
        entry_zt = tl.load(
            trajectory_tangent + slot + long_plane, mask=state_mask, other=0.0
        )

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event)
        dt_value = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        dt_tangent = tl.load(
            dot_duration + event_base + event, mask=active_atom, other=0.0
        )
        e1_value = tl.exp(-rate1_value * dt_value)
        e1_tangent = -e1_value * (rate1_value * dt_tangent + rate1_tangent * dt_value)
        e2_value = tl.exp(-rate2_value * dt_value)
        e2_tangent = -e2_value * (rate2_value * dt_tangent + rate2_tangent * dt_value)
        damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
            atom_damping, atom_dot_damping, dt_value, dt_tangent, order
        )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t

        # Replay the intra-event stages from the recorded entry state.
        stage_pv = entry_pv * e2_value
        stage_pt = entry_pv * e2_tangent + entry_pt * e2_value
        stage_mv = entry_mv * e2_value
        stage_mt = entry_mv * e2_tangent + entry_mt * e2_value
        stage_zv = entry_zv * e1_value + tl.where(state == 0, recovery_value, 0.0)
        stage_zt = entry_zv * e1_tangent + entry_zt * e1_value
        stage_zt += tl.where(state == 0, recovery_tangent, 0.0)

        pre_shift = (event_action & 1) != 0
        shifted_pv, shifted_mv = _shift_real(
            stage_pv, stage_mv, scratch_p, scratch_m, state, state_mask, state_count
        )
        shifted_pt, shifted_mt = _shift_real(
            stage_pt, stage_mt, scratch_p, scratch_m, state, state_mask, state_count
        )
        stage_pv = tl.where(pre_shift, shifted_pv, stage_pv)
        stage_mv = tl.where(pre_shift, shifted_mv, stage_mv)
        stage_pt = tl.where(pre_shift, shifted_pt, stage_pt)
        stage_mt = tl.where(pre_shift, shifted_mt, stage_mt)

        # Undo the trailing spoil or shift.
        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        spoil = (event_action & 8) != 0
        adjoint_pv, adjoint_mv = _shift_real_adjoint(
            plus_bar_value, minus_bar_value, scratch_p, scratch_m, state,
            state_mask, state_count,
        )
        adjoint_pt, adjoint_mt = _shift_real_adjoint(
            plus_bar_tangent, minus_bar_tangent, scratch_p, scratch_m, state,
            state_mask, state_count,
        )
        trailing = do_shift & ~spoil
        plus_bar_value = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_pv, plus_bar_value)
        )
        minus_bar_value = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_mv, minus_bar_value)
        )
        plus_bar_tangent = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_pt, plus_bar_tangent)
        )
        minus_bar_tangent = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_mt, minus_bar_tangent)
        )

        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        inversion_gain = -tl.sum(
            tl.where(invert, long_bar_value * stage_zv, 0.0), axis=1
        )[:, None]
        inversion_gain_tangent = -tl.sum(
            tl.where(
                invert,
                long_bar_value * stage_zt + long_bar_tangent * stage_zv,
                0.0,
            ),
            axis=1,
        )[:, None]
        grad_inversion_value += inversion_gain
        grad_inversion_tangent += inversion_gain_tangent
        inverted_bar_value = -atom_inversion * long_bar_value
        inverted_bar_tangent = (
            -atom_inversion * long_bar_tangent - atom_dot_inversion * long_bar_value
        )
        long_bar_value = tl.where(invert, inverted_bar_value, long_bar_value)
        long_bar_tangent = tl.where(invert, inverted_bar_tangent, long_bar_tangent)

        event_flip = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        event_dot_flip = tl.load(
            dot_flip + event_base + event, mask=active_atom, other=0.0
        )
        alpha_value = event_flip * atom_b1
        alpha_tangent = event_dot_flip * atom_b1 + event_flip * atom_dot_b1
        cosine_value = tl.cos(alpha_value)
        sine_value = tl.sin(alpha_value)
        cosine_tangent = -sine_value * alpha_tangent
        sine_tangent = cosine_value * alpha_tangent
        chs_value = 0.5 * (1.0 + cosine_value)
        chs_tangent = 0.5 * cosine_tangent
        shs_value = 0.5 * (1.0 - cosine_value)
        shs_tangent = -0.5 * cosine_tangent
        half_sine_value = 0.5 * sine_value
        half_sine_tangent = 0.5 * sine_tangent

        # d/dalpha of each output row, contracted with the adjoint.
        row_p_value = half_sine_value * stage_mv - half_sine_value * stage_pv
        row_p_value -= cosine_value * stage_zv
        row_p_tangent = half_sine_value * stage_mt + half_sine_tangent * stage_mv
        row_p_tangent -= half_sine_value * stage_pt + half_sine_tangent * stage_pv
        row_p_tangent -= cosine_value * stage_zt + cosine_tangent * stage_zv
        row_m_value = half_sine_value * stage_pv - half_sine_value * stage_mv
        row_m_value += cosine_value * stage_zv
        row_m_tangent = half_sine_value * stage_pt + half_sine_tangent * stage_pv
        row_m_tangent -= half_sine_value * stage_mt + half_sine_tangent * stage_mv
        row_m_tangent += cosine_value * stage_zt + cosine_tangent * stage_zv
        row_z_value = 0.5 * cosine_value * stage_pv - 0.5 * cosine_value * stage_mv
        row_z_value -= sine_value * stage_zv
        row_z_tangent = 0.5 * (cosine_value * stage_pt + cosine_tangent * stage_pv)
        row_z_tangent -= 0.5 * (cosine_value * stage_mt + cosine_tangent * stage_mv)
        row_z_tangent -= sine_value * stage_zt + sine_tangent * stage_zv

        alpha_bar_terms_value = plus_bar_value * row_p_value
        alpha_bar_terms_value += minus_bar_value * row_m_value
        alpha_bar_terms_value += long_bar_value * row_z_value
        alpha_bar_terms_tangent = plus_bar_value * row_p_tangent
        alpha_bar_terms_tangent += plus_bar_tangent * row_p_value
        alpha_bar_terms_tangent += minus_bar_value * row_m_tangent
        alpha_bar_terms_tangent += minus_bar_tangent * row_m_value
        alpha_bar_terms_tangent += long_bar_value * row_z_tangent
        alpha_bar_terms_tangent += long_bar_tangent * row_z_value
        rotate = is_rf & ~is_inversion
        grad_alpha_value = tl.sum(
            tl.where(rotate, alpha_bar_terms_value, 0.0), axis=1
        )[:, None]
        grad_alpha_tangent = tl.sum(
            tl.where(rotate, alpha_bar_terms_tangent, 0.0), axis=1
        )[:, None]

        # Transpose of the rotation.
        rotated_pbv = chs_value * plus_bar_value + shs_value * minus_bar_value
        rotated_pbv += half_sine_value * long_bar_value
        rotated_pbt = chs_value * plus_bar_tangent + chs_tangent * plus_bar_value
        rotated_pbt += shs_value * minus_bar_tangent + shs_tangent * minus_bar_value
        rotated_pbt += half_sine_value * long_bar_tangent
        rotated_pbt += half_sine_tangent * long_bar_value
        rotated_mbv = shs_value * plus_bar_value + chs_value * minus_bar_value
        rotated_mbv -= half_sine_value * long_bar_value
        rotated_mbt = shs_value * plus_bar_tangent + shs_tangent * plus_bar_value
        rotated_mbt += chs_value * minus_bar_tangent + chs_tangent * minus_bar_value
        rotated_mbt -= half_sine_value * long_bar_tangent
        rotated_mbt -= half_sine_tangent * long_bar_value
        rotated_zbv = -sine_value * plus_bar_value + sine_value * minus_bar_value
        rotated_zbv += cosine_value * long_bar_value
        rotated_zbt = -sine_value * plus_bar_tangent - sine_tangent * plus_bar_value
        rotated_zbt += sine_value * minus_bar_tangent + sine_tangent * minus_bar_value
        rotated_zbt += cosine_value * long_bar_tangent + cosine_tangent * long_bar_value

        plus_bar_value = tl.where(rotate, rotated_pbv, plus_bar_value)
        plus_bar_tangent = tl.where(rotate, rotated_pbt, plus_bar_tangent)
        minus_bar_value = tl.where(rotate, rotated_mbv, minus_bar_value)
        minus_bar_tangent = tl.where(rotate, rotated_mbt, minus_bar_tangent)
        long_bar_value = tl.where(rotate, rotated_zbv, long_bar_value)
        long_bar_tangent = tl.where(rotate, rotated_zbt, long_bar_tangent)

        flip_gain_value = grad_alpha_value * atom_b1
        flip_gain_tangent = grad_alpha_tangent * atom_b1
        flip_gain_tangent += grad_alpha_value * atom_dot_b1
        writes_flip = active_atom & rotate
        tl.atomic_add(
            grad_flip_value + event_base + event, flip_gain_value, mask=writes_flip
        )
        tl.atomic_add(
            grad_flip_tangent + event_base + event, flip_gain_tangent, mask=writes_flip
        )
        grad_b1_value += tl.where(rotate, grad_alpha_value * event_flip, 0.0)
        grad_b1_tangent += tl.where(
            rotate,
            grad_alpha_tangent * event_flip + grad_alpha_value * event_dot_flip,
            0.0,
        )

        # The sample is i * m0 * plus[0]; only the imaginary seed acts.
        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        seed = tl.load(
            grad_output_imag + problem * output_count + out,
            mask=active_atom & record & (out >= 0),
            other=0.0,
        )
        grad_m0_value += tl.sum(
            tl.where(state == 0, seed * stage_pv, 0.0), axis=1
        )[:, None]
        grad_m0_tangent += tl.sum(
            tl.where(state == 0, seed * stage_pt, 0.0), axis=1
        )[:, None]
        plus_bar_value += tl.where(state == 0, seed * atom_m0, 0.0)
        plus_bar_tangent += tl.where(state == 0, seed * atom_dot_m0, 0.0)

        adjoint_pv, adjoint_mv = _shift_real_adjoint(
            plus_bar_value, minus_bar_value, scratch_p, scratch_m, state,
            state_mask, state_count,
        )
        adjoint_pt, adjoint_mt = _shift_real_adjoint(
            plus_bar_tangent, minus_bar_tangent, scratch_p, scratch_m, state,
            state_mask, state_count,
        )
        plus_bar_value = tl.where(pre_shift, adjoint_pv, plus_bar_value)
        minus_bar_value = tl.where(pre_shift, adjoint_mv, minus_bar_value)
        plus_bar_tangent = tl.where(pre_shift, adjoint_pt, plus_bar_tangent)
        minus_bar_tangent = tl.where(pre_shift, adjoint_mt, minus_bar_tangent)

        cot2_value = plus_bar_value * entry_pv + minus_bar_value * entry_mv
        cot2_tangent = (
            plus_bar_value * entry_pt
            + plus_bar_tangent * entry_pv
            + minus_bar_value * entry_mt
            + minus_bar_tangent * entry_mv
        )
        cot1_value = long_bar_value * entry_zv
        cot1_tangent = long_bar_value * entry_zt + long_bar_tangent * entry_zv
        grad_e2_value = tl.sum(cot2_value * damp_t, axis=1)[:, None]
        grad_e2_tangent = tl.sum(
            cot2_value * damp_t_tangent + cot2_tangent * damp_t, axis=1
        )[:, None]
        grad_e1_value = tl.sum(cot1_value * damp_z, axis=1)[:, None]
        grad_e1_value -= tl.sum(
            tl.where(state == 0, long_bar_value, 0.0), axis=1
        )[:, None]
        grad_e1_tangent = tl.sum(
            cot1_value * damp_z_tangent + cot1_tangent * damp_z, axis=1
        )[:, None]
        grad_e1_tangent -= tl.sum(
            tl.where(state == 0, long_bar_tangent, 0.0), axis=1
        )[:, None]

        # The rate and the interval multiply every order's b-weight, so both
        # take a weighted sum. Order zero has no longitudinal weight, which
        # keeps recovery out of this.
        weighted_value = (
            cot1_value * bare1_value * damp_z * longitudinal_weight
            + cot2_value * bare2_value * damp_t * transverse_weight
        )
        weighted_tangent = (
            cot1_tangent * bare1_value * damp_z
            + cot1_value * bare1_tangent * damp_z
            + cot1_value * bare1_value * damp_z_tangent
        ) * longitudinal_weight + (
            cot2_tangent * bare2_value * damp_t
            + cot2_value * bare2_tangent * damp_t
            + cot2_value * bare2_value * damp_t_tangent
        ) * transverse_weight
        spread_value = tl.sum(weighted_value, axis=1)[:, None]
        spread_tangent = tl.sum(weighted_tangent, axis=1)[:, None]
        grad_damping_value += -spread_value * dt_value
        grad_damping_tangent += -(
            spread_value * dt_tangent + spread_tangent * dt_value
        )

        plus_bar_tangent = plus_bar_value * e2_tangent + plus_bar_tangent * e2_value
        plus_bar_value = plus_bar_value * e2_value
        minus_bar_tangent = minus_bar_value * e2_tangent + minus_bar_tangent * e2_value
        minus_bar_value = minus_bar_value * e2_value
        long_bar_tangent = long_bar_value * e1_tangent + long_bar_tangent * e1_value
        long_bar_value = long_bar_value * e1_value

        inverse1_value = 1000.0 / (atom_t1 * atom_t1)
        inverse1_tangent = -2000.0 * atom_dot_t1 / (atom_t1 * atom_t1 * atom_t1)
        inverse2_value = 1000.0 / (atom_t2 * atom_t2)
        inverse2_tangent = -2000.0 * atom_dot_t2 / (atom_t2 * atom_t2 * atom_t2)
        scale1_value = bare1_value * dt_value * inverse1_value
        scale1_tangent = bare1_tangent * dt_value * inverse1_value
        scale1_tangent += bare1_value * dt_tangent * inverse1_value
        scale1_tangent += bare1_value * dt_value * inverse1_tangent
        scale2_value = bare2_value * dt_value * inverse2_value
        scale2_tangent = bare2_tangent * dt_value * inverse2_value
        scale2_tangent += bare2_value * dt_tangent * inverse2_value
        scale2_tangent += bare2_value * dt_value * inverse2_tangent
        grad_t1_value += grad_e1_value * scale1_value
        grad_t1_tangent += grad_e1_value * scale1_tangent
        grad_t1_tangent += grad_e1_tangent * scale1_value
        grad_t2_value += grad_e2_value * scale2_value
        grad_t2_tangent += grad_e2_value * scale2_tangent
        grad_t2_tangent += grad_e2_tangent * scale2_value

        decay1_value = rate1_value * bare1_value
        decay1_tangent = (
            rate1_value * bare1_tangent
            + rate1_tangent * bare1_value
        )
        decay2_value = rate2_value * bare2_value
        decay2_tangent = (
            rate2_value * bare2_tangent
            + rate2_tangent * bare2_value
        )
        duration_gain_value = -grad_e1_value * decay1_value
        duration_gain_value -= grad_e2_value * decay2_value
        duration_gain_tangent = -(
            grad_e1_value * decay1_tangent + grad_e1_tangent * decay1_value
        )
        duration_gain_tangent -= (
            grad_e2_value * decay2_tangent + grad_e2_tangent * decay2_value
        )
        duration_gain_value += -spread_value * atom_damping
        duration_gain_tangent += -(
            spread_value * atom_dot_damping + spread_tangent * atom_damping
        )
        tl.atomic_add(
            grad_duration_value + event_base + event,
            duration_gain_value,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_duration_tangent + event_base + event,
            duration_gain_tangent,
            mask=active_atom,
        )

    tl.atomic_add(grad_tissue_value + atom, grad_t1_value, mask=active_atom)
    tl.atomic_add(grad_tissue_tangent + atom, grad_t1_tangent, mask=active_atom)
    tl.atomic_add(
        grad_tissue_value + atom_count + atom, grad_t2_value, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_tangent + atom_count + atom, grad_t2_tangent, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_value + 2 * atom_count + atom, grad_m0_value, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_tangent + 2 * atom_count + atom, grad_m0_tangent, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_value + 3 * atom_count + atom, grad_b1_value, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_tangent + 3 * atom_count + atom, grad_b1_tangent, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_value + 6 * atom_count + atom,
        grad_inversion_value,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue_tangent + 6 * atom_count + atom,
        grad_inversion_tangent,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue_value + 7 * atom_count + atom,
        grad_damping_value,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue_tangent + 7 * atom_count + atom,
        grad_damping_tangent,
        mask=active_atom,
    )


@triton.jit
def _epg_real_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    output_real,
    output_imag,
    scratch_plus,
    scratch_minus,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    train = problem // atom_count
    workspace_offset = problem * state_count
    scratch_p = scratch_plus + workspace_offset
    scratch_m = scratch_minus + workspace_offset

    empty = tl.zeros((problems, block_states), tl.float32)
    plus = empty
    minus = empty
    longitudinal = empty + tl.where(state == 0, 1.0, 0.0)

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = tl.load(m0 + atom, mask=active_atom, other=0.0)
    atom_b1 = tl.load(b1 + atom, mask=active_atom, other=1.0)
    atom_inversion = tl.load(
        inversion_efficiency + atom, mask=active_atom, other=1.0
    )
    rate1 = 1000.0 / atom_t1
    rate2 = 1000.0 / atom_t2
    atom_damping = tl.load(diffusion + atom, mask=active_atom, other=0.0)
    order = state.to(tl.float32)

    event_base = train * event_count
    for event in range(0, event_count):
        dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        e1 = tl.exp(-rate1 * dt)
        e2 = tl.exp(-rate2 * dt)
        damp_z, damp_t = _damping(atom_damping, dt, order)
        recovery = 1.0 - e1
        plus *= e2 * damp_t
        minus *= e2 * damp_t
        longitudinal = longitudinal * (e1 * damp_z) + tl.where(
            state == 0, recovery, 0.0
        )

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_p, shifted_m = _shift_real(
            plus, minus, scratch_p, scratch_m, state, state_mask, state_count
        )
        plus = tl.where(pre_shift, shifted_p, plus)
        minus = tl.where(pre_shift, shifted_m, minus)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        longitudinal = tl.where(
            invert, -atom_inversion * longitudinal, longitudinal
        )

        alpha = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        alpha *= atom_b1
        cosine = tl.cos(alpha)
        sine = tl.sin(alpha)
        cosine_half_sq = 0.5 * (1.0 + cosine)
        sine_half_sq = 0.5 * (1.0 - cosine)
        half_sine = 0.5 * sine
        rotated_p = cosine_half_sq * plus + sine_half_sq * minus - sine * longitudinal
        rotated_m = sine_half_sq * plus + cosine_half_sq * minus + sine * longitudinal
        rotated_z = half_sine * plus - half_sine * minus + cosine * longitudinal

        rotate = is_rf & ~is_inversion
        plus = tl.where(rotate, rotated_p, plus)
        minus = tl.where(rotate, rotated_m, minus)
        longitudinal = tl.where(rotate, rotated_z, longitudinal)

        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        output_offset = problem * output_count + out
        output_mask = active_atom & (state == 0) & record & (out >= 0)
        tl.store(output_real + output_offset + state, empty, mask=output_mask)
        tl.store(output_imag + output_offset + state, atom_m0 * plus, mask=output_mask)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_p, shifted_m = _shift_real(
            plus, minus, scratch_p, scratch_m, state, state_mask, state_count
        )
        plus = tl.where(do_shift, shifted_p, plus)
        minus = tl.where(do_shift, shifted_m, minus)
        spoil = (event_action & 8) != 0
        plus = tl.where(spoil, 0.0, plus)
        minus = tl.where(spoil, 0.0, minus)


@triton.jit
def _epg_real_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    tangent_t1,
    tangent_t2,
    tangent_m0,
    tangent_b1,
    tangent_inversion_efficiency,
    tangent_diffusion,
    tangent_duration,
    tangent_flip,
    output_real,
    output_imag,
    scratch_plus,
    scratch_minus,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    train = problem // atom_count
    workspace_offset = problem * state_count
    scratch_p = scratch_plus + workspace_offset
    scratch_m = scratch_minus + workspace_offset

    empty = tl.zeros((problems, block_states), tl.float32)
    plus = empty
    minus = empty
    longitudinal = empty + tl.where(state == 0, 1.0, 0.0)
    dot_plus = empty
    dot_minus = empty
    dot_longitudinal = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = tl.load(m0 + atom, mask=active_atom, other=0.0)
    atom_b1 = tl.load(b1 + atom, mask=active_atom, other=1.0)
    atom_inversion = tl.load(
        inversion_efficiency + atom, mask=active_atom, other=1.0
    )
    dot_t1 = tl.load(tangent_t1 + atom, mask=active_atom, other=0.0)
    dot_t2 = tl.load(tangent_t2 + atom, mask=active_atom, other=0.0)
    dot_m0 = tl.load(tangent_m0 + atom, mask=active_atom, other=0.0)
    dot_b1 = tl.load(tangent_b1 + atom, mask=active_atom, other=0.0)
    dot_inversion = tl.load(
        tangent_inversion_efficiency + atom, mask=active_atom, other=0.0
    )
    rate1 = 1000.0 / atom_t1
    rate2 = 1000.0 / atom_t2
    atom_damping = tl.load(diffusion + atom, mask=active_atom, other=0.0)
    dot_damping = tl.load(tangent_diffusion + atom, mask=active_atom, other=0.0)
    order = state.to(tl.float32)

    event_base = train * event_count
    for event in range(0, event_count):
        dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        dot_dt = tl.load(
            tangent_duration + event_base + event, mask=active_atom, other=0.0
        )
        e1 = tl.exp(-rate1 * dt)
        e2 = tl.exp(-rate2 * dt)
        dot_e1 = e1 * (1000.0 * dt * dot_t1 / (atom_t1 * atom_t1) - rate1 * dot_dt)
        dot_e2 = e2 * (1000.0 * dt * dot_t2 / (atom_t2 * atom_t2) - rate2 * dot_dt)
        damp_z, ddamp_z, damp_t, ddamp_t = _damping_jvp(
            atom_damping, dot_damping, dt, dot_dt, order
        )
        # Order zero is undamped, so the recovery term keeps the bare factor.
        recovery, dot_recovery = 1.0 - e1, -dot_e1
        dot_e1 = dot_e1 * damp_z + e1 * ddamp_z
        e1 = e1 * damp_z
        dot_e2 = dot_e2 * damp_t + e2 * ddamp_t
        e2 = e2 * damp_t
        dot_plus = dot_plus * e2 + plus * dot_e2
        dot_minus = dot_minus * e2 + minus * dot_e2
        dot_longitudinal = dot_longitudinal * e1 + longitudinal * dot_e1
        dot_longitudinal += tl.where(state == 0, dot_recovery, 0.0)
        plus *= e2
        minus *= e2
        longitudinal = longitudinal * e1 + tl.where(state == 0, recovery, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_p, shifted_m = _shift_real(
            plus, minus, scratch_p, scratch_m, state, state_mask, state_count
        )
        shifted_dp, shifted_dm = _shift_real(
            dot_plus, dot_minus, scratch_p, scratch_m, state, state_mask, state_count
        )
        plus = tl.where(pre_shift, shifted_p, plus)
        minus = tl.where(pre_shift, shifted_m, minus)
        dot_plus = tl.where(pre_shift, shifted_dp, dot_plus)
        dot_minus = tl.where(pre_shift, shifted_dm, dot_minus)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        dot_longitudinal = tl.where(
            invert,
            -atom_inversion * dot_longitudinal - dot_inversion * longitudinal,
            dot_longitudinal,
        )
        longitudinal = tl.where(
            invert, -atom_inversion * longitudinal, longitudinal
        )

        event_flip = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        dot_flip = tl.load(
            tangent_flip + event_base + event, mask=active_atom, other=0.0
        )
        alpha = event_flip * atom_b1
        dot_alpha = dot_flip * atom_b1 + event_flip * dot_b1
        cosine = tl.cos(alpha)
        sine = tl.sin(alpha)
        cosine_half_sq = 0.5 * (1.0 + cosine)
        sine_half_sq = 0.5 * (1.0 - cosine)
        half_sine = 0.5 * sine
        dot_cosine = -sine * dot_alpha
        dot_sine = cosine * dot_alpha
        dot_cosine_half_sq = -0.5 * sine * dot_alpha
        dot_sine_half_sq = 0.5 * sine * dot_alpha
        dot_half_sine = 0.5 * cosine * dot_alpha

        rotated_dp = cosine_half_sq * dot_plus + dot_cosine_half_sq * plus
        rotated_dp += sine_half_sq * dot_minus + dot_sine_half_sq * minus
        rotated_dp -= sine * dot_longitudinal + dot_sine * longitudinal
        rotated_dm = sine_half_sq * dot_plus + dot_sine_half_sq * plus
        rotated_dm += cosine_half_sq * dot_minus + dot_cosine_half_sq * minus
        rotated_dm += sine * dot_longitudinal + dot_sine * longitudinal
        rotated_dz = half_sine * dot_plus + dot_half_sine * plus
        rotated_dz -= half_sine * dot_minus + dot_half_sine * minus
        rotated_dz += cosine * dot_longitudinal + dot_cosine * longitudinal
        rotated_p = cosine_half_sq * plus + sine_half_sq * minus - sine * longitudinal
        rotated_m = sine_half_sq * plus + cosine_half_sq * minus + sine * longitudinal
        rotated_z = half_sine * plus - half_sine * minus + cosine * longitudinal

        rotate = is_rf & ~is_inversion
        plus = tl.where(rotate, rotated_p, plus)
        minus = tl.where(rotate, rotated_m, minus)
        longitudinal = tl.where(rotate, rotated_z, longitudinal)
        dot_plus = tl.where(rotate, rotated_dp, dot_plus)
        dot_minus = tl.where(rotate, rotated_dm, dot_minus)
        dot_longitudinal = tl.where(rotate, rotated_dz, dot_longitudinal)

        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        output_offset = problem * output_count + out
        output_mask = active_atom & (state == 0) & record & (out >= 0)
        signal_imag = dot_m0 * plus + atom_m0 * dot_plus
        tl.store(output_real + output_offset + state, empty, mask=output_mask)
        tl.store(output_imag + output_offset + state, signal_imag, mask=output_mask)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_p, shifted_m = _shift_real(
            plus, minus, scratch_p, scratch_m, state, state_mask, state_count
        )
        shifted_dp, shifted_dm = _shift_real(
            dot_plus, dot_minus, scratch_p, scratch_m, state, state_mask, state_count
        )
        plus = tl.where(do_shift, shifted_p, plus)
        minus = tl.where(do_shift, shifted_m, minus)
        dot_plus = tl.where(do_shift, shifted_dp, dot_plus)
        dot_minus = tl.where(do_shift, shifted_dm, dot_minus)
        spoil = (event_action & 8) != 0
        plus = tl.where(spoil, 0.0, plus)
        minus = tl.where(spoil, 0.0, minus)
        dot_plus = tl.where(spoil, 0.0, dot_plus)
        dot_minus = tl.where(spoil, 0.0, dot_minus)


@triton.jit
def _epg_kernel(
    t1,
    t2,
    m0,
    b1,
    b1_phase,
    b0,
    inversion_efficiency,
    diffusion,
    velocity,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    output_real,
    output_imag,
    scratch_fplus_real,
    scratch_fplus_imag,
    scratch_fminus_real,
    scratch_fminus_imag,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    state_count: tl.constexpr,
    shims: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    # A partial block carries lanes with no problem behind them; they must not
    # touch the scratch rows, which only exist for real problems.
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    train = problem // atom_count
    workspace_offset = problem * state_count
    scratch_pr = scratch_fplus_real + workspace_offset
    scratch_pi = scratch_fplus_imag + workspace_offset
    scratch_mr = scratch_fminus_real + workspace_offset
    scratch_mi = scratch_fminus_imag + workspace_offset

    empty = tl.zeros((problems, block_states), tl.float32)
    fplus_real = empty
    fplus_imag = empty
    fminus_real = empty
    fminus_imag = empty
    longitudinal_real = empty + tl.where(state == 0, 1.0, 0.0)
    longitudinal_imag = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = tl.load(m0 + atom, mask=active_atom, other=0.0)
    atom_b1 = tl.load(b1 + atom, mask=active_atom, other=1.0)
    atom_b1_phase = tl.load(b1_phase + atom, mask=active_atom, other=0.0)
    atom_b0 = tl.load(b0 + atom, mask=active_atom, other=0.0)
    atom_inversion = tl.load(
        inversion_efficiency + atom, mask=active_atom, other=1.0
    )
    atom_damping = tl.load(diffusion + atom, mask=active_atom, other=0.0)
    atom_velocity = tl.load(velocity + atom, mask=active_atom, other=0.0)
    atom_flow = atom_velocity * flow_scale
    atom_washout = tl.abs(atom_velocity) * washout_scale
    order = state.to(tl.float32)

    event_base = train * event_count
    for event in range(0, event_count):
        dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        wout = _washout(atom_washout, dt)
        e1 = tl.exp(-(1000.0 / atom_t1) * dt) * wout
        e2 = tl.exp(-(1000.0 / atom_t2) * dt) * wout
        damp_z, damp_t = _damping(atom_damping, dt, order)
        turn_z, turn_t = _flow(atom_flow, dt, order)
        recovery = 1.0 - e1
        e1 = e1 * damp_z
        e2 = e2 * damp_t
        # Flow winds the transverse states through the same rotation
        # off-resonance does, so the two phases add before either is taken.
        off_phase = -2.0 * 3.141592653589793 * atom_b0 * dt + turn_t
        off_cos = tl.cos(off_phase)
        off_sin = tl.sin(off_phase)
        old_real = fplus_real
        fplus_real = e2 * (old_real * off_cos - fplus_imag * off_sin)
        fplus_imag = e2 * (old_real * off_sin + fplus_imag * off_cos)
        old_real = fminus_real
        fminus_real = e2 * (old_real * off_cos + fminus_imag * off_sin)
        fminus_imag = e2 * (-old_real * off_sin + fminus_imag * off_cos)
        # The longitudinal states carry a phase of their own, which nothing
        # else in the state machine gives them.
        turn_cos = tl.cos(turn_z)
        turn_sin = tl.sin(turn_z)
        old_real = longitudinal_real
        longitudinal_real = e1 * (old_real * turn_cos - longitudinal_imag * turn_sin)
        longitudinal_imag = e1 * (old_real * turn_sin + longitudinal_imag * turn_cos)
        longitudinal_real += tl.where(state == 0, recovery, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fplus_real,
            fplus_imag,
            fminus_real,
            fminus_imag,
            scratch_pr,
            scratch_pi,
            scratch_mr,
            scratch_mi,
            state,
            state_mask,
            state_count,
        )
        fplus_real = tl.where(pre_shift, shifted_pr, fplus_real)
        fplus_imag = tl.where(pre_shift, shifted_pi, fplus_imag)
        fminus_real = tl.where(pre_shift, shifted_mr, fminus_real)
        fminus_imag = tl.where(pre_shift, shifted_mi, fminus_imag)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        longitudinal_real = tl.where(
            invert, -atom_inversion * longitudinal_real, longitudinal_real
        )
        longitudinal_imag = tl.where(
            invert, -atom_inversion * longitudinal_imag, longitudinal_imag
        )

        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shims > 1:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            atom_b1_phase = tl.load(
                b1_phase + row + atom, mask=active_atom, other=0.0
            )
        alpha = tl.load(flip + event_base + event, mask=active_atom, other=0.0) * atom_b1
        phi = tl.load(phase + event_base + event, mask=active_atom, other=0.0) + atom_b1_phase
        cosine = tl.cos(alpha)
        sine = tl.sin(alpha)
        cosine_half_sq = 0.5 * (1.0 + cosine)
        sine_half_sq = 0.5 * (1.0 - cosine)
        cos_phi = tl.cos(phi)
        sin_phi = tl.sin(phi)
        cos_2phi = tl.cos(2.0 * phi)
        sin_2phi = tl.sin(2.0 * phi)

        fp_r = fplus_real
        fp_i = fplus_imag
        fm_r = fminus_real
        fm_i = fminus_imag
        z_r = longitudinal_real
        z_i = longitudinal_imag

        rotated_pr = cosine_half_sq * fp_r
        rotated_pr += sine_half_sq * (cos_2phi * fm_r - sin_2phi * fm_i)
        rotated_pr += sine * (sin_phi * z_r + cos_phi * z_i)
        rotated_pi = cosine_half_sq * fp_i
        rotated_pi += sine_half_sq * (sin_2phi * fm_r + cos_2phi * fm_i)
        rotated_pi += sine * (sin_phi * z_i - cos_phi * z_r)

        rotated_mr = sine_half_sq * (cos_2phi * fp_r + sin_2phi * fp_i)
        rotated_mr += cosine_half_sq * fm_r
        rotated_mr += sine * (sin_phi * z_r - cos_phi * z_i)
        rotated_mi = sine_half_sq * (-sin_2phi * fp_r + cos_2phi * fp_i)
        rotated_mi += cosine_half_sq * fm_i
        rotated_mi += sine * (cos_phi * z_r + sin_phi * z_i)

        rotated_zr = -0.5 * sine * (sin_phi * fp_r - cos_phi * fp_i)
        rotated_zr += -0.5 * sine * (sin_phi * fm_r + cos_phi * fm_i)
        rotated_zr += cosine * z_r
        rotated_zi = -0.5 * sine * (cos_phi * fp_r + sin_phi * fp_i)
        rotated_zi += 0.5 * sine * (cos_phi * fm_r - sin_phi * fm_i)
        rotated_zi += cosine * z_i

        rotate = is_rf & ~is_inversion
        fplus_real = tl.where(rotate, rotated_pr, fplus_real)
        fplus_imag = tl.where(rotate, rotated_pi, fplus_imag)
        fminus_real = tl.where(rotate, rotated_mr, fminus_real)
        fminus_imag = tl.where(rotate, rotated_mi, fminus_imag)
        longitudinal_real = tl.where(rotate, rotated_zr, longitudinal_real)
        longitudinal_imag = tl.where(rotate, rotated_zi, longitudinal_imag)

        record = ((event_action & 32) != 0) & (event_kind == 2)
        adc_phase = tl.load(phase + event_base + event, mask=active_atom, other=0.0)
        adc_cos = tl.cos(adc_phase)
        adc_sin = tl.sin(adc_phase)
        signal_real = atom_m0 * (fplus_real * adc_cos + fplus_imag * adc_sin)
        signal_imag = atom_m0 * (fplus_imag * adc_cos - fplus_real * adc_sin)
        out = tl.load(output_index + event)
        output_offset = problem * output_count + out
        output_mask = active_atom & (state == 0) & record & (out >= 0)
        tl.store(output_real + output_offset + state, signal_real, mask=output_mask)
        tl.store(output_imag + output_offset + state, signal_imag, mask=output_mask)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fplus_real,
            fplus_imag,
            fminus_real,
            fminus_imag,
            scratch_pr,
            scratch_pi,
            scratch_mr,
            scratch_mi,
            state,
            state_mask,
            state_count,
        )
        fplus_real = tl.where(do_shift, shifted_pr, fplus_real)
        fplus_imag = tl.where(do_shift, shifted_pi, fplus_imag)
        fminus_real = tl.where(do_shift, shifted_mr, fminus_real)
        fminus_imag = tl.where(do_shift, shifted_mi, fminus_imag)
        spoil = (event_action & 8) != 0
        fplus_real = tl.where(spoil, 0.0, fplus_real)
        fplus_imag = tl.where(spoil, 0.0, fplus_imag)
        fminus_real = tl.where(spoil, 0.0, fminus_real)
        fminus_imag = tl.where(spoil, 0.0, fminus_imag)


@triton.jit
def _epg_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    b1_phase,
    b0,
    inversion_efficiency,
    diffusion,
    velocity,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    tangent_t1,
    tangent_t2,
    tangent_m0,
    tangent_b1,
    tangent_b1_phase,
    tangent_b0,
    tangent_inversion_efficiency,
    tangent_diffusion,
    tangent_velocity,
    tangent_duration,
    tangent_flip,
    tangent_phase,
    output_real,
    output_imag,
    scratch_fplus_real,
    scratch_fplus_imag,
    scratch_fminus_real,
    scratch_fminus_imag,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    state_count: tl.constexpr,
    shims: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    # A partial block carries lanes with no problem behind them; they must not
    # touch the scratch rows, which only exist for real problems.
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    train = problem // atom_count
    workspace_offset = problem * state_count
    scratch_pr = scratch_fplus_real + workspace_offset
    scratch_pi = scratch_fplus_imag + workspace_offset
    scratch_mr = scratch_fminus_real + workspace_offset
    scratch_mi = scratch_fminus_imag + workspace_offset

    empty = tl.zeros((problems, block_states), tl.float32)
    fpr = empty
    fpi = empty
    fmr = empty
    fmi = empty
    zr = empty + tl.where(state == 0, 1.0, 0.0)
    zi = empty
    dfpr = empty
    dfpi = empty
    dfmr = empty
    dfmi = empty
    dzr = empty
    dzi = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = tl.load(m0 + atom, mask=active_atom, other=0.0)
    atom_b1 = tl.load(b1 + atom, mask=active_atom, other=1.0)
    atom_b1_phase = tl.load(b1_phase + atom, mask=active_atom, other=0.0)
    atom_b0 = tl.load(b0 + atom, mask=active_atom, other=0.0)
    atom_inversion = tl.load(
        inversion_efficiency + atom, mask=active_atom, other=1.0
    )
    atom_damping = tl.load(diffusion + atom, mask=active_atom, other=0.0)
    d_damping = tl.load(tangent_diffusion + atom, mask=active_atom, other=0.0)
    atom_velocity = tl.load(velocity + atom, mask=active_atom, other=0.0)
    d_velocity = tl.load(tangent_velocity + atom, mask=active_atom, other=0.0)
    atom_flow = atom_velocity * flow_scale
    d_flow = d_velocity * flow_scale
    # |v| has no derivative at the origin, so a still voxel contributes none.
    direction = (atom_velocity > 0.0).to(tl.float32) - (atom_velocity < 0.0).to(
        tl.float32
    )
    atom_washout = tl.abs(atom_velocity) * washout_scale
    d_washout = direction * d_velocity * washout_scale
    order = state.to(tl.float32)
    dt1 = tl.load(tangent_t1 + atom, mask=active_atom, other=0.0)
    dt2 = tl.load(tangent_t2 + atom, mask=active_atom, other=0.0)
    dm0 = tl.load(tangent_m0 + atom, mask=active_atom, other=0.0)
    db1 = tl.load(tangent_b1 + atom, mask=active_atom, other=0.0)
    db1_phase = tl.load(tangent_b1_phase + atom, mask=active_atom, other=0.0)
    db0 = tl.load(tangent_b0 + atom, mask=active_atom, other=0.0)
    dinversion = tl.load(
        tangent_inversion_efficiency + atom, mask=active_atom, other=0.0
    )

    event_base = train * event_count
    for event in range(0, event_count):
        event_dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        ddt = tl.load(tangent_duration + event_base + event, mask=active_atom, other=0.0)
        r1 = 1000.0 / atom_t1
        r2 = 1000.0 / atom_t2
        wout, dwout = _washout_jvp(atom_washout, d_washout, event_dt, ddt)
        dry1 = tl.exp(-r1 * event_dt)
        dry2 = tl.exp(-r2 * event_dt)
        e1 = dry1 * wout
        e2 = dry2 * wout
        de1 = e1 * (
            1000.0 * event_dt * dt1 / (atom_t1 * atom_t1) - r1 * ddt
        ) + dry1 * dwout
        de2 = e2 * (
            1000.0 * event_dt * dt2 / (atom_t2 * atom_t2) - r2 * ddt
        ) + dry2 * dwout
        damp_z, ddamp_z, damp_t, ddamp_t = _damping_jvp(
            atom_damping, d_damping, event_dt, ddt, order
        )
        # Order zero is undamped, so the recovery term keeps the bare factor.
        recovery, drecovery = 1.0 - e1, -de1
        de1 = de1 * damp_z + e1 * ddamp_z
        e1 = e1 * damp_z
        de2 = de2 * damp_t + e2 * ddamp_t
        e2 = e2 * damp_t
        turn_z, turn_t = _flow(atom_flow, event_dt, order)
        d_turn = d_flow * event_dt + atom_flow * ddt
        dturn_z = -order * d_turn
        dturn_t = -(order + 0.5) * d_turn
        # Flow winds the transverse states through the same rotation
        # off-resonance does, so the two phases add before either is taken.
        off_phase = -2.0 * 3.141592653589793 * atom_b0 * event_dt + turn_t
        doff_phase = -2.0 * 3.141592653589793 * (
            db0 * event_dt + atom_b0 * ddt
        ) + dturn_t
        off_cos = tl.cos(off_phase)
        off_sin = tl.sin(off_phase)
        doff_cos = -off_sin * doff_phase
        doff_sin = off_cos * doff_phase

        old_fpr = fpr
        old_fpi = fpi
        old_dfpr = dfpr
        old_dfpi = dfpi
        fpr = e2 * (old_fpr * off_cos - old_fpi * off_sin)
        fpi = e2 * (old_fpr * off_sin + old_fpi * off_cos)
        dfpr = de2 * (old_fpr * off_cos - old_fpi * off_sin)
        dfpr += e2 * (
            old_dfpr * off_cos
            + old_fpr * doff_cos
            - old_dfpi * off_sin
            - old_fpi * doff_sin
        )
        dfpi = de2 * (old_fpr * off_sin + old_fpi * off_cos)
        dfpi += e2 * (
            old_dfpr * off_sin
            + old_fpr * doff_sin
            + old_dfpi * off_cos
            + old_fpi * doff_cos
        )

        old_fmr = fmr
        old_fmi = fmi
        old_dfmr = dfmr
        old_dfmi = dfmi
        fmr = e2 * (old_fmr * off_cos + old_fmi * off_sin)
        fmi = e2 * (-old_fmr * off_sin + old_fmi * off_cos)
        dfmr = de2 * (old_fmr * off_cos + old_fmi * off_sin)
        dfmr += e2 * (
            old_dfmr * off_cos
            + old_fmr * doff_cos
            + old_dfmi * off_sin
            + old_fmi * doff_sin
        )
        dfmi = de2 * (-old_fmr * off_sin + old_fmi * off_cos)
        dfmi += e2 * (
            -old_dfmr * off_sin
            - old_fmr * doff_sin
            + old_dfmi * off_cos
            + old_fmi * doff_cos
        )

        # The longitudinal states carry a phase of their own, which nothing
        # else in the state machine gives them.
        turn_cos = tl.cos(turn_z)
        turn_sin = tl.sin(turn_z)
        dturn_cos = -turn_sin * dturn_z
        dturn_sin = turn_cos * dturn_z
        old_zr = zr
        old_zi = zi
        old_dzr = dzr
        old_dzi = dzi
        spun_zr = old_zr * turn_cos - old_zi * turn_sin
        spun_zi = old_zr * turn_sin + old_zi * turn_cos
        dspun_zr = (
            old_dzr * turn_cos
            + old_zr * dturn_cos
            - old_dzi * turn_sin
            - old_zi * dturn_sin
        )
        dspun_zi = (
            old_dzr * turn_sin
            + old_zr * dturn_sin
            + old_dzi * turn_cos
            + old_zi * dturn_cos
        )
        dzr = dspun_zr * e1 + spun_zr * de1 + tl.where(state == 0, drecovery, 0.0)
        dzi = dspun_zi * e1 + spun_zi * de1
        zr = spun_zr * e1 + tl.where(state == 0, recovery, 0.0)
        zi = spun_zi * e1

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fpr,
            fpi,
            fmr,
            fmi,
            scratch_pr,
            scratch_pi,
            scratch_mr,
            scratch_mi,
            state,
            state_mask,
            state_count,
        )
        shifted_dpr, shifted_dpi, shifted_dmr, shifted_dmi = _shift(
            dfpr,
            dfpi,
            dfmr,
            dfmi,
            scratch_pr,
            scratch_pi,
            scratch_mr,
            scratch_mi,
            state,
            state_mask,
            state_count,
        )
        fpr = tl.where(pre_shift, shifted_pr, fpr)
        fpi = tl.where(pre_shift, shifted_pi, fpi)
        fmr = tl.where(pre_shift, shifted_mr, fmr)
        fmi = tl.where(pre_shift, shifted_mi, fmi)
        dfpr = tl.where(pre_shift, shifted_dpr, dfpr)
        dfpi = tl.where(pre_shift, shifted_dpi, dfpi)
        dfmr = tl.where(pre_shift, shifted_dmr, dfmr)
        dfmi = tl.where(pre_shift, shifted_dmi, dfmi)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        dzr = tl.where(invert, -dinversion * zr - atom_inversion * dzr, dzr)
        dzi = tl.where(invert, -dinversion * zi - atom_inversion * dzi, dzi)
        zr = tl.where(invert, -atom_inversion * zr, zr)
        zi = tl.where(invert, -atom_inversion * zi, zi)

        event_flip = tl.load(flip + event_base + event, mask=active_atom, other=0.0)
        event_phase = tl.load(phase + event_base + event, mask=active_atom, other=0.0)
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shims > 1:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            atom_b1_phase = tl.load(
                b1_phase + row + atom, mask=active_atom, other=0.0
            )
            db1 = tl.load(tangent_b1 + row + atom, mask=active_atom, other=0.0)
            db1_phase = tl.load(
                tangent_b1_phase + row + atom, mask=active_atom, other=0.0
            )
        alpha = event_flip * atom_b1
        dalpha = tl.load(tangent_flip + event_base + event, mask=active_atom, other=0.0) * atom_b1 + event_flip * db1
        phi = event_phase + atom_b1_phase
        dphi = tl.load(tangent_phase + event_base + event, mask=active_atom, other=0.0) + db1_phase
        cosine = tl.cos(alpha)
        sine = tl.sin(alpha)
        dcosine = -sine * dalpha
        dsine = cosine * dalpha
        ch = 0.5 * (1.0 + cosine)
        sh = 0.5 * (1.0 - cosine)
        dch = 0.5 * dcosine
        dsh = -0.5 * dcosine
        cos_phi = tl.cos(phi)
        sin_phi = tl.sin(phi)
        cos_2phi = tl.cos(2.0 * phi)
        sin_2phi = tl.sin(2.0 * phi)
        dcos_phi = -sin_phi * dphi
        dsin_phi = cos_phi * dphi
        dcos_2phi = -2.0 * sin_2phi * dphi
        dsin_2phi = 2.0 * cos_2phi * dphi

        pr_a = cos_2phi * fmr - sin_2phi * fmi
        dpr_a = dcos_2phi * fmr + cos_2phi * dfmr
        dpr_a -= dsin_2phi * fmi + sin_2phi * dfmi
        pr_b = sin_phi * zr + cos_phi * zi
        dpr_b = dsin_phi * zr + sin_phi * dzr + dcos_phi * zi + cos_phi * dzi
        rotated_pr = ch * fpr + sh * pr_a + sine * pr_b
        rotated_dpr = dch * fpr + ch * dfpr + dsh * pr_a + sh * dpr_a
        rotated_dpr += dsine * pr_b + sine * dpr_b

        pi_a = sin_2phi * fmr + cos_2phi * fmi
        dpi_a = dsin_2phi * fmr + sin_2phi * dfmr
        dpi_a += dcos_2phi * fmi + cos_2phi * dfmi
        pi_b = sin_phi * zi - cos_phi * zr
        dpi_b = dsin_phi * zi + sin_phi * dzi - dcos_phi * zr - cos_phi * dzr
        rotated_pi = ch * fpi + sh * pi_a + sine * pi_b
        rotated_dpi = dch * fpi + ch * dfpi + dsh * pi_a + sh * dpi_a
        rotated_dpi += dsine * pi_b + sine * dpi_b

        mr_a = cos_2phi * fpr + sin_2phi * fpi
        dmr_a = dcos_2phi * fpr + cos_2phi * dfpr
        dmr_a += dsin_2phi * fpi + sin_2phi * dfpi
        mr_b = sin_phi * zr - cos_phi * zi
        dmr_b = dsin_phi * zr + sin_phi * dzr - dcos_phi * zi - cos_phi * dzi
        rotated_mr = sh * mr_a + ch * fmr + sine * mr_b
        rotated_dmr = dsh * mr_a + sh * dmr_a + dch * fmr + ch * dfmr
        rotated_dmr += dsine * mr_b + sine * dmr_b

        mi_a = -sin_2phi * fpr + cos_2phi * fpi
        dmi_a = -dsin_2phi * fpr - sin_2phi * dfpr
        dmi_a += dcos_2phi * fpi + cos_2phi * dfpi
        mi_b = cos_phi * zr + sin_phi * zi
        dmi_b = dcos_phi * zr + cos_phi * dzr + dsin_phi * zi + sin_phi * dzi
        rotated_mi = sh * mi_a + ch * fmi + sine * mi_b
        rotated_dmi = dsh * mi_a + sh * dmi_a + dch * fmi + ch * dfmi
        rotated_dmi += dsine * mi_b + sine * dmi_b

        zr_a = sin_phi * fpr - cos_phi * fpi
        dzr_a = dsin_phi * fpr + sin_phi * dfpr - dcos_phi * fpi - cos_phi * dfpi
        zr_b = sin_phi * fmr + cos_phi * fmi
        dzr_b = dsin_phi * fmr + sin_phi * dfmr + dcos_phi * fmi + cos_phi * dfmi
        rotated_zr = -0.5 * sine * zr_a - 0.5 * sine * zr_b + cosine * zr
        rotated_dzr = -0.5 * (dsine * zr_a + sine * dzr_a)
        rotated_dzr -= 0.5 * (dsine * zr_b + sine * dzr_b)
        rotated_dzr += dcosine * zr + cosine * dzr

        zi_a = cos_phi * fpr + sin_phi * fpi
        dzi_a = dcos_phi * fpr + cos_phi * dfpr + dsin_phi * fpi + sin_phi * dfpi
        zi_b = cos_phi * fmr - sin_phi * fmi
        dzi_b = dcos_phi * fmr + cos_phi * dfmr - dsin_phi * fmi - sin_phi * dfmi
        rotated_zi = -0.5 * sine * zi_a + 0.5 * sine * zi_b + cosine * zi
        rotated_dzi = -0.5 * (dsine * zi_a + sine * dzi_a)
        rotated_dzi += 0.5 * (dsine * zi_b + sine * dzi_b)
        rotated_dzi += dcosine * zi + cosine * dzi

        rotate = is_rf & ~is_inversion
        fpr = tl.where(rotate, rotated_pr, fpr)
        fpi = tl.where(rotate, rotated_pi, fpi)
        fmr = tl.where(rotate, rotated_mr, fmr)
        fmi = tl.where(rotate, rotated_mi, fmi)
        zr = tl.where(rotate, rotated_zr, zr)
        zi = tl.where(rotate, rotated_zi, zi)
        dfpr = tl.where(rotate, rotated_dpr, dfpr)
        dfpi = tl.where(rotate, rotated_dpi, dfpi)
        dfmr = tl.where(rotate, rotated_dmr, dfmr)
        dfmi = tl.where(rotate, rotated_dmi, dfmi)
        dzr = tl.where(rotate, rotated_dzr, dzr)
        dzi = tl.where(rotate, rotated_dzi, dzi)

        record = ((event_action & 32) != 0) & (event_kind == 2)
        adc_cos = tl.cos(event_phase)
        adc_sin = tl.sin(event_phase)
        dadc_phase = tl.load(tangent_phase + event_base + event, mask=active_atom, other=0.0)
        dadc_cos = -adc_sin * dadc_phase
        dadc_sin = adc_cos * dadc_phase
        signal_real = dm0 * (fpr * adc_cos + fpi * adc_sin)
        signal_real += atom_m0 * (
            dfpr * adc_cos + fpr * dadc_cos + dfpi * adc_sin + fpi * dadc_sin
        )
        signal_imag = dm0 * (fpi * adc_cos - fpr * adc_sin)
        signal_imag += atom_m0 * (
            dfpi * adc_cos + fpi * dadc_cos - dfpr * adc_sin - fpr * dadc_sin
        )
        out = tl.load(output_index + event)
        output_offset = problem * output_count + out
        output_mask = active_atom & (state == 0) & record & (out >= 0)
        tl.store(output_real + output_offset + state, signal_real, mask=output_mask)
        tl.store(output_imag + output_offset + state, signal_imag, mask=output_mask)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fpr,
            fpi,
            fmr,
            fmi,
            scratch_pr,
            scratch_pi,
            scratch_mr,
            scratch_mi,
            state,
            state_mask,
            state_count,
        )
        shifted_dpr, shifted_dpi, shifted_dmr, shifted_dmi = _shift(
            dfpr,
            dfpi,
            dfmr,
            dfmi,
            scratch_pr,
            scratch_pi,
            scratch_mr,
            scratch_mi,
            state,
            state_mask,
            state_count,
        )
        fpr = tl.where(do_shift, shifted_pr, fpr)
        fpi = tl.where(do_shift, shifted_pi, fpi)
        fmr = tl.where(do_shift, shifted_mr, fmr)
        fmi = tl.where(do_shift, shifted_mi, fmi)
        dfpr = tl.where(do_shift, shifted_dpr, dfpr)
        dfpi = tl.where(do_shift, shifted_dpi, dfpi)
        dfmr = tl.where(do_shift, shifted_dmr, dfmr)
        dfmi = tl.where(do_shift, shifted_dmi, dfmi)
        spoil = (event_action & 8) != 0
        fpr = tl.where(spoil, 0.0, fpr)
        fpi = tl.where(spoil, 0.0, fpi)
        fmr = tl.where(spoil, 0.0, fmr)
        fmi = tl.where(spoil, 0.0, fmi)
        dfpr = tl.where(spoil, 0.0, dfpr)
        dfpi = tl.where(spoil, 0.0, dfpi)
        dfmr = tl.where(spoil, 0.0, dfmr)
        dfmi = tl.where(spoil, 0.0, dfmi)


def _problems_per_program(total: int, block_states: int) -> int:
    """How many independent problems to carry on one program's lane axis.

    A warp's lanes cost about the same whether they are used or not, so packing
    several problems into one program is close to free -- but only while enough
    programs remain to fill the device. Below that the packing starves the GPU
    of parallelism and costs more than the lanes save.

    The result indexes a ``tl.arange``, so it must be a power of two.
    """
    widest = max(1, 64 // block_states)
    packed = max(1, min(widest, total // 1024))
    return 1 << (packed.bit_length() - 1)


def _output_shape(
    train_count: int, atom_count: int, output_count: int
) -> tuple[int, ...]:
    """Signal shape, matching what the CPU kernels return."""
    if train_count == 1:
        return (atom_count, output_count)
    return (train_count, atom_count, output_count)


def _scratch(
    count: int, problems: int, device: torch.device, state_count: int
) -> list[torch.Tensor]:
    """Per-problem staging rows for the shift, one set per state plane.

    The shift moves data between lanes of the state axis, which registers
    cannot do, so it goes out to memory and back.
    """
    return [
        torch.empty((problems, state_count), dtype=torch.float32, device=device)
        for _ in range(count)
    ]


def simulate(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
) -> torch.Tensor:
    """Run a packed state machine on CUDA and return complex signals.

    ``real_axis`` of 1 selects the real-subspace kernel; see
    ``real_subspace_axis`` for when that is legitimate.
    """
    train_count = _train_count(events)
    atom_count = tissue[0].numel()
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.empty_like(output_real)
    simulate_into(
        tissue,
        events,
        output_real,
        output_imag,
        None,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
    )
    return torch.complex(output_real, output_imag)


def simulate_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    scratch: list[torch.Tensor] | None,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
) -> None:
    """Run the forward machine into buffers the caller owns.

    Streaming reuses one set of buffers per chunk, so allocating here would put
    an allocation in the loop -- and an allocation that reaches ``cudaMalloc``
    synchronizes the device, which is exactly what the streams exist to avoid.

    ``atom_count`` is given rather than taken from ``tissue`` because a chunk's
    buffers are sized for the largest chunk and the last one is shorter.
    Passing ``None`` for ``scratch`` allocates it for this call alone.
    """
    (
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, diffusion, velocity,
    ) = tissue
    duration, kind, flip, phase, action, output_index, shim_index = events
    train_count = _train_count(events)
    shims = _shim_count(tissue)
    block_states = triton.next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(total, block_states)
    grid = (triton.cdiv(total, problems),)
    planes = 2 if real_axis == 1 else 4
    if scratch is None:
        scratch = _scratch(planes, total, t1.device, state_count)

    if real_axis == 1:
        _epg_real_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            output_real,
            output_imag,
            *scratch,
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        return

    _epg_kernel[grid](
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        output_real,
        output_imag,
        *scratch,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        state_count=state_count,
        shims=shims,
        block_states=block_states,
        problems=problems,
        num_warps=1,
    )


def simulate_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
) -> torch.Tensor:
    """Run one fused state-machine Jacobian-vector product on CUDA.

    ``real_axis`` of 1 selects the real-subspace kernel, which produces no
    derivative along ``b1_phase``, ``b0`` or the RF phase -- seeds along those
    directions leave the subspace, so the caller must rule them out.
    """
    train_count = _train_count(events)
    atom_count = tissue[0].numel()
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.empty_like(output_real)
    simulate_jvp_into(
        tissue,
        events,
        tissue_tangents,
        event_tangents,
        output_real,
        output_imag,
        None,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
    )
    return torch.complex(output_real, output_imag)


def simulate_jvp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    scratch: list[torch.Tensor] | None,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
) -> None:
    """Run one Jacobian-vector product into buffers the caller owns.

    See ``simulate_into`` for why the streaming path needs this.
    """
    (
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, diffusion, velocity,
    ) = tissue
    duration, kind, flip, phase, action, output_index, shim_index = events
    tangent_duration, tangent_flip, tangent_phase = event_tangents
    train_count = _train_count(events)
    shims = _shim_count(tissue)
    block_states = triton.next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(total, block_states)
    grid = (triton.cdiv(total, problems),)
    if scratch is None:
        scratch = _scratch(
            2 if real_axis == 1 else 4, total, t1.device, state_count
        )

    if real_axis == 1:
        _epg_real_jvp_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            tissue_tangents[0],
            tissue_tangents[1],
            tissue_tangents[2],
            tissue_tangents[3],
            tissue_tangents[6],
            tissue_tangents[7],
            tangent_duration,
            tangent_flip,
            output_real,
            output_imag,
            *scratch,
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        return

    _epg_jvp_kernel[grid](
        *tissue,
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        *tissue_tangents,
        tangent_duration,
        tangent_flip,
        tangent_phase,
        output_real,
        output_imag,
        *scratch,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        state_count=state_count,
        shims=shims,
        block_states=block_states,
        problems=problems,
        num_warps=1,
    )


# How much device memory the recorded trajectory may hold at once. Beyond this
# the problems are run in waves, which the gradient buffers absorb because they
# accumulate rather than being written.
_TRAJECTORY_BUDGET_BYTES = 256 << 20


def _trajectory_wave(
    event_count: int, state_count: int, total: int, planes: int
) -> int:
    """How many problems can record their trajectory in one launch."""
    per_problem = event_count * 3 * state_count * planes * 4
    return max(1, min(total, _TRAJECTORY_BUDGET_BYTES // max(1, per_problem)))


class AdjointBuffers:
    """Device memory a forward-over-reverse pass writes into.

    Sized for ``chunk`` voxels and reusable for any narrower one. Per-voxel
    gradients are cleared before each pass; per-event gradients accumulate over
    every pass the buffers serve and are read out with ``event_gradients``.

    ``real_axis`` of 1 halves the state planes, so buffers built for one
    representation cannot be handed to the other.
    """

    def __init__(
        self,
        events: tuple[torch.Tensor, ...],
        chunk: int,
        *,
        state_count: int,
        output_count: int,
        real_axis: int | None = None,
        shims: int = 1,
    ) -> None:
        duration, kind, flip, phase, _action, _output_index, _shim = events
        device = kind.device
        train_count = _train_count(events)
        event_count = kind.numel()
        self.planes = 2 if real_axis == 1 else 4
        self.chunk = chunk
        self.shims = shims
        self.rows = tissue_gradient_height(shims)
        self.state_count = state_count
        self.output_count = output_count
        self.train_count = train_count
        # One dual accumulator per plane: value is the gradient w.r.t. the
        # tangent inputs, tangent the gradient w.r.t. the primal ones.
        self.tissue = [
            torch.zeros(self.rows * chunk, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.flip = [torch.zeros_like(flip) for _ in range(2)]
        self.duration = [torch.zeros_like(duration) for _ in range(2)]
        self.phase = [torch.zeros_like(phase) for _ in range(2)]
        self.cotangent = [
            torch.empty(
                train_count * chunk * output_count,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self.wave = _trajectory_wave(
            event_count, state_count, train_count * chunk, self.planes
        )
        self.trajectory = [
            torch.empty(
                (self.wave, event_count * 3 * state_count),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.planes)
        ]
        self.scratch = _scratch(self.planes, self.wave, device, state_count)

    def tissue_gradients(self, atom_count: int) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The per-voxel gradients of the last pass, one entry per parameter.

        Each is flat and as wide as the buffer it belongs to, so the transmit
        pair spans every shim. Ordered to match ``event_gradients``: tangent
        plane first.
        """
        return tuple(
            tuple(
                self.tissue[plane][
                    base * atom_count : (base + rows) * atom_count
                ]
                for base, rows in zip(
                    tissue_gradient_bases(self.shims),
                    tissue_gradient_rows(self.shims),
                    strict=True,
                )
            )
            for plane in (1, 0)
        )

    def event_gradients(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The per-event gradients summed over every pass so far.

        Ordered ``(duration, flip, phase)`` to match the tail of the
        differentiable-input order, tangent plane first.
        """
        return tuple(
            (self.duration[plane], self.flip[plane], self.phase[plane])
            for plane in (1, 0)
        )


def simulate_vjp_jvp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: AdjointBuffers,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Forward-over-reverse for one chunk of voxels, into caller-owned buffers.

    Returns the per-voxel gradients of this chunk -- tangent plane first, one
    entry per tissue parameter, views into ``buffers`` that the next call
    overwrites. The per-event gradients accumulate inside ``buffers`` instead,
    because every chunk contributes to all of them.

    ``atom_count`` is this chunk's width, which may be narrower than the one
    the buffers were built for.
    """
    (
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, diffusion, velocity,
    ) = tissue
    duration, kind, flip, phase, action, output_index, shim_index = events
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = triton.next_power_of_2(state_count)
    real = real_axis == 1

    grad_output = grad_output.resolve_conj()
    size = total * output_count
    grad_real, grad_imag = (
        plane[:size].view(grad_output.shape) for plane in buffers.cotangent
    )
    grad_real.copy_(grad_output.real)
    grad_imag.copy_(grad_output.imag)
    grad_tissue = [plane[: buffers.rows * atom_count] for plane in buffers.tissue]
    for plane in grad_tissue:
        plane.zero_()
    grad_flip, grad_duration, grad_phase = buffers.flip, buffers.duration, buffers.phase
    trajectory, scratch = buffers.trajectory, buffers.scratch

    wave = buffers.wave
    problems = _problems_per_program(wave, block_states)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        grid = (triton.cdiv(span, problems),)
        shape = dict(
            state_count=state_count,
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        if real:
            _epg_real_vjp_jvp_kernel[grid](
                t1,
                t2,
                m0,
                b1,
                inversion_efficiency,
                diffusion,
                duration,
                kind,
                flip,
                action,
                output_index,
                tangents[0],
                tangents[1],
                tangents[2],
                tangents[3],
                tangents[6],
                tangents[7],
                tangents[9],
                tangents[10],
                grad_imag,
                *grad_tissue,
                *grad_flip,
                *grad_duration,
                *trajectory,
                *scratch,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                **shape,
            )
        else:
            _epg_vjp_jvp_kernel[grid](
                *tissue,
                *events,
                *tangents,
                grad_real,
                grad_imag,
                *grad_tissue,
                *grad_flip,
                *grad_phase,
                *grad_duration,
                *trajectory,
                *scratch,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                geometry.flow_scale,
                geometry.washout_scale,
                shims=_shim_count(tissue),
                **shape,
            )

    # Plane 1 is the tangent part -> d/d(primal inputs); plane 0 the value part.
    return buffers.tissue_gradients(atom_count)


def simulate_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse through the state machine on CUDA.

    ``tangents`` follows the differentiable-input order ``(t1, t2, m0, b1,
    b1_phase, b0, inversion_efficiency, duration, flip, phase)``, and the two
    returned tuples -- gradients with respect to the primal inputs, then to the
    tangent inputs -- follow it too.

    ``real_axis`` of 1 selects the real-subspace adjoint. That representation
    divides the RF phase out, so it leaves ``b1_phase``, ``b0`` and ``phase`` at
    zero and callers must not ask for those; the complex adjoint produces all
    ten.

    Gradients land through atomic accumulation, so repeated runs agree to
    floating-point tolerance rather than bit for bit.
    """
    atom_count = tissue[0].numel()
    buffers = AdjointBuffers(
        events,
        atom_count,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        shims=_shim_count(tissue),
    )
    voxel_grads = simulate_vjp_jvp_into(
        tissue,
        events,
        tangents,
        grad_output,
        buffers,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
    )
    return tuple(
        (*voxels, *per_event)
        for voxels, per_event in zip(
            voxel_grads, buffers.event_gradients(), strict=True
        )
    )
