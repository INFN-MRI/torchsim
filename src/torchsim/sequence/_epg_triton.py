"""Fused Triton kernel for inference-only EPG state machines."""

from __future__ import annotations

__all__: list[str] = []

import torch

from ._accelerators import _train_count
import triton
import triton.language as tl


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


@triton.jit
def _epg_real_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
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

    event_base = train * event_count
    for event in range(0, event_count):
        dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        e1 = tl.exp(-rate1 * dt)
        e2 = tl.exp(-rate2 * dt)
        plus *= e2
        minus *= e2
        longitudinal = longitudinal * e1 + tl.where(state == 0, 1.0 - e1, 0.0)

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
        dot_plus = dot_plus * e2 + plus * dot_e2
        dot_minus = dot_minus * e2 + minus * dot_e2
        dot_longitudinal = dot_longitudinal * e1 + longitudinal * dot_e1
        dot_longitudinal -= tl.where(state == 0, dot_e1, 0.0)
        plus *= e2
        minus *= e2
        longitudinal = longitudinal * e1 + tl.where(state == 0, 1.0 - e1, 0.0)

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
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
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
    state_count: tl.constexpr,
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

    event_base = train * event_count
    for event in range(0, event_count):
        dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        e1 = tl.exp(-(1000.0 / atom_t1) * dt)
        e2 = tl.exp(-(1000.0 / atom_t2) * dt)
        off_phase = -2.0 * 3.141592653589793 * atom_b0 * dt
        off_cos = tl.cos(off_phase)
        off_sin = tl.sin(off_phase)
        old_real = fplus_real
        fplus_real = e2 * (old_real * off_cos - fplus_imag * off_sin)
        fplus_imag = e2 * (old_real * off_sin + fplus_imag * off_cos)
        old_real = fminus_real
        fminus_real = e2 * (old_real * off_cos + fminus_imag * off_sin)
        fminus_imag = e2 * (-old_real * off_sin + fminus_imag * off_cos)
        longitudinal_real *= e1
        longitudinal_imag *= e1
        longitudinal_real += tl.where(state == 0, 1.0 - e1, 0.0)

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
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    tangent_t1,
    tangent_t2,
    tangent_m0,
    tangent_b1,
    tangent_b1_phase,
    tangent_b0,
    tangent_inversion_efficiency,
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
    state_count: tl.constexpr,
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
        e1 = tl.exp(-r1 * event_dt)
        e2 = tl.exp(-r2 * event_dt)
        de1 = e1 * (1000.0 * event_dt * dt1 / (atom_t1 * atom_t1) - r1 * ddt)
        de2 = e2 * (1000.0 * event_dt * dt2 / (atom_t2 * atom_t2) - r2 * ddt)
        off_phase = -2.0 * 3.141592653589793 * atom_b0 * event_dt
        doff_phase = -2.0 * 3.141592653589793 * (
            db0 * event_dt + atom_b0 * ddt
        )
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

        old_zr = zr
        old_zi = zi
        dzr = dzr * e1 + old_zr * de1 - tl.where(state == 0, de1, 0.0)
        dzi = dzi * e1 + old_zi * de1
        zr = old_zr * e1 + tl.where(state == 0, 1.0 - e1, 0.0)
        zi = old_zi * e1

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
) -> torch.Tensor:
    """Run a packed state machine on CUDA and return complex signals.

    ``real_axis`` of 1 selects the real-subspace kernel; see
    ``real_subspace_axis`` for when that is legitimate.
    """
    t1, t2, m0, b1, b1_phase, b0, inversion_efficiency = tissue
    duration, kind, flip, phase, action, output_index = events
    atom_count = t1.numel()
    train_count = _train_count(events)
    block_states = triton.next_power_of_2(state_count)
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=t1.device,
    )
    output_imag = torch.empty_like(output_real)
    total = train_count * atom_count
    problems = _problems_per_program(total, block_states)
    grid = (triton.cdiv(total, problems),)

    if real_axis == 1:
        _epg_real_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            duration,
            kind,
            flip,
            action,
            output_index,
            output_real,
            output_imag,
            *_scratch(2, total, t1.device, state_count),
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        return torch.complex(output_real, output_imag)

    scratch = _scratch(4, total, t1.device, state_count)
    _epg_kernel[grid](
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        duration,
        kind,
        flip,
        phase,
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
    return torch.complex(output_real, output_imag)


def simulate_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
) -> torch.Tensor:
    """Run one fused state-machine Jacobian-vector product on CUDA.

    ``real_axis`` of 1 selects the real-subspace kernel, which produces no
    derivative along ``b1_phase``, ``b0`` or the RF phase -- seeds along those
    directions leave the subspace, so the caller must rule them out.
    """
    t1, t2, m0, b1, b1_phase, b0, inversion_efficiency = tissue
    duration, kind, flip, phase, action, output_index = events
    tangent_duration, tangent_flip, tangent_phase = event_tangents
    atom_count = t1.numel()
    train_count = _train_count(events)
    block_states = triton.next_power_of_2(state_count)
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=t1.device,
    )
    output_imag = torch.empty_like(output_real)
    total = train_count * atom_count
    problems = _problems_per_program(total, block_states)
    grid = (triton.cdiv(total, problems),)

    if real_axis == 1:
        _epg_real_jvp_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
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
            tangent_duration,
            tangent_flip,
            output_real,
            output_imag,
            *_scratch(2, total, t1.device, state_count),
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        return torch.complex(output_real, output_imag)

    scratch = _scratch(4, total, t1.device, state_count)
    _epg_jvp_kernel[grid](
        *tissue,
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
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
        state_count=state_count,
        block_states=block_states,
        problems=problems,
        num_warps=1,
    )
    return torch.complex(output_real, output_imag)
