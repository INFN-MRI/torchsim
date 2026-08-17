"""Fused Triton kernel for inference-only EPG state machines."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

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
from ._parameters import BOUND_POOL_INPUTS as _BOUND_POOL_INPUTS
from ._parameters import FLOAT_NAMES as _FLOAT_NAMES
from ._parameters import TISSUE_COUNT as _TISSUE_PARAMETERS
from ._parameters import TRANSMIT_INPUTS as _TRANSMIT_INPUTS

# Triton reads globals only through its own constexpr wrapper.
_TISSUE_COUNT = tl.constexpr(_TISSUE_PARAMETERS)

# How many tissue parameters the free pool alone accounts for. The bound
# pool's sit past them, and are written by the bound-pool kernels alone; a
# single-pool run leaves their planes at the zero they were cleared to.
_FREE_POOL_COUNT = tl.constexpr(_TISSUE_PARAMETERS - len(_BOUND_POOL_INPUTS))
_BOUND_ROW = tl.constexpr(_BOUND_POOL_INPUTS[0])

# The gradient plane holds a row of voxels per tissue parameter, except that
# the transmit pair holds one per shim. Both sit ahead of everything that
# widens, so a plane's row is its parameter index shifted by the rows the pair
# added ahead of it.
_B1_ROW = tl.constexpr(_TRANSMIT_INPUTS[0])
_B1_PHASE_ROW = tl.constexpr(_TRANSMIT_INPUTS[1])

# Where the two event directions the real-subspace adjoint follows sit among
# the differentiable inputs. Named rather than counted, so a tissue parameter
# added ahead of them moves them instead of silently renaming a neighbour.
_DURATION_SEED = _FLOAT_NAMES.index("duration")
_FLIP_SEED = _FLOAT_NAMES.index("flip")


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
def _two_pool_step(r1_free, r1_bound, exchange, bound, dt, attenuation):
    """The two-pool longitudinal operator over one interval, and its recovery.

    ``expm((K - diag(R1)) t)`` in the exact 2x2 closed form. Its discriminant
    is a square plus a product of two non-negative rates, so the root is real
    and the branch a general exponential would need does not exist here.
    ``sinh(d)/d`` is taken by series near the origin, where the root has no
    derivative of its own.

    The equilibrium each pool relaxes toward is its own fraction, so the
    recovery is ``(I - E1) (1 - f, f)`` and needs no solve. Returned as
    ``(e11, e12, e21, e22, recovery_free, recovery_bound)``.
    """
    free = 1.0 - bound
    kab = exchange * bound
    kba = exchange * free
    l11 = (-kab - r1_free) * dt
    l12 = kba * dt
    l21 = kab * dt
    l22 = (-kba - r1_bound) * dt

    half_trace = 0.5 * (l11 + l22)
    half_gap = 0.5 * (l11 - l22)
    square = half_gap * half_gap + l12 * l21
    # tau +/- d are the eigenvalues, both non-positive for a decaying system,
    # so their exponentials are bounded by one. Formed that way rather than as
    # e^tau cosh(d), which over a long interval is an underflow times an
    # overflow.
    root = tl.sqrt(tl.maximum(square, 0.0))
    upper = tl.exp(half_trace + root)
    lower = tl.exp(half_trace - root)
    cosine = 0.5 * (upper + lower)
    turning = square > 1e-12
    guarded = tl.where(turning, root, 1.0)
    scale = tl.where(
        turning,
        0.5 * (upper - lower) / guarded,
        tl.exp(half_trace) * (1.0 + square / 6.0 + square * square / 120.0),
    )
    e11 = attenuation * (cosine + scale * half_gap)
    e12 = attenuation * scale * l12
    e21 = attenuation * scale * l21
    e22 = attenuation * (cosine - scale * half_gap)
    return (
        e11,
        e12,
        e21,
        e22,
        free - (e11 * free + e12 * bound),
        bound - (e21 * free + e22 * bound),
    )


@triton.jit
def _lineshape_at(lineshape, offset_hz, bins: tl.constexpr, step):
    """How well the bound pool absorbs a pulse this far off its resonance.

    Cubic Hermite between the two knots bracketing the offset, taken in
    magnitude because the lineshape is even, and clamped at the far end. Each
    knot is two floats -- the value then its slope -- so the two a read needs
    are four contiguous ones.
    """
    last = bins - 1
    scaled = tl.minimum(tl.abs(offset_hz) / step, last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    base = lower.to(tl.int64) * 2
    near = tl.load(lineshape + base)
    near_slope = tl.load(lineshape + base + 1)
    far = tl.load(lineshape + base + 2)
    far_slope = tl.load(lineshape + base + 3)
    return (
        (2.0 * u3 - 3.0 * u2 + 1.0) * near
        + (u3 - 2.0 * u2 + u) * step * near_slope
        + (-2.0 * u3 + 3.0 * u2) * far
        + (u3 - u2) * step * far_slope
    )


@triton.jit
def _two_pool_step_jvp(
    r1_free,
    d_r1_free,
    r1_bound,
    d_r1_bound,
    exchange,
    d_exchange,
    bound,
    d_bound,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
):
    """The two-pool operator and its directional derivative.

    The same closed form :func:`_two_pool_step` evaluates, carried alongside a
    tangent. Returned as the six outputs then their six tangents.
    """
    free = 1.0 - bound
    d_free = -d_bound
    kab = exchange * bound
    d_kab = d_exchange * bound + exchange * d_bound
    kba = exchange * free
    d_kba = d_exchange * free + exchange * d_free
    l11 = (-kab - r1_free) * dt
    d_l11 = (-d_kab - d_r1_free) * dt + (-kab - r1_free) * d_dt
    l12 = kba * dt
    d_l12 = d_kba * dt + kba * d_dt
    l21 = kab * dt
    d_l21 = d_kab * dt + kab * d_dt
    l22 = (-kba - r1_bound) * dt
    d_l22 = (-d_kba - d_r1_bound) * dt + (-kba - r1_bound) * d_dt

    half_trace = 0.5 * (l11 + l22)
    d_half_trace = 0.5 * (d_l11 + d_l22)
    half_gap = 0.5 * (l11 - l22)
    d_half_gap = 0.5 * (d_l11 - d_l22)
    square = half_gap * half_gap + l12 * l21
    d_square = 2.0 * half_gap * d_half_gap + d_l12 * l21 + l12 * d_l21

    turning = square > 1e-12
    root = tl.sqrt(tl.maximum(square, 0.0))
    guarded = tl.where(turning, root, 1.0)
    d_root = tl.where(turning, 0.5 * d_square / guarded, 0.0)

    upper = tl.exp(half_trace + root)
    d_upper = upper * (d_half_trace + d_root)
    lower = tl.exp(half_trace - root)
    d_lower = lower * (d_half_trace - d_root)
    cosine = 0.5 * (upper + lower)
    d_cosine = 0.5 * (d_upper + d_lower)

    # sinh(d)/d by series where the root has no derivative of its own.
    plain = tl.exp(half_trace)
    d_plain = plain * d_half_trace
    poly = 1.0 + square / 6.0 + square * square / 120.0
    d_poly = d_square / 6.0 + square * d_square / 60.0
    scale = tl.where(turning, 0.5 * (upper - lower) / guarded, plain * poly)
    d_scale = tl.where(
        turning,
        0.5 * (d_upper - d_lower) / guarded
        - 0.5 * (upper - lower) * d_root / (guarded * guarded),
        d_plain * poly + plain * d_poly,
    )

    e11 = attenuation * (cosine + scale * half_gap)
    d_e11 = d_attenuation * (cosine + scale * half_gap) + attenuation * (
        d_cosine + d_scale * half_gap + scale * d_half_gap
    )
    e12 = attenuation * scale * l12
    d_e12 = (
        d_attenuation * scale * l12
        + attenuation * d_scale * l12
        + attenuation * scale * d_l12
    )
    e21 = attenuation * scale * l21
    d_e21 = (
        d_attenuation * scale * l21
        + attenuation * d_scale * l21
        + attenuation * scale * d_l21
    )
    e22 = attenuation * (cosine - scale * half_gap)
    d_e22 = d_attenuation * (cosine - scale * half_gap) + attenuation * (
        d_cosine - d_scale * half_gap - scale * d_half_gap
    )

    grow_free = free - (e11 * free + e12 * bound)
    d_grow_free = d_free - (
        d_e11 * free + e11 * d_free + d_e12 * bound + e12 * d_bound
    )
    grow_bound = bound - (e21 * free + e22 * bound)
    d_grow_bound = d_bound - (
        d_e21 * free + e21 * d_free + d_e22 * bound + e22 * d_bound
    )
    return (
        e11, e12, e21, e22, grow_free, grow_bound,
        d_e11, d_e12, d_e21, d_e22, d_grow_free, d_grow_bound,
    )


@triton.jit
def _lineshape_at_slope(lineshape, offset_hz, bins: tl.constexpr, step):
    """The lineshape and its derivative in the *signed* offset.

    The table covers the magnitude, so the slope changes sign with the offset;
    past the last knot the read is constant and the slope is zero.
    """
    last = bins - 1
    magnitude = tl.abs(offset_hz) / step
    scaled = tl.minimum(magnitude, last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    base = lower.to(tl.int64) * 2
    near = tl.load(lineshape + base)
    near_slope = tl.load(lineshape + base + 1)
    far = tl.load(lineshape + base + 2)
    far_slope = tl.load(lineshape + base + 3)
    value = (
        (2.0 * u3 - 3.0 * u2 + 1.0) * near
        + (u3 - 2.0 * u2 + u) * step * near_slope
        + (-2.0 * u3 + 3.0 * u2) * far
        + (u3 - u2) * step * far_slope
    )
    direction = tl.where(offset_hz < 0.0, -1.0, 1.0)
    slope = direction * (
        (6.0 * u2 - 6.0 * u) * near / step
        + (3.0 * u2 - 4.0 * u + 1.0) * near_slope
        + (-6.0 * u2 + 6.0 * u) * far / step
        + (3.0 * u2 - 2.0 * u) * far_slope
    )
    return value, tl.where(magnitude > last, 0.0, slope)


@triton.jit
def _lineshape_at_curve(lineshape, offset_hz, bins: tl.constexpr, step):
    """The lineshape, its slope and its curvature, from the same cubic.

    The table covers the magnitude, so the slope changes sign with the offset
    and the curvature does not: an even function's second derivative is even.
    """
    last = bins - 1
    magnitude = tl.abs(offset_hz) / step
    scaled = tl.minimum(magnitude, last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    base = lower.to(tl.int64) * 2
    near = tl.load(lineshape + base)
    near_slope = tl.load(lineshape + base + 1)
    far = tl.load(lineshape + base + 2)
    far_slope = tl.load(lineshape + base + 3)
    value = (
        (2.0 * u3 - 3.0 * u2 + 1.0) * near
        + (u3 - 2.0 * u2 + u) * step * near_slope
        + (-2.0 * u3 + 3.0 * u2) * far
        + (u3 - u2) * step * far_slope
    )
    direction = tl.where(offset_hz < 0.0, -1.0, 1.0)
    slope = direction * (
        (6.0 * u2 - 6.0 * u) * near / step
        + (3.0 * u2 - 4.0 * u + 1.0) * near_slope
        + (-6.0 * u2 + 6.0 * u) * far / step
        + (3.0 * u2 - 2.0 * u) * far_slope
    )
    curve = (
        (12.0 * u - 6.0) * near / (step * step)
        + (6.0 * u - 4.0) * near_slope / step
        + (-12.0 * u + 6.0) * far / (step * step)
        + (6.0 * u - 2.0) * far_slope / step
    )
    beyond = magnitude > last
    return (
        value,
        tl.where(beyond, 0.0, slope),
        tl.where(beyond, 0.0, curve),
    )


@triton.jit
def _two_pool_step_adjoint_jvp(
    r1_free,
    d_r1_free,
    r1_bound,
    d_r1_bound,
    exchange,
    d_exchange,
    bound,
    d_bound,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
    bar_e11,
    d_bar_e11,
    bar_e12,
    d_bar_e12,
    bar_e21,
    d_bar_e21,
    bar_e22,
    d_bar_e22,
    bar_free,
    d_bar_free,
    bar_bound,
    d_bar_bound,
):
    """The reverse sweep of :func:`_two_pool_step`, carried on a direction.

    Recomputes the forward rather than carrying it across the event: the whole
    thing is a handful of transcendentals once per interval, against a state
    loop that runs per dephasing order.

    Where the discriminant is small the value is still formed from the two
    eigenvalues -- a sum, which loses nothing -- but the derivative is taken
    from the series, because ``d cosh(d)/d(d^2)`` reached through
    ``(e^{t+d} - e^{t-d})/2d`` is a cancellation divided by a small number.

    Returned as the gradients w.r.t. ``(r1_free, r1_bound, exchange, bound,
    dt, attenuation)`` then their six tangents.
    """
    free = 1.0 - bound
    d_free = -d_bound
    kab = exchange * bound
    d_kab = d_exchange * bound + exchange * d_bound
    kba = exchange * free
    d_kba = d_exchange * free + exchange * d_free
    l11 = (-kab - r1_free) * dt
    d_l11 = (-d_kab - d_r1_free) * dt + (-kab - r1_free) * d_dt
    l12 = kba * dt
    d_l12 = d_kba * dt + kba * d_dt
    l21 = kab * dt
    d_l21 = d_kab * dt + kab * d_dt
    l22 = (-kba - r1_bound) * dt
    d_l22 = (-d_kba - d_r1_bound) * dt + (-kba - r1_bound) * d_dt

    half_trace = 0.5 * (l11 + l22)
    d_half_trace = 0.5 * (d_l11 + d_l22)
    half_gap = 0.5 * (l11 - l22)
    d_half_gap = 0.5 * (d_l11 - d_l22)
    square = half_gap * half_gap + l12 * l21
    d_square = 2.0 * half_gap * d_half_gap + d_l12 * l21 + l12 * d_l21

    turning = square > 1e-12
    root = tl.sqrt(tl.maximum(square, 0.0))
    guarded = tl.where(turning, root, 1.0)
    d_root = tl.where(turning, 0.5 * d_square / guarded, 0.0)
    upper = tl.exp(half_trace + root)
    d_upper = upper * (d_half_trace + d_root)
    lower = tl.exp(half_trace - root)
    d_lower = lower * (d_half_trace - d_root)
    cosine = 0.5 * (upper + lower)
    d_cosine = 0.5 * (d_upper + d_lower)
    plain = tl.exp(half_trace)
    d_plain = plain * d_half_trace
    poly = 1.0 + square / 6.0 + square * square / 120.0
    d_poly = d_square / 6.0 + square * d_square / 60.0
    scale = tl.where(turning, 0.5 * (upper - lower) / guarded, plain * poly)
    d_scale = tl.where(
        turning,
        0.5 * (d_upper - d_lower) / guarded
        - 0.5 * (upper - lower) * d_root / (guarded * guarded),
        d_plain * poly + plain * d_poly,
    )

    bare11 = cosine + scale * half_gap
    d_bare11 = d_cosine + d_scale * half_gap + scale * d_half_gap
    bare12 = scale * l12
    d_bare12 = d_scale * l12 + scale * d_l12
    bare21 = scale * l21
    d_bare21 = d_scale * l21 + scale * d_l21
    bare22 = cosine - scale * half_gap
    d_bare22 = d_cosine - d_scale * half_gap - scale * d_half_gap

    # The recovery reaches the operator's four entries and the two fractions.
    carried11 = bar_e11 - bar_free * free
    d_carried11 = d_bar_e11 - (d_bar_free * free + bar_free * d_free)
    carried12 = bar_e12 - bar_free * bound
    d_carried12 = d_bar_e12 - (d_bar_free * bound + bar_free * d_bound)
    carried21 = bar_e21 - bar_bound * free
    d_carried21 = d_bar_e21 - (d_bar_bound * free + bar_bound * d_free)
    carried22 = bar_e22 - bar_bound * bound
    d_carried22 = d_bar_e22 - (d_bar_bound * bound + bar_bound * d_bound)

    e11 = attenuation * bare11
    d_e11 = d_attenuation * bare11 + attenuation * d_bare11
    e12 = attenuation * bare12
    d_e12 = d_attenuation * bare12 + attenuation * d_bare12
    e21 = attenuation * bare21
    d_e21 = d_attenuation * bare21 + attenuation * d_bare21
    e22 = attenuation * bare22
    d_e22 = d_attenuation * bare22 + attenuation * d_bare22

    back_free = bar_free * (1.0 - e11) - bar_bound * e21
    d_back_free = (
        d_bar_free * (1.0 - e11) - bar_free * d_e11
        - (d_bar_bound * e21 + bar_bound * d_e21)
    )
    back_bound = bar_bound * (1.0 - e22) - bar_free * e12
    d_back_bound = (
        d_bar_bound * (1.0 - e22) - bar_bound * d_e22
        - (d_bar_free * e12 + bar_free * d_e12)
    )

    back_attenuation = (
        carried11 * bare11 + carried12 * bare12
        + carried21 * bare21 + carried22 * bare22
    )
    d_back_attenuation = (
        d_carried11 * bare11 + carried11 * d_bare11
        + d_carried12 * bare12 + carried12 * d_bare12
        + d_carried21 * bare21 + carried21 * d_bare21
        + d_carried22 * bare22 + carried22 * d_bare22
    )

    scaled11 = attenuation * carried11
    d_scaled11 = d_attenuation * carried11 + attenuation * d_carried11
    scaled12 = attenuation * carried12
    d_scaled12 = d_attenuation * carried12 + attenuation * d_carried12
    scaled21 = attenuation * carried21
    d_scaled21 = d_attenuation * carried21 + attenuation * d_carried21
    scaled22 = attenuation * carried22
    d_scaled22 = d_attenuation * carried22 + attenuation * d_carried22

    bar_cosine = scaled11 + scaled22
    d_bar_cosine = d_scaled11 + d_scaled22
    gap = scaled11 - scaled22
    d_gap = d_scaled11 - d_scaled22
    bar_scale = gap * half_gap + scaled12 * l12 + scaled21 * l21
    d_bar_scale = (
        d_gap * half_gap + gap * d_half_gap
        + d_scaled12 * l12 + scaled12 * d_l12
        + d_scaled21 * l21 + scaled21 * d_l21
    )
    bar_half_gap = scale * gap
    d_bar_half_gap = d_scale * gap + scale * d_gap
    bar_l12 = scale * scaled12
    d_bar_l12 = d_scale * scaled12 + scale * d_scaled12
    bar_l21 = scale * scaled21
    d_bar_l21 = d_scale * scaled21 + scale * d_scaled21

    series_trace = bar_cosine * cosine + bar_scale * scale
    d_series_trace = (
        d_bar_cosine * cosine + bar_cosine * d_cosine
        + d_bar_scale * scale + bar_scale * d_scale
    )
    cosine_poly = 0.5 + square / 12.0
    d_cosine_poly = d_square / 12.0
    scale_poly = 1.0 / 6.0 + square / 60.0
    d_scale_poly = d_square / 60.0
    series_square = plain * (bar_cosine * cosine_poly + bar_scale * scale_poly)
    d_series_square = d_plain * (
        bar_cosine * cosine_poly + bar_scale * scale_poly
    ) + plain * (
        d_bar_cosine * cosine_poly + bar_cosine * d_cosine_poly
        + d_bar_scale * scale_poly + bar_scale * d_scale_poly
    )

    inverse = tl.where(turning, 1.0 / guarded, 0.0)
    d_inverse = tl.where(turning, -d_root / (guarded * guarded), 0.0)
    bar_upper = 0.5 * (bar_cosine + bar_scale * inverse)
    d_bar_upper = 0.5 * (
        d_bar_cosine + d_bar_scale * inverse + bar_scale * d_inverse
    )
    bar_lower = 0.5 * (bar_cosine - bar_scale * inverse)
    d_bar_lower = 0.5 * (
        d_bar_cosine - d_bar_scale * inverse - bar_scale * d_inverse
    )
    root_trace = bar_upper * upper + bar_lower * lower
    d_root_trace = (
        d_bar_upper * upper + bar_upper * d_upper
        + d_bar_lower * lower + bar_lower * d_lower
    )
    bar_root = (
        bar_upper * upper - bar_lower * lower - bar_scale * scale * inverse
    )
    d_bar_root = (
        d_bar_upper * upper + bar_upper * d_upper
        - d_bar_lower * lower - bar_lower * d_lower
        - (
            d_bar_scale * scale * inverse
            + bar_scale * d_scale * inverse
            + bar_scale * scale * d_inverse
        )
    )
    root_square = 0.5 * bar_root * inverse
    d_root_square = 0.5 * (d_bar_root * inverse + bar_root * d_inverse)

    bar_half_trace = tl.where(turning, root_trace, series_trace)
    d_bar_half_trace = tl.where(turning, d_root_trace, d_series_trace)
    bar_square = tl.where(turning, root_square, series_square)
    d_bar_square = tl.where(turning, d_root_square, d_series_square)

    bar_half_gap += 2.0 * bar_square * half_gap
    d_bar_half_gap += 2.0 * (d_bar_square * half_gap + bar_square * d_half_gap)
    bar_l12 += bar_square * l21
    d_bar_l12 += d_bar_square * l21 + bar_square * d_l21
    bar_l21 += bar_square * l12
    d_bar_l21 += d_bar_square * l12 + bar_square * d_l12

    bar_l11 = 0.5 * (bar_half_trace + bar_half_gap)
    d_bar_l11 = 0.5 * (d_bar_half_trace + d_bar_half_gap)
    bar_l22 = 0.5 * (bar_half_trace - bar_half_gap)
    d_bar_l22 = 0.5 * (d_bar_half_trace - d_bar_half_gap)

    bar_kab = (bar_l21 - bar_l11) * dt
    d_bar_kab = (d_bar_l21 - d_bar_l11) * dt + (bar_l21 - bar_l11) * d_dt
    bar_kba = (bar_l12 - bar_l22) * dt
    d_bar_kba = (d_bar_l12 - d_bar_l22) * dt + (bar_l12 - bar_l22) * d_dt
    back_dt = (
        bar_l11 * (-kab - r1_free) + bar_l12 * kba + bar_l21 * kab
        + bar_l22 * (-kba - r1_bound)
    )
    d_back_dt = (
        d_bar_l11 * (-kab - r1_free) + bar_l11 * (-d_kab - d_r1_free)
        + d_bar_l12 * kba + bar_l12 * d_kba
        + d_bar_l21 * kab + bar_l21 * d_kab
        + d_bar_l22 * (-kba - r1_bound) + bar_l22 * (-d_kba - d_r1_bound)
    )

    back_bound += bar_kab * exchange
    d_back_bound += d_bar_kab * exchange + bar_kab * d_exchange
    back_free += bar_kba * exchange
    d_back_free += d_bar_kba * exchange + bar_kba * d_exchange

    return (
        -bar_l11 * dt,
        -bar_l22 * dt,
        bar_kab * bound + bar_kba * free,
        back_bound - back_free,
        back_dt,
        back_attenuation,
        -(d_bar_l11 * dt + bar_l11 * d_dt),
        -(d_bar_l22 * dt + bar_l22 * d_dt),
        d_bar_kab * bound + bar_kab * d_bound
        + d_bar_kba * free + bar_kba * d_free,
        d_back_bound - d_back_free,
        d_back_dt,
        d_back_attenuation,
    )


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
def _table_row(profile_index, event, location, locations: tl.constexpr):
    """Which row of the stacked tables this pulse reads.

    Its own shape's block of ``locations`` rows, then the voxel's place along
    the slice.
    """
    return tl.load(profile_index + event).to(tl.int64) * locations + location


@triton.jit
def _profile_pair(profile, row, theta, bins: tl.constexpr, step):
    """The Cayley-Klein pair the transition table holds at this flip angle.

    Cubic Hermite between the two knots bracketing ``theta``, clamped at both
    ends: a cubic run off its grid leaves the unit circle. Each knot is eight
    floats -- the pair then its slope, real before imaginary -- so the two a
    read needs are sixteen contiguous ones.
    """
    last = bins - 1
    scaled = tl.minimum(tl.maximum(theta / step, 0.0), last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * step
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * step

    base = (row * bins + lower.to(tl.int64)) * 8
    pair = ()
    for component in tl.static_range(4):
        near = tl.load(profile + base + component)
        near_slope = tl.load(profile + base + 4 + component)
        far = tl.load(profile + base + 8 + component)
        far_slope = tl.load(profile + base + 12 + component)
        pair = pair + (h00 * near + h10 * near_slope + h01 * far + h11 * far_slope,)
    return pair


@triton.jit
def _profile_pair_slope(profile, row, theta, bins: tl.constexpr, step):
    """The pair and its derivative in the flip angle, from the same cubic.

    The derivative of a Hermite segment is another polynomial in the same four
    knot values, so reading both costs one extra combination rather than a
    second table. Returned interleaved: each component's value then its slope,
    in the order ``a`` real, ``a`` imaginary, ``b`` real, ``b`` imaginary.
    """
    last = bins - 1
    scaled = tl.minimum(tl.maximum(theta / step, 0.0), last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * step
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * step
    # d/dtheta is d/du over the knot spacing.
    g00 = (6.0 * u2 - 6.0 * u) / step
    g10 = 3.0 * u2 - 4.0 * u + 1.0
    g01 = (6.0 * u - 6.0 * u2) / step
    g11 = 3.0 * u2 - 2.0 * u

    base = (row * bins + lower.to(tl.int64)) * 8
    read = ()
    for component in tl.static_range(4):
        near = tl.load(profile + base + component)
        near_slope = tl.load(profile + base + 4 + component)
        far = tl.load(profile + base + 8 + component)
        far_slope = tl.load(profile + base + 12 + component)
        read = read + (
            h00 * near + h10 * near_slope + h01 * far + h11 * far_slope,
            g00 * near + g10 * near_slope + g01 * far + g11 * far_slope,
        )
    return read


@triton.jit
def _profile_pair_curve(profile, row, theta, bins: tl.constexpr, step):
    """The pair, its slope and its curvature in the flip angle.

    The second-order pass differentiates the read twice, and a Hermite segment
    is a cubic, so all three come from the same four knot values. Returned in
    threes per component: value, slope, curvature.
    """
    last = bins - 1
    scaled = tl.minimum(tl.maximum(theta / step, 0.0), last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * step
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * step
    g00 = (6.0 * u2 - 6.0 * u) / step
    g10 = 3.0 * u2 - 4.0 * u + 1.0
    g01 = (6.0 * u - 6.0 * u2) / step
    g11 = 3.0 * u2 - 2.0 * u
    c00 = (12.0 * u - 6.0) / (step * step)
    c10 = (6.0 * u - 4.0) / step
    c01 = (6.0 - 12.0 * u) / (step * step)
    c11 = (6.0 * u - 2.0) / step

    base = (row * bins + lower.to(tl.int64)) * 8
    read = ()
    for component in tl.static_range(4):
        near = tl.load(profile + base + component)
        near_slope = tl.load(profile + base + 4 + component)
        far = tl.load(profile + base + 8 + component)
        far_slope = tl.load(profile + base + 12 + component)
        read = read + (
            h00 * near + h10 * near_slope + h01 * far + h11 * far_slope,
            g00 * near + g10 * near_slope + g01 * far + g11 * far_slope,
            c00 * near + c10 * near_slope + c01 * far + c11 * far_slope,
        )
    return read


@triton.jit
def _dual_conj(z):
    """A dual complex number's conjugate, both halves."""
    return (z[0], -z[1], z[2], -z[3])


@triton.jit
def _dual_weigh(z, factor):
    """A dual complex number scaled by a real constant."""
    return (factor * z[0], factor * z[1], factor * z[2], factor * z[3])


@triton.jit
def _dual_sum(first, second, third, fourth):
    """Four dual complex numbers added."""
    return (
        first[0] + second[0] + third[0] + fourth[0],
        first[1] + second[1] + third[1] + fourth[1],
        first[2] + second[2] + third[2] + fourth[2],
        first[3] + second[3] + third[3] + fourth[3],
    )


@triton.jit
def _dual_product(x, y):
    """Two dual complex numbers multiplied."""
    return _dual_mul(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3])


@triton.jit
def _profiled_pair_dual(
    profile, row, alpha_value, alpha_tangent, phi_value, phi_tangent,
    bins: tl.constexpr, step,
):
    """The pair a shaped pulse turns through, and its slope, as duals.

    The flip angle carries the tangent into the table, so the pair's tangent is
    the stored slope and the slope's own tangent is the segment's curvature.
    The RF phase turns the axis once the pair is out, and so reaches ``b``.
    """
    read = _profile_pair_curve(profile, row, alpha_value, bins, step)
    a = (read[0], read[3], read[1] * alpha_tangent, read[4] * alpha_tangent)
    slope_a = (read[1], read[4], read[2] * alpha_tangent, read[5] * alpha_tangent)
    b = (read[6], read[9], read[7] * alpha_tangent, read[10] * alpha_tangent)
    slope_b = (read[7], read[10], read[8] * alpha_tangent, read[11] * alpha_tangent)
    turn = _dual_polar(-phi_value, -phi_tangent)
    return a, _dual_product(b, turn), slope_a, _dual_product(slope_b, turn)


@triton.jit
def _spinor_adjoint_dual(a, b, sp, sm, rz, pb, mb, zb):
    """The spinor rotation's adjoint, on dual numbers.

    Returns the cotangent on the Cayley-Klein pair and the three state
    cotangents sent back through the conjugate transpose. Every entry of the
    matrix is a product of two factors drawn from the pair and its conjugate,
    so the pair's two Wirtinger halves are linear in the outer product of the
    seed with the state the rotation acted on -- which is why this is a closed
    form rather than a differentiated matrix.
    """
    t00, t01, t02, t10, t11, t12, t20, t21, t22 = _spinor_coefficients(
        a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3]
    )

    conj_pb = _dual_conj(pb)
    conj_mb = _dual_conj(mb)
    conj_zb = _dual_conj(zb)
    m00 = _dual_product(conj_pb, sp)
    m01 = _dual_product(conj_pb, sm)
    m02 = _dual_product(conj_pb, rz)
    m10 = _dual_product(conj_mb, sp)
    m11 = _dual_product(conj_mb, sm)
    m12 = _dual_product(conj_mb, rz)
    m20 = _dual_product(conj_zb, sp)
    m21 = _dual_product(conj_zb, sm)
    m22 = _dual_product(conj_zb, rz)

    conj_a = _dual_conj(a)
    conj_b = _dual_conj(b)
    holding_conj_a = _dual_sum(
        _dual_weigh(_dual_product(a, m11), 2.0),
        _dual_weigh(_dual_product(b, m12), -2.0),
        _dual_product(conj_b, m21),
        _dual_product(conj_a, m22),
    )
    holding_a = _dual_sum(
        _dual_weigh(_dual_product(conj_a, m00), 2.0),
        _dual_weigh(_dual_product(conj_b, m02), -2.0),
        _dual_product(b, m20),
        _dual_product(a, m22),
    )
    holding_conj_b = _dual_sum(
        _dual_weigh(_dual_product(b, m10), -2.0),
        _dual_weigh(_dual_product(a, m12), -2.0),
        _dual_product(conj_a, m20),
        _dual_weigh(_dual_product(conj_b, m22), -1.0),
    )
    holding_b = _dual_sum(
        _dual_weigh(_dual_product(conj_b, m01), -2.0),
        _dual_weigh(_dual_product(conj_a, m02), -2.0),
        _dual_product(a, m21),
        _dual_weigh(_dual_product(b, m22), -1.0),
    )
    zero = _dual_weigh(m00, 0.0)
    grad_a = _dual_sum(_dual_conj(holding_conj_a), holding_a, zero, zero)
    grad_b = _dual_sum(_dual_conj(holding_conj_b), holding_b, zero, zero)

    next_pb = _dual_sum(
        _dual_product(_dual_conj(t00), pb),
        _dual_product(_dual_conj(t10), mb),
        _dual_product(_dual_conj(t20), zb),
        zero,
    )
    next_mb = _dual_sum(
        _dual_product(_dual_conj(t01), pb),
        _dual_product(_dual_conj(t11), mb),
        _dual_product(_dual_conj(t21), zb),
        zero,
    )
    next_zb = _dual_sum(
        _dual_product(_dual_conj(t02), pb),
        _dual_product(_dual_conj(t12), mb),
        _dual_product(_dual_conj(t22), zb),
        zero,
    )
    return grad_a, grad_b, next_pb, next_mb, next_zb


@triton.jit
def _spinor_coefficients(ar, ai, br, bi, dar, dai, dbr, dbi):
    """The rotation's nine coefficients and their tangents.

    Every entry is a product of two factors drawn from the pair and its
    conjugate, so five products carry all nine: ``a^2``, ``b^2``, ``a b``,
    ``a conj(b)`` and the norm difference.
    """
    aa_r = ar * ar - ai * ai
    aa_i = 2.0 * ar * ai
    daa_r = 2.0 * (ar * dar - ai * dai)
    daa_i = 2.0 * (dar * ai + ar * dai)

    bb_r = br * br - bi * bi
    bb_i = 2.0 * br * bi
    dbb_r = 2.0 * (br * dbr - bi * dbi)
    dbb_i = 2.0 * (dbr * bi + br * dbi)

    ab_r = ar * br - ai * bi
    ab_i = ar * bi + ai * br
    dab_r = dar * br + ar * dbr - dai * bi - ai * dbi
    dab_i = dar * bi + ar * dbi + dai * br + ai * dbr

    cross_r = ar * br + ai * bi
    cross_i = ar * bi - ai * br
    dcross_r = dar * br + ar * dbr + dai * bi + ai * dbi
    dcross_i = dar * bi + ar * dbi - dai * br - ai * dbr

    t22 = ar * ar + ai * ai - br * br - bi * bi
    dt22 = 2.0 * (ar * dar + ai * dai - br * dbr - bi * dbi)

    return (
        (aa_r, -aa_i, daa_r, -daa_i),
        (-bb_r, bb_i, -dbb_r, dbb_i),
        (-2.0 * ab_r, 2.0 * ab_i, -2.0 * dab_r, 2.0 * dab_i),
        (-bb_r, -bb_i, -dbb_r, -dbb_i),
        (aa_r, aa_i, daa_r, daa_i),
        (-2.0 * ab_r, -2.0 * ab_i, -2.0 * dab_r, -2.0 * dab_i),
        (cross_r, cross_i, dcross_r, dcross_i),
        (cross_r, -cross_i, dcross_r, -dcross_i),
        (t22, 0.0 * t22, dt22, 0.0 * dt22),
    )


@triton.jit
def _rotate_spinor_dual(
    ar, ai, br, bi,
    dar, dai, dbr, dbi,
    fp_r, fp_i, fm_r, fm_i, z_r, z_i,
    dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i,
):
    """The spinor rotation carrying a forward-mode tangent.

    Both the states and the pair naming the rotation move, so the tangent is
    ``T dx + dT x``.
    """
    t00, t01, t02, t10, t11, t12, t20, t21, t22 = _spinor_coefficients(
        ar, ai, br, bi, dar, dai, dbr, dbi
    )

    out_pr, out_pi = _dual_row(t00, t01, t02, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    out_mr, out_mi = _dual_row(t10, t11, t12, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    out_zr, out_zi = _dual_row(t20, t21, t22, fp_r, fp_i, fm_r, fm_i, z_r, z_i)

    dpr, dpi = _dual_row(
        t00, t01, t02, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i
    )
    dmr, dmi = _dual_row(
        t10, t11, t12, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i
    )
    dzr, dzi = _dual_row(
        t20, t21, t22, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i
    )
    tpr, tpi = _tangent_row(t00, t01, t02, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    tmr, tmi = _tangent_row(t10, t11, t12, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    tzr, tzi = _tangent_row(t20, t21, t22, fp_r, fp_i, fm_r, fm_i, z_r, z_i)

    return (
        out_pr, out_pi, out_mr, out_mi, out_zr, out_zi,
        dpr + tpr, dpi + tpi, dmr + tmr, dmi + tmi, dzr + tzr, dzi + tzi,
    )


@triton.jit
def _dual_row(first, second, third, fp_r, fp_i, fm_r, fm_i, z_r, z_i):
    """One row of the rotation applied to the states, values only."""
    real = (
        first[0] * fp_r - first[1] * fp_i
        + second[0] * fm_r - second[1] * fm_i
        + third[0] * z_r - third[1] * z_i
    )
    imag = (
        first[0] * fp_i + first[1] * fp_r
        + second[0] * fm_i + second[1] * fm_r
        + third[0] * z_i + third[1] * z_r
    )
    return real, imag


@triton.jit
def _tangent_row(first, second, third, fp_r, fp_i, fm_r, fm_i, z_r, z_i):
    """The same row built from the coefficients' tangents instead."""
    real = (
        first[2] * fp_r - first[3] * fp_i
        + second[2] * fm_r - second[3] * fm_i
        + third[2] * z_r - third[3] * z_i
    )
    imag = (
        first[2] * fp_i + first[3] * fp_r
        + second[2] * fm_i + second[3] * fm_r
        + third[2] * z_i + third[3] * z_r
    )
    return real, imag


@triton.jit
def _rotate_spinor(ar, ai, br, bi, fp_r, fp_i, fm_r, fm_i, z_r, z_i):
    """The rotation named by its Cayley-Klein pair, applied to the states.

        T = [ conj(a)^2   -conj(b)^2   -2 conj(a b) ]
            [ -b^2         a^2         -2 a b       ]
            [ conj(a) b    a conj(b)   |a|^2-|b|^2  ]
    """
    aa_r = ar * ar - ai * ai
    aa_i = 2.0 * ar * ai
    bb_r = br * br - bi * bi
    bb_i = 2.0 * br * bi
    ab_r = ar * br - ai * bi
    ab_i = ar * bi + ai * br

    t00_r, t00_i = aa_r, -aa_i
    t01_r, t01_i = -bb_r, bb_i
    t02_r, t02_i = -2.0 * ab_r, 2.0 * ab_i
    t10_r, t10_i = -bb_r, -bb_i
    t11_r, t11_i = aa_r, aa_i
    t12_r, t12_i = -2.0 * ab_r, -2.0 * ab_i
    cross_r = ar * br + ai * bi
    cross_i = ar * bi - ai * br
    t20_r, t20_i = cross_r, cross_i
    t21_r, t21_i = cross_r, -cross_i
    t22 = ar * ar + ai * ai - br * br - bi * bi

    out_pr = (
        t00_r * fp_r - t00_i * fp_i
        + t01_r * fm_r - t01_i * fm_i
        + t02_r * z_r - t02_i * z_i
    )
    out_pi = (
        t00_r * fp_i + t00_i * fp_r
        + t01_r * fm_i + t01_i * fm_r
        + t02_r * z_i + t02_i * z_r
    )
    out_mr = (
        t10_r * fp_r - t10_i * fp_i
        + t11_r * fm_r - t11_i * fm_i
        + t12_r * z_r - t12_i * z_i
    )
    out_mi = (
        t10_r * fp_i + t10_i * fp_r
        + t11_r * fm_i + t11_i * fm_r
        + t12_r * z_i + t12_i * z_r
    )
    out_zr = (
        t20_r * fp_r - t20_i * fp_i
        + t21_r * fm_r - t21_i * fm_i
        + t22 * z_r
    )
    out_zi = (
        t20_r * fp_i + t20_i * fp_r
        + t21_r * fm_i + t21_i * fm_r
        + t22 * z_i
    )
    return out_pr, out_pi, out_mr, out_mi, out_zr, out_zi


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
    bound_fraction,
    exchange_rate,
    t1_bound,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    profile,
    profile_index,
    lineshape,
    dot_t1,
    dot_t2,
    dot_m0,
    dot_b1,
    dot_b1_phase,
    dot_b0,
    dot_inversion_efficiency,
    dot_diffusion,
    dot_velocity,
    dot_bound_fraction,
    dot_exchange_rate,
    dot_t1_bound,
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
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    shims: tl.constexpr,
    locations: tl.constexpr,
    profile_bins: tl.constexpr,
    lineshape_bins: tl.constexpr,
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
    # Voxels are spread over the slice voxel-major, so a voxel's place along
    # the slice is its index modulo the profile's width. One pulse shape holds
    # that many consecutive rows, and the event says which shape it drives.
    location = atom % locations
    local = problem - problem_base
    scratch_offset = local * state_count
    sp_r = scratch_pr + scratch_offset
    sp_i = scratch_pi + scratch_offset
    sm_r = scratch_mr + scratch_offset
    sm_i = scratch_mi + scratch_offset
    # The bound pool rides along as a fourth plane: it enters an event as its
    # own vector and the RF operator scales it, so the reverse sweep cannot
    # replay it from the free pool's.
    record_stride = (4 if lineshape_bins > 0 else 3) * state_count
    trajectory = local * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count
    bound_plane = 3 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    pvr = empty
    pvi = empty
    ptr = empty
    pti = empty
    mvr = empty
    mvi = empty
    mtr = empty
    mti = empty
    bvr = empty
    bvi = empty
    btr = empty
    bti = empty
    if lineshape_bins > 0:
        atom_bound = tl.load(bound_fraction + atom, mask=active_atom, other=0.0)
        d_boundf = tl.load(
            dot_bound_fraction + atom, mask=active_atom, other=0.0
        )
        atom_exchange = tl.load(exchange_rate + atom, mask=active_atom, other=0.0)
        d_exchange = tl.load(
            dot_exchange_rate + atom, mask=active_atom, other=0.0
        )
        atom_t1b = tl.load(t1_bound + atom, mask=active_atom, other=1.0)
        d_t1b = tl.load(dot_t1_bound + atom, mask=active_atom, other=0.0)
        r1b_value = 1000.0 / atom_t1b
        r1b_tangent = -1000.0 * d_t1b / (atom_t1b * atom_t1b)
        zvr = empty + tl.where(state == 0, 1.0 - atom_bound, 0.0)
        ztr = empty + tl.where(state == 0, -d_boundf, 0.0)
        bvr = empty + tl.where(state == 0, atom_bound, 0.0)
        btr = empty + tl.where(state == 0, d_boundf, 0.0)
    else:
        zvr = empty + tl.where(state == 0, 1.0, 0.0)
        ztr = empty
    zvi = empty
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
        if lineshape_bins > 0:
            tl.store(trajectory_vr + slot + bound_plane, bvr, mask=state_mask)
            tl.store(trajectory_vi + slot + bound_plane, bvi, mask=state_mask)
            tl.store(trajectory_tr + slot + bound_plane, btr, mask=state_mask)
            tl.store(trajectory_ti + slot + bound_plane, bti, mask=state_mask)

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
        if lineshape_bins > 0:
            # The exchange operator is a property of the interval, not of a
            # dephasing order, so it is formed once and the per-order damping
            # and turn multiply it.
            (
                pe11, pe12, pe21, pe22, prec_f, prec_b,
                de11, de12, de21, de22, drec_f, drec_b,
            ) = _two_pool_step_jvp(
                r1_value, r1_tangent, r1b_value, r1b_tangent,
                atom_exchange, d_exchange, atom_bound, d_boundf,
                dt_value, dt_tangent, wout_value, wout_tangent,
            )
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti)
            free_part = _dual_scale(pe11, de11, zvr, zvi, ztr, zti)
            cross_in = _dual_scale(pe12, de12, bvr, bvi, btr, bti)
            cross_out = _dual_scale(pe21, de21, zvr, zvi, ztr, zti)
            bound_part = _dual_scale(pe22, de22, bvr, bvi, btr, bti)
            zvr, zvi, ztr, zti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3],
                free_part[0] + cross_in[0], free_part[1] + cross_in[1],
                free_part[2] + cross_in[2], free_part[3] + cross_in[3],
            )
            bvr, bvi, btr, bti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3],
                cross_out[0] + bound_part[0], cross_out[1] + bound_part[1],
                cross_out[2] + bound_part[2], cross_out[3] + bound_part[3],
            )
            zvr += tl.where(state == 0, prec_f, 0.0)
            ztr += tl.where(state == 0, drec_f, 0.0)
            bvr += tl.where(state == 0, prec_b, 0.0)
            btr += tl.where(state == 0, drec_b, 0.0)
        else:
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
        if lineshape_bins > 0:
            # The bound pool absorbs the power the pulse deposits, so it reads
            # the bare flip the transmit field gives the voxel -- not the
            # slice-shaped rotation the free pool takes from the table.
            offset_value = tl.load(rf_frequency + event) - atom_b0
            shape_value, shape_slope = _lineshape_at_slope(
                lineshape, offset_value, lineshape_bins, lineshape_step
            )
            shape_tangent = shape_slope * -d_b0
            event_saturation = tl.load(saturation + event)
            power_value = event_saturation * alpha_value * alpha_value
            power_tangent = (
                event_saturation * 2.0 * alpha_value * alpha_tangent
            )
            absorbed_value = tl.exp(power_value * shape_value)
            absorbed_tangent = absorbed_value * (
                power_tangent * shape_value + power_value * shape_tangent
            )
            sat_b = _dual_scale(
                absorbed_value, absorbed_tangent, bvr, bvi, btr, bti
            )
            saturating = is_rf & ~is_inversion
            bvr = tl.where(saturating, sat_b[0], bvr)
            bvi = tl.where(saturating, sat_b[1], bvi)
            btr = tl.where(saturating, sat_b[2], btr)
            bti = tl.where(saturating, sat_b[3], bti)
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

        turned_pvr = a0[0] + a1[0] + a2[0]
        turned_pvi = a0[1] + a1[1] + a2[1]
        turned_ptr = a0[2] + a1[2] + a2[2]
        turned_pti = a0[3] + a1[3] + a2[3]
        turned_mvr = b0_[0] + b1_[0] + b2[0]
        turned_mvi = b0_[1] + b1_[1] + b2[1]
        turned_mtr = b0_[2] + b1_[2] + b2[2]
        turned_mti = b0_[3] + b1_[3] + b2[3]
        turned_zvr = c0[0] + c1[0] + c2[0]
        turned_zvi = c0[1] + c1[1] + c2[1]
        turned_ztr = c0[2] + c1[2] + c2[2]
        turned_zti = c0[3] + c1[3] + c2[3]
        if profile_bins > 0:
            shaped_a, shaped_b, _, _ = _profiled_pair_dual(
                profile,
                _table_row(profile_index, event, location, locations),
                alpha_value, alpha_tangent, phi_value, phi_tangent,
                profile_bins, profile_step,
            )
            (
                turned_pvr, turned_pvi, turned_mvr, turned_mvi, turned_zvr,
                turned_zvi, turned_ptr, turned_pti, turned_mtr, turned_mti,
                turned_ztr, turned_zti,
            ) = _rotate_spinor_dual(
                shaped_a[0], shaped_a[1], shaped_b[0], shaped_b[1],
                shaped_a[2], shaped_a[3], shaped_b[2], shaped_b[3],
                pvr, pvi, mvr, mvi, zvr, zvi,
                ptr, pti, mtr, mti, ztr, zti,
            )

        rotate = is_rf & ~is_inversion
        pvr = tl.where(rotate, turned_pvr, pvr)
        pvi = tl.where(rotate, turned_pvi, pvi)
        ptr = tl.where(rotate, turned_ptr, ptr)
        pti = tl.where(rotate, turned_pti, pti)
        mvr = tl.where(rotate, turned_mvr, mvr)
        mvi = tl.where(rotate, turned_mvi, mvi)
        mtr = tl.where(rotate, turned_mtr, mtr)
        mti = tl.where(rotate, turned_mti, mti)
        zvr = tl.where(rotate, turned_zvr, zvr)
        zvi = tl.where(rotate, turned_zvi, zvi)
        ztr = tl.where(rotate, turned_ztr, ztr)
        zti = tl.where(rotate, turned_zti, zti)

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
    bbvr = empty
    bbvi = empty
    bbtr = empty
    bbti = empty
    zero = tl.zeros((problems, 1), tl.float32)
    g_boundv = zero
    g_boundt = zero
    g_exchv = zero
    g_excht = zero
    g_t1bv = zero
    g_t1bt = zero
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
        xbvr = empty
        xbvi = empty
        xbtr = empty
        xbti = empty
        if lineshape_bins > 0:
            xbvr = tl.load(
                trajectory_vr + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbvi = tl.load(
                trajectory_vi + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbtr = tl.load(
                trajectory_tr + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbti = tl.load(
                trajectory_ti + slot + bound_plane, mask=state_mask, other=0.0
            )

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
        rbvr = empty
        rbvi = empty
        rbtr = empty
        rbti = empty
        if lineshape_bins > 0:
            (
                pe11, pe12, pe21, pe22, prec_f, prec_b,
                de11, de12, de21, de22, drec_f, drec_b,
            ) = _two_pool_step_jvp(
                r1_value, r1_tangent, r1b_value, r1b_tangent,
                atom_exchange, d_exchange, atom_bound, d_boundf,
                dt_value, dt_tangent, wout_value, wout_tangent,
            )
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti)
            free_part = _dual_scale(pe11, de11, xzvr, xzvi, xztr, xzti)
            cross_in = _dual_scale(pe12, de12, xbvr, xbvi, xbtr, xbti)
            cross_out = _dual_scale(pe21, de21, xzvr, xzvi, xztr, xzti)
            bound_part = _dual_scale(pe22, de22, xbvr, xbvi, xbtr, xbti)
            mixed_free = (
                free_part[0] + cross_in[0], free_part[1] + cross_in[1],
                free_part[2] + cross_in[2], free_part[3] + cross_in[3],
            )
            mixed_bound = (
                cross_out[0] + bound_part[0], cross_out[1] + bound_part[1],
                cross_out[2] + bound_part[2], cross_out[3] + bound_part[3],
            )
            rzvr, rzvi, rztr, rzti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_free
            )
            rbvr, rbvi, rbtr, rbti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_bound
            )
            rzvr += tl.where(state == 0, prec_f, 0.0)
            rztr += tl.where(state == 0, drec_f, 0.0)
            rbvr += tl.where(state == 0, prec_b, 0.0)
            rbtr += tl.where(state == 0, drec_b, 0.0)
        else:
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
        sat_alpha_v = zero
        sat_alpha_t = zero
        sat_b0_v = zero
        sat_b0_t = zero
        if lineshape_bins > 0:
            # The pulse scales every order of the bound pool by one real
            # number, so its cotangent is a single sum over the states it
            # multiplied. The lineshape's own slope is differentiated too,
            # which is what the curvature the reader returns is for.
            offset_value = tl.load(rf_frequency + event) - atom_b0
            shape_value, shape_slope, shape_curve = _lineshape_at_curve(
                lineshape, offset_value, lineshape_bins, lineshape_step
            )
            shape_tangent = shape_slope * -d_b0
            slope_tangent = shape_curve * -d_b0
            event_saturation = tl.load(saturation + event)
            power_value = event_saturation * alpha_value * alpha_value
            power_tangent = (
                event_saturation * 2.0 * alpha_value * alpha_tangent
            )
            absorbed_value = tl.exp(power_value * shape_value)
            absorbed_tangent = absorbed_value * (
                power_tangent * shape_value + power_value * shape_tangent
            )
            per_state_v, per_state_t = _dual_real_conj_mul(
                bbvr, bbvi, bbtr, bbti, rbvr, rbvi, rbtr, rbti
            )
            grad_absorbed_v = tl.sum(per_state_v, axis=1)[:, None]
            grad_absorbed_t = tl.sum(per_state_t, axis=1)[:, None]
            grad_exponent_v = grad_absorbed_v * absorbed_value
            grad_exponent_t = (
                grad_absorbed_t * absorbed_value
                + grad_absorbed_v * absorbed_tangent
            )
            twice = event_saturation * 2.0
            sat_alpha_v = grad_exponent_v * (twice * alpha_value * shape_value)
            sat_alpha_t = grad_exponent_t * (
                twice * alpha_value * shape_value
            ) + grad_exponent_v * twice * (
                alpha_tangent * shape_value + alpha_value * shape_tangent
            )
            # The lineshape is read at the pulse's offset from the voxel, so a
            # step in the voxel's own off-resonance moves the read the other
            # way.
            sat_b0_v = -grad_exponent_v * (power_value * shape_slope)
            sat_b0_t = -(
                grad_exponent_t * (power_value * shape_slope)
                + grad_exponent_v * (
                    power_tangent * shape_slope + power_value * slope_tangent
                )
            )
            damped = _dual_scale(
                absorbed_value, absorbed_tangent, bbvr, bbvi, bbtr, bbti
            )
            saturating = is_rf & ~is_inversion
            bbvr = tl.where(saturating, damped[0], bbvr)
            bbvi = tl.where(saturating, damped[1], bbvi)
            bbtr = tl.where(saturating, damped[2], bbtr)
            bbti = tl.where(saturating, damped[3], bbti)
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

        if profile_bins > 0:
            shaped_a, shaped_b, shaped_slope_a, shaped_slope_b = (
                _profiled_pair_dual(
                    profile,
                    _table_row(profile_index, event, location, locations),
                    alpha_value, alpha_tangent, phi_value, phi_tangent,
                    profile_bins, profile_step,
                )
            )
            grad_a, grad_b, shaped_pb, shaped_mb, shaped_zb = (
                _spinor_adjoint_dual(
                    shaped_a, shaped_b,
                    (spvr, spvi, sptr, spti),
                    (smvr, smvi, smtr, smti),
                    (rzvr, rzvi, rztr, rzti),
                    (pbvr, pbvi, pbtr, pbti),
                    (mbvr, mbvi, mbtr, mbti),
                    (zbvr, zbvi, zbtr, zbti),
                )
            )
            alpha_v, alpha_t = _dual_real_conj_mul(
                grad_a[0], grad_a[1], grad_a[2], grad_a[3],
                shaped_slope_a[0], shaped_slope_a[1],
                shaped_slope_a[2], shaped_slope_a[3],
            )
            part_v, part_t = _dual_real_conj_mul(
                grad_b[0], grad_b[1], grad_b[2], grad_b[3],
                shaped_slope_b[0], shaped_slope_b[1],
                shaped_slope_b[2], shaped_slope_b[3],
            )
            alpha_v += part_v
            alpha_t += part_t
            # d(b e^{-i phi})/dphi is -i times it, and nothing else moves.
            turn_r, turn_i, turn_tr, turn_ti = _dual_times_i(
                shaped_b[0], shaped_b[1], shaped_b[2], shaped_b[3]
            )
            phi_v, phi_t = _dual_real_conj_mul(
                grad_b[0], grad_b[1], grad_b[2], grad_b[3],
                -turn_r, -turn_i, -turn_tr, -turn_ti,
            )

        rotate = is_rf & ~is_inversion
        grad_alpha_v = tl.sum(tl.where(rotate, alpha_v, 0.0), axis=1)[:, None]
        grad_alpha_t = tl.sum(tl.where(rotate, alpha_t, 0.0), axis=1)[:, None]
        grad_phi_v = tl.sum(tl.where(rotate, phi_v, 0.0), axis=1)[:, None]
        grad_phi_t = tl.sum(tl.where(rotate, phi_t, 0.0), axis=1)[:, None]
        if lineshape_bins > 0:
            turning = tl.where(rotate, 1.0, 0.0)
            grad_alpha_v += sat_alpha_v * turning
            grad_alpha_t += sat_alpha_t * turning
            g_b0v += sat_b0_v * turning
            g_b0t += sat_b0_t * turning

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

        back_pb = (n0[0] + n1[0] + n2[0], n0[1] + n1[1] + n2[1],
                   n0[2] + n1[2] + n2[2], n0[3] + n1[3] + n2[3])
        back_mb = (q0[0] + q1[0] + q2[0], q0[1] + q1[1] + q2[1],
                   q0[2] + q1[2] + q2[2], q0[3] + q1[3] + q2[3])
        back_zb = (w0[0] + w1[0] + w2[0], w0[1] + w1[1] + w2[1],
                   w0[2] + w1[2] + w2[2], w0[3] + w1[3] + w2[3])
        if profile_bins > 0:
            back_pb = shaped_pb
            back_mb = shaped_mb
            back_zb = shaped_zb

        pbvr = tl.where(rotate, back_pb[0], pbvr)
        pbvi = tl.where(rotate, back_pb[1], pbvi)
        pbtr = tl.where(rotate, back_pb[2], pbtr)
        pbti = tl.where(rotate, back_pb[3], pbti)
        mbvr = tl.where(rotate, back_mb[0], mbvr)
        mbvi = tl.where(rotate, back_mb[1], mbvi)
        mbtr = tl.where(rotate, back_mb[2], mbtr)
        mbti = tl.where(rotate, back_mb[3], mbti)
        zbvr = tl.where(rotate, back_zb[0], zbvr)
        zbvi = tl.where(rotate, back_zb[1], zbvi)
        zbtr = tl.where(rotate, back_zb[2], zbtr)
        zbti = tl.where(rotate, back_zb[3], zbti)

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

        grad_e1_v = zero
        grad_e1_t = zero
        attenuation_v = zero
        attenuation_t = zero
        two_pool_dt_v = zero
        two_pool_dt_t = zero
        if lineshape_bins > 0:
            # The four entries of the exchange operator and the two recoveries,
            # summed over the orders that share them, then pushed back through
            # the closed form once for the whole interval.
            free_bar = (zbvr, zbvi, zbtr, zbti)
            bound_bar = (bbvr, bbvi, bbtr, bbti)
            spun_free = _dual_mul(spin[0], spin[1], spin[2], spin[3],
                                  xzvr, xzvi, xztr, xzti)
            spun_bound = _dual_mul(spin[0], spin[1], spin[2], spin[3],
                                   xbvr, xbvi, xbtr, xbti)
            e11_v, e11_t = _dual_real_conj_mul(*free_bar, *spun_free)
            e12_v, e12_t = _dual_real_conj_mul(*free_bar, *spun_bound)
            e21_v, e21_t = _dual_real_conj_mul(*bound_bar, *spun_free)
            e22_v, e22_t = _dual_real_conj_mul(*bound_bar, *spun_bound)
            bar_e11_v = tl.sum(e11_v, axis=1)[:, None]
            bar_e11_t = tl.sum(e11_t, axis=1)[:, None]
            bar_e12_v = tl.sum(e12_v, axis=1)[:, None]
            bar_e12_t = tl.sum(e12_t, axis=1)[:, None]
            bar_e21_v = tl.sum(e21_v, axis=1)[:, None]
            bar_e21_t = tl.sum(e21_t, axis=1)[:, None]
            bar_e22_v = tl.sum(e22_v, axis=1)[:, None]
            bar_e22_t = tl.sum(e22_t, axis=1)[:, None]
            rec_f_v = tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
            rec_f_t = tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None]
            rec_b_v = tl.sum(tl.where(state == 0, bbvr, 0.0), axis=1)[:, None]
            rec_b_t = tl.sum(tl.where(state == 0, bbtr, 0.0), axis=1)[:, None]
            (
                back_r1_v, back_r1b_v, back_exch_v, back_bound_v, back_dt_v,
                back_att_v,
                back_r1_t, back_r1b_t, back_exch_t, back_bound_t, back_dt_t,
                back_att_t,
            ) = _two_pool_step_adjoint_jvp(
                r1_value, r1_tangent, r1b_value, r1b_tangent,
                atom_exchange, d_exchange, atom_bound, d_boundf,
                dt_value, dt_tangent, wout_value, wout_tangent,
                bar_e11_v, bar_e11_t, bar_e12_v, bar_e12_t,
                bar_e21_v, bar_e21_t, bar_e22_v, bar_e22_t,
                rec_f_v, rec_f_t, rec_b_v, rec_b_t,
            )
            # r1 = 1000/t1, so a rate gradient reaches the time through the
            # square of it.
            slope1_v = -1000.0 / (atom_t1 * atom_t1)
            slope1_t = 2000.0 * d_t1 / (atom_t1 * atom_t1 * atom_t1)
            slope1b_v = -1000.0 / (atom_t1b * atom_t1b)
            slope1b_t = 2000.0 * d_t1b / (atom_t1b * atom_t1b * atom_t1b)
            g_t1v += back_r1_v * slope1_v
            g_t1t += back_r1_t * slope1_v + back_r1_v * slope1_t
            g_t1bv += back_r1b_v * slope1b_v
            g_t1bt += back_r1b_t * slope1b_v + back_r1b_v * slope1b_t
            g_exchv += back_exch_v
            g_excht += back_exch_t
            g_boundv += back_bound_v
            g_boundt += back_bound_t
            attenuation_v = back_att_v
            attenuation_t = back_att_t
            two_pool_dt_v = back_dt_v
            two_pool_dt_t = back_dt_t
            # Both pools take the same per-order damping and turn, so each
            # collects the cotangent of the mixture that reached it.
            damp_pair_v, damp_pair_t = _dual_real_conj_mul(*free_bar, *_dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_free
            ))
            other_v, other_t = _dual_real_conj_mul(*bound_bar, *_dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_bound
            ))
            long_damp_v = damp_pair_v + other_v
            long_damp_t = damp_pair_t + other_t
            spun_mix_free = _dual_times_i(*_dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_free
            ))
            spun_mix_bound = _dual_times_i(*_dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_bound
            ))
            zangle_v, zangle_t = _dual_real_conj_mul(*free_bar, *spun_mix_free)
            part_v, part_t = _dual_real_conj_mul(*bound_bar, *spun_mix_bound)
            zangle_v += part_v
            zangle_t += part_t
            back_z = _dual_mul(
                pe11 * spin[0], -(pe11 * spin[1]),
                de11 * spin[0] + pe11 * spin[2],
                -(de11 * spin[1] + pe11 * spin[3]),
                *free_bar,
            )
            cross_z = _dual_mul(
                pe21 * spin[0], -(pe21 * spin[1]),
                de21 * spin[0] + pe21 * spin[2],
                -(de21 * spin[1] + pe21 * spin[3]),
                *bound_bar,
            )
            back_b = _dual_mul(
                pe12 * spin[0], -(pe12 * spin[1]),
                de12 * spin[0] + pe12 * spin[2],
                -(de12 * spin[1] + pe12 * spin[3]),
                *free_bar,
            )
            cross_b = _dual_mul(
                pe22 * spin[0], -(pe22 * spin[1]),
                de22 * spin[0] + pe22 * spin[2],
                -(de22 * spin[1] + pe22 * spin[3]),
                *bound_bar,
            )
            next_zbvr = back_z[0] + cross_z[0]
            next_zbvi = back_z[1] + cross_z[1]
            next_zbtr = back_z[2] + cross_z[2]
            next_zbti = back_z[3] + cross_z[3]
            bbvr = back_b[0] + cross_b[0]
            bbvi = back_b[1] + cross_b[1]
            bbtr = back_b[2] + cross_b[2]
            bbti = back_b[3] + cross_b[3]
            zbvr = next_zbvr
            zbvi = next_zbvi
            zbtr = next_zbtr
            zbti = next_zbti
        else:
            spun = _dual_mul(szr, szi, sztr, szti, xzvr, xzvi, xztr, xzti)
            e1_v, e1_t = _dual_real_conj_mul(
                zbvr, zbvi, zbtr, zbti, spun[0], spun[1], spun[2], spun[3]
            )
            grad_e1_v = tl.sum(e1_v * damp_z, axis=1)[:, None]
            grad_e1_t = tl.sum(
                e1_v * damp_z_tangent + e1_t * damp_z, axis=1
            )[:, None]
            grad_e1_v -= tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
            grad_e1_t -= tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None]
            long_damp_v = e1_v * bare1_value * damp_z
            long_damp_t = (
                e1_t * bare1_value * damp_z
                + e1_v * bare1_tangent * damp_z
                + e1_v * bare1_value * damp_z_tangent
            )
            # The longitudinal states turn too, and by a whole order rather
            # than the transverse half-order more.
            zo = _dual_mul(lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti)
            zo = _dual_times_i(zo[0], zo[1], zo[2], zo[3])
            zangle_v, zangle_t = _dual_real_conj_mul(
                zbvr, zbvi, zbtr, zbti, zo[0], zo[1], zo[2], zo[3]
            )
            zbvr, zbvi, zbtr, zbti = _dual_mul(
                lvr, -lvi, ltr, -lti, zbvr, zbvi, zbtr, zbti
            )

        # The rate and the interval multiply every order's b-weight, so both
        # take a weighted sum rather than one scalar. Order zero carries no
        # longitudinal weight, which keeps recovery out of this.
        weighted_v = (
            long_damp_v * longitudinal_weight
            + cot2_v * bare2_value * damp_t * transverse_weight
        )
        weighted_t = long_damp_t * longitudinal_weight + (
            cot2_t * bare2_value * damp_t
            + cot2_v * bare2_tangent * damp_t
            + cot2_v * bare2_value * damp_t_tangent
        ) * transverse_weight
        spread_v = tl.sum(weighted_v, axis=1)[:, None]
        spread_t = tl.sum(weighted_t, axis=1)[:, None]
        g_diffv += -spread_v * dt_value
        g_difft += -(spread_v * dt_tangent + spread_t * dt_value)

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
        wash_v = -washing * (
            grad_e1_v * dry1_value + grad_e2_v * dry2_value + attenuation_v
        )
        wash_t = -washing * (
            grad_e1_v * dry1_tangent + grad_e1_t * dry1_value
            + grad_e2_v * dry2_tangent + grad_e2_t * dry2_value
            + attenuation_t
        )
        g_washv += wash_v * dt_value
        g_washt += wash_v * dt_tangent + wash_t * dt_value

        pbvr, pbvi, pbtr, pbti = _dual_mul(
            ovr, -ovi, otr, -oti, pbvr, pbvi, pbtr, pbti
        )
        mbvr, mbvi, mbtr, mbti = _dual_mul(
            ovr, ovi, otr, oti, mbvr, mbvi, mbtr, mbti
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
        duration_v += grad_angle_v * (turn * atom_b0) + two_pool_dt_v
        duration_t = -(grad_e1_v * decay1_tangent + grad_e1_t * decay1_value)
        duration_t -= grad_e2_v * decay2_tangent + grad_e2_t * decay2_value
        duration_t += grad_angle_v * (turn * d_b0) + grad_angle_t * (turn * atom_b0)
        duration_t += two_pool_dt_t
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
    if lineshape_bins > 0:
        # The fraction also sets where each pool starts, which the walk back
        # reaches last.
        g_boundv += tl.sum(
            tl.where(state == 0, bbvr - zbvr, 0.0), axis=1
        )[:, None]
        g_boundt += tl.sum(
            tl.where(state == 0, bbtr - zbtr, 0.0), axis=1
        )[:, None]
        tl.atomic_add(
            grad_tissue_value + (_BOUND_ROW + 2 * (shims - 1)) * atom_count + atom,
            g_boundv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + (_BOUND_ROW + 2 * (shims - 1)) * atom_count + atom,
            g_boundt,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_value + ((_BOUND_ROW + 2 * (shims - 1)) + 1) * atom_count + atom,
            g_exchv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + ((_BOUND_ROW + 2 * (shims - 1)) + 1) * atom_count + atom,
            g_excht,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_value + ((_BOUND_ROW + 2 * (shims - 1)) + 2) * atom_count + atom,
            g_t1bv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + ((_BOUND_ROW + 2 * (shims - 1)) + 2) * atom_count + atom,
            g_t1bt,
            mask=active_atom,
        )
    values = (
        g_t1v, g_t2v, g_m0v, g_b1v, g_b1pv, g_b0v, g_invv, g_diffv, velocity_v,
    )
    tangents = (
        g_t1t, g_t2t, g_m0t, g_b1t, g_b1pt, g_b0t, g_invt, g_difft, velocity_t,
    )
    for parameter in tl.static_range(_FREE_POOL_COUNT):
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
    bound_fraction,
    exchange_rate,
    t1_bound,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    profile,
    profile_index,
    lineshape,
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
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    shims: tl.constexpr,
    locations: tl.constexpr,
    profile_bins: tl.constexpr,
    lineshape_bins: tl.constexpr,
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
    # Voxels are spread over the slice voxel-major, so a voxel's place along
    # the slice is its index modulo the profile's width. One pulse shape holds
    # that many consecutive rows, and the event says which shape it drives.
    location = atom % locations
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
    # The bound pool holds its own share of the equilibrium, and carries
    # longitudinal states alone: nothing dephases it, so it reaches the higher
    # orders only through exchange with the free pool's.
    atom_bound = 0.0
    atom_exchange = 0.0
    atom_r1_bound = 0.0
    if lineshape_bins > 0:
        atom_bound = tl.load(bound_fraction + atom, mask=active_atom, other=0.0)
        atom_exchange = tl.load(exchange_rate + atom, mask=active_atom, other=0.0)
        atom_r1_bound = 1000.0 / tl.load(
            t1_bound + atom, mask=active_atom, other=1.0
        )
    longitudinal_real = empty + tl.where(state == 0, 1.0 - atom_bound, 0.0)
    longitudinal_imag = empty
    bound_real = empty + tl.where(state == 0, atom_bound + 0.0, 0.0)
    bound_imag = empty

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
        if lineshape_bins > 0:
            # The exchange operator is a property of the interval, not of a
            # dephasing order, so it is formed once and the per-order damping
            # multiplies it. Both pools take that damping and the flow phase:
            # their order-n states describe one dephasing configuration, and the
            # bound pool has no diffusion coefficient of its own to damp by.
            e11, e12, e21, e22, grow_free, grow_bound = _two_pool_step(
                1000.0 / atom_t1, atom_r1_bound, atom_exchange, atom_bound, dt, wout
            )
            free_real = e11 * longitudinal_real + e12 * bound_real
            free_imag = e11 * longitudinal_imag + e12 * bound_imag
            held_real = e21 * longitudinal_real + e22 * bound_real
            held_imag = e21 * longitudinal_imag + e22 * bound_imag
            longitudinal_real = damp_z * (
                free_real * turn_cos - free_imag * turn_sin
            )
            longitudinal_imag = damp_z * (
                free_real * turn_sin + free_imag * turn_cos
            )
            bound_real = damp_z * (held_real * turn_cos - held_imag * turn_sin)
            bound_imag = damp_z * (held_real * turn_sin + held_imag * turn_cos)
            longitudinal_real += tl.where(state == 0, grow_free, 0.0)
            bound_real += tl.where(state == 0, grow_bound, 0.0)
        else:
            old_real = longitudinal_real
            longitudinal_real = e1 * (
                old_real * turn_cos - longitudinal_imag * turn_sin
            )
            longitudinal_imag = e1 * (
                old_real * turn_sin + longitudinal_imag * turn_cos
            )
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
        if profile_bins > 0:
            # The table is built at zero RF phase, which turns the rotation
            # axis and so reaches ``b`` alone.
            pair = _profile_pair(
                profile, _table_row(profile_index, event, location, locations),
                alpha, profile_bins, profile_step,
            )
            turn_r = tl.cos(phi)
            turn_i = -tl.sin(phi)
            spun_br = pair[2] * turn_r - pair[3] * turn_i
            spun_bi = pair[2] * turn_i + pair[3] * turn_r
            (
                shaped_pr, shaped_pi, shaped_mr, shaped_mi, shaped_zr, shaped_zi
            ) = _rotate_spinor(
                pair[0], pair[1], spun_br, spun_bi,
                fplus_real, fplus_imag, fminus_real, fminus_imag,
                longitudinal_real, longitudinal_imag,
            )
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

        if profile_bins > 0:
            rotated_pr = shaped_pr
            rotated_pi = shaped_pi
            rotated_mr = shaped_mr
            rotated_mi = shaped_mi
            rotated_zr = shaped_zr
            rotated_zi = shaped_zi

        rotate = is_rf & ~is_inversion
        if lineshape_bins > 0:
            # The bound pool absorbs the power the pulse deposits, so it reads
            # the bare flip the transmit field gives the voxel -- not the
            # slice-shaped rotation the free pool takes from the table.
            offset = tl.load(rf_frequency + event) - atom_b0
            absorbed = tl.exp(
                tl.load(saturation + event)
                * alpha
                * alpha
                * _lineshape_at(lineshape, offset, lineshape_bins, lineshape_step)
            )
            bound_real = tl.where(rotate, absorbed * bound_real, bound_real)
            bound_imag = tl.where(rotate, absorbed * bound_imag, bound_imag)
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
    # The bound pool's three. A bound pool is outside the real subspace this
    # kernel stands for, so the dispatch never sends one here; the pointers
    # hold the ABI's shape.
    bound_fraction,
    exchange_rate,
    t1_bound,
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
    tangent_bound_fraction,
    tangent_exchange_rate,
    tangent_t1_bound,
    tangent_duration,
    tangent_flip,
    tangent_phase,
    saturation,
    rf_frequency,
    profile,
    profile_index,
    lineshape,
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
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    shims: tl.constexpr,
    locations: tl.constexpr,
    profile_bins: tl.constexpr,
    lineshape_bins: tl.constexpr,
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
    # Voxels are spread over the slice voxel-major, so a voxel's place along
    # the slice is its index modulo the profile's width. One pulse shape holds
    # that many consecutive rows, and the event says which shape it drives.
    location = atom % locations
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
    # Equilibrium is split between the pools, so a direction along the bound
    # fraction moves magnetization from one to the other before a single event
    # has run.
    atom_bound = 0.0
    d_bound = 0.0
    atom_exchange = 0.0
    d_exchange = 0.0
    atom_r1_bound = 0.0
    d_r1_bound = 0.0
    if lineshape_bins > 0:
        atom_bound = tl.load(bound_fraction + atom, mask=active_atom, other=0.0)
        d_bound = tl.load(
            tangent_bound_fraction + atom, mask=active_atom, other=0.0
        )
        atom_exchange = tl.load(exchange_rate + atom, mask=active_atom, other=0.0)
        d_exchange = tl.load(
            tangent_exchange_rate + atom, mask=active_atom, other=0.0
        )
        held_t1 = tl.load(t1_bound + atom, mask=active_atom, other=1.0)
        atom_r1_bound = 1000.0 / held_t1
        d_r1_bound = -1000.0 * tl.load(
            tangent_t1_bound + atom, mask=active_atom, other=0.0
        ) / (held_t1 * held_t1)
    zr = empty + tl.where(state == 0, 1.0 - atom_bound, 0.0)
    zi = empty
    br = empty + tl.where(state == 0, atom_bound + 0.0, 0.0)
    bi = empty
    dfpr = empty
    dfpi = empty
    dfmr = empty
    dfmi = empty
    dzr = empty + tl.where(state == 0, -d_bound, 0.0)
    dzi = empty
    dbr = empty + tl.where(state == 0, d_bound + 0.0, 0.0)
    dbi = empty

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
        if lineshape_bins > 0:
            # The exchange operator belongs to the interval, not to a dephasing
            # order, so it is formed once and carries its own tangent; the
            # per-order damping multiplies both pools, whose order-n states
            # describe one dephasing configuration.
            (
                e11, e12, e21, e22, grow_free, grow_bound,
                d_e11, d_e12, d_e21, d_e22, d_grow_free, d_grow_bound,
            ) = _two_pool_step_jvp(
                r1,
                -1000.0 * dt1 / (atom_t1 * atom_t1),
                atom_r1_bound,
                d_r1_bound,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_bound,
                event_dt,
                ddt,
                wout,
                dwout,
            )
            old_br = br
            old_bi = bi
            old_dbr = dbr
            old_dbi = dbi
            spun_hr = old_br * turn_cos - old_bi * turn_sin
            spun_hi = old_br * turn_sin + old_bi * turn_cos
            dspun_hr = (
                old_dbr * turn_cos
                + old_br * dturn_cos
                - old_dbi * turn_sin
                - old_bi * dturn_sin
            )
            dspun_hi = (
                old_dbr * turn_sin
                + old_br * dturn_sin
                + old_dbi * turn_cos
                + old_bi * dturn_cos
            )
            free_r = e11 * spun_zr + e12 * spun_hr
            free_i = e11 * spun_zi + e12 * spun_hi
            held_r = e21 * spun_zr + e22 * spun_hr
            held_i = e21 * spun_zi + e22 * spun_hi
            d_free_r = (
                d_e11 * spun_zr + e11 * dspun_zr + d_e12 * spun_hr + e12 * dspun_hr
            )
            d_free_i = (
                d_e11 * spun_zi + e11 * dspun_zi + d_e12 * spun_hi + e12 * dspun_hi
            )
            d_held_r = (
                d_e21 * spun_zr + e21 * dspun_zr + d_e22 * spun_hr + e22 * dspun_hr
            )
            d_held_i = (
                d_e21 * spun_zi + e21 * dspun_zi + d_e22 * spun_hi + e22 * dspun_hi
            )
            zr = damp_z * free_r + tl.where(state == 0, grow_free, 0.0)
            zi = damp_z * free_i
            dzr = (
                ddamp_z * free_r
                + damp_z * d_free_r
                + tl.where(state == 0, d_grow_free, 0.0)
            )
            dzi = ddamp_z * free_i + damp_z * d_free_i
            br = damp_z * held_r + tl.where(state == 0, grow_bound, 0.0)
            bi = damp_z * held_i
            dbr = (
                ddamp_z * held_r
                + damp_z * d_held_r
                + tl.where(state == 0, d_grow_bound, 0.0)
            )
            dbi = ddamp_z * held_i + damp_z * d_held_i
        else:
            dzr = (
                dspun_zr * e1
                + spun_zr * de1
                + tl.where(state == 0, drecovery, 0.0)
            )
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
        if lineshape_bins > 0:
            # The bound pool absorbs the power the pulse deposits, so it reads
            # the bare flip the transmit field gives the voxel. The offset
            # reaches it through the voxel's own off-resonance, which is where
            # the lineshape's slope enters a forward direction.
            offset = tl.load(rf_frequency + event) - atom_b0
            shape, shape_slope = _lineshape_at_slope(
                lineshape, offset, lineshape_bins, lineshape_step
            )
            deposited = tl.load(saturation + event)
            absorbed = tl.exp(deposited * alpha * alpha * shape)
            d_exponent = deposited * (
                2.0 * alpha * dalpha * shape
                - alpha * alpha * shape_slope * db0
            )
            saturating = is_rf & ~is_inversion
            dbr = tl.where(saturating, absorbed * (dbr + br * d_exponent), dbr)
            dbi = tl.where(saturating, absorbed * (dbi + bi * d_exponent), dbi)
            br = tl.where(saturating, absorbed * br, br)
            bi = tl.where(saturating, absorbed * bi, bi)
        if profile_bins > 0:
            read = _profile_pair_slope(
                profile, _table_row(profile_index, event, location, locations),
                alpha, profile_bins, profile_step,
            )
            # The flip angle carries the tangent into the table; the RF phase
            # turns the axis after the pair comes out, and so reaches ``b``.
            turn_r = tl.cos(phi)
            turn_i = -tl.sin(phi)
            spun_br = read[4] * turn_r - read[6] * turn_i
            spun_bi = read[4] * turn_i + read[6] * turn_r
            slope_br = read[5] * dalpha
            slope_bi = read[7] * dalpha
            (
                shaped_pr, shaped_pi, shaped_mr, shaped_mi, shaped_zr,
                shaped_zi, shaped_dpr, shaped_dpi, shaped_dmr, shaped_dmi,
                shaped_dzr, shaped_dzi,
            ) = _rotate_spinor_dual(
                read[0], read[2], spun_br, spun_bi,
                read[1] * dalpha, read[3] * dalpha,
                slope_br * turn_r - slope_bi * turn_i + dphi * spun_bi,
                slope_br * turn_i + slope_bi * turn_r - dphi * spun_br,
                fpr, fpi, fmr, fmi, zr, zi,
                dfpr, dfpi, dfmr, dfmi, dzr, dzi,
            )
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

        if profile_bins > 0:
            rotated_pr = shaped_pr
            rotated_pi = shaped_pi
            rotated_mr = shaped_mr
            rotated_mi = shaped_mi
            rotated_zr = shaped_zr
            rotated_zi = shaped_zi
            rotated_dpr = shaped_dpr
            rotated_dpi = shaped_dpi
            rotated_dmr = shaped_dmr
            rotated_dmi = shaped_dmi
            rotated_dzr = shaped_dzr
            rotated_dzi = shaped_dzi

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
    profile: Any = None,
    lineshape: Any = None,
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
        profile=profile,
        lineshape=lineshape,
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
    profile: Any = None,
    lineshape: Any = None,
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
        bound_fraction, exchange_rate, t1_bound,
    ) = tissue
    (
        duration, kind, flip, phase, action, output_index, shim_index,
        saturation, rf_frequency,
    ) = events
    train_count = _train_count(events)
    shims = _shim_count(tissue)
    block_states = triton.next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(total, block_states)
    grid = (triton.cdiv(total, problems),)
    planes = 2 if real_axis == 1 else 4
    if scratch is None:
        scratch = _scratch(planes, total, t1.device, state_count)
    # A kernel argument has to be a tensor even where the branch reading it is
    # compiled out, so an unprofiled launch passes one it already has.
    table = None if profile is None else profile.packed(t1.device)
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)

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
        bound_fraction,
        exchange_rate,
        t1_bound,
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
        t1 if table is None else table,
        kind if table_rows is None else table_rows,
        t1 if absorption is None else absorption,
        output_real,
        output_imag,
        *scratch,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        state_count=state_count,
        shims=shims,
        locations=1 if profile is None else profile.points,
        profile_bins=0 if profile is None else profile.bins,
        lineshape_bins=0 if lineshape is None else lineshape.bins,
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
    profile: Any = None,
    lineshape: Any = None,
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
        profile=profile,
        lineshape=lineshape,
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
    profile: Any = None,
    lineshape: Any = None,
) -> None:
    """Run one Jacobian-vector product into buffers the caller owns.

    See ``simulate_into`` for why the streaming path needs this.
    """
    (
        t1, t2, m0, b1, b1_phase, b0, inversion_efficiency, diffusion, velocity,
        _bound_fraction, _exchange_rate, _t1_bound,
    ) = tissue
    (
        duration, kind, flip, phase, action, output_index, shim_index,
        saturation, rf_frequency,
    ) = events
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

    table = None if profile is None else profile.packed(t1.device)
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
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
        saturation,
        rf_frequency,
        t1 if table is None else table,
        kind if table_rows is None else table_rows,
        t1 if absorption is None else absorption,
        output_real,
        output_imag,
        *scratch,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        state_count=state_count,
        shims=shims,
        locations=1 if profile is None else profile.points,
        profile_bins=0 if profile is None else profile.bins,
        lineshape_bins=0 if lineshape is None else lineshape.bins,
        block_states=block_states,
        problems=problems,
        num_warps=1,
    )


# How much device memory the recorded trajectory may hold at once. Beyond this
# the problems are run in waves, which the gradient buffers absorb because they
# accumulate rather than being written.
_TRAJECTORY_BUDGET_BYTES = 256 << 20


def _trajectory_wave(
    event_count: int, state_count: int, total: int, planes: int, blocks: int = 3
) -> int:
    """How many problems can record their trajectory in one launch."""
    per_problem = event_count * blocks * state_count * planes * 4
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
        pools: int = 1,
    ) -> None:
        (
            duration, kind, flip, phase, _action, _output_index, _shim,
            _saturation, _rf_frequency,
        ) = events
        device = kind.device
        train_count = _train_count(events)
        event_count = kind.numel()
        self.planes = 2 if real_axis == 1 else 4
        # A bound pool records a fourth block of states per event: the RF
        # operator scales it, so the reverse sweep cannot replay it from the
        # free pool's.
        self.blocks = 3 + (pools - 1)
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
            event_count, state_count, train_count * chunk, self.planes,
            self.blocks,
        )
        self.trajectory = [
            torch.empty(
                (self.wave, event_count * self.blocks * state_count),
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
    profile: Any = None,
    lineshape: Any = None,
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
        _bound_fraction, _exchange_rate, _t1_bound,
    ) = tissue
    (
        duration, kind, flip, phase, action, output_index, shim_index,
        _saturation, _rf_frequency,
    ) = events
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
    table = None if profile is None else profile.packed(t1.device)
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)

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
                tangents[_DURATION_SEED],
                tangents[_FLIP_SEED],
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
                t1 if table is None else table,
                kind if table_rows is None else table_rows,
                t1 if absorption is None else absorption,
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
                1.0 if profile is None else profile.step,
                1.0 if lineshape is None else lineshape.step,
                shims=_shim_count(tissue),
                locations=1 if profile is None else profile.points,
                profile_bins=0 if profile is None else profile.bins,
                lineshape_bins=0 if lineshape is None else lineshape.bins,
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
    profile: Any = None,
    lineshape: Any = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse through the state machine on CUDA.

    ``tangents`` follows the differentiable-input order -- every tissue
    property, then event duration, flip and phase -- and the two returned
    tuples, gradients with respect to the primal inputs then to the tangent
    inputs, follow it too.

    ``real_axis`` of 1 selects the real-subspace adjoint. That representation
    divides the RF phase out, so it leaves ``b1_phase``, ``b0`` and ``phase`` at
    zero and callers must not ask for those; the complex adjoint produces every
    one of them.

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
        pools=1 if lineshape is None else 2,
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
        profile=profile,
        lineshape=lineshape,
    )
    return tuple(
        (*voxels, *per_event)
        for voxels, per_event in zip(
            voxel_grads, buffers.event_gradients(), strict=True
        )
    )
