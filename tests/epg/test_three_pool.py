"""The three-pool longitudinal operator the kernels have to reproduce.

Free water beside both second pools at once -- a semisolid pool that carries
only ``Z``, and a chemically exchanging pool that carries ``F+`` and ``F-`` as
well. Only the longitudinal axis sees all three, so the transverse step stays
the 2x2 of :mod:`test_two_pool_transverse` and what is new here is

    E1 = expm((K - diag(R1)) t)

for a 3x3 ``K``. Free water exchanges with each second pool and the two second
pools do not exchange with each other, which is what makes each two-pool
system a limit of this one rather than a different model.

Four things about the closed form matter enough to pin before a kernel exists:

* **The eigenvalues are real and non-positive.** Detailed balance makes the
  generator similar to a symmetric matrix, and that matrix is negative
  semi-definite. So the cubic always has three real roots, the trigonometric
  solution applies, and no complex arithmetic is needed anywhere.
* **The equilibrium is still a fixed point**, so the affine recovery term is
  ``(I - E1)`` against the fractions and no 3x3 solve is needed.
* **Float32 does not reach the answer.** A 3x3 loses accuracy like the square
  of the interval where the 2x2 loses it like the interval, which is why this
  operator is the one thing in the state machine formed in double.
* **The two branches are both differentiable.** The series branch is a
  polynomial in the two invariants and forms no root at all; the eigenvalue
  branch keeps ``arccos`` off its vertical tangent, which is where two roots
  meet.
"""

from __future__ import annotations

import math

import pytest
import torch

from utils import epg

REAL = torch.float64
PI = math.pi

# How far the discriminant may sit from the origin before the series gives way
# to the eigenvalues, and how far ``arccos`` is kept from its vertical tangent.
# The guard is what stops a double root returning a NaN derivative; measured at
# 1e-16 it leaves the operator bit for bit what an unguarded one returns and
# costs 4e-11 in the gradient, four orders under the float32 that follows.
SPREAD_CUT = 1.0
ARG_GUARD = 1e-16
SINCH_CUT = 1e-4


def generator(t1a_ms, t1b_ms, t1c_ms, exchange_b, exchange_c, fraction_b, fraction_c):
    """``K - diag(R1)``, the generator of the three-pool longitudinal step.

    Free water is pool a, the chemically exchanging pool is b and the semisolid
    pool is c. Each second pool exchanges with the free water and not with the
    other, so a pool leaves at the rate scaled by the *other* pool's fraction
    and the exchange part conserves the total magnetization on its own.
    """
    free = 1.0 - fraction_b - fraction_c
    kab, kba = exchange_b * fraction_b, exchange_b * free
    kac, kca = exchange_c * fraction_c, exchange_c * free
    r1a, r1b, r1c = 1000.0 / t1a_ms, 1000.0 / t1b_ms, 1000.0 / t1c_ms
    zero = torch.zeros_like(free)
    return torch.stack(
        (
            torch.stack((-kab - kac - r1a, kba, kca), dim=-1),
            torch.stack((kab, -kba - r1b, zero), dim=-1),
            torch.stack((kac, zero, -kca - r1c), dim=-1),
        ),
        dim=-2,
    )


def _minor_sum(matrix):
    """The sum of a 3x3's principal 2x2 minors, its second invariant."""
    return (
        matrix[..., 0, 0] * matrix[..., 1, 1]
        - matrix[..., 0, 1] * matrix[..., 1, 0]
        + matrix[..., 0, 0] * matrix[..., 2, 2]
        - matrix[..., 0, 2] * matrix[..., 2, 0]
        + matrix[..., 1, 1] * matrix[..., 2, 2]
        - matrix[..., 1, 2] * matrix[..., 2, 1]
    )


def _determinant(matrix):
    return (
        matrix[..., 0, 0]
        * (
            matrix[..., 1, 1] * matrix[..., 2, 2]
            - matrix[..., 1, 2] * matrix[..., 2, 1]
        )
        - matrix[..., 0, 1]
        * (
            matrix[..., 1, 0] * matrix[..., 2, 2]
            - matrix[..., 1, 2] * matrix[..., 2, 0]
        )
        + matrix[..., 0, 2]
        * (
            matrix[..., 1, 0] * matrix[..., 2, 1]
            - matrix[..., 1, 1] * matrix[..., 2, 0]
        )
    )


def eigenvalues(scaled):
    """The three real roots, ascending, by the trigonometric solution.

    ``arccos`` is kept a hair off its endpoints. That is where two roots meet,
    and there the sorted roots have a vertical tangent the operator itself does
    not -- so the guard buys a finite derivative for a bias far under the
    float32 the answer is narrowed to.
    """
    third = scaled.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    shifted = scaled - third[..., None, None] * torch.eye(3, dtype=scaled.dtype)
    depressed = _minor_sum(shifted)
    constant = -_determinant(shifted)
    radius = torch.sqrt(torch.clamp(-depressed / 3.0, min=1e-30))
    argument = torch.clamp(
        -0.5 * constant / radius**3, -1.0 + ARG_GUARD, 1.0 - ARG_GUARD
    )
    angle = torch.acos(argument) / 3.0
    roots = torch.stack(
        [2.0 * radius * torch.cos(angle - 2.0 * PI * turn / 3.0) for turn in (0, 1, 2)],
        dim=-1,
    )
    return torch.sort(roots, dim=-1).values + third[..., None]


def _divided_difference(lower, upper):
    """``[a, b] exp``, by the series where the two points nearly meet.

    ``sinh(d)/d`` is even in the gap, so the series is a polynomial in its
    square and the branch is differentiable through the coalescence.
    """
    half = 0.5 * (upper - lower)
    near = half.abs() < SINCH_CUT
    gap = torch.where(near, torch.ones_like(half), upper - lower)
    series = sum(half ** (2 * term) / math.factorial(2 * term + 1) for term in range(6))
    return torch.where(
        near,
        torch.exp(0.5 * (lower + upper)) * series,
        (torch.exp(upper) - torch.exp(lower)) / gap,
    )


def three_pool_step(scaled, terms: int = 16):
    """``expm(scaled)`` for a three-pool longitudinal generator.

    Two branches, chosen by how far apart the eigenvalues are.

    Where they are close the exponential's own series is reduced modulo the
    characteristic polynomial ``x^3 + u x - v``, which needs a three-term
    recurrence in the two invariants and forms no root at all -- so it stays
    differentiable exactly where the roots stop being.

    Where they are far apart the interpolating polynomial is taken in Newton
    form at the three roots, whose divided differences are bounded because
    every root is non-positive and every exponential of one is therefore at
    most one.
    """
    eye = torch.eye(3, dtype=scaled.dtype).expand_as(scaled)
    third = scaled.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    shifted = scaled - third[..., None, None] * eye
    # The shifted generator's eigenvalues sum to zero, so the sum of their
    # squares is -2u and the largest is no bigger than the root of it.
    depressed = _minor_sum(shifted)
    constant = _determinant(shifted)
    close = (-2.0 * depressed) < SPREAD_CUT * SPREAD_CUT

    first = torch.ones_like(depressed)
    second = torch.zeros_like(depressed)
    third_power = torch.zeros_like(depressed)
    flat, linear, square = first.clone(), second.clone(), third_power.clone()
    factorial = 1.0
    for order in range(1, terms):
        first, second, third_power = (
            third_power * constant,
            first - third_power * depressed,
            second,
        )
        factorial *= order
        flat = flat + first / factorial
        linear = linear + second / factorial
        square = square + third_power / factorial
    series = torch.exp(third)[..., None, None] * (
        flat[..., None, None] * eye
        + linear[..., None, None] * shifted
        + square[..., None, None] * (shifted @ shifted)
    )

    roots = eigenvalues(scaled)
    low, middle, high = roots[..., 0], roots[..., 1], roots[..., 2]
    span = high - low
    second_difference = (
        _divided_difference(middle, high) - _divided_difference(low, middle)
    ) / torch.where(span.abs() < 1e-30, torch.ones_like(span), span)
    from_low = scaled - low[..., None, None] * eye
    from_middle = scaled - middle[..., None, None] * eye
    spectral = (
        torch.exp(low)[..., None, None] * eye
        + _divided_difference(low, middle)[..., None, None] * from_low
        + second_difference[..., None, None] * (from_low @ from_middle)
    )
    return torch.where(close[..., None, None], series, spectral)


def _draw(count, seed, *, identical=None, exchange=None):
    """Physically plausible three-pool parameters, in a spread wide enough to
    reach both branches and both degeneracies."""
    generator_ = torch.Generator().manual_seed(seed)

    def uniform(low, high):
        return low + (high - low) * torch.rand(count, generator=generator_, dtype=REAL)

    def log_uniform(low, high):
        return torch.exp(uniform(math.log(low), math.log(high)))

    t1a = uniform(200.0, 5000.0)
    t1b = t1a.clone() if identical else uniform(50.0, 3000.0)
    t1c = t1a.clone() if identical == "all" else uniform(0.5, 3000.0)
    rates = (
        (torch.zeros(count, dtype=REAL), torch.zeros(count, dtype=REAL))
        if exchange == "none"
        else (log_uniform(1e-3, 2e5), log_uniform(1e-3, 2e5))
    )
    fraction_b = uniform(0.0, 0.5)
    fraction_c = uniform(0.0, 0.3)
    dt = log_uniform(1e-6, 5.0)
    return (t1a, t1b, t1c, *rates, fraction_b, fraction_c), dt


def _scaled(values, dt):
    return generator(*values) * dt[:, None, None]


# --- what the generator is ---


def test_the_eigenvalues_never_leave_the_real_line() -> None:
    """Detailed balance makes the generator similar to a symmetric matrix.

    That is what lets the kernels solve a cubic with the trigonometric formula
    and skip complex arithmetic entirely; a single complex pair anywhere in
    the physical range would sink the whole design.
    """
    values, dt = _draw(40000, 0)
    spectrum = torch.linalg.eigvals(_scaled(values, dt))

    relative = spectrum.imag.abs() / spectrum.abs().clamp(min=1e-30)
    assert float(relative.max()) == 0.0


def test_the_eigenvalues_never_grow() -> None:
    """Every root is non-positive, so every exponential of one is at most one.

    The Newton form is formed from those exponentials directly rather than from
    a mean times a spread, which over a long interval would be an underflow
    times an overflow.
    """
    values, dt = _draw(40000, 1)
    spectrum = torch.linalg.eigvals(_scaled(values, dt))

    assert float(spectrum.real.max()) <= 0.0


def test_the_equilibrium_is_a_fixed_point() -> None:
    """``L m0 = -C``, which is what saves the kernels a 3x3 solve.

    The recovery over an interval is then ``(I - E1) m0`` rather than the
    textbook ``(E1 - I) L^-1 C``, exactly as it is for two pools.
    """
    values, dt = _draw(40000, 2)
    (t1a, t1b, t1c, _, _, fraction_b, fraction_c) = values
    equilibrium = torch.stack(
        (1.0 - fraction_b - fraction_c, fraction_b, fraction_c), -1
    )
    source = torch.stack(
        (
            (1.0 - fraction_b - fraction_c) * 1000.0 / t1a,
            fraction_b * 1000.0 / t1b,
            fraction_c * 1000.0 / t1c,
        ),
        dim=-1,
    )
    driven = (generator(*values) @ equilibrium[..., None])[..., 0]

    residual = (driven + source).abs().amax(-1)
    assert float(source.abs().max()) > 0.0
    assert float((residual / source.abs().amax(-1).clamp(min=1e-30)).max()) < 1e-9


# --- the closed form ---


def test_the_closed_form_agrees_with_a_general_matrix_exponential() -> None:
    """Against Pade with scaling and squaring, which shares no algebra with it.

    The sweep spans exchange from a millihertz to 200 kHz and intervals from a
    microsecond to five seconds, so both branches and the whole conditioning
    range are covered.
    """
    values, dt = _draw(120000, 3)
    scaled = _scaled(values, dt)
    expected = torch.matrix_exp(scaled)

    error = (three_pool_step(scaled) - expected).abs().amax((-2, -1))
    relative = error / expected.abs().amax((-2, -1)).clamp(min=1e-30)
    assert float(torch.quantile(relative, 0.99)) < 1e-7
    assert float(relative.max()) < 1e-5


def test_both_branches_agree_where_they_meet() -> None:
    """The series and the eigenvalues describe the same operator.

    Read either side of the cut on a sweep built to straddle it, so the branch
    is a choice between two right answers rather than a seam.
    """
    values, dt = _draw(4000, 4)
    scaled = _scaled(values, dt)
    third = scaled.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    shifted = scaled - third[..., None, None] * torch.eye(3, dtype=REAL)
    spread = torch.sqrt(torch.clamp(-2.0 * _minor_sum(shifted), min=0.0))
    # Rescale each draw so its spread lands within a factor of two of the cut.
    scaled = scaled * (SPREAD_CUT / spread.clamp(min=1e-12))[..., None, None]

    expected = torch.matrix_exp(scaled)
    relative = (three_pool_step(scaled) - expected).abs().amax(
        (-2, -1)
    ) / expected.abs().amax((-2, -1)).clamp(min=1e-30)
    assert float(relative.max()) < 1e-12


@pytest.mark.parametrize(
    "identical,exchange",
    [("two", "none"), ("all", "none"), ("two", "live"), ("all", "live")],
)
def test_coalescent_roots_keep_the_operator_exact(identical, exchange) -> None:
    """Pools that relax alike put two or three roots on top of each other.

    Nothing exotic -- two pools with the same T1 and no exchange between them
    is the first thing a user tries -- so the branch that handles it is read
    against the general algorithm rather than merely checked for finiteness.

    Held to the envelope the general sweep holds to, which is the statement
    worth making: a degeneracy costs nothing.
    """
    values, dt = _draw(4000, 5, identical=identical, exchange=exchange)
    scaled = _scaled(values, dt)
    expected = torch.matrix_exp(scaled)

    relative = (three_pool_step(scaled) - expected).abs().amax(
        (-2, -1)
    ) / expected.abs().amax((-2, -1)).clamp(min=1e-30)
    assert float(torch.quantile(relative, 0.99)) < 1e-7
    assert float(relative.max()) < 1e-5


# --- why the kernels form this one operator in double ---


def test_float32_does_not_reach_the_answer() -> None:
    """The measurement that decides where the state machine changes precision.

    A 2x2 loses accuracy like the interval; a 3x3 loses it like the square of
    the interval, because the answer's entries are order one while the terms
    that build them are order ``|L dt|^2``. Cancellation of that shape is not a
    tuning problem, so the operator is formed in double and narrowed once it
    is an operator.

    Read at ``|L dt|`` of a thousand -- a one second inversion delay at a
    hundred hertz of exchange, which is exactly what a three-pool model gets
    reached for.
    """
    values, dt = _draw(60000, 6)
    scaled = _scaled(values, dt)
    expected = torch.matrix_exp(scaled)
    scale = expected.abs().amax((-2, -1)).clamp(min=1e-30)
    reach = scaled.abs().sum(-1).amax(-1)

    def worst(below):
        band = reach < below
        assert int(band.sum()) > 100
        single = three_pool_step(scaled.to(torch.float32)).to(REAL)
        double = three_pool_step(scaled)
        return (
            float(((single - expected).abs().amax((-2, -1)) / scale)[band].max()),
            float(((double - expected).abs().amax((-2, -1)) / scale)[band].max()),
        )

    single_close, double_close = worst(10.0)
    single_far, double_far = worst(1000.0)

    # Near the identity float32 is fine and the choice would not matter.
    assert single_close < 1e-4
    # Over a preparation interval it is not, by four orders.
    assert single_far > 1e-3
    assert double_far < 1e-9
    assert double_close < 1e-11


# --- derivatives ---


def _gradient(values, dt, step):
    """``d/dparameters`` of a fixed contraction of the operator."""
    weights = torch.randn(3, 3, generator=torch.Generator().manual_seed(7), dtype=REAL)
    leaves = tuple(value.detach().clone().requires_grad_(True) for value in values)
    operator = step((generator(*leaves) * dt)[None])[0]
    return torch.autograd.grad((weights * operator).sum(), leaves)


@pytest.mark.parametrize(
    "name,t1s,rates,fractions,dt",
    [
        ("well separated", (1000.0, 400.0, 800.0), (30.0, 60.0), (0.2, 0.1), 0.05),
        ("triple root", (900.0, 900.0, 900.0), (0.0, 0.0), (1 / 3.0, 1 / 3.0), 2.0),
        ("double root, wide spread", (900.0, 900.0, 1.0), (0.0, 0.0), (0.3, 0.2), 0.5),
        (
            "double root, long interval",
            (900.0, 900.0, 0.6),
            (0.0, 0.0),
            (0.3, 0.2),
            3.0,
        ),
        ("double root, exchanging", (900.0, 900.0, 1.0), (40.0, 40.0), (0.3, 0.2), 0.5),
        ("no exchange at all", (900.0, 400.0, 3.0), (0.0, 0.0), (0.3, 0.2), 5.0),
        ("extreme exchange", (900.0, 400.0, 3.0), (2e5, 2e5), (0.3, 0.2), 0.5),
    ],
)
def test_the_derivatives_match_autograd(name, t1s, rates, fractions, dt) -> None:
    """Where two roots meet the sorted roots have a vertical tangent and the
    operator does not, so a derivative taken through them returns NaN unless
    the branch is guarded. Every degeneracy a user can reach by giving two
    pools the same relaxation is read here.
    """
    values = tuple(
        torch.tensor(entry, dtype=REAL) for entry in (*t1s, *rates, *fractions)
    )
    delay = torch.tensor(dt, dtype=REAL)

    measured = _gradient(values, delay, three_pool_step)
    expected = _gradient(values, delay, torch.matrix_exp)

    scale = max(float(entry.abs().max()) for entry in expected)
    assert scale > 0.0
    for got, want in zip(measured, expected, strict=True):
        assert torch.isfinite(got).all(), name
        assert float((got - want).abs().max()) / scale < 1e-8, name


# --- the two-pool systems are limits of this one ---


@pytest.mark.parametrize("absent", ["semisolid", "exchanging"])
def test_dropping_a_pool_leaves_the_two_pool_operator(absent: str) -> None:
    """A fraction of zero empties one pool and decouples it.

    This is what makes the three-pool kernel safe to add: each two-pool system
    is a limit of it rather than a neighbour, so the answer a user already had
    is the answer they keep.
    """
    values, dt = _draw(3000, 8)
    (t1a, t1b, t1c, exchange_b, exchange_c, fraction_b, fraction_c) = values
    zero = torch.zeros_like(fraction_b)
    kept = slice(0, 2) if absent == "semisolid" else slice(0, 3, 2)
    values = (
        (t1a, t1b, t1c, exchange_b, exchange_c, fraction_b, zero)
        if absent == "semisolid"
        else (t1a, t1b, t1c, exchange_b, exchange_c, zero, fraction_c)
    )
    scaled = _scaled(values, dt)
    operator = three_pool_step(scaled)

    # The remaining pair evolves as the two-pool operator on its own block.
    # Read against the whole operator's scale rather than the block's: an
    # emptied pool that drains fast leaves its block eighteen orders under the
    # column that still carries magnetization, and no arithmetic resolves that.
    # The envelope is the general sweep's, which is the point -- taking the
    # limit costs nothing beyond what the operator already carries.
    pair = scaled[:, kept, kept]
    expected = torch.matrix_exp(pair)
    scale = operator.abs().amax((-2, -1)).clamp(min=1e-30)
    relative = (operator[:, kept, kept] - expected).abs().amax((-2, -1)) / scale
    assert float(torch.quantile(relative, 0.99)) < 1e-7
    assert float(relative.max()) < 1e-5
    # The decoupling itself is exact, and says so at a tolerance conditioning
    # cannot reach: an emptied pool neither receives nor sends anything.
    gone = 2 if absent == "semisolid" else 1
    assert float(operator[:, gone, kept].abs().max()) < 1e-12


# --- the package's own operator ---


def test_the_closed_form_agrees_with_the_package_operator() -> None:
    """The convention check, at the precision that operator holds.

    ``longitudinal_relaxation_exchange_op`` is generic in the number of pools
    and says nothing about which exchanges with which, so handing it this
    file's exchange matrix is what settles that the rates are read the way the
    kernels read them. It reaches its exponential through an eigendecomposition,
    which is ill-conditioned where two roots meet -- the corner the series
    branch exists for -- so this pins the convention rather than the digits.
    """
    values, dt = _draw(200, 9, exchange="live")
    (t1a, t1b, t1c, exchange_b, exchange_c, fraction_b, fraction_c) = values
    free = 1.0 - fraction_b - fraction_c
    worst = 0.0
    for index in range(len(dt)):
        weight = torch.stack((free[index], fraction_b[index], fraction_c[index]))
        rates = torch.stack(
            (1000.0 / t1a[index], 1000.0 / t1b[index], 1000.0 / t1c[index])
        )
        matrix = generator(
            t1a[index],
            t1b[index],
            t1c[index],
            exchange_b[index],
            exchange_c[index],
            fraction_b[index],
            fraction_c[index],
        ) + torch.diag(rates)
        operator, _ = epg.longitudinal_relaxation_exchange_op(
            weight, matrix, rates, dt[index]
        )
        mine = three_pool_step(
            (generator(*(value[index] for value in values)) * dt[index])[None]
        )[0]
        worst = max(
            worst,
            float(
                (mine - operator).abs().max() / operator.abs().max().clamp(min=1e-30)
            ),
        )
    assert worst < 1e-6
