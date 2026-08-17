"""The two-pool longitudinal operator the MT kernels have to reproduce.

A bound proton pool exchanges magnetization with the free water, so the
longitudinal step stops being a scalar decay and becomes a 2x2 one: ``Z <-
E1 Z``, with ``E1 = expm((K - diag(R1)) t)`` and a recovery term to match.
The kernels cannot call a matrix exponential, so they need that exponential
in closed form -- and they need its derivative, four passes deep.

For a 2x2 the closed form is exact:

    tau = tr(L)/2      d^2 = tau^2 - det(L)      D = L - tau I
    expm(L) = e^tau [ cosh(d) I + (sinh(d)/d) D ]

and the discriminant is a sum of two non-negative terms,

    d^2 = ((L11 - L22)/2)^2 + k_ba k_ab,

so it never leaves the real line. That is what makes this cheap: no complex
branch, no eigendecomposition, and a derivative that can be written down.

These tests pin the closed form and its derivatives before either kernel
exists, so implementing it is checking against a test rather than trusting a
derivation.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import epg
from torchsim.base.config.relax_model import build_two_pool_exchange_matrix

DOUBLE = torch.float64


def _rates(t1a_ms, t1b_ms):
    """Longitudinal rates in 1/s, as the kernels form them."""
    return torch.stack((1000.0 / t1a_ms, 1000.0 / t1b_ms), dim=-1)


def _lambda(t1a_ms, t1b_ms, exchange, bound):
    """``K - diag(R1)``, the generator of the two-pool longitudinal step.

    ``K`` is the exchange matrix in the kernels' own terms: a pool leaves at
    the rate scaled by the *other* pool's fraction, so the columns sum to zero
    and the total magnetization is conserved by exchange alone.
    """
    free = 1.0 - bound
    kab = exchange * bound
    kba = exchange * free
    rates = _rates(t1a_ms, t1b_ms)
    return torch.stack(
        (
            torch.stack((-kab - rates[0], kba)),
            torch.stack((kab, -kba - rates[1])),
        )
    )


def _two_pool_step(t1a_ms, t1b_ms, exchange, bound, dt):
    """The closed form the kernels have to reproduce.

    Returns ``(E1, rE1)``: the 2x2 relaxation-and-exchange operator and the
    recovery it adds to the zeroth dephasing order.
    """
    generator = _lambda(t1a_ms, t1b_ms, exchange, bound)
    scaled = generator * dt
    eye = torch.eye(2, dtype=scaled.dtype)

    half_trace = 0.5 * (scaled[0, 0] + scaled[1, 1])
    determinant = scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0]
    square = half_trace * half_trace - determinant
    # Both terms of the discriminant are non-negative, so the square root is
    # real; it is taken through the squared value near the origin, where the
    # root itself has no derivative.
    turning = square > 1e-18
    guarded = torch.where(turning, square, torch.ones_like(square))
    root = torch.sqrt(guarded)

    # e^tau cosh(d) and e^tau sinh(d)/d, formed from the two eigenvalues
    # rather than as a product. Written the obvious way, a long interval
    # multiplies an underflowed e^tau by an overflowed cosh and returns NaN --
    # tau +/- d are the eigenvalues themselves, both non-positive for a
    # decaying system, so their exponentials are bounded by one.
    upper = torch.exp(half_trace + root)
    lower = torch.exp(half_trace - root)
    cosine = 0.5 * (upper + lower)
    scale = torch.where(
        turning,
        0.5 * (upper - lower) / root,
        torch.exp(half_trace) * (1.0 + square / 6.0 + square**2 / 120.0),
    )
    operator = cosine * eye + scale * (scaled - half_trace * eye)

    # Recovery: (E1 - I) L^-1 C, with L unscaled and C the equilibrium each
    # pool relaxes toward, which is its own fraction.
    free = 1.0 - bound
    rates = _rates(t1a_ms, t1b_ms)
    source = torch.stack((free * rates[0], bound * rates[1]))
    inverse_determinant = 1.0 / (
        generator[0, 0] * generator[1, 1] - generator[0, 1] * generator[1, 0]
    )
    solved = inverse_determinant * torch.stack(
        (
            generator[1, 1] * source[0] - generator[0, 1] * source[1],
            -generator[1, 0] * source[0] + generator[0, 0] * source[1],
        )
    )
    recovery = (operator - eye) @ solved
    return operator, recovery


def _draw(generator, *, equal_rates=False, exchange=None, bound=None):
    """One physically plausible set of two-pool parameters."""

    def uniform(low, high):
        return low + (high - low) * torch.rand((), generator=generator, dtype=DOUBLE)

    t1a = uniform(300.0, 2000.0)
    t1b = t1a.clone() if equal_rates else uniform(200.0, 1500.0)
    k = uniform(1.0, 80.0) if exchange is None else torch.tensor(exchange, dtype=DOUBLE)
    fb = uniform(0.02, 0.25) if bound is None else torch.tensor(bound, dtype=DOUBLE)
    dt = uniform(1e-4, 2e-2)
    return t1a, t1b, k, fb, dt


def _oracle(t1a_ms, t1b_ms, exchange, bound, dt):
    """The same operator through the package's own exchange operator."""
    free = 1.0 - bound
    weight = torch.stack((free, bound))
    matrix = build_two_pool_exchange_matrix(weight, exchange)
    return epg.longitudinal_relaxation_exchange_op(
        weight, matrix, _rates(t1a_ms, t1b_ms), dt
    )


def test_the_closed_form_is_the_matrix_exponential() -> None:
    """The anchor: the exponential reached through its spectral projector.

    For distinct real eigenvalues that projector is the exact answer, so this
    pins the rearrangement -- the one that keeps the origin finite and the
    long interval from overflowing -- against the plain algebra.
    """
    generator = torch.Generator().manual_seed(0)
    eye = torch.eye(2, dtype=DOUBLE)
    worst_operator = worst_recovery = 0.0
    for _ in range(200):
        values = _draw(generator)
        operator, recovery = _two_pool_step(*values)

        generator_matrix = _lambda(*values[:4])
        scaled = generator_matrix * values[4]
        half_trace = 0.5 * (scaled[0, 0] + scaled[1, 1])
        root = torch.sqrt(
            half_trace**2 - (scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0])
        )
        upper, lower = half_trace + root, half_trace - root
        expected = (
            torch.exp(upper) * (scaled - lower * eye)
            - torch.exp(lower) * (scaled - upper * eye)
        ) / (upper - lower)

        free, bound = 1.0 - values[3], values[3]
        source = torch.stack((free * 1000.0 / values[0], bound * 1000.0 / values[1]))
        expected_recovery = (expected - eye) @ torch.linalg.solve(
            generator_matrix, source
        )
        worst_operator = max(worst_operator, float((operator - expected).abs().max()))
        worst_recovery = max(
            worst_recovery, float((recovery - expected_recovery).abs().max())
        )
    assert worst_operator < 1e-14
    assert worst_recovery < 1e-14


def test_the_closed_form_beats_a_general_matrix_exponential() -> None:
    """Arbitrated at fifty digits, because the two disagree at eleven.

    ``torch.matrix_exp`` is Pade with scaling and squaring, and on these
    generators it carries about 1e-11. The closed form lands at the double
    epsilon. Worth knowing which way round that is: the kernels are not
    settling for an approximation of a better answer, they have the better
    answer.
    """
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50

    generator = torch.Generator().manual_seed(6)
    worst_closed = worst_pade = 0.0
    for _ in range(25):
        values = _draw(generator)
        scaled = _lambda(*values[:4]) * values[4]
        reference = mpmath.expm(mpmath.matrix(scaled.tolist()))
        operator, _ = _two_pool_step(*values)
        pade = torch.matrix_exp(scaled)
        for row in range(2):
            for column in range(2):
                exact = float(reference[row, column])
                worst_closed = max(
                    worst_closed, abs(float(operator[row, column]) - exact)
                )
                worst_pade = max(worst_pade, abs(float(pade[row, column]) - exact))
    assert worst_closed < 1e-15
    assert worst_closed < worst_pade


def test_the_closed_form_agrees_with_the_package_operator() -> None:
    """The convention check, at the precision that operator actually holds.

    ``epg.longitudinal_relaxation_exchange_op`` casts to ``complex64`` and
    reaches the exponential through an eigendecomposition, so it pins the
    convention -- which pool leaves at which rate, what the recovery is
    measured against -- rather than the digits.
    """
    generator = torch.Generator().manual_seed(5)
    worst = 0.0
    for _ in range(100):
        values = _draw(generator)
        operator, recovery = _two_pool_step(*values)
        expected_operator, expected_recovery = _oracle(*values)
        worst = max(
            worst,
            float((operator - expected_operator.real).abs().max()),
            float((recovery - expected_recovery.real).abs().max()),
        )
    assert worst < 1e-6


def test_a_long_interval_does_not_overflow_its_way_to_a_nan() -> None:
    """``e^tau cosh(d)`` is a vanishing number times a huge one.

    Formed as a product it is ``0 * inf`` once the interval is long against
    T1, which a sequence with a slow recovery reaches easily. Formed from the
    two eigenvalues it is bounded.
    """
    for seconds in (1.0, 10.0, 60.0, 600.0, 1e4):
        operator, recovery = _two_pool_step(
            torch.tensor(900.0, dtype=DOUBLE),
            torch.tensor(400.0, dtype=DOUBLE),
            torch.tensor(40.0, dtype=DOUBLE),
            torch.tensor(0.1, dtype=DOUBLE),
            torch.tensor(seconds, dtype=DOUBLE),
        )
        assert torch.isfinite(operator).all(), seconds
        assert torch.isfinite(recovery).all(), seconds


def test_the_discriminant_never_leaves_the_real_line() -> None:
    """No oscillatory branch, so the kernels need no complex arithmetic.

    ``d^2 = ((L11 - L22)/2)^2 + k_ba k_ab`` adds a square to a product of two
    rates that are non-negative by construction, so this is algebra rather
    than luck -- but the sweep covers the corners where a sign error would
    show: equal and wildly unequal T1, and bound fractions at both ends.
    """
    generator = torch.Generator().manual_seed(1)
    smallest = float("inf")
    for index in range(300):
        values = _draw(
            generator,
            equal_rates=index % 3 == 0,
            bound=0.001 if index % 5 == 0 else None,
            exchange=0.0 if index % 7 == 0 else None,
        )
        scaled = _lambda(*values[:4]) * values[4]
        half_trace = 0.5 * (scaled[0, 0] + scaled[1, 1])
        determinant = scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0]
        smallest = min(smallest, float(half_trace * half_trace - determinant))
    assert smallest >= 0.0


def test_the_degenerate_point_is_a_number_in_value_and_in_slope() -> None:
    """Equal rates and no exchange put the discriminant at exactly zero.

    ``sinh(d)/d`` is continuous there and its square root is not, which is the
    same corner that poisoned the transition table's slopes. The series has to
    carry both the value and its derivative.
    """
    t1 = torch.tensor(900.0, dtype=DOUBLE, requires_grad=True)
    k = torch.tensor(0.0, dtype=DOUBLE, requires_grad=True)
    fb = torch.tensor(0.1, dtype=DOUBLE, requires_grad=True)
    dt = torch.tensor(5e-3, dtype=DOUBLE, requires_grad=True)

    operator, recovery = _two_pool_step(t1, t1, k, fb, dt)
    assert torch.isfinite(operator).all()
    assert torch.isfinite(recovery).all()

    gradients = torch.autograd.grad(operator.sum() + recovery.sum(), (t1, k, fb, dt))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_no_bound_pool_is_the_single_pool_operator() -> None:
    """The identity that makes an MT-off kernel bitwise checkable.

    With no bound pool nothing leaves the free pool, so its own entry must be
    the scalar decay the single-pool kernel already computes and its recovery
    the ``1 - E1`` that goes with it.
    """
    generator = torch.Generator().manual_seed(2)
    for _ in range(50):
        t1a, t1b, k, _, dt = _draw(generator)
        bound = torch.zeros((), dtype=DOUBLE)
        operator, recovery = _two_pool_step(t1a, t1b, k, bound, dt)
        decay = torch.exp(-1000.0 / t1a * dt)

        assert abs(float(operator[0, 0]) - float(decay)) < 1e-12
        assert abs(float(operator[1, 0])) < 1e-12
        assert abs(float(recovery[0]) - float(1.0 - decay)) < 1e-12


def test_exchange_alone_conserves_the_magnetization() -> None:
    """With relaxation switched off, what leaves one pool reaches the other."""
    generator = torch.Generator().manual_seed(3)
    for _ in range(50):
        _, _, k, fb, dt = _draw(generator)
        # An infinite T1 is no relaxation, which leaves exchange on its own.
        huge = torch.tensor(1e12, dtype=DOUBLE)
        operator, _ = _two_pool_step(huge, huge, k, fb, dt)
        state = torch.stack((1.0 - fb, fb))

        moved = operator @ state
        assert abs(float(moved.sum() - state.sum())) < 1e-10


def test_a_long_interval_reaches_the_weighted_equilibrium() -> None:
    """Each pool relaxes toward its own fraction, exchange or not."""
    generator = torch.Generator().manual_seed(4)
    for _ in range(20):
        t1a, t1b, k, fb, _ = _draw(generator)
        dt = torch.tensor(60.0, dtype=DOUBLE)
        operator, recovery = _two_pool_step(t1a, t1b, k, fb, dt)
        settled = operator @ torch.zeros(2, dtype=DOUBLE) + recovery

        assert abs(float(settled[0]) - float(1.0 - fb)) < 1e-6
        assert abs(float(settled[1]) - float(fb)) < 1e-6


@pytest.mark.parametrize("index", range(4))
def test_the_derivatives_match_the_matrix_exponential(index: int) -> None:
    """Pointwise equality is not enough; the kernels differentiate this.

    Checked against ``torch.matrix_exp`` rather than the package operator,
    whose eigendecomposition is itself ill-conditioned where two eigenvalues
    meet -- which is exactly the corner the closed form exists to handle. The
    tolerance is set by Pade's own accuracy on these generators, not by the
    closed form's.
    """
    generator = torch.Generator().manual_seed(10 + index)
    t1a, t1b, k, fb, dt = (
        value.clone().requires_grad_(True) for value in _draw(generator)
    )
    seed = torch.randn(2, 2, generator=generator, dtype=DOUBLE)
    seed_recovery = torch.randn(2, generator=generator, dtype=DOUBLE)

    operator, recovery = _two_pool_step(t1a, t1b, k, fb, dt)
    actual = torch.autograd.grad(
        (operator * seed).sum() + (recovery * seed_recovery).sum(),
        (t1a, t1b, k, fb, dt),
    )

    def reference(t1a, t1b, k, fb, dt):
        generator_matrix = _lambda(t1a, t1b, k, fb)
        exponential = torch.matrix_exp(generator_matrix * dt)
        free = 1.0 - fb
        source = torch.stack((free * 1000.0 / t1a, fb * 1000.0 / t1b))
        solved = torch.linalg.solve(generator_matrix, source)
        rest = (exponential - torch.eye(2, dtype=DOUBLE)) @ solved
        return (exponential * seed).sum() + (rest * seed_recovery).sum()

    expected = torch.autograd.grad(
        reference(t1a, t1b, k, fb, dt), (t1a, t1b, k, fb, dt)
    )
    for name, want, got in zip(
        ("t1a", "t1b", "k", "bound", "dt"), expected, actual, strict=True
    ):
        scale = max(abs(float(want)), 1.0)
        assert abs(float(want - got)) < 1e-12 * scale, name
