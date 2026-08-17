"""The two-pool transverse operator the exchange kernels have to reproduce.

Two pools that exchange while both carrying transverse magnetization turn the
transverse step into a 2x2 one, ``F+ <- E2 F+``, with

    E2 = expm((K - diag(R2) - 2 pi i diag(df)) t)

and ``F-`` taking the elementwise conjugate of that operator. The chemical
shift is what makes this generator complex, and complex is what separates it
from the longitudinal pair in :mod:`test_two_pool`: there the discriminant is
a sum of a square and a product of two non-negative rates and never leaves the
real line, here it leaves it as soon as the pools sit at different offsets.

Three things about the closed form matter enough to pin before a kernel exists:

* **It is the same closed form.** The rearrangement that keeps the origin
  finite and a long interval from overflowing does not care that the numbers
  became complex.
* **The square root's branch does not reach the answer.** ``cosh(d)`` and
  ``sinh(d)/d`` are both even, so the operator is a function of ``d^2`` alone
  and either root gives it. A complex square root has a branch cut; this
  operator cannot see it.
* **The eigenvalues still decay.** Forming the pair from ``e^(tau +/- d)``
  rather than as ``e^tau cosh(d)`` needs their real parts non-positive, which
  survives an imaginary diagonal because an anti-Hermitian part contributes
  nothing to them.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import epg
from torchsim.base.config.relax_model import build_two_pool_exchange_matrix

DOUBLE = torch.complex128
REAL = torch.float64


def _rates(t2a_ms, t2b_ms):
    """Transverse rates in 1/s, as the kernels form them."""
    return torch.stack((1000.0 / t2a_ms, 1000.0 / t2b_ms), dim=-1)


def _lambda(t2a_ms, t2b_ms, exchange, bound, shift_hz):
    """``K - diag(R2) - 2 pi i diag(df)``, the transverse generator.

    Pool a sits at the scanner's own off-resonance, which the free precession
    already carries, so only pool b's offset from it appears here.
    """
    free = 1.0 - bound
    kab = exchange * bound
    kba = exchange * free
    rates = _rates(t2a_ms, t2b_ms)
    turn = 2.0 * torch.pi * shift_hz
    zero = torch.zeros((), dtype=REAL)
    return torch.stack(
        (
            torch.stack((
                torch.complex(-kab - rates[0], zero),
                torch.complex(kba, zero),
            )),
            torch.stack((
                torch.complex(kab, zero),
                torch.complex(-kba - rates[1], -turn),
            )),
        )
    )


def _transverse_step(t2a_ms, t2b_ms, exchange, bound, shift_hz, dt, branch=1.0):
    """The closed form the kernels have to reproduce.

    ``branch`` picks which square root is taken, which the operator is not
    supposed to be able to tell apart.
    """
    scaled = _lambda(t2a_ms, t2b_ms, exchange, bound, shift_hz) * dt
    eye = torch.eye(2, dtype=DOUBLE)

    half_trace = 0.5 * (scaled[0, 0] + scaled[1, 1])
    determinant = scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0]
    square = half_trace * half_trace - determinant
    turning = square.abs() > 1e-18
    guarded = torch.where(turning, square, torch.ones_like(square))
    root = branch * torch.sqrt(guarded)

    # tau +/- d are the eigenvalues themselves, whose real parts are
    # non-positive for a decaying system, so their exponentials are bounded.
    # Written as e^tau cosh(d), a long interval is an underflow times an
    # overflow and returns NaN.
    upper = torch.exp(half_trace + root)
    lower = torch.exp(half_trace - root)
    cosine = 0.5 * (upper + lower)
    scale = torch.where(
        turning,
        0.5 * (upper - lower) / root,
        torch.exp(half_trace) * (1.0 + square / 6.0 + square**2 / 120.0),
    )
    return cosine * eye + scale * (scaled - half_trace * eye)


def _draw(generator, *, equal_rates=False, shift=None, exchange=None, bound=None):
    """One physically plausible set of two-pool transverse parameters."""

    def uniform(low, high):
        return low + (high - low) * torch.rand((), generator=generator, dtype=REAL)

    t2a = uniform(20.0, 200.0)
    t2b = t2a.clone() if equal_rates else uniform(5.0, 150.0)
    k = uniform(1.0, 100.0) if exchange is None else torch.tensor(exchange, dtype=REAL)
    fb = uniform(0.02, 0.4) if bound is None else torch.tensor(bound, dtype=REAL)
    df = uniform(-800.0, 800.0) if shift is None else torch.tensor(shift, dtype=REAL)
    dt = uniform(1e-4, 2e-2)
    return t2a, t2b, k, fb, df, dt


def _oracle(t2a_ms, t2b_ms, exchange, bound, shift_hz, dt):
    """The same operator through the package's own exchange operator.

    ``transverse_relaxation_exchange_op`` documents its offset as rad/s and
    then multiplies it by ``2 pi``, so what it wants is Hz; passing it the
    same number this file calls ``shift_hz`` is the convention check.
    """
    free = 1.0 - bound
    weight = torch.stack((free, bound))
    matrix = build_two_pool_exchange_matrix(weight, exchange)
    return epg.transverse_relaxation_exchange_op(
        matrix,
        _rates(t2a_ms, t2b_ms),
        dt,
        torch.stack((torch.zeros((), dtype=REAL), shift_hz)),
    )


def test_the_closed_form_is_the_matrix_exponential() -> None:
    """The anchor: the exponential reached through its spectral projector.

    For distinct eigenvalues that projector is exact, so this pins the
    rearrangement -- the one that keeps the origin finite and the long
    interval from overflowing -- against the plain algebra, complex numbers
    and all.
    """
    generator = torch.Generator().manual_seed(0)
    eye = torch.eye(2, dtype=DOUBLE)
    worst = 0.0
    for _ in range(200):
        values = _draw(generator)
        operator = _transverse_step(*values)

        scaled = _lambda(*values[:5]) * values[5]
        half_trace = 0.5 * (scaled[0, 0] + scaled[1, 1])
        root = torch.sqrt(
            half_trace**2
            - (scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0])
        )
        upper, lower = half_trace + root, half_trace - root
        expected = (
            torch.exp(upper) * (scaled - lower * eye)
            - torch.exp(lower) * (scaled - upper * eye)
        ) / (upper - lower)
        worst = max(worst, float((operator - expected).abs().max()))
    assert worst < 1e-13


def test_the_closed_form_agrees_with_a_general_matrix_exponential() -> None:
    """Against Pade with scaling and squaring, which shares no algebra with it.

    The tolerance is set by Pade's own accuracy on these generators rather
    than by the closed form's; the next test says which way round that is.
    """
    generator = torch.Generator().manual_seed(6)
    worst = 0.0
    for _ in range(200):
        values = _draw(generator)
        scaled = _lambda(*values[:5]) * values[5]
        worst = max(
            worst,
            float((_transverse_step(*values) - torch.matrix_exp(scaled)).abs().max()),
        )
    assert worst < 1e-9


def test_the_closed_form_beats_a_general_matrix_exponential() -> None:
    """Arbitrated at fifty digits, because the two disagree at eleven.

    Worth knowing which way round it is: the kernels are not settling for an
    approximation of a better answer, they have the better answer, and the
    tolerances above are the other algorithm's error rather than theirs.
    """
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50

    generator = torch.Generator().manual_seed(8)
    worst_closed = worst_pade = 0.0
    for _ in range(25):
        values = _draw(generator)
        scaled = _lambda(*values[:5]) * values[5]
        reference = mpmath.expm(
            mpmath.matrix([[complex(entry) for entry in row] for row in scaled])
        )
        operator = _transverse_step(*values)
        pade = torch.matrix_exp(scaled)
        for row in range(2):
            for column in range(2):
                exact = complex(reference[row, column])
                worst_closed = max(
                    worst_closed, abs(complex(operator[row, column]) - exact)
                )
                worst_pade = max(worst_pade, abs(complex(pade[row, column]) - exact))
    assert worst_closed < 5e-14
    assert worst_closed < worst_pade


def test_the_closed_form_agrees_with_the_package_operator() -> None:
    """The convention check, at the precision that operator holds.

    Which pool leaves at which rate, and whether the offset is read in Hz or
    in rad/s -- the package's own docstring says one and its arithmetic the
    other, so the agreement is what settles it.

    ``epg.matrix_exp`` reaches the exponential through an eigendecomposition,
    which is ill-conditioned where two eigenvalues meet -- exactly the corner
    the closed form exists to handle -- so this pins the convention rather
    than the digits.
    """
    generator = torch.Generator().manual_seed(5)
    worst = 0.0
    for _ in range(100):
        values = _draw(generator)
        worst = max(
            worst,
            float((_transverse_step(*values) - _oracle(*values)).abs().max()),
        )
    assert worst < 1e-9


def test_the_square_root_branch_does_not_reach_the_answer() -> None:
    """A complex square root has a branch cut; this operator cannot see it.

    ``cosh(d)`` and ``sinh(d)/d`` are both even, so the operator is a function
    of ``d^2`` alone and either root produces it -- bitwise, not nearly. That
    is what lets the kernels take whichever root the hardware gives them and
    carry no branch logic at all.
    """
    generator = torch.Generator().manual_seed(1)
    worst = 0.0
    for index in range(300):
        values = _draw(
            generator,
            equal_rates=index % 3 == 0,
            bound=0.001 if index % 5 == 0 else None,
            exchange=0.0 if index % 7 == 0 else None,
        )
        positive = _transverse_step(*values, branch=1.0)
        negative = _transverse_step(*values, branch=-1.0)
        worst = max(worst, float((positive - negative).abs().max()))
    assert worst == 0.0


def test_the_discriminant_does_leave_the_real_line() -> None:
    """The contrast with the longitudinal pair, and the reason for a separate
    code path.

    There the discriminant is a square plus a product of two non-negative
    rates and stays real; here the chemical shift makes the generator complex
    and takes it off the line, so the kernels need complex arithmetic whatever
    else they reuse.
    """
    generator = torch.Generator().manual_seed(2)
    largest = 0.0
    for _ in range(300):
        values = _draw(generator)
        scaled = _lambda(*values[:5]) * values[5]
        half_trace = 0.5 * (scaled[0, 0] + scaled[1, 1])
        determinant = scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0]
        largest = max(largest, abs(float((half_trace**2 - determinant).imag)))
    assert largest > 1.0


def test_the_eigenvalues_never_grow() -> None:
    """What makes the overflow-safe rearrangement available.

    An imaginary diagonal is anti-Hermitian and contributes nothing to an
    eigenvalue's real part, leaving ``K_sym - diag(R2)``. The sweep covers the
    corners a proof would have to: rates three orders apart, exchange far
    faster than either, fractions at both ends, shifts well past any real
    chemical shift.
    """
    generator = torch.Generator().manual_seed(3)
    largest = -float("inf")
    for _ in range(2000):

        def uniform(low, high):
            return low + (high - low) * torch.rand(
                (), generator=generator, dtype=REAL
            )

        scaled = _lambda(
            uniform(0.5, 3000.0),
            uniform(0.5, 3000.0),
            uniform(0.0, 5000.0),
            uniform(0.0, 1.0),
            uniform(-1.0e4, 1.0e4),
        )
        largest = max(largest, float(torch.linalg.eigvals(scaled).real.max()))
    assert largest <= 0.0


def test_a_long_interval_does_not_overflow_its_way_to_a_nan() -> None:
    """``e^tau cosh(d)`` is a vanishing number times a huge one."""
    for seconds in (1.0, 10.0, 60.0, 600.0, 1e4):
        operator = _transverse_step(
            torch.tensor(80.0, dtype=REAL),
            torch.tensor(20.0, dtype=REAL),
            torch.tensor(40.0, dtype=REAL),
            torch.tensor(0.2, dtype=REAL),
            torch.tensor(-420.0, dtype=REAL),
            torch.tensor(seconds, dtype=REAL),
        )
        assert torch.isfinite(operator.real).all(), seconds
        assert torch.isfinite(operator.imag).all(), seconds


def test_the_degenerate_point_is_a_number_in_value_and_in_slope() -> None:
    """Equal rates, no exchange and no shift put the discriminant at zero.

    ``sinh(d)/d`` is continuous there and its square root is not, which is the
    same corner that poisoned the longitudinal pair. The series has to carry
    both the value and its derivative through it.
    """
    t2 = torch.tensor(80.0, dtype=REAL, requires_grad=True)
    k = torch.tensor(0.0, dtype=REAL, requires_grad=True)
    fb = torch.tensor(0.2, dtype=REAL, requires_grad=True)
    df = torch.tensor(0.0, dtype=REAL, requires_grad=True)
    dt = torch.tensor(5e-3, dtype=REAL, requires_grad=True)

    operator = _transverse_step(t2, t2, k, fb, df, dt)
    assert torch.isfinite(operator.real).all()
    assert torch.isfinite(operator.imag).all()

    gradients = torch.autograd.grad(operator.real.sum(), (t2, k, fb, df, dt))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_no_second_pool_is_the_scalar_decay() -> None:
    """The identity that makes an exchange-off kernel bitwise checkable.

    With no second pool nothing leaves the free water, so its own entry must
    be the scalar decay the single-pool kernel already computes -- and the
    chemical shift, which belongs to a pool that is not there, must not reach
    it.
    """
    generator = torch.Generator().manual_seed(4)
    for _ in range(50):
        t2a, t2b, k, _, df, dt = _draw(generator)
        bound = torch.zeros((), dtype=REAL)
        operator = _transverse_step(t2a, t2b, k, bound, df, dt)
        decay = torch.exp(-1000.0 / t2a * dt)

        assert abs(complex(operator[0, 0]) - complex(decay)) < 1e-12
        assert abs(complex(operator[1, 0])) < 1e-12


def test_exchange_alone_conserves_the_magnetization() -> None:
    """With relaxation and shift switched off, what leaves one pool reaches
    the other -- and stays in phase while doing so.
    """
    generator = torch.Generator().manual_seed(7)
    for _ in range(50):
        _, _, k, fb, _, dt = _draw(generator)
        huge = torch.tensor(1e12, dtype=REAL)
        still = torch.zeros((), dtype=REAL)
        operator = _transverse_step(huge, huge, k, fb, still, dt)
        state = torch.stack((1.0 - fb, fb)).to(DOUBLE)

        assert abs(complex((operator @ state).sum() - state.sum())) < 1e-10


def test_a_shift_without_exchange_turns_one_pool_and_not_the_other() -> None:
    """The offset belongs to pool b alone, pool a already sitting at whatever
    off-resonance the free precession carries.
    """
    dt = torch.tensor(2.0e-3, dtype=REAL)
    shift = torch.tensor(-420.0, dtype=REAL)
    huge = torch.tensor(1e12, dtype=REAL)
    operator = _transverse_step(
        huge, huge, torch.zeros((), dtype=REAL), torch.tensor(0.3, dtype=REAL),
        shift, dt,
    )
    turned = torch.exp(-2j * torch.pi * shift.to(DOUBLE) * dt)

    assert abs(complex(operator[0, 0]) - 1.0) < 1e-9
    assert abs(complex(operator[1, 1]) - complex(turned)) < 1e-9
    assert abs(complex(operator[0, 1])) < 1e-12
    assert abs(complex(operator[1, 0])) < 1e-12


@pytest.mark.parametrize("index", range(4))
def test_the_derivatives_match_the_matrix_exponential(index: int) -> None:
    """Pointwise equality is not enough; the kernels differentiate this.

    Checked against ``torch.matrix_exp``, whose Pade is a different algorithm
    and is differentiated by autograd rather than by the closed form's own
    algebra. Both halves of the cotangent are seeded, since a complex operator
    has two.
    """
    generator = torch.Generator().manual_seed(20 + index)
    t2a, t2b, k, fb, df, dt = (
        value.clone().requires_grad_(True) for value in _draw(generator)
    )
    seed = torch.complex(
        torch.randn(2, 2, generator=generator, dtype=REAL),
        torch.randn(2, 2, generator=generator, dtype=REAL),
    )

    def contracted(operator):
        return (operator.real * seed.real + operator.imag * seed.imag).sum()

    inputs = (t2a, t2b, k, fb, df, dt)
    actual = torch.autograd.grad(
        contracted(_transverse_step(t2a, t2b, k, fb, df, dt)), inputs
    )
    expected = torch.autograd.grad(
        contracted(torch.matrix_exp(_lambda(t2a, t2b, k, fb, df) * dt)), inputs
    )
    for name, want, got in zip(
        ("t2a", "t2b", "k", "bound", "shift", "dt"), expected, actual, strict=True
    ):
        scale = max(abs(float(want)), 1.0)
        assert abs(float(want - got)) < 1e-10 * scale, name
