"""Gauss-Newton inversion of the forward operator.

Each step solves the linearized problem

    x_{n+1} = x_n + argmin_d || DF(x_n) d + F(x_n) - y ||^2 + a_n R(x_n + d)

with a caller-chosen linear solve (:func:`direct`, :func:`iterative`, or any
callable of the same signature) and damping policy (:class:`Schedule`,
:class:`TrustRegion`).
"""

from __future__ import annotations

__all__ = [
    "GaussNewton",
    "LeastSquares",
    "Linearization",
    "Schedule",
    "Solution",
    "TrustRegion",
    "direct",
    "iterative",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

#: Guards a division whose denominator the algorithm has already tested.
_TINY = torch.finfo(torch.float32).tiny


@dataclass(frozen=True)
class Linearization:
    """The forward operator's derivative at one point, as products.

    Attributes
    ----------
    matvec : callable
        ``d -> J d``, from map space to measurement space.
    rmatvec : callable
        ``v -> J^H v``, real.
    blocks : torch.Tensor, default=None
        ``(voxels, channels, contrasts)`` per-voxel Jacobian; ``None`` where an
        encoding has mixed the voxels.
    """

    matvec: Callable[[torch.Tensor], torch.Tensor]
    rmatvec: Callable[[torch.Tensor], torch.Tensor]
    blocks: torch.Tensor | None = None


@dataclass(frozen=True)
class Solution:
    """Result of :meth:`GaussNewton.minimize`.

    Attributes
    ----------
    x : torch.Tensor
        Solved-for variables; pass through
        :meth:`~blochsim.recon.ModelOperator.split` for the properties.
    cost : torch.Tensor
        Squared residual before the first step and after each step.
    damping : torch.Tensor
        Damping of each step, averaged over voxels where it is per voxel.
    iterations : int
        Steps taken.
    unconverged : int
        Voxels still moving when the loop stopped (0 under an encoding).
    """

    x: torch.Tensor
    cost: torch.Tensor
    damping: torch.Tensor
    iterations: int
    unconverged: int


# %% inner solves


@runtime_checkable
class LeastSquares(Protocol):
    """Regularized linear least-squares solve, in deepinv's calling shape.

    Minimizes ``||A d - y||^2 + (1/gamma) ||d - z||^2`` from products with the
    operator and its adjoint. ``deepinv.optim.linear.least_squares`` conforms.

    Parameters
    ----------
    A : callable
        Forward product ``d -> A d``.
    AT : callable
        Adjoint product ``r -> A^H r``.
    y : torch.Tensor
        Right-hand side.
    z : torch.Tensor
        Point the regularizer pulls towards.
    gamma : float or torch.Tensor or None
        Reciprocal of the damping; ``None`` for no damping.
    max_iter : int
        Maximum iterations.
    tol : float
        Relative residual tolerance.
    """

    def __call__(
        self,
        *,
        A: Callable[[torch.Tensor], torch.Tensor],
        AT: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        z: torch.Tensor,
        gamma: float | torch.Tensor | None,
        max_iter: int,
        tol: float,
    ) -> torch.Tensor:
        """Return the step ``d`` that minimizes the damped residual."""


def iterative(
    solver: LeastSquares | None = None, *, max_iter: int = 50, tol: float = 1e-6
) -> Callable[..., torch.Tensor]:
    """Build an inner solve from a regularized least-squares routine.

    Parameters
    ----------
    solver : LeastSquares, default=None
        Routine minimizing the damped linear problem. ``None`` is deepinv's
        ``least_squares`` (conjugate gradients); ``partial(least_squares,
        solver="lsqr")`` selects another.
    max_iter : int, default=50
        Maximum inner iterations per Gauss-Newton step.
    tol : float, default=1e-6
        Relative residual tolerance.

    Returns
    -------
    callable
        Inner solve, for :class:`GaussNewton` ``solve=``.

    Raises
    ------
    ImportError
        If no solver is given and deepinv is not installed.
    """

    def solve(
        linearization: Linearization,
        rhs: torch.Tensor,
        damping: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        routine = solver
        if routine is None:
            try:
                from deepinv.optim.linear import least_squares
            except ImportError as error:  # pragma: no cover - depends on env
                raise ImportError(
                    "iterative() falls back to deepinv's least_squares, and "
                    "BlochSim does not depend on deepinv. pip install "
                    "deepinv, pass a LeastSquares of your own, or use "
                    "direct() where the model stands alone."
                ) from error
            routine = least_squares
        weight = torch.as_tensor(damping)
        if not bool((weight > 0).any()):
            gamma = None
        elif weight.numel() == 1:
            # A single weight goes over as a number. Some of deepinv's solvers
            # take a scalar regularization and not a map of one.
            gamma = 1.0 / float(weight.reshape(()))
        else:
            gamma = 1.0 / torch.broadcast_to(weight, reference.shape)
        return routine(
            A=linearization.matvec,
            AT=linearization.rmatvec,
            y=rhs,
            z=reference,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
        )

    return solve


def direct(
    linearization: Linearization,
    rhs: torch.Tensor,
    damping: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Solve every voxel's damped linearized system by batched least squares.

    Valid only for a voxel-diagonal Jacobian. The damping enters as extra rows
    ``sqrt(a) I d = sqrt(a) z`` rather than as a ridge on the normal equations.
    A complex contrast counts as two real measurements.

    Parameters
    ----------
    linearization : Linearization
        Derivative carrying its ``blocks``.
    rhs : torch.Tensor
        ``(voxels, contrasts)``, ``y - F(x)``.
    damping : torch.Tensor
        ``(voxels, 1)`` or a single value.
    reference : torch.Tensor
        ``(voxels, channels)``, point the damping pulls towards.

    Returns
    -------
    torch.Tensor
        ``(voxels, channels)`` step.

    Raises
    ------
    ValueError
        If the linearization carries no blocks.
    """
    blocks = linearization.blocks
    if blocks is None:
        raise ValueError(
            "direct() needs the per-voxel Jacobian, which only a model with "
            "no encoding in front of it has; use iterative() instead"
        )
    rows, columns = _as_real(blocks, rhs)
    root = torch.broadcast_to(damping, reference.shape).sqrt()
    eye = torch.eye(rows.shape[-2], dtype=rows.dtype, device=rows.device)
    system = torch.cat((rows.mT, root[..., None] * eye), dim=-2)
    target = torch.cat((columns, root * reference), dim=-1)
    return torch.linalg.lstsq(system, target[..., None]).solution.squeeze(-1)


# %% damping policies


@dataclass
class Schedule:
    """One damping for the whole volume, lowered geometrically.

    Iteratively regularized Gauss-Newton: steps are pulled towards the
    starting guess with a weight that decays each step. Every step is accepted.

    Parameters
    ----------
    initial : float, default=1.0
        First weight, on the squared norm of the step from the starting guess
        in the solved-for variables.
    factor : float, default=1/3
        Multiplier applied after each step, in (0, 1).
    minimum : float, default=1e-3
        Floor of the weight, above 0.
    """

    initial: float = 1.0
    factor: float = 1.0 / 3.0
    minimum: float = 1e-3
    #: Damping pulls the step towards the starting guess, not towards nothing.
    anchored: bool = True
    #: Every voxel shares one weight, so nothing is retired on its own.
    per_voxel: bool = False

    def __post_init__(self) -> None:
        if self.initial <= 0.0:
            raise ValueError(f"initial must be positive, got {self.initial}")
        if not 0.0 < self.factor < 1.0:
            raise ValueError(f"factor must lower the damping, got {self.factor}")
        if self.minimum <= 0.0:
            raise ValueError(f"minimum must be positive, got {self.minimum}")

    def begin(self, curvature: torch.Tensor) -> torch.Tensor:
        """Return the initial weight."""
        return torch.as_tensor(
            self.initial, dtype=torch.float32, device=curvature.device
        )

    def judge(
        self, damping: torch.Tensor, gain: torch.Tensor, singular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept every step and lower the weight."""
        return (
            torch.ones_like(gain, dtype=torch.bool),
            (damping * self.factor).clamp_min(self.minimum),
        )


@dataclass
class TrustRegion:
    """Levenberg-Marquardt: per-voxel damping, raised where a step does not pay.

    A step is accepted per voxel only if it lowers the residual. Requires
    independent voxels, so no encoding operator.

    Parameters
    ----------
    tau : float, default=1e-2
        First damping, as a multiple of the largest diagonal element of the
        Gauss-Newton curvature ``J^H J`` of each voxel at the start. Above 0.
    """

    tau: float = 1e-2
    #: Damping shortens the step; it does not pull it anywhere.
    anchored: bool = False
    #: Each voxel carries its own weight and retires on its own.
    per_voxel: bool = True
    _rising: torch.Tensor | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")

    def begin(self, curvature: torch.Tensor) -> torch.Tensor:
        """Return the initial per-voxel damping from the curvature."""
        damping = self.tau * curvature.diagonal(dim1=-2, dim2=-1).amax(-1)
        self._rising = torch.full_like(damping, 2.0)
        return damping

    def judge(
        self, damping: torch.Tensor, gain: torch.Tensor, singular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept steps with positive gain and update the damping (Nielsen)."""
        better = (gain > 0) & ~singular
        rising = self._rising
        lowered = damping * (1.0 - (2.0 * gain - 1.0).pow(3)).clamp_min(1.0 / 3.0)
        self._rising = torch.where(better, torch.full_like(rising, 2.0), rising * 2.0)
        return better, torch.where(better, lowered, damping * rising)

    def retire(self, keep: torch.Tensor) -> None:
        """Drop the state of retired voxels."""
        self._rising = self._rising[keep]


# %% the loop


class GaussNewton:
    """Solve a nonlinear inverse problem by repeated linearization.

    Parameters
    ----------
    damping : Schedule or TrustRegion, default=None
        Damping policy: ``Schedule`` for iteratively regularized Gauss-Newton,
        ``TrustRegion`` for Levenberg-Marquardt. ``None`` is ``Schedule()``.
    solve : callable, default=None
        Inner solve, ``solve(linearization, rhs, damping, reference)``
        returning the step. ``None`` is :func:`direct` without an encoding and
        :func:`iterative` with one.
    max_iterations : int, default=8
        Maximum outer steps, at least 1.
    gradient_tolerance, step_tolerance : float, default=1e-8
        A voxel is done when its gradient magnitude is below the first or its
        step is short relative to its norm by the second.

    Raises
    ------
    ValueError
        If ``max_iterations`` is below 1.

    Examples
    --------
    .. code-block:: python

        loop = GaussNewton(Schedule(initial=1.0))
        found = loop.minimize(operator, kspace, start, encoding=nufft)
        maps = operator.split(found.x)
    """

    def __init__(
        self,
        damping: Any = None,
        *,
        solve: Callable[..., torch.Tensor] | None = None,
        max_iterations: int = 8,
        gradient_tolerance: float = 1e-8,
        step_tolerance: float = 1e-8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        self.damping = damping if damping is not None else Schedule()
        self.solve = solve
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = gradient_tolerance
        self.step_tolerance = step_tolerance

    @torch.no_grad()
    def minimize(
        self,
        operator: Any,
        measured: torch.Tensor,
        initial: torch.Tensor,
        *,
        encoding: Any = None,
    ) -> Solution:
        """Solve for the maps that explain ``measured``.

        Parameters
        ----------
        operator : ModelOperator
            Forward model.
        measured : torch.Tensor
            Data: ``(..., contrasts)`` without an encoding, otherwise the
            encoding's output.
        initial : torch.Tensor
            ``(..., channels)`` start; see
            :meth:`~blochsim.recon.ModelOperator.initial`.
        encoding : object, default=None
            Anything with ``A`` and ``A_adjoint`` (deepinv ``LinearPhysics``).
            It receives the contrast axis at axis 1.

        Returns
        -------
        Solution
            Solved-for variables ``x`` of shape ``(..., channels)``, with the
            cost history and iteration counts.

        Raises
        ------
        ValueError
            If the damping is per voxel and an encoding is given.
        """
        solve = self.solve or (direct if encoding is None else iterative())
        if encoding is not None and getattr(self.damping, "per_voxel", False):
            raise ValueError(
                f"{type(self.damping).__name__} steps each voxel on its own, "
                "which an encoding operator has made impossible; use "
                "Schedule() instead"
            )
        shape = initial.shape[:-1]
        # An encoding operator reads the maps as a picture, so the voxel axes
        # stay as they were. With nothing in front of the model they are only
        # a list, and flattening them is what lets a converged voxel drop out.
        if encoding is None:
            start = initial.reshape(-1, initial.shape[-1])
            data = measured.reshape(start.shape[0], -1)
        else:
            start, data = initial, measured
        retiring = encoding is None and getattr(self.damping, "per_voxel", False)

        answer = start.clone()
        live = torch.arange(start.shape[0], device=start.device)
        x = start.clone()
        residual, linearization = self._at(operator, x, data, encoding)
        cost = _cost(residual, encoding)
        damping = self.damping.begin(_curvature(linearization))
        history: list[torch.Tensor] = [cost.sum()]
        weights: list[torch.Tensor] = []

        iterations = 0
        for _ in range(self.max_iterations):
            gradient = linearization.rmatvec(residual)
            flat = _per_voxel(gradient.abs(), encoding, torch.amax) < (
                self.gradient_tolerance
            )
            if retiring:
                x, live, data, residual, linearization, cost, damping = _retire(
                    ~flat,
                    answer,
                    live,
                    x,
                    data,
                    residual,
                    linearization,
                    cost,
                    damping,
                    self.damping,
                )
                if x.shape[0] == 0:
                    break
                if x.shape[0] != gradient.shape[0]:
                    operator = operator.select(~flat)
                # Retirement closed the rows up under it.
                gradient = linearization.rmatvec(residual)
            elif bool(flat.all()):
                break
            iterations += 1
            weights.append(damping.detach().reshape(-1).mean())

            reference = (
                start[live] - x
                if getattr(self.damping, "anchored", False)
                else torch.zeros_like(x)
            )
            step = solve(linearization, -residual, _shaped(damping), reference)
            walked = _norm(step, encoding)
            short = walked < self.step_tolerance * (
                _norm(x, encoding) + self.step_tolerance
            )
            singular = ~_per_voxel(torch.isfinite(step), encoding, torch.all)

            candidate = x + step
            trial, trial_linearization = self._at(operator, candidate, data, encoding)
            trial_cost = _cost(trial, encoding)
            predicted = _per_voxel(
                step * (_shaped(damping) * step - gradient), encoding, torch.sum
            )
            gain = torch.where(
                predicted > 0,
                (cost - trial_cost) / predicted.clamp_min(_TINY),
                torch.zeros_like(predicted),
            )
            better, damping = self.damping.judge(damping, gain, singular)

            taken = better[..., None] if better.ndim else better
            x = torch.where(taken, candidate, x)
            residual = torch.where(_spread(better, trial.shape), trial, residual)
            linearization = _chosen(better, trial_linearization, linearization)
            cost = torch.where(better, trial_cost, cost)
            history.append(cost.sum())

            if retiring:
                staying = ~(short & ~singular)
                x, live, data, residual, linearization, cost, damping = _retire(
                    staying,
                    answer,
                    live,
                    x,
                    data,
                    residual,
                    linearization,
                    cost,
                    damping,
                    self.damping,
                )
                if x.shape[0] == 0:
                    break
                if not bool(staying.all()):
                    operator = operator.select(staying)
            elif bool((short & ~singular).all()):
                break

        unconverged = int(x.shape[0]) if retiring else 0
        if retiring:
            if unconverged:
                answer[live] = x
        else:
            answer = x
        return Solution(
            x=answer.reshape(*shape, answer.shape[-1]) if encoding is None else answer,
            cost=torch.stack(history),
            damping=(
                torch.stack(weights) if weights else torch.zeros(0, device=start.device)
            ),
            iterations=iterations,
            unconverged=unconverged,
        )

    def _at(
        self, operator: Any, x: torch.Tensor, data: Any, encoding: Any
    ) -> tuple[torch.Tensor, Linearization]:
        """The residual at ``x`` and the derivative there."""
        predicted = operator.A(x)
        if encoding is None:
            # Promoted, never narrowed. A real model measured on complex data
            # is a modelling error, but throwing the imaginary half away would
            # hide it behind a plausible answer instead of a large residual.
            dtype = torch.promote_types(predicted.dtype, data.dtype)
            blocks = operator.jacobian(x).to(dtype)
            return predicted.to(dtype) - data.to(dtype), _diagonal(blocks)
        return (
            encoding.A(_forward(predicted)) - data,
            Linearization(
                matvec=lambda d: encoding.A(_forward(operator.A_jvp(x, d))),
                rmatvec=lambda v: operator.A_vjp(x, _backward(encoding.A_adjoint(v))),
            ),
        )


# %% private module subroutines


def _per_voxel(values: torch.Tensor, encoding: Any, reduce: Any) -> torch.Tensor:
    """The reduction the problem's shape calls for.

    Over the channels where the voxels are independent, over all of them where
    an encoding operator has tied them together.

    Convergence, the gain ratio and the damping are per-voxel questions only
    when the voxels are separate problems. Under an encoding there is one
    problem, and one answer to each.
    """
    if encoding is None:
        return reduce(values, -1)
    return reduce(values)


def _norm(values: torch.Tensor, encoding: Any) -> torch.Tensor:
    """The length of a step, per voxel or over the whole volume."""
    if encoding is None:
        return torch.linalg.vector_norm(values, dim=-1)
    return torch.linalg.vector_norm(values)


def _diagonal(blocks: torch.Tensor) -> Linearization:
    """The derivative of a voxel-diagonal operator, as products with it."""
    return Linearization(
        matvec=lambda d: torch.einsum("vpt,vp->vt", blocks, d.to(blocks.dtype)),
        rmatvec=lambda v: (
            torch.einsum("vpt,vt->vp", blocks.conj(), v.to(blocks.dtype)).real
        ),
        blocks=blocks,
    )


def _forward(contrasts: torch.Tensor) -> torch.Tensor:
    """Contrast-last, as the model gives it, to contrast-first as encoding wants."""
    return contrasts.movedim(-1, 1)


def _backward(contrasts: torch.Tensor) -> torch.Tensor:
    """The way back."""
    return contrasts.movedim(1, -1)


def _as_real(
    blocks: torch.Tensor, rhs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A complex contrast as the two real measurements it is."""
    if not torch.is_complex(blocks):
        return blocks, rhs.real if torch.is_complex(rhs) else rhs
    return (
        torch.cat((blocks.real, blocks.imag), dim=-1),
        torch.cat((rhs.real, rhs.imag), dim=-1),
    )


def _cost(residual: torch.Tensor, encoding: Any) -> torch.Tensor:
    """The squared residual, per voxel where the voxels are independent."""
    squared = residual.real.square() + (
        residual.imag.square() if torch.is_complex(residual) else 0.0
    )
    return squared.sum(-1) if encoding is None else squared.sum()


def _curvature(linearization: Linearization) -> torch.Tensor:
    """What the damping is scaled against at the starting point."""
    blocks = linearization.blocks
    if blocks is None:
        return torch.zeros(1)
    rows = (
        torch.cat((blocks.real, blocks.imag), dim=-1)
        if torch.is_complex(blocks)
        else blocks
    )
    return rows @ rows.mT


def _shaped(damping: torch.Tensor) -> torch.Tensor:
    """A per-voxel weight, shaped to broadcast against a map."""
    return damping[..., None] if damping.ndim == 1 else damping


def _spread(better: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """An accept flag, widened to whatever it is selecting between."""
    if not better.ndim:
        return better
    return better.reshape(better.shape[0], *(1,) * (len(shape) - 1))


def _chosen(
    better: torch.Tensor, trial: Linearization, standing: Linearization
) -> Linearization:
    """The derivative at wherever each voxel ended up.

    A rejected step leaves that voxel where it was, so its old derivative
    still stands and is kept rather than recomputed. Where there is no
    per-voxel structure the whole step was taken or none of it was.
    """
    if standing.blocks is None or trial.blocks is None:
        return trial if bool(better.all()) else standing
    return _diagonal(torch.where(better[:, None, None], trial.blocks, standing.blocks))


def _retire(
    keep: torch.Tensor,
    answer: torch.Tensor,
    live: torch.Tensor,
    x: torch.Tensor,
    data: torch.Tensor,
    residual: torch.Tensor,
    linearization: Linearization,
    cost: torch.Tensor,
    damping: torch.Tensor,
    policy: Any,
) -> tuple[Any, ...]:
    """Write finished voxels out and close the rest up.

    Late iterations then cost what is left rather than what was started with,
    which is most of the saving on a volume whose background converges at
    once.
    """
    done = ~keep
    if bool(done.any()):
        answer[live[done]] = x[done]
    if bool(keep.all()):
        return (x, live, data, residual, linearization, cost, damping)
    blocks = linearization.blocks[keep]
    if hasattr(policy, "retire"):
        policy.retire(keep)
    return (
        x[keep],
        live[keep],
        data[keep],
        residual[keep],
        _diagonal(blocks),
        cost[keep],
        damping[keep],
    )
