"""Inverting the forward operator, one linearized problem at a time.

Every Newton-type method for a nonlinear inverse problem does the same thing:
linearize where it stands, solve the linear problem that leaves, step, repeat.

    x_{n+1} = x_n + argmin_d || DF(x_n) d + F(x_n) - y ||^2 + a_n R(x_n + d)

What separates the methods is *how the linear problem is solved* and *where the
damping comes from*, and both belong to the caller.

The linear solve is a callable, and mostly it is somebody else's.
:func:`iterative` hands the linearized problem to deepinv, whose
``least_squares`` minimizes exactly what a Gauss-Newton step leaves --
``||A d - b||^2 + (1/g) ||d - z||^2`` -- by conjugate gradients, LSQR,
BiCGStab or MINRES, and asks for nothing but products with the Jacobian. There
is no conjugate gradient written here and there should not be: a package that
ships its own is one more copy of an algorithm nobody wanted three of.

:func:`direct` is the exception, and only because it is not a general linear
solver: where the operator is voxel-diagonal the problem is a stack of small
independent least-squares systems, and one batched factorization over all of
them is the Levenberg-Marquardt step itself. That factorization is
:func:`torch.linalg.lstsq`.

Anything with the same signature does, and wrapping an external proximal
solver is how a regularizer enters -- which is the difference between an
iteratively regularized Gauss-Newton solved by conjugate gradients and one
solved by FISTA under an L1 prior.

The damping is a policy. :class:`Schedule` lowers one number geometrically and
accepts every step, which is the iteratively regularized Gauss-Newton the
model-based reconstruction literature runs. :class:`TrustRegion` gives each
voxel its own damping, accepts or rejects on whether the step actually helped,
and retires voxels as they converge -- which is Levenberg-Marquardt. The
second is the first with the bookkeeping that independent rows make possible,
and the two land in the same place on a problem both can solve.
"""

from __future__ import annotations

__all__ = [
    "GaussNewton",
    "Linearization",
    "Schedule",
    "Solution",
    "TrustRegion",
    "direct",
    "iterative",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


#: Guards a division whose denominator the algorithm has already tested.
_TINY = torch.finfo(torch.float32).tiny


@dataclass(frozen=True)
class Linearization:
    """The forward operator's derivative at one point, as products with it.

    A Gauss-Newton step never needs the Jacobian, only what it does to a
    vector, so that is what this carries. :attr:`blocks` is the exception and
    is filled in only where the operator is voxel-diagonal: there the Jacobian
    really is a stack of small independent matrices and factorizing them
    outright beats iterating.

    Attributes
    ----------
    matvec : callable
        ``d -> J d``, from map space to measurement space.
    rmatvec : callable
        ``v -> J^H v``, back again, real because the maps are.
    blocks : torch.Tensor, optional
        ``(voxels, channels, contrasts)`` where the operator is voxel-diagonal,
        ``None`` where an encoding has mixed the voxels together.
    """

    matvec: Callable[[torch.Tensor], torch.Tensor]
    rmatvec: Callable[[torch.Tensor], torch.Tensor]
    blocks: torch.Tensor | None = None


@dataclass(frozen=True)
class Solution:
    """What a solve returned, and what it did to get there.

    Attributes
    ----------
    x : torch.Tensor
        The maps, as the variables that were solved for. Pass them through
        :meth:`~torchsim.recon.ModelOperator.split` for the properties.
    cost : torch.Tensor
        The squared residual after each step, so convergence is read rather
        than assumed.
    damping : torch.Tensor
        The damping each step ran at, averaged over the voxels where each
        carries its own.
    iterations : int
        How many steps were taken.
    unconverged : int
        How many voxels were still moving when the loop stopped. Zero for a
        solve that ran to its tolerance.
    """

    x: torch.Tensor
    cost: torch.Tensor
    damping: torch.Tensor
    iterations: int
    unconverged: int


# %% inner solves


def iterative(
    solver: str = "CG", *, max_iter: int = 50, tol: float = 1e-6
) -> Callable[..., torch.Tensor]:
    """The inner solve, by deepinv's linear solvers.

    ``deepinv.optim.linear.least_squares`` minimizes
    ``||A d - b||^2 + (1/g) ||d - z||^2``, which is what a Gauss-Newton step
    leaves once the damping and its reference are in it. It asks only for
    products with the Jacobian, so it does not care whether an encoding
    operator sits in front of the model.

    Parameters
    ----------
    solver : str, optional
        Which of deepinv's solvers to use: ``"CG"``, ``"lsqr"``,
        ``"BiCGStab"`` or ``"minres"``. Which one suits depends on the
        problem and is worth measuring: conjugate gradients works on the
        normal equations, so on a small well-scaled system it stops on a
        tolerance the squared condition number has already spoiled, while
        LSQR and MINRES reach what a direct solve reaches -- and on a large
        encoded problem inside a fixed iteration budget the ordering reverses.
    max_iter : int, optional
        Most inner iterations per Gauss-Newton step.
    tol : float, optional
        Stop when the residual has fallen this far relative to the data.

    Returns
    -------
    callable
        An inner solve, to give :class:`GaussNewton` as ``solve=``.

    Raises
    ------
    ImportError
        If deepinv is not installed. TorchSim does not depend on it, and
        :func:`direct` needs nothing beyond torch.
    """

    def solve(
        linearization: Linearization,
        rhs: torch.Tensor,
        damping: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from deepinv.optim.linear import least_squares
        except ImportError as error:  # pragma: no cover - depends on the env
            raise ImportError(
                "iterative() solves through deepinv; TorchSim does not depend "
                "on it. pip install deepinv, or use direct() where the model "
                "stands alone."
            ) from error
        weight = torch.as_tensor(damping)
        if not bool((weight > 0).any()):
            gamma = None
        elif weight.numel() == 1:
            # A single weight goes over as a number. Some of deepinv's solvers
            # take a scalar regularization and not a map of one.
            gamma = 1.0 / float(weight.reshape(()))
        else:
            gamma = 1.0 / torch.broadcast_to(weight, reference.shape)
        return least_squares(
            A=linearization.matvec,
            AT=linearization.rmatvec,
            y=rhs,
            z=reference,
            gamma=gamma,
            solver=solver,
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
    """Solve every voxel's linearized system outright, all of them at once.

    Where nothing has mixed the voxels together the problem is a stack of
    small independent least-squares systems, one per voxel, and a batched
    factorization over the whole stack is the Levenberg-Marquardt step itself
    -- not a general linear solver, which is why this is here and a conjugate
    gradient is not.

    The damping enters as extra rows rather than as a ridge on the normal
    equations:

        [ J^T        ] d  =  [ rhs           ]
        [ sqrt(a) I  ]       [ sqrt(a) z     ]

    which is the same minimizer without squaring the condition number, and
    which has full column rank for any positive damping -- so there is no
    singular case to detect and none to work around.

    A complex contrast is two real measurements, because what is minimized is
    the squared modulus.

    Parameters
    ----------
    linearization : Linearization
        The derivative, which must carry its blocks.
    rhs : torch.Tensor
        ``(voxels, contrasts)``, ``y - F(x)``.
    damping : torch.Tensor
        ``(voxels, 1)`` or one number.
    reference : torch.Tensor
        ``(voxels, channels)``, what the damping pulls towards.

    Returns
    -------
    torch.Tensor
        ``(voxels, channels)``.

    Raises
    ------
    ValueError
        If the linearization carries no blocks, which is what an encoding
        operator in front of the model leaves.
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

    The iteratively regularized part of an iteratively regularized
    Gauss-Newton: early steps are held close to the starting guess, where the
    linearization is trustworthy, and the hold is released as the iterate
    settles. Every step is accepted.

    Attributes
    ----------
    initial : float
        The first weight.
    factor : float
        What it is multiplied by after each step.
    minimum : float
        The floor, so late steps stay regularized rather than becoming a bare
        Gauss-Newton on an ill-posed problem.
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
            raise ValueError(
                f"factor must lower the damping, got {self.factor}"
            )
        if self.minimum <= 0.0:
            raise ValueError(f"minimum must be positive, got {self.minimum}")

    def begin(self, curvature: torch.Tensor) -> torch.Tensor:
        """The weight to start at, given the curvature at the starting point."""
        return torch.as_tensor(
            self.initial, dtype=torch.float32, device=curvature.device
        )

    def judge(
        self, damping: torch.Tensor, gain: torch.Tensor, singular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take the step, and lower the weight for the next one."""
        return (
            torch.ones_like(gain, dtype=torch.bool),
            (damping * self.factor).clamp_min(self.minimum),
        )


@dataclass
class TrustRegion:
    """Each voxel its own damping, raised when a step does not pay.

    Levenberg-Marquardt. A step is taken only where it actually lowered the
    residual; where it did not, the damping goes up and the step is tried
    again shorter. That needs the voxels to be independent, so it applies to a
    model with no encoding in front of it.

    Attributes
    ----------
    tau : float
        Sets the first damping, as this times the largest curvature the
        starting point shows. Small where the guess is good.
    """

    tau: float = 1e-2
    #: Damping shortens the step; it does not pull it anywhere.
    anchored: bool = False
    #: Each voxel carries its own weight and retires on its own.
    per_voxel: bool = True
    _rising: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")

    def begin(self, curvature: torch.Tensor) -> torch.Tensor:
        """Scale the first damping to the curvature the starting point shows."""
        damping = self.tau * curvature.diagonal(dim1=-2, dim2=-1).amax(-1)
        self._rising = torch.full_like(damping, 2.0)
        return damping

    def judge(
        self, damping: torch.Tensor, gain: torch.Tensor, singular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept where the step paid, and move the damping either way.

        The Nielsen update: a step that predicted its own gain well lets the
        damping fall a long way, one that barely paid lets it fall a little,
        and a rejected step doubles it and doubles the doubling.
        """
        better = (gain > 0) & ~singular
        rising = self._rising
        lowered = damping * (1.0 - (2.0 * gain - 1.0).pow(3)).clamp_min(1.0 / 3.0)
        self._rising = torch.where(better, torch.full_like(rising, 2.0), rising * 2.0)
        return better, torch.where(better, lowered, damping * rising)

    def retire(self, keep: torch.Tensor) -> None:
        """Close up after the converged voxels have been written out."""
        self._rising = self._rising[keep]


# %% the loop


class GaussNewton:
    """Solve a nonlinear inverse problem by repeated linearization.

    Parameters
    ----------
    damping : Schedule or TrustRegion, optional
        :class:`Schedule` for an iteratively regularized Gauss-Newton,
        :class:`TrustRegion` for Levenberg-Marquardt.
    solve : callable, optional
        The inner solve, called as
        ``solve(linearization, rhs, damping, reference)``. Left out, it is
        :func:`direct` where the model stands alone and ``iterative()`` --
        deepinv's conjugate gradients -- where an encoding operator does not
        leave independent voxels. A closure around anything else is how a
        regularizer enters.
    max_iterations : int, optional
        Most outer steps to take.
    gradient_tolerance, step_tolerance : float, optional
        A voxel is done when its gradient is flat or its step is short
        relative to where it stands.

    Examples
    --------
    .. code-block:: python

        loop = GaussNewton(Schedule(initial=1.0))
        found = loop.minimize(operator, kspace, start, encoding=nufft)
        maps = operator.split(found.x)

    Raises
    ------
    ValueError
        If a per-voxel damping is asked for under an encoding operator, which
        has mixed the voxels together and left no independent rows.
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
            raise ValueError(
                f"max_iterations must be positive, got {max_iterations}"
            )
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
        operator:
            A :class:`~torchsim.recon.ModelOperator`.
        measured:
            The data. Without an encoding, ``(..., contrasts)`` -- the
            contrast images themselves. With one, whatever the encoding
            produces.
        initial:
            ``(..., channels)``, where to start. Build it with
            :meth:`~torchsim.recon.ModelOperator.initial`.
        encoding:
            Anything with ``A`` and ``A_adjoint`` -- a deepinv
            ``LinearPhysics``, an mri-nufft operator through its deepinv
            bridge. It sees the contrast axis first, at axis 1, which is the
            convention on that side of the line; the maps keep it last, which
            is the convention on this one.

        Returns
        -------
        Solution
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
                    ~flat, answer, live, x, data, residual, linearization,
                    cost, damping, self.damping,
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
                start[live] - x if getattr(self.damping, "anchored", False)
                else torch.zeros_like(x)
            )
            step = solve(linearization, -residual, _shaped(damping), reference)
            walked = _norm(step, encoding)
            short = walked < self.step_tolerance * (
                _norm(x, encoding) + self.step_tolerance
            )
            singular = ~_per_voxel(torch.isfinite(step), encoding, torch.all)

            candidate = x + step
            trial, trial_linearization = self._at(
                operator, candidate, data, encoding
            )
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
            residual = torch.where(
                _spread(better, trial.shape), trial, residual
            )
            linearization = _chosen(better, trial_linearization, linearization)
            cost = torch.where(better, trial_cost, cost)
            history.append(cost.sum())

            if retiring:
                staying = ~(short & ~singular)
                x, live, data, residual, linearization, cost, damping = _retire(
                    staying, answer, live, x, data, residual, linearization,
                    cost, damping, self.damping,
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
            x=answer.reshape(*shape, answer.shape[-1])
            if encoding is None
            else answer,
            cost=torch.stack(history),
            damping=(
                torch.stack(weights)
                if weights
                else torch.zeros(0, device=start.device)
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
                rmatvec=lambda v: operator.A_vjp(
                    x, _backward(encoding.A_adjoint(v))
                ),
            ),
        )


# %% private module subroutines


def _per_voxel(values: torch.Tensor, encoding: Any, reduce: Any) -> torch.Tensor:
    """Reduce over the channels where the voxels are independent, over all of
    them where an encoding operator has tied them together.

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
        rmatvec=lambda v: torch.einsum(
            "vpt,vt->vp", blocks.conj(), v.to(blocks.dtype)
        ).real,
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
    return _diagonal(
        torch.where(better[:, None, None], trial.blocks, standing.blocks)
    )


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
