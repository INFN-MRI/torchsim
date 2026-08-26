"""Fitting a model to every voxel at once, by damped Gauss-Newton."""

from __future__ import annotations

__all__ = ["NonlinearLeastSquares"]

from collections.abc import Mapping, Sequence
from typing import Any

import torch


class NonlinearLeastSquares(torch.nn.Module):
    """Levenberg-Marquardt, stepping every voxel together.

    Where a dictionary spans a grid whose size is the product of the parameter
    ranges, a nonlinear fit walks downhill from a starting guess and pays
    nothing for a third parameter beyond a third column of the Jacobian. What
    it gives up is the guarantee: it finds a local minimum of the residual,
    and which one depends on where it started.

    Voxels are independent, so they are not solved one after another. Every
    voxel takes its step in the same pass, carries its own damping, and
    accepts or rejects on its own; a voxel that has converged drops out and
    the remaining ones close up, so late iterations cost only what is left.

    Parameters
    ----------
    bounds : mapping, optional
        ``{name: (low, high)}``, either end ``None`` for unbounded. A bound is
        kept by fitting a transformed variable rather than by clipping a
        result, so no iterate ever leaves the interval and the bound cannot be
        sitting exactly on the answer. It also puts every parameter on the
        same scale whatever its units, which is what the damping term assumes.
    initial : mapping, optional
        ``{name: value}`` to start from, which must be strictly inside that
        property's bound. Without one, :meth:`fit` takes the median of the
        parameters the training set drew.
    max_iterations : int, optional
        Most steps any voxel takes.
    tau : float, optional
        Sets the first damping, as ``tau`` times the largest curvature the
        starting point shows. Small where the guess is good.
    gradient_tolerance, step_tolerance : float, optional
        A voxel is done when its gradient is flat or its step is short
        relative to where it stands.

    Notes
    -----
    **Equality constraints are written into the model, not declared here.**
    A constraint that fixes one parameter in terms of the others removes a
    degree of freedom, so the way to impose it is to not have that freedom:
    write the model in terms of the parameters that remain. For a fat-water
    fit where the two fractions must sum to one, make the fat fraction ``f``
    the only unknown and write water as ``1 - f`` inside the model. The
    constraint then holds identically at every iterate, rather than being
    restored after each one.

    The solve is iterative and runs without building a graph, so an estimate
    carries no gradient with respect to the measurement.

    Examples
    --------
    .. code-block:: python

        mapping = ParameterMapping(
            Acquisition(FSESimulator(ESP=5.0, TR=1800.0), flip=train),
            T1=(200.0, 3000.0),
            T2=(10.0, 300.0),
        )
        mapping.train(NonlinearLeastSquares(bounds={"T2": (1.0, 500.0)}))
        maps = mapping(volume)
    """

    def __init__(
        self,
        *,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        initial: Mapping[str, float] | None = None,
        max_iterations: int = 20,
        tau: float = 1e-2,
        gradient_tolerance: float = 1e-8,
        step_tolerance: float = 1e-8,
    ) -> None:
        super().__init__()
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.bounds = dict(bounds or {})
        self.initial = dict(initial or {})
        self.max_iterations = int(max_iterations)
        self.tau = float(tau)
        self.gradient_tolerance = float(gradient_tolerance)
        self.step_tolerance = float(step_tolerance)
        self.unknown: tuple[str, ...] = ()
        self.known: tuple[str, ...] = ()
        self._acquisition: Any = None
        self._subspace: Any = None
        self._start: torch.Tensor | None = None
        #: Steps the last solve took, and how many voxels ran out of them.
        self.iterations = 0
        self.unconverged = 0

    @property
    def fitted(self) -> bool:
        """Whether a model and a starting point are both in place."""
        return self._acquisition is not None and self._start is not None

    def bind(
        self,
        acquisition: Any,
        unknown: Sequence[str],
        known: Sequence[str] = (),
        subspace: Any = None,
    ) -> None:
        """Take the model to fit, and what its unknowns are called.

        A method that learns needs a training set; one that fits needs the
        model itself. :class:`~torchsim.ParameterMapping` calls this on any
        method that has it, before handing over the training set.

        Parameters
        ----------
        acquisition:
            The sequence being inverted, as an
            :class:`~torchsim.Acquisition`.
        unknown:
            The property names being estimated, in the order the maps come
            back in.
        known:
            The property names measured separately, in the order their columns
            arrive in.
        subspace:
            The compression the measurements have already been through, if
            any. Predictions go through it too, so the residual is taken where
            the measurement lives.

        Raises
        ------
        ValueError
            If a bound or a starting value names something that is not an
            unknown.
        """
        self._acquisition = acquisition
        self.unknown = tuple(unknown)
        self.known = tuple(known)
        self._subspace = subspace
        for label, given in (("bounds", self.bounds), ("initial", self.initial)):
            stray = set(given) - set(self.unknown)
            if stray:
                raise ValueError(
                    f"{label} names {sorted(stray)}, which "
                    f"{'is' if len(stray) == 1 else 'are'} not being estimated"
                )

    def fit(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> NonlinearLeastSquares:
        """Take a starting point from the parameters the training set drew.

        The training signals are not needed -- the model is what this fits
        against -- but the parameters say what range the answer is in, and
        their median is a better first guess than the middle of a bound.

        Parameters
        ----------
        signals:
            Ignored. Accepted so that a fit and a learned method are called
            the same way.
        parameters:
            ``(samples, parameters)``, in the order given to :meth:`bind`.
        known:
            Ignored, for the same reason.
        noise_std:
            Accepted and unused. A least-squares fit weights every contrast
            alike, which is what uniform noise implies.

        Returns
        -------
        NonlinearLeastSquares
            This estimator, ready to be called.

        Raises
        ------
        ValueError
            If a starting value is not strictly inside its bound.
        RuntimeError
            If no model has been bound.
        """
        del signals, known, noise_std
        if self._acquisition is None:
            raise RuntimeError(
                "no model to fit; give this estimator to a ParameterMapping, "
                "or call bind() with an Acquisition"
            )
        drawn = torch.as_tensor(parameters).reshape(-1, len(self.unknown))
        median = drawn.to(torch.float32).median(dim=0).values
        start = []
        for index, name in enumerate(self.unknown):
            stated = name in self.initial
            value = float(self.initial[name]) if stated else float(median[index])
            low, high = _bound_of(self.bounds, name)
            # The transformed variable is infinite at a bound, so a fit that
            # started there would have no direction to move in.
            if (low is not None and value <= low) or (
                high is not None and value >= high
            ):
                whose = (
                    "the starting value"
                    if stated
                    else "the median of the training range"
                )
                raise ValueError(
                    f"{name}: {whose}, {value:g}, is not strictly inside its "
                    f"bound ({low}, {high})"
                )
            start.append(torch.as_tensor(value, dtype=torch.float32))
        self._start = torch.stack(start)
        return self

    def forward(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the parameters that best explain each signal.

        Parameters
        ----------
        signals:
            ``(..., contrasts)``.
        known:
            ``(..., knowns)``, the properties measured separately.

        Returns
        -------
        torch.Tensor
            ``(..., parameters)``, in the order given to :meth:`bind`.

        Raises
        ------
        RuntimeError
            If no model has been bound or no starting point chosen.
        """
        if not self.fitted:
            raise RuntimeError(
                "the estimator has no model and starting point to fit from"
            )
        signals = torch.as_tensor(signals)
        shape = signals.shape[:-1]
        measured = signals.reshape(-1, signals.shape[-1])
        given = (
            torch.as_tensor(known).reshape(-1, len(self.known))
            if known is not None
            else None
        )
        found = self._solve(measured, given)
        return found.reshape(*shape, len(self.unknown))

    # -- the solve ----------------------------------------------------------

    @torch.no_grad()
    def _solve(
        self, measured: torch.Tensor, known: torch.Tensor | None
    ) -> torch.Tensor:
        """Levenberg-Marquardt over every voxel, compacting as they finish."""
        voxels = measured.shape[0]
        device = measured.device
        start = self._start.to(device)
        answer = start.expand(voxels, -1).clone()
        # Which of the original voxels each row still being solved belongs to.
        live = torch.arange(voxels, device=device)
        free = _to_free(answer, self.bounds, self.unknown, device)

        residual, jacobian = self._residual(free, measured, known)
        curvature = jacobian @ jacobian.mT
        gradient = (jacobian @ residual[..., None]).squeeze(-1)
        damping = self.tau * curvature.diagonal(dim1=-2, dim2=-1).amax(-1)
        rising = torch.full_like(damping, 2.0)
        cost = residual.square().sum(-1)

        self.iterations = 0
        for _ in range(self.max_iterations):
            flat = gradient.abs().amax(-1) < self.gradient_tolerance
            free, live, measured, known, residual, jacobian, curvature, \
                gradient, damping, rising, cost, answer = _retire(
                    ~flat, answer, live, free,
                    (measured, known, residual, jacobian, curvature,
                     gradient, damping, rising, cost),
                    self.bounds, self.unknown,
                )
            if free.shape[0] == 0:
                break
            self.iterations += 1

            step, singular = _step(curvature, gradient, damping)
            short = step.norm(dim=-1) < self.step_tolerance * (
                free.norm(dim=-1) + self.step_tolerance
            )
            # A singular normal-equation system is not a converged voxel; it
            # is one whose damping is too small to make the system definite.
            done = short & ~singular

            candidate = free + step
            trial, trial_jacobian = self._residual(candidate, measured, known)
            trial_cost = trial.square().sum(-1)
            predicted = (
                step * (damping[:, None] * step - gradient)
            ).sum(-1)
            gain = torch.where(
                predicted > 0, (cost - trial_cost) / predicted.clamp_min(_TINY),
                torch.zeros_like(predicted),
            )
            better = (gain > 0) & ~singular

            free = torch.where(better[:, None], candidate, free)
            residual = torch.where(better[:, None], trial, residual)
            jacobian = torch.where(better[:, None, None], trial_jacobian, jacobian)
            cost = torch.where(better, trial_cost, cost)
            curvature = jacobian @ jacobian.mT
            gradient = (jacobian @ residual[..., None]).squeeze(-1)
            damping = torch.where(
                better,
                damping * (1.0 - (2.0 * gain - 1.0).pow(3)).clamp_min(1.0 / 3.0),
                damping * rising,
            )
            rising = torch.where(better, torch.full_like(rising, 2.0), rising * 2.0)

            free, live, measured, known, residual, jacobian, curvature, \
                gradient, damping, rising, cost, answer = _retire(
                    ~done, answer, live, free,
                    (measured, known, residual, jacobian, curvature,
                     gradient, damping, rising, cost),
                    self.bounds, self.unknown,
                )
            if free.shape[0] == 0:
                break

        self.unconverged = int(free.shape[0])
        if self.unconverged:
            answer[live] = _to_natural(free, self.bounds, self.unknown)
        return answer

    def _residual(
        self,
        free: torch.Tensor,
        measured: torch.Tensor,
        known: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``predicted - measured`` and its derivative, both real.

        A complex contrast is two real residuals, because what is minimized is
        the squared modulus. The chain rule through the bound transformation
        rides on the Jacobian, so the solve never sees the natural parameter.
        """
        natural = _to_natural(free, self.bounds, self.unknown)
        given: dict[str, Any] = {
            name: natural[:, index] for index, name in enumerate(self.unknown)
        }
        if known is not None:
            given |= {
                name: known[:, index] for index, name in enumerate(self.known)
            }
        # A sequence of names keeps the parameter axis whatever its length,
        # which a single name would collapse.
        predicted, derivative = self._acquisition.jacobian(self.unknown, **given)
        if self._subspace is not None:
            predicted = self._subspace.project(predicted)
            derivative = self._subspace.project(derivative)

        residual = predicted - measured.to(predicted.dtype)
        slope = derivative * _widen(self.bounds, self.unknown, free)[..., None]
        if torch.is_complex(residual):
            residual = torch.cat((residual.real, residual.imag), dim=-1)
            slope = torch.cat((slope.real, slope.imag), dim=-1)
        return residual.to(torch.float32), slope.to(torch.float32)


# %% private module subroutines

#: Guards a division whose denominator the algorithm has already tested.
_TINY = torch.finfo(torch.float32).tiny


def _step(
    curvature: torch.Tensor, gradient: torch.Tensor, damping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``(JJ' + mu I) d = -g`` per voxel, flagging the ones that cannot.

    A Cholesky reports failure per voxel rather than raising, so one flat
    voxel in a volume does not stop the others.
    """
    size = curvature.shape[-1]
    eye = torch.eye(size, dtype=curvature.dtype, device=curvature.device)
    system = curvature + damping[:, None, None] * eye
    factor, info = torch.linalg.cholesky_ex(system)
    singular = info != 0
    safe = torch.where(singular[:, None, None], eye.expand_as(factor), factor)
    step = torch.cholesky_solve((-gradient)[..., None], safe).squeeze(-1)
    return torch.where(singular[:, None], torch.zeros_like(step), step), singular


def _retire(
    keep: torch.Tensor,
    answer: torch.Tensor,
    live: torch.Tensor,
    free: torch.Tensor,
    carried: tuple[Any, ...],
    bounds: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[Any, ...]:
    """Write finished voxels out and close the rest up.

    Late iterations then cost what is left rather than what was started with,
    which is most of the saving on a volume where the background converges at
    once.
    """
    finished = ~keep
    if bool(finished.any()):
        answer[live[finished]] = _to_natural(free[finished], bounds, names)
    if bool(keep.all()):
        return (free, live, *carried, answer)
    return (
        free[keep],
        live[keep],
        *(None if value is None else value[keep] for value in carried),
        answer,
    )


def _bound_of(
    bounds: Mapping[str, Any], name: str
) -> tuple[float | None, float | None]:
    """The pair for one property, absent meaning unbounded either way."""
    low, high = bounds.get(name, (None, None))
    if low is not None and high is not None and not high > low:
        raise ValueError(f"{name}: the bound ({low}, {high}) is not increasing")
    return low, high


def _to_natural(
    free: torch.Tensor, bounds: Mapping[str, Any], names: Sequence[str]
) -> torch.Tensor:
    """Map the unconstrained variables back to the properties they stand for."""
    if not bounds:
        return free
    columns = []
    for index, name in enumerate(names):
        low, high = _bound_of(bounds, name)
        value = free[:, index]
        if low is not None and high is not None:
            columns.append(low + (high - low) * torch.sigmoid(value))
        elif low is not None:
            columns.append(low + torch.nn.functional.softplus(value))
        elif high is not None:
            columns.append(high - torch.nn.functional.softplus(-value))
        else:
            columns.append(value)
    return torch.stack(columns, dim=-1)


def _to_free(
    natural: torch.Tensor,
    bounds: Mapping[str, Any],
    names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Map properties to the unconstrained variables that stand for them.

    The clamp guards the arithmetic at the very edge of the interval; a
    starting value actually sitting on a bound is refused where it is stated.
    """
    if not bounds:
        return natural.to(device)
    columns = []
    for index, name in enumerate(names):
        low, high = _bound_of(bounds, name)
        value = natural[:, index]
        if low is not None and high is not None:
            span = high - low
            inside = ((value - low) / span).clamp(_EDGE, 1.0 - _EDGE)
            columns.append(torch.log(inside) - torch.log1p(-inside))
        elif low is not None:
            columns.append(_softplus_inverse((value - low).clamp_min(_EDGE)))
        elif high is not None:
            columns.append(-_softplus_inverse((high - value).clamp_min(_EDGE)))
        else:
            columns.append(value)
    return torch.stack(columns, dim=-1).to(device)


def _widen(
    bounds: Mapping[str, Any], names: Sequence[str], free: torch.Tensor
) -> torch.Tensor:
    """The derivative of each property with respect to its free variable."""
    if not bounds:
        return torch.ones_like(free)
    columns = []
    for index, name in enumerate(names):
        low, high = _bound_of(bounds, name)
        value = free[:, index]
        if low is not None and high is not None:
            opened = torch.sigmoid(value)
            columns.append((high - low) * opened * (1.0 - opened))
        elif low is not None:
            columns.append(torch.sigmoid(value))
        elif high is not None:
            columns.append(torch.sigmoid(-value))
        else:
            columns.append(torch.ones_like(value))
    return torch.stack(columns, dim=-1)


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """``log(exp(x) - 1)``, written so a large ``x`` does not overflow."""
    return value + torch.log(-torch.expm1(-value))


#: Keeps the transform's argument finite at the very edge of an interval.
_EDGE = 1e-6
