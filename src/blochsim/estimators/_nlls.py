"""Fitting a model to every voxel at once, by damped Gauss-Newton."""

from __future__ import annotations

__all__ = ["NonlinearLeastSquares"]

from collections.abc import Mapping
from typing import Any

import torch

from .._bounds import bound_of
from ..optim._design import crlb
from ..recon import GaussNewton, ModelOperator, TrustRegion, direct
from ._mapping import Estimator


class NonlinearLeastSquares(Estimator):
    """Per-voxel Levenberg-Marquardt fit of the signal model.

    Fits the model itself, so each extra unknown adds one Jacobian column
    rather than a dictionary axis. The result is a local minimum that depends
    on the starting point. All voxels step together with their own trust
    region and leave the loop on convergence. A face on
    :class:`~blochsim.recon.GaussNewton`.

    Parameters
    ----------
    acquisition : Simulator, default=None
        The sequence being inverted, with every property that is neither
        unknown nor measured separately already fixed.
    bounds : mapping, default=None
        ``{name: (low, high)}`` in the property's units, either end ``None`` for
        unbounded; ``None`` leaves every unknown unbounded. Enforced by
        a variable transform, so iterates stay inside.
    initial : mapping, default=None
        ``{name: value}`` start, strictly inside the bound, in the property's
        units. ``None`` starts every unknown from the median of the training
        draw.
    loop : GaussNewton, default=None
        The solver. ``None`` is a :class:`~blochsim.recon.TrustRegion` over
        :func:`~blochsim.recon.direct`, 20 iterations.

    Notes
    -----
    An equality constraint is written into the model by removing the degree
    of freedom it fixes: for a fat-water fit with fractions summing to one,
    make the fat fraction ``f`` the only unknown and use ``1 - f`` for water.

    The estimate carries no gradient with respect to the measurement. The
    stated uncertainty is the inverse-Fisher standard error at the solution
    (a linearization, as :func:`~blochsim.crlb`), zero when the training noise
    is zero.

    Examples
    --------
    .. code-block:: python

        fit = NonlinearLeastSquares(
            FSESimulator(ESP=5.0, TR=1800.0, flip=train),
            bounds={"T2": (1.0, 500.0)},
        ).fit(T1=(200.0, 3000.0), T2=(10.0, 300.0))
        maps = fit.map(volume)

    A solve that needs more room, or a different one entirely:

    .. code-block:: python

        NonlinearLeastSquares(
            bounds={"T2": (1.0, 500.0)},
            loop=GaussNewton(TrustRegion(tau=1e-3), solve=direct,
                             max_iterations=60),
        )
    """

    def __init__(
        self,
        acquisition: Any = None,
        *,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        initial: Mapping[str, float] | None = None,
        loop: GaussNewton | None = None,
    ) -> None:
        super().__init__(acquisition)
        self.bounds = dict(bounds or {})
        self.initial = dict(initial or {})
        self.loop = (
            loop
            if loop is not None
            else GaussNewton(TrustRegion(), solve=direct, max_iterations=20)
        )
        self._start: torch.Tensor | None = None
        #: Steps the last solve took, and how many voxels ran out of them.
        self.iterations = 0
        self.unconverged = 0

    @property
    def fitted(self) -> bool:
        """Whether a model and a starting point are in place."""
        return self.acquisition is not None and self._start is not None

    def _fit_arrays(
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
        signals : torch.Tensor
            Ignored. Accepted so that a fit and a learned method are called
            the same way.
        parameters : torch.Tensor
            ``(samples, parameters)``, in the order given to :meth:`bind`.
        known : torch.Tensor, default=None
            Ignored, for the same reason.
        noise_std : float or torch.Tensor, default=0.0
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
        stray = {name for given in (self.bounds, self.initial) for name in given} - set(
            self.unknown
        )
        if stray:
            raise ValueError(
                f"bounds or initial name {sorted(stray)}, which "
                f"{'is' if len(stray) == 1 else 'are'} not being estimated"
            )
        if self.acquisition is None:
            raise RuntimeError(
                "no model to fit; give this estimator the acquisition it is "
                "inverting when it is made"
            )
        drawn = torch.as_tensor(parameters).reshape(-1, len(self.unknown))
        median = drawn.to(torch.float32).median(dim=0).values
        start = []
        for index, name in enumerate(self.unknown):
            stated = name in self.initial
            value = float(self.initial[name]) if stated else float(median[index])
            low, high = bound_of(self.bounds, name)
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

    def _estimate_arrays(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the parameters that best explain each signal.

        Parameters
        ----------
        signals : torch.Tensor
            ``(..., contrasts)``.
        known : torch.Tensor, default=None
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

    def _uncertainty_arrays(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor,
    ) -> torch.Tensor:
        """The standard error of the fit, from its own sensitivity.

        A least-squares solution moves with the noise by as much as the model
        is insensitive to the parameters there, which is the inverse Fisher
        matrix at the solution -- the same quantity :func:`~blochsim.crlb`
        bounds an unbiased estimate by, read at the answer rather than at a
        truth nobody has. It is a linearization about the solution, so it is
        the standard error a fit reports and holds where the residual is small
        enough that the model is straight across it.
        """
        scale = torch.as_tensor(
            self.noise_std, dtype=torch.float32, device=values.device
        )
        if not torch.any(scale != 0):
            return torch.zeros_like(values)
        acquisition = self.acquisition
        if known is not None:
            acquisition = acquisition.bind(
                **{name: known[:, index] for index, name in enumerate(self.known)}
            )
        _, sensitivity = acquisition.jacobian(
            list(self.unknown),
            **{name: values[..., i] for i, name in enumerate(self.unknown)},
        )
        if self.subspace is not None:
            sensitivity = self.subspace.project(sensitivity)
        # Per-voxel noise divides the sensitivity rather than the variance,
        # which is the same thing and takes a map as readily as a number.
        sensitivity = sensitivity / scale.reshape(
            *scale.shape, *((1,) * (sensitivity.ndim - scale.ndim))
        )
        return (
            crlb(sensitivity, noise_variance=1.0, singular="infinite")
            .clamp_min(0.0)
            .sqrt()
        )

    def _solve(
        self, measured: torch.Tensor, known: torch.Tensor | None
    ) -> torch.Tensor:
        """Levenberg-Marquardt over every voxel, compacting as they finish.

        The loop is :class:`~blochsim.recon.GaussNewton` under a per-voxel
        trust region, and the model, its bounds and its derivative are a
        :class:`~blochsim.recon.ModelOperator` -- the same two pieces a
        model-based reconstruction is built from, with nothing encoding the
        voxels together.
        """
        operator = self._operator(measured.shape[0], known)
        found = self.loop.minimize(operator, measured, self._at(operator, measured))
        self.iterations = found.iterations
        self.unconverged = found.unconverged
        maps = operator.split(found.x)
        return torch.stack([maps[name] for name in self.unknown], dim=-1)

    def _operator(self, voxels: int, known: torch.Tensor | None) -> ModelOperator:
        """The model to fit, with anything measured separately held on it."""
        acquisition = self.acquisition
        if known is not None:
            acquisition = acquisition.bind(
                **{name: known[:, index] for index, name in enumerate(self.known)}
            )
        return ModelOperator(
            acquisition,
            *self.unknown,
            bounds=self.bounds,
            amplitude=False,
            subspace=self.subspace,
        )

    def _at(self, operator: ModelOperator, measured: torch.Tensor) -> torch.Tensor:
        """Every voxel started from the same point, as variables to solve for."""
        start = operator.initial(
            measured.shape[:1],
            **{
                name: float(self._start[index])
                for index, name in enumerate(self.unknown)
            },
        )
        return start.to(measured.device)
