"""The estimator base class: stating a parameter-mapping problem, fitting and mapping.

What is unknown, over what range, and how noisy belongs to :meth:`Estimator.fit`,
so one estimator can be refitted over a different sampling::

    fitter = PERK(acquisition, n_features=1000)
    fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
    maps = fitter.map(volume)

Training signals are simulated from the acquisition. Alternatively
``fit(signals=..., parameters=...)`` takes arrays, and ``map`` then returns the
parameter columns as a tensor rather than named maps.
"""

from __future__ import annotations

__all__ = ["Estimator"]

from collections.abc import Mapping
from typing import Any

import torch

from .._subspace import Subspace
from ..model import Simulator

#: How many samples are simulated at once while a training set is drawn.
_CHUNK = 4096
#: Training draws, where nothing states a grid of its own.
_SAMPLES = 10_000


class Estimator(torch.nn.Module):
    """Base class of the mapping methods.

    :meth:`fit` names the unknowns and draws the training set from
    :attr:`acquisition`; :meth:`map` returns one named map per unknown. A new
    method subclasses this and implements ``_fit_arrays`` and
    ``_estimate_arrays``, which see signals and parameters as matrices.

    Parameters
    ----------
    acquisition : Simulator, default=None
        The sequence being inverted, with every property that is neither
        unknown nor measured separately already fixed. Omit to fit from signals.

    Attributes
    ----------
    acquisition : Simulator or None
        The sequence being inverted.
    noise_std : float or torch.Tensor
        Training noise standard deviation passed to :meth:`fit`.
    rank : int or None
        Subspace rank passed to :meth:`fit`.
    seed : int or None
        Training-draw seed passed to :meth:`fit`.
    """

    def __init__(self, acquisition: Simulator | None = None) -> None:
        super().__init__()
        self.acquisition = acquisition
        self.noise_std: float | torch.Tensor = 0.0
        self.rank: int | None = None
        self.seed: int | None = None
        self._unknown: dict[str, Any] = {}
        self._known: dict[str, Any] = {}
        self._subspace: Subspace | None = None
        self._fitted = False

    @property
    def unknown(self) -> tuple[str, ...]:
        """The properties being estimated, in the order the maps come back."""
        return tuple(self._unknown)

    @property
    def known(self) -> tuple[str, ...]:
        """The properties measured separately, in the order they are given."""
        return tuple(self._known)

    @property
    def subspace(self) -> Subspace | None:
        """The temporal basis in use, or ``None`` when working on the full contrasts."""
        return self._subspace

    @property
    def trained(self) -> bool:
        """Whether the estimator can map.

        True after :meth:`fit`, or when the subclass holds a fitted state
        otherwise, such as a dictionary given to the constructor.
        """
        return self._fitted or bool(getattr(self, "fitted", False))

    # -- fitting ---------------------------------------------------------

    def fit(
        self,
        unknown: Mapping[str, Any] | None = None,
        *,
        known: Mapping[str, Any] | None = None,
        noise_std: float | torch.Tensor = 0.0,
        rank: int | None = None,
        subspace: Subspace | None = None,
        seed: int | None = None,
        samples: int | None = None,
        chunk: int = _CHUNK,
        signals: torch.Tensor | None = None,
        parameters: torch.Tensor | None = None,
        **ranges: Any,
    ) -> Any:
        """Fit the estimator over a sampling of the tissue it will meet.

        Training signals are simulated from :attr:`acquisition` unless
        ``signals`` is given.

        Parameters
        ----------
        unknown : mapping, default=None
            Properties to estimate, ``{name: (low, high)}`` to draw uniformly
            or ``{name: values}`` to use as given. Order sets the order of the
            maps. Equivalent to the keyword form; use it for a name that
            collides with a keyword of ``fit``.
        known : mapping or torch.Tensor, default=None
            Properties measured separately, ``{name: range}`` or
            ``{name: values}``; pass the measured maps again to :meth:`map`.
            A tensor of columns when ``signals`` is given.
        noise_std : float or torch.Tensor, default=0.0
            Standard deviation of the noise added to training signals, in
            signal units (a fully relaxed voxel is 1).
        rank : int, default=None
            Fit a temporal :class:`~blochsim.Subspace` of this rank to the
            training signals and work in it. Not with ``subspace``.
        subspace : Subspace, default=None
            Work in a basis fitted elsewhere. Not with ``rank``.
        seed : int, default=None
            Seed of the training draw.
        samples : int, default=None
            Number of tissues to draw. Defaults to the length of any
            properties given as values, else 10000.
        chunk : int, default=4096
            Samples simulated at a time.
        signals : torch.Tensor, default=None
            Training signals ``(samples, contrasts)``, used instead of
            simulating. :meth:`map` then returns a tensor unless names were
            given.
        parameters : torch.Tensor, default=None
            Unknowns ``(samples, unknowns)`` accompanying ``signals``.
        **ranges
            Unknown properties as keywords, as for ``unknown``.

        Returns
        -------
        Estimator
            This estimator, fitted.

        Raises
        ------
        RuntimeError
            If there are no signals and no acquisition to simulate them.
        ValueError
            If a name is both unknown and known, a range is not a pair,
            ``rank`` is not positive, or both ``rank`` and ``subspace`` are
            given.
        """
        properties = {**(unknown or {}), **ranges}
        measured = known if isinstance(known, Mapping) else None
        overlap = set(properties) & set(measured or ())
        if overlap:
            raise ValueError(f"{sorted(overlap)} cannot be both unknown and known")
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank is not None and subspace is not None:
            raise ValueError(
                "give a rank to fit a basis here, or a subspace to work in one "
                "fitted elsewhere -- not both"
            )
        self._subspace = subspace
        if properties:
            self._unknown = dict(properties)
        if measured is not None:
            self._known = dict(measured)
        self.noise_std = noise_std
        self.rank = rank
        self.seed = seed

        if signals is None:
            if self.acquisition is None:
                raise RuntimeError(
                    "nothing to fit from: give this estimator the acquisition "
                    "it is inverting, or hand fit() the signals directly"
                )
            if not self._unknown:
                raise ValueError("name at least one property to estimate")
            if samples is None:
                samples = self._stated_samples() or _SAMPLES
            self._fit_simulated(samples, chunk=chunk, rank=rank, noise_std=noise_std)
        else:
            given = known if isinstance(known, torch.Tensor) else None
            if self._subspace is not None:
                signals = self._subspace.project(torch.as_tensor(signals))
            self._fit_arrays(signals, parameters, given, noise_std=noise_std)
        self._fitted = True
        return self

    def _fit_simulated(
        self,
        samples: int,
        *,
        chunk: int,
        rank: int | None,
        noise_std: float | torch.Tensor,
    ) -> None:
        """Simulate a training set, fit a basis over it if asked, then fit to it.

        The signals live only for the length of this call. They are what a
        basis is estimated from and what the method is fitted to, and nothing
        holds them afterwards.
        """
        signals, parameters, known = self.training_set(samples, chunk=chunk)
        if rank is not None:
            self._subspace = Subspace.fit(signals, rank)
        if self._subspace is not None:
            signals = self._subspace.project(signals)
        self._fit_arrays(signals, parameters, known, noise_std=noise_std)

    # -- the training set ------------------------------------------------

    def _stated_samples(self) -> int | None:
        """How many samples the given value arrays already fix, if any.

        Grids stated outright -- which is how a lookup table is built -- carry
        their own length, and repeating it as ``samples`` would be one more
        place for it to disagree. ``None`` where everything is a range to draw
        from, which fixes nothing.
        """
        lengths = {
            torch.as_tensor(spec).reshape(-1).numel()
            for spec in (*self._unknown.values(), *self._known.values())
            if not (isinstance(spec, (tuple, list)) and len(spec) == 2)
        }
        lengths.discard(1)
        if len(lengths) > 1:
            raise ValueError(
                f"properties given as values disagree on length: {sorted(lengths)}"
            )
        if not lengths:
            return None
        return int(lengths.pop())

    def training_set(
        self, samples: int, *, chunk: int = _CHUNK
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulate a training set from the properties given to :meth:`fit`.

        Parameters
        ----------
        samples : int
            Number of tissues to draw.
        chunk : int, default=4096
            Samples simulated at a time.

        Returns
        -------
        tuple
            ``(signals, parameters, known)``; ``known`` is ``None`` when
            nothing is measured separately.

        Raises
        ------
        RuntimeError
            If there is no acquisition, or no unknowns have been named.
        ValueError
            If ``samples`` or ``chunk`` is not positive, or a property given as
            values has the wrong length.
        """
        if self.acquisition is None:
            raise RuntimeError(
                "this estimator has no acquisition to simulate from; give one "
                "when it is made, or hand fit() the signals directly"
            )
        if not self._unknown:
            raise RuntimeError(
                "this estimator has not been told what is unknown; name the "
                "properties when fit() is called"
            )
        if samples < 1:
            raise ValueError(f"samples must be positive, got {samples}")
        if chunk < 1:
            raise ValueError(f"chunk must be positive, got {chunk}")
        drawn, parameters, known = self._draw_parameters(samples)
        pieces = []
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            given = {name: value[start:stop] for name, value in drawn.items()}
            pieces.append(torch.as_tensor(self.acquisition.simulate(**given)))
        signals = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        return signals, parameters, known

    def _draw_parameters(
        self, samples: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
        """The tissue a training set covers, before anything is simulated."""
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        drawn = {
            name: _draw(name, spec, samples, generator)
            for name, spec in (*self._unknown.items(), *self._known.items())
        }
        parameters = torch.stack([drawn[name] for name in self._unknown], dim=-1)
        known = (
            torch.stack([drawn[name] for name in self._known], dim=-1)
            if self._known
            else None
        )
        return drawn, parameters, known

    # -- mapping ---------------------------------------------------------

    def map(
        self,
        volume: Any,
        known: Any = None,
        *,
        uncertainty: bool = False,
    ) -> Any:
        """Estimate the tissue parameters of a measured volume.

        Parameters
        ----------
        volume : array-like
            Measurement ``(..., contrasts)``; every leading axis is a voxel
            axis and is kept on the maps.
        known : mapping or torch.Tensor, default=None
            Measured maps under the names given to :meth:`fit`, each
            broadcast to the voxel shape.
        uncertainty : bool, default=False
            Also return the uncertainty of each map; see
            :meth:`uncertainty_of`.

        Returns
        -------
        dict or torch.Tensor
            ``{name: map}`` with the shape of ``volume`` without its contrast
            axis; a tensor of parameter columns if the estimator was fitted
            from arrays without names. With ``uncertainty=True``, a
            ``(maps, uncertainties)`` pair.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        NotImplementedError
            If the method states no uncertainty.
        ValueError
            If a measured map is missing or has the wrong voxel count.
        """
        if not self._unknown:
            values = self._estimate_arrays(volume, known)
            if not uncertainty:
                return values
            signals = torch.as_tensor(volume)
            return values, self._spread(
                signals.reshape(-1, signals.shape[-1]), known, values
            )
        return self._named(volume, known, project=True, uncertainty=uncertainty)

    def forward(
        self,
        volume: Any,
        known: Any = None,
        *,
        uncertainty: bool = False,
    ) -> Any:
        """Alias of :meth:`map`."""
        return self.map(volume, known, uncertainty=uncertainty)

    def uncertainty_of(self, volume: Any, known: Any = None) -> Any:
        """Uncertainty of each map, as a standard deviation in the map's units.

        What it measures depends on the method. :class:`~blochsim.PERK` states
        the learned total error, prior and noise together, rescaled to a
        standard deviation for a Gaussian error.
        :class:`~blochsim.NonlinearLeastSquares` states the standard error from
        the inverse Fisher matrix at the solution, which is zero when the
        training noise is zero. Compare with :func:`~blochsim.crlb`.

        Parameters
        ----------
        volume : array-like
            Measurement ``(..., contrasts)``, as for :meth:`map`.
        known : mapping or torch.Tensor, default=None
            Measured maps, as for :meth:`map`.

        Returns
        -------
        dict or torch.Tensor
            One uncertainty per unknown, shaped like the maps.

        Raises
        ------
        NotImplementedError
            If the method states no uncertainty (:class:`~blochsim.DictionaryMatcher`,
            :class:`~blochsim.LookupTable`).
        """
        _, spread = self.map(volume, known, uncertainty=True)
        return spread

    def from_coefficients(
        self, coefficients: Any, known: Mapping[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        """Map coefficients already in this estimator's basis.

        Unlike :meth:`map`, the input is not projected onto :attr:`subspace`
        first.

        Parameters
        ----------
        coefficients : array-like
            Subspace coefficients ``(..., rank)``; leading axes are voxel axes.
        known : mapping, default=None
            Measured maps, as for :meth:`map`.

        Returns
        -------
        dict
            ``{name: map}``, shaped like ``coefficients`` without its last
            axis.

        Raises
        ------
        RuntimeError
            If the estimator is unfitted or has no :attr:`subspace`.
        ValueError
            If the last axis is not the basis rank.
        """
        if self._subspace is None:
            raise RuntimeError(
                "this estimator has no subspace, so there are no coefficients "
                "to read; state a rank when it is fitted"
            )
        values = torch.as_tensor(coefficients)
        if values.shape[-1] != self._subspace.rank:
            raise ValueError(
                f"the basis has rank {self._subspace.rank}, "
                f"got {values.shape[-1]} coefficients"
            )
        return self._named(values, known, project=False)

    def _named(
        self,
        volume: Any,
        known: Mapping[str, Any] | None,
        *,
        project: bool,
        uncertainty: bool = False,
    ) -> Any:
        """Fill in the maps, from contrasts or from coefficients."""
        if not self.trained:
            raise RuntimeError("an estimator must be fitted before it maps")
        signals = torch.as_tensor(volume)
        # Where the maps belong is where the volume is, not where the method
        # happens to have been fitted.
        home = signals.device
        shape = signals.shape[:-1]
        voxels = int(torch.tensor(shape).prod()) if shape else 1
        signals = signals.reshape(voxels, signals.shape[-1])
        columns = _known_matrix(known, self._known, voxels)
        seen = signals
        if project and self._subspace is not None:
            seen = self._subspace.project(signals)
        values = self._estimate_arrays(seen, columns)

        def laid_out(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                name: matrix[..., column].reshape(shape).to(home)
                for column, name in enumerate(self._unknown)
            }

        maps = laid_out(values) | {
            name: value.reshape(shape).to(home)
            for name, value in self._extra_maps(signals, values).items()
        }
        if not uncertainty:
            return maps
        return maps, laid_out(self._spread(seen, columns, values, measured=signals))

    def _spread(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The standard deviations, with the noise the fit was told about."""
        spread = self._uncertainty_arrays(
            signals, known, values, measured=signals if measured is None else measured
        )
        if spread is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not state an uncertainty. A method "
                f"whose answer is a grid point moves in steps rather than "
                f"smoothly, so propagating the noise through it gives zero "
                f"rather than an error bar; PERK and NonlinearLeastSquares do "
                f"state one, and blochsim.crlb bounds what any unbiased method "
                f"could reach on this sequence."
            )
        return spread

    # -- what a subclass supplies ----------------------------------------

    def _fit_arrays(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None,
        *,
        noise_std: float | torch.Tensor,
    ) -> None:
        """Fit to plain tensors."""
        raise NotImplementedError

    def _estimate_arrays(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate ``(..., unknowns)`` from plain tensors."""
        raise NotImplementedError

    def _extra_maps(
        self, measured: torch.Tensor, values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Maps a method answers for besides the properties it was asked for."""
        return {}

    def _uncertainty_arrays(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor,
    ) -> torch.Tensor | None:
        """One standard deviation per unknown, or ``None`` where there is none.

        ``signals`` are as :meth:`_estimate_arrays` saw them -- projected into
        the basis where there is one -- and ``values`` is what it answered, so
        a method that linearizes about its own estimate has it to hand.
        ``measured`` is the measurement before any basis was applied, which is
        the domain the noise is independent in and carries whether the data is
        complex at all.
        """
        return None


# %% private module subroutines


def _draw(
    name: str, spec: Any, samples: int, generator: torch.Generator
) -> torch.Tensor:
    """One property's training values: a range to draw over, or values given."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        low, high = (float(value) for value in spec)
        if not high > low:
            raise ValueError(f"{name}: the range {spec} is not increasing")
        return low + (high - low) * torch.rand(samples, generator=generator)
    values = torch.as_tensor(spec, dtype=torch.float32).reshape(-1)
    if values.numel() == 1:
        return values.expand(samples).clone()
    if values.numel() != samples:
        raise ValueError(
            f"{name}: {values.numel()} values for {samples} training samples"
        )
    return values


def _known_matrix(
    given: Mapping[str, Any] | None,
    names: Mapping[str, Any],
    voxels: int,
) -> torch.Tensor | None:
    """The measured maps as ``(voxels, known)``, in the estimator's own order."""
    if not names:
        return None
    if given is None:
        raise ValueError(f"this estimator needs {sorted(names)} measured")
    if isinstance(given, torch.Tensor):
        return given
    columns = []
    for name in names:
        if name not in given:
            raise ValueError(f"{name} was not given")
        value = torch.as_tensor(given[name], dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            value = value.expand(voxels)
        elif value.numel() != voxels:
            raise ValueError(f"{name}: {value.numel()} values for {voxels} voxels")
        columns.append(value)
    return torch.stack(columns, dim=-1)
