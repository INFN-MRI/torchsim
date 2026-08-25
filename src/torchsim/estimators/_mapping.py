"""Stating a parameter-mapping problem, and filling it in.

A mapping problem is the inverse of a design problem and is stated the same
way. :class:`~torchsim.Acquisition` says what the scanner plays and what it is
being asked about; the mapping says which tissue properties are unknown and
over what range, which are measured separately, and how noisy the measurement
is. A **method** -- a kernel regression, a dictionary match -- fills it in.

What a method is, is small: something that can be fitted to simulated signals
and then called on measured ones. That is the whole of :class:`Estimator`, and
it is why swapping one for another is a one-word change rather than a rewrite.

The training set comes from the same simulator the design side optimizes, so
what an estimator is trained on cannot drift from what the scanner will
produce.
"""

from __future__ import annotations

__all__ = ["Estimator", "ParameterMapping"]

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

from .._subspace import Subspace
from ..model import Acquisition

#: How many samples are simulated at once while a training set is drawn.
_CHUNK = 4096


@runtime_checkable
class Estimator(Protocol):
    """What a mapping method has to be able to do.

    Two things: be fitted to simulated signals whose parameters are known, and
    be called on measured ones. :class:`~torchsim.PERK` and
    :class:`~torchsim.DictionaryMatcher` both are one, and so is anything a
    user writes.
    """

    def fit(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> Any:
        """Fit to ``(samples, contrasts)`` signals and ``(samples, parameters)``."""

    def __call__(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate ``(..., parameters)`` from ``(..., contrasts)``."""


class ParameterMapping:
    """A quantitative mapping: what is unknown, what is known, how noisy.

    Parameters
    ----------
    acquisition:
        The sequence being inverted, with any tissue property that is neither
        unknown nor measured already fixed on it.
    known:
        Properties measured separately, as ``{name: range}`` or
        ``{name: values}``. They are drawn over during training and reach the
        simulator, so the training signals carry their effect, and the
        measured maps are given again at :meth:`__call__`.
    noise_std:
        Standard deviation of the noise added to the training signals, in the
        units the signal is in -- a fully relaxed voxel is 1. This is what
        teaches a method how far to trust a measurement, so it should be the
        noise the scan actually has.
    rank:
        Fit a temporal :class:`~torchsim.Subspace` of this rank to the
        training signals and work in it. Every contrast dropped is arithmetic
        neither training nor mapping has to do; :attr:`subspace` reports what
        the compression kept.
    seed:
        Seed for the training draw.
    unknown:
        The properties being estimated, as ``{name: (low, high)}`` to draw
        uniformly over, or an array of values to use as given. The order is
        the order of the maps that come back.

    Examples
    --------
    .. code-block:: python

        mapping = ParameterMapping(
            Acquisition(MRFSimulator(TR=10.0, TI=20.0), flip=schedule),
            T1=(200.0, 3000.0),
            T2=(10.0, 300.0),
            noise_std=0.01,
            rank=16,
        )
        mapping.train(PERK(n_features=1000), samples=100_000)
        maps = mapping(volume)      # {"T1": ..., "T2": ...}

    Raises
    ------
    ValueError
        If nothing is named unknown, if a range is not a pair, or if a name is
        both unknown and known.
    """

    def __init__(
        self,
        acquisition: Acquisition,
        *,
        known: Mapping[str, Any] | None = None,
        noise_std: float | torch.Tensor = 0.0,
        rank: int | None = None,
        seed: int | None = None,
        **unknown: Any,
    ) -> None:
        if not unknown:
            raise ValueError("name at least one property to estimate")
        overlap = set(unknown) & set(known or ())
        if overlap:
            raise ValueError(
                f"{sorted(overlap)} cannot be both unknown and known"
            )
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
        self.acquisition = acquisition
        self.noise_std = noise_std
        self.rank = rank
        self.seed = seed
        self._unknown = dict(unknown)
        self._known = dict(known or {})
        self._subspace: Subspace | None = None
        self._method: Estimator | None = None

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
        """The temporal basis in use, or ``None`` if the mapping works in full."""
        return self._subspace

    @property
    def method(self) -> Estimator | None:
        """The fitted method, or ``None`` before :meth:`train`."""
        return self._method

    @property
    def trained(self) -> bool:
        """Whether a method has been fitted."""
        return self._method is not None

    def training_set(
        self, samples: int, *, chunk: int = _CHUNK
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulate a training set: signals, unknowns, and knowns.

        Memory is ``samples`` by contrasts, because the signals are held once
        rather than resimulated for each pass a method makes over them.

        Parameters
        ----------
        samples:
            How many tissues to draw.
        chunk:
            How many to simulate at a time.

        Returns
        -------
        tuple
            ``(signals, parameters, known)``, the last ``None`` when nothing
            is measured separately.

        Raises
        ------
        ValueError
            If ``samples`` or ``chunk`` is not positive, or if an array given
            for a property has a different length.
        """
        if samples < 1:
            raise ValueError(f"samples must be positive, got {samples}")
        if chunk < 1:
            raise ValueError(f"chunk must be positive, got {chunk}")
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        drawn = {
            name: _draw(name, spec, samples, generator)
            for name, spec in (*self._unknown.items(), *self._known.items())
        }
        pieces = []
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            given = {name: value[start:stop] for name, value in drawn.items()}
            pieces.append(
                torch.as_tensor(self.acquisition.simulate(**given))
            )
        signals = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        parameters = torch.stack(
            [drawn[name] for name in self._unknown], dim=-1
        )
        known = (
            torch.stack([drawn[name] for name in self._known], dim=-1)
            if self._known
            else None
        )
        return signals, parameters, known

    def train(
        self, method: Estimator, *, samples: int = 10_000, chunk: int = _CHUNK
    ) -> ParameterMapping:
        """Draw a training set and fit ``method`` to it.

        Parameters
        ----------
        method:
            Anything satisfying :class:`Estimator`.
        samples:
            How many tissues to train over.
        chunk:
            How many to simulate at a time.

        Returns
        -------
        ParameterMapping
            This mapping, trained.
        """
        signals, parameters, known = self.training_set(samples, chunk=chunk)
        if self.rank is not None:
            self._subspace = Subspace.fit(signals, self.rank)
            signals = self._subspace.project(signals)
        method.fit(signals, parameters, known, noise_std=self.noise_std)
        self._method = method
        return self

    def __call__(
        self, volume: Any, known: Mapping[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        """Map a measured volume, returning one named map per unknown.

        Parameters
        ----------
        volume:
            ``(..., contrasts)``. Every leading axis is the voxel axis and
            comes back on the maps unchanged.
        known:
            The measured maps, under the names the mapping was given. Each is
            broadcast to the voxel shape.

        Returns
        -------
        dict
            ``{name: map}``, each shaped like ``volume`` without its contrast
            axis.

        Raises
        ------
        RuntimeError
            If the mapping has not been trained.
        ValueError
            If a measured map is missing, or has the wrong voxel count.
        """
        if self._method is None:
            raise RuntimeError("a mapping must be trained before it maps")
        signals = torch.as_tensor(volume)
        # Where the maps belong is where the volume is, not where the method
        # happens to have been fitted.
        home = signals.device
        shape = signals.shape[:-1]
        voxels = int(torch.tensor(shape).prod()) if shape else 1
        signals = signals.reshape(voxels, signals.shape[-1])
        if self._subspace is not None:
            signals = self._subspace.project(signals)
        values = self._method(signals, _known_matrix(known, self._known, voxels))
        return {
            name: values[..., column].reshape(shape).to(home)
            for column, name in enumerate(self._unknown)
        }


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
    """The measured maps as ``(voxels, known)``, in the mapping's own order."""
    if not names:
        return None
    if given is None:
        raise ValueError(f"this mapping needs {sorted(names)} measured")
    columns = []
    for name in names:
        if name not in given:
            raise ValueError(f"{name} was not given")
        value = torch.as_tensor(given[name], dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            value = value.expand(voxels)
        elif value.numel() != voxels:
            raise ValueError(
                f"{name}: {value.numel()} values for {voxels} voxels"
            )
        columns.append(value)
    return torch.stack(columns, dim=-1)
