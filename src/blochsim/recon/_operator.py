"""The signal model as an operator a reconstruction can compose with.

A physics-based reconstruction writes the forward operator as a chain
``F = P . F . C . M``: sampling, Fourier encoding, coil sensitivities and the
signal model ``M``. :class:`ModelOperator` supplies ``M`` for any simulator and
composes with an external encoding (mri-nufft, deepinv) rather than
reimplementing it.

The model is voxel-diagonal, so ``J d`` is one forward-mode pass and ``J^H v``
one reverse-mode pass over the volume; no Jacobian is built except by
:meth:`ModelOperator.jacobian`. Proton density and receive phase are carried as
one complex amplitude multiplying the model output; its two Jacobian columns
are the output times one and times i.
"""

from __future__ import annotations

__all__ = ["ModelOperator"]

from collections.abc import Mapping, Sequence
from copy import copy as shallow_copy
from typing import Any

import torch

from .._bounds import bound_of, to_free, to_natural, widen
from .._derivative import forward_mode
from .._execution import PER_VOXEL_CROSSOVER, per_voxel

#: What the amplitude occupies, when it is carried: real part then imaginary.
_AMPLITUDE = ("amplitude.real", "amplitude.imag")


class ModelOperator(torch.nn.Module):
    """A signal model over parameter maps, with its derivatives.

    Maps are stacked on the last axis: ``(..., channels)`` in,
    ``(..., contrasts)`` out, every leading axis a voxel axis. Only
    :meth:`physics` moves the channel axis to the front, as deepinv expects.

    Parameters
    ----------
    acquisition : Simulator
        Sequence being inverted, with every property not solved for already
        bound. A property bound as a map is one value per voxel.
    *unknown : str
        Names of the properties solved for, in channel order. At least one.
    bounds : mapping, default=None
        ``{name: (low, high)}`` in the property's units; either end may be
        ``None``, and ``None`` leaves every unknown unbounded. Enforced by
        solving for a transformed variable, so no iterate leaves the bounds.
    scale : mapping, default=None
        ``{name: value}``, step scale of an unbounded parameter in its own
        units; unlisted ones use 1. Bounded parameters are scaled by their
        interval.
    amplitude : bool, default=True
        Carry a complex amplitude multiplying the model output, as two extra
        channels (real, imaginary).
    subspace : Subspace, default=None
        Basis the prediction is projected onto, to solve in the temporal
        basis rather than in the contrasts.

    Raises
    ------
    ValueError
        If nothing is unknown, a bound or scale names a property that is not
        unknown, a bound is not increasing, or a scale is not positive.

    Notes
    -----
    The operator follows the device of the maps it is called with; ``to()``
    moves nothing.

    An equality constraint is imposed by writing the model on the remaining
    parameters. For fat-water fractions that sum to one, make the fat fraction
    the only unknown and write water as ``1 - f`` inside the model; the
    constraint then holds exactly at every iterate.

    Examples
    --------
    .. code-block:: python

        operator = ModelOperator(
            MultiEchoSimulator(TE=echo_times, T1=1000.0),
            "T2",
            bounds={"T2": (10.0, 300.0)},
        )
        maps = operator.initial((128, 128), T2=80.0)
        images = operator.A(maps)          # (128, 128, echoes), complex
    """

    def __init__(
        self,
        acquisition: Any,
        *unknown: str,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        scale: Mapping[str, float] | None = None,
        amplitude: bool = True,
        subspace: Any = None,
    ) -> None:
        super().__init__()
        if not unknown:
            raise ValueError("name at least one property to solve for")
        self.acquisition = acquisition
        self.unknown = tuple(unknown)
        self.bounds = dict(bounds or {})
        self.amplitude = bool(amplitude)
        self.subspace = subspace
        for label, given in (("bounds", self.bounds), ("scale", scale or {})):
            stray = set(given) - set(self.unknown)
            if stray:
                raise ValueError(
                    f"{label} names {sorted(stray)}, which "
                    f"{'is' if len(stray) == 1 else 'are'} not being solved for"
                )
        for name in self.unknown:
            bound_of(self.bounds, name)
        self.scale = tuple(float((scale or {}).get(name, 1.0)) for name in self.unknown)
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("every scale must be positive")
        self._elsewhere: dict[str, Any] = {}

    # -- what the maps are --------------------------------------------------

    @property
    def channels(self) -> int:
        """Number of map channels of ``x``."""
        return len(self.unknown) + 2 * self.amplitude

    @property
    def names(self) -> tuple[str, ...]:
        """Channel names of ``x``, in order."""
        return self.unknown + (_AMPLITUDE if self.amplitude else ())

    def split(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert solved-for variables to properties in their own units.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``, the variables.

        Returns
        -------
        dict
            One entry per unknown, inside its bound, plus complex
            ``"amplitude"`` where carried.
        """
        given, rho = self._named(x)
        return given if rho is None else {**given, "amplitude": rho}

    def initial(self, shape: Sequence[int] = (), **values: Any) -> torch.Tensor:
        """Build starting variables from property values.

        Inverse of :meth:`split`.

        Parameters
        ----------
        shape : sequence of int, default=()
            Voxel shape. Empty gives one voxel.
        **values : float or array-like
            ``{name: value}`` in the property's units, scalar or broadcastable
            to ``shape``. An unknown left out starts at the middle of its
            two-sided bound. ``amplitude`` (complex) defaults to 1.

        Returns
        -------
        torch.Tensor
            ``(*shape, channels)``.

        Raises
        ------
        ValueError
            If an unknown with no two-sided bound is left out, or a value is
            not strictly inside its bound.
        """
        stray = set(values) - set(self.unknown) - {"amplitude"}
        if stray:
            raise ValueError(f"{sorted(stray)} is not a channel of this operator")
        columns = []
        for name in self.unknown:
            low, high = bound_of(self.bounds, name)
            if name in values:
                start = torch.as_tensor(values[name], dtype=torch.float32)
            elif low is not None and high is not None:
                start = torch.tensor(0.5 * (low + high))
            else:
                raise ValueError(
                    f"{name} has no two-sided bound, so give it a starting value"
                )
            start = torch.broadcast_to(start, tuple(shape)).to(torch.float32)
            outside = (low is not None and bool((start <= low).any())) or (
                high is not None and bool((start >= high).any())
            )
            if outside:
                raise ValueError(
                    f"{name}: {start.min() if low is not None else start.max()} "
                    f"is not strictly inside its bound ({low}, {high})"
                )
            columns.append(start)
        natural = torch.stack(columns, dim=-1)
        scale = torch.tensor(self.scale)
        free = to_free(natural, self.bounds, self.unknown) / scale
        if not self.amplitude:
            return free
        start = torch.as_tensor(values.get("amplitude", 1.0 + 0.0j))
        start = torch.broadcast_to(start, tuple(shape)).to(torch.complex64)
        return torch.cat((free, start.real[..., None], start.imag[..., None]), dim=-1)

    # -- the operator -------------------------------------------------------

    def select(self, keep: torch.Tensor) -> ModelOperator:
        """Restrict the operator to a subset of the voxels.

        Properties bound as per-voxel maps are indexed by ``keep``; other
        bound values are unchanged.

        Parameters
        ----------
        keep : torch.Tensor
            Index or boolean mask over the voxel axis.

        Returns
        -------
        ModelOperator
            Copy over those voxels; ``self`` if no bound property is a map.
        """
        declared = set(self.acquisition.exposes)
        narrowed = {
            name: value[keep]
            for name, value in self.acquisition.bound.items()
            if name in declared and torch.is_tensor(value) and value.ndim
        }
        if not narrowed:
            return self
        chosen = shallow_copy(self)
        chosen.acquisition = self.acquisition.bind(**narrowed)
        chosen._elsewhere = {}
        return chosen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the contrasts of the maps.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``.

        Returns
        -------
        torch.Tensor
            ``(..., contrasts)``, complex.
        """
        return self._voxelwise("A", (x,), lambda chunk: (self._predict(chunk[0]),))

    #: The name a reconstruction knows the forward operator by.
    A = forward

    def A_jvp(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Directional derivative ``J d`` by one forward-mode pass.

        Parameters
        ----------
        x, d : torch.Tensor
            ``(..., channels)``, the point and the direction.

        Returns
        -------
        torch.Tensor
            ``(..., contrasts)``, complex.
        """
        forward_mode()
        return self._voxelwise(
            "jvp",
            (x, d),
            lambda chunk: (torch.func.jvp(self._predict, (chunk[0],), (chunk[1],))[1],),
        )

    def A_vjp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Adjoint product ``J^H v`` by one reverse-mode pass.

        The result is the real part of the conjugate product, the adjoint of
        :meth:`A_jvp` under the real inner product.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``, the point.
        v : torch.Tensor
            ``(..., contrasts)``, the cotangent.

        Returns
        -------
        torch.Tensor
            ``(..., channels)``, real.
        """
        return self._voxelwise(
            "vjp",
            (x, v),
            lambda chunk: (torch.func.vjp(self._predict, chunk[0])[1](chunk[1])[0],),
        )

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Build the derivative block of every voxel.

        Used by voxel-wise fits; use :meth:`A_jvp` and :meth:`A_vjp` under an
        encoding operator.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``.

        Returns
        -------
        torch.Tensor
            ``(..., channels, contrasts)``, complex.
        """
        return self._voxelwise(
            "jacobian", (x,), lambda chunk: (self._blocks(chunk[0]),)
        )

    def physics(self, **kwargs: Any) -> Any:
        """Wrap the operator as a :class:`deepinv.physics.Physics`.

        The channel axis moves to the front: ``(batch, channels, *xyz)`` in,
        ``(batch, contrasts, *xyz)`` out. ``encoding * operator.physics()``
        composes with an encoding operator.

        Parameters
        ----------
        **kwargs
            Passed to :class:`deepinv.physics.Physics`, e.g. a noise model.

        Returns
        -------
        deepinv.physics.Physics

        Raises
        ------
        ImportError
            If deepinv is not installed.

        Notes
        -----
        A deepinv ``ComposedPhysics`` differentiates the chain by autograd, not
        by the analytic products of this class. :class:`~blochsim.recon.GaussNewton`
        chains the analytic products.
        """
        try:
            from deepinv.physics import Physics
        except ImportError as error:  # pragma: no cover - depends on the env
            raise ImportError(
                "physics() needs deepinv; BlochSim does not depend on it. "
                "pip install deepinv"
            ) from error

        operator = self

        class _ModelPhysics(Physics):
            def A(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
                return _to_channels(operator.A(_to_trailing(x)))

            def A_vjp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return _to_channels(operator.A_vjp(_to_trailing(x), _to_trailing(v)))

        return _ModelPhysics(**kwargs)

    # -- what a caller does not write ---------------------------------------

    def _parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The free variables and the scale that turns them into properties."""
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"this operator has {self.channels} channels, got {x.shape[-1]}"
            )
        free = x[..., : len(self.unknown)]
        scale = torch.tensor(self.scale, dtype=free.dtype, device=free.device)
        return free, scale

    def _on(self, device: torch.device) -> Any:
        """The acquisition, wherever the maps being evaluated are.

        A chunk that has crossed to a card has left the sequence behind: the
        echo times and flip train live on the simulator, not on the maps, and
        an exponential of one against the other has to find both in the same
        place. Each device's copy is made once and kept.
        """
        key = str(device)
        if key not in self._elsewhere:
            self._elsewhere[key] = self.acquisition.to(device)
        return self._elsewhere[key]

    def _named(self, x: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor | None]:
        """The properties to simulate at, and the amplitude to scale by."""
        free, scale = self._parts(x)
        natural = to_natural(free * scale, self.bounds, self.unknown)
        given = {name: natural[..., index] for index, name in enumerate(self.unknown)}
        rho = torch.complex(x[..., -2], x[..., -1]) if self.amplitude else None
        return given, rho

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """``(voxels, channels)`` to ``(voxels, contrasts)``."""
        given, rho = self._named(x)
        signal = torch.as_tensor(self._on(x.device).simulate(**given))
        if self.subspace is not None:
            signal = self.subspace.project(signal)
        if rho is None:
            return signal
        return (
            signal.to(torch.promote_types(signal.dtype, rho.dtype)) * (rho[..., None])
        )

    def _blocks(self, x: torch.Tensor) -> torch.Tensor:
        """``(voxels, channels, contrasts)``, built column by column.

        The model's own columns cost one forward-mode pass each. The
        amplitude's two cost nothing: the model output *is* its derivative,
        times one and times i.
        """
        given, rho = self._named(x)
        free, scale = self._parts(x)
        signal, derivative = self._on(x.device).jacobian(self.unknown, **given)
        signal = torch.as_tensor(signal)
        derivative = torch.as_tensor(derivative)
        if self.subspace is not None:
            signal = self.subspace.project(signal)
            derivative = self.subspace.project(derivative)
        chain = widen(self.bounds, self.unknown, free * scale) * scale
        columns = derivative * chain[..., None]
        if rho is None:
            return columns
        dtype = torch.promote_types(signal.dtype, rho.dtype)
        columns = columns.to(dtype) * rho[..., None, None]
        signal = signal.to(dtype)
        return torch.cat(
            (columns, signal[..., None, :], (1j * signal)[..., None, :]), dim=-2
        )

    def _voxelwise(
        self, kind: str, inputs: Sequence[torch.Tensor], body: Any
    ) -> torch.Tensor:
        """Run a per-voxel body wherever the execution policy says to.

        Every axis but the last is a voxel axis, so the work is flattened to
        one, handed to :func:`~blochsim._execution.per_voxel`, and folded
        back. Where no policy is in force it runs as written.

        So does a call whose inputs carry a derivative, and that covers two
        cases at once: a streamed chunk is copied through a reused pinned
        buffer, which autograd cannot be walked back through, and a tensor
        wrapped by an outer ``torch.func`` transform -- which is how a caller
        differentiates through a composed operator -- reports the same way.
        """
        shape = inputs[0].shape[:-1]
        flat = [value.reshape(-1, value.shape[-1]) for value in inputs]
        answer = None
        differentiating = torch.is_grad_enabled() and any(
            value.requires_grad for value in flat
        )
        if not differentiating:
            contrasts = self._contrasts(flat[0])
            with torch.no_grad():
                answer = per_voxel(
                    flat,
                    bytes_per_voxel=(self.channels + 2 * contrasts) * 4,
                    work=int(flat[0].shape[0]) * contrasts,
                    crossover=PER_VOXEL_CROSSOVER,
                    body=lambda chunk, device: body(
                        [value.to(device) for value in chunk]
                    ),
                )
        if answer is None:
            answer = body(flat)
        return answer[0].reshape(*shape, *answer[0].shape[1:])

    def _contrasts(self, flat: torch.Tensor) -> int:
        """How many samples one voxel records, measured once and kept."""
        cached = getattr(self, "_contrast_count", None)
        if cached is None:
            with torch.no_grad():
                cached = int(self._predict(flat[:1]).shape[-1])
            self._contrast_count = cached
        return cached


# %% private module subroutines


def _to_trailing(x: torch.Tensor) -> torch.Tensor:
    """Channel-first, as deepinv passes it, to channel-last as we read it."""
    return x.movedim(1, -1)


def _to_channels(x: torch.Tensor) -> torch.Tensor:
    """The way back."""
    return x.movedim(-1, 1)
