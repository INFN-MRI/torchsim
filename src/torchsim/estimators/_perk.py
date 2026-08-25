"""Scalable parameter estimation via Gaussian random Fourier features."""

from __future__ import annotations

__all__ = ["PERK"]

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal

import torch

from .._execution import per_voxel
from ._calibrate import crossover


class PERK(torch.nn.Module):
    """Parameter estimation via regression with kernels.

    This is the random-Fourier-feature form of PERK. Training accumulates the
    required covariances in chunks, so memory depends on the feature order
    rather than the number of simulated training samples. Estimation uses only
    ``cos`` and matrix multiplication and therefore dispatches to optimized
    Torch CPU/CUDA kernels without a separate native implementation.

    Parameters
    ----------
    n_features : int, optional
        Number of random Fourier features. The default is ``1000``.
    length_scale : float or tensor, optional
        Gaussian-kernel length scale for every signal/known-parameter feature.
        ``None`` estimates it from the training standard deviation.
    regularization : float, optional
        Tikhonov parameter added to the feature covariance. The default is
        ``1e-5``.
    chunk_size : int, optional
        Maximum samples transformed at once. The default is ``65536``.
    seed : int, optional
        Seed used for the fixed random feature map.
    complex_mode : {"cartesian", "magnitude"}, optional
        Representation used for complex signals. Magnitude data are often
        preferable when image phase is a nuisance parameter.
    normalize : bool, optional
        Normalize every signal vector before feature projection. This removes
        an unknown global scale when it is not itself a target.

    Notes
    -----
    Cartesian complex signals are represented as interleaved real/imaginary
    features. The learned estimator remains differentiable with respect to its
    input, which permits its use inside a larger reconstruction network.
    """

    def __init__(
        self,
        n_features: int = 1000,
        length_scale: float | torch.Tensor | None = None,
        regularization: float = 1e-5,
        chunk_size: int = 65536,
        seed: int | None = None,
        complex_mode: Literal["cartesian", "magnitude"] = "cartesian",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if complex_mode not in {"cartesian", "magnitude"}:
            raise ValueError("complex_mode must be 'cartesian' or 'magnitude'")
        self.n_features = int(n_features)
        self.regularization = float(regularization)
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self.complex_mode = complex_mode
        self.normalize = bool(normalize)
        self._requested_length_scale = length_scale
        self.register_buffer("frequency", torch.empty(0))
        # The same frequencies laid out for the host kernel's inner loop.
        self.register_buffer("frequency_t", torch.empty(0))
        self.register_buffer("phase", torch.empty(0))
        self.register_buffer("feature_mean", torch.empty(0))
        self.register_buffer("parameter_mean", torch.empty(0))
        self.register_buffer("weight", torch.empty(0))
        self.register_buffer("length_scale", torch.empty(0))
        # Copies of the fitted tensors, one per device a mapping has reached.
        self._replicas: dict[str, tuple[torch.Tensor, ...]] = {}

    @property
    def fitted(self) -> bool:
        """Whether training statistics have been fitted."""
        return self.weight.numel() != 0

    @torch.no_grad()
    def fit(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> PERK:
        """Fit the estimator from simulated signals and parameter targets.

        Parameters
        ----------
        signals : torch.Tensor
            Training signals shaped ``(samples, ..., contrasts)``. All leading
            sample dimensions are flattened.
        parameters : torch.Tensor
            Unknown target parameters shaped ``(samples, ..., n_parameters)``.
        known : torch.Tensor, optional
            Known parameters appended to the regression features.
        noise_std : float or torch.Tensor, optional
            Standard deviation of Gaussian training noise. Independent noise
            is added to real and imaginary channels for complex signals.

        Returns
        -------
        PERK
            The fitted estimator.
        """
        signals = torch.as_tensor(signals)
        parameters = _parameter_matrix(parameters, signals)
        sample_count = parameters.shape[0]
        if signals.numel() % sample_count:
            raise ValueError("signals and parameters have different sample counts")
        signals = signals.reshape(sample_count, -1)
        known = _known_matrix(known, signals, sample_count)
        random_seed = _random_seed(self.seed)

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, sample_count, self.chunk_size):
                stop = min(start + self.chunk_size, sample_count)
                generator = _generator(signals.device, random_seed + start)
                signal_chunk = _add_noise(
                    signals[start:stop], noise_std, generator=generator
                )
                known_chunk = None if known is None else known[start:stop]
                yield (
                    _feature_matrix(
                        signal_chunk,
                        known_chunk,
                        stop - start,
                        complex_mode=self.complex_mode,
                        normalize=self.normalize,
                    ),
                    parameters[start:stop],
                )

        return self._fit_batches(batches, sample_count)

    @torch.no_grad()
    def fit_simulator(
        self,
        simulator: Any,
        properties: Mapping[str, Any],
        known: Mapping[str, Any] | None = None,
        *,
        simulation_chunk_size: int | None = None,
        noise_std: float | torch.Tensor = 0.0,
        **sequence: Any,
    ) -> PERK:
        """Train on signals a simulator generates, without ever holding them.

        Parameters
        ----------
        simulator : SignalModel
            The sequence being inverted -- the same object sequence design and
            a reconstruction pipeline take, so the training signals cannot
            drift from the ones the scanner will produce.
        properties : mapping
            What is being estimated, as ``{name: value per training sample}``
            under the names the simulator exposes. The order is the order of
            the estimator's output columns.
        known : mapping, optional
            Properties measured separately and given to the estimator along
            with the signal -- a transmit map, say. They reach the simulator
            too, so the training signals carry their effect.
        simulation_chunk_size : int, optional
            How many samples to simulate at once. Memory follows this rather
            than the number of training samples.
        noise_std : float or tensor, optional
            Noise added to the training signals, which is what teaches the
            estimator how far to trust them.
        sequence
            Anything else the simulator takes, passed on unchanged.

        Returns
        -------
        PERK
            This estimator, fitted.

        Raises
        ------
        ValueError
            If ``properties`` is empty, if the values disagree on how many
            training samples there are, or if the chunk size is not positive.
        """
        if not properties:
            raise ValueError("name at least one property to estimate")
        chunk_size = simulation_chunk_size or self.chunk_size
        if chunk_size < 1:
            raise ValueError("simulation_chunk_size must be positive")
        columns = _sample_columns(properties)
        parameters = torch.stack(columns, dim=-1)
        count = parameters.shape[0]
        known_names = tuple(known or ())
        known_matrix = (
            torch.stack(_sample_columns(known, count), dim=-1) if known else None
        )
        random_seed = _random_seed(self.seed)
        names = tuple(properties)

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                known_chunk = (
                    None if known_matrix is None else known_matrix[start:stop]
                )
                given = {
                    name: parameters[start:stop, column]
                    for column, name in enumerate(names)
                }
                if known_chunk is not None:
                    given.update(
                        {
                            name: known_chunk[:, column]
                            for column, name in enumerate(known_names)
                        }
                    )
                signals = torch.as_tensor(
                    simulator.simulate(**given, **sequence),
                    device=parameters.device,
                )
                generator = _generator(parameters.device, random_seed + start)
                signals = _add_noise(signals, noise_std, generator=generator)
                yield (
                    _feature_matrix(
                        signals.reshape(stop - start, -1),
                        known_chunk,
                        stop - start,
                        complex_mode=self.complex_mode,
                        normalize=self.normalize,
                    ),
                    parameters[start:stop],
                )

        return self._fit_batches(batches, count)

    def forward(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate parameters from measured signals."""
        if not self.fitted:
            raise RuntimeError("PERK must be fitted before estimation")
        signals = torch.as_tensor(signals, device=self.frequency.device)
        sample_shape = signals.shape[:-1]
        sample_count = math.prod(sample_shape) if sample_shape else 1
        inputs = _feature_matrix(
            signals,
            known,
            sample_count,
            complex_mode=self.complex_mode,
            normalize=self.normalize,
        )
        placed = self._placed(inputs)
        if placed is not None:
            return placed[0].reshape(*sample_shape, -1)
        if self._fused(inputs):
            values = _FusedRegression.apply(
                inputs,
                self.frequency,
                self.frequency_t,
                self.phase,
                self.feature_mean,
                self.weight,
                self.parameter_mean,
            )
            return values.reshape(*sample_shape, -1)
        outputs = []
        for chunk in inputs.split(self.chunk_size):
            features = _rff(chunk, self.frequency, self.phase)
            outputs.append(
                self.parameter_mean + (features - self.feature_mean) @ self.weight.mT
            )
        return torch.cat(outputs, dim=0).reshape(*sample_shape, -1)

    def _placed(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        """Run under the execution policy, or ``None`` if none applies.

        Voxels are independent, so a volume too large for a card is streamed
        through it and a machine with two cards uses both. What the policy
        cannot carry is a gradient: a streamed chunk is copied through a
        reused pinned buffer, which is not something autograd can be walked
        back through. A call that wants a derivative gets the ordinary path.
        """
        if torch.is_grad_enabled() and inputs.requires_grad:
            return None
        contrasts = int(inputs.shape[-1])
        parameters = int(self.weight.shape[0])
        with torch.no_grad():
            return per_voxel(
                [inputs],
                bytes_per_voxel=(contrasts + parameters) * 4,
                work=int(inputs.shape[0]) * contrasts * self.n_features,
                crossover=lambda device: crossover(
                    (contrasts, self.n_features, parameters),
                    device,
                    self._probe(contrasts, parameters),
                    contrasts * self.n_features,
                ),
                body=lambda chunk, device: (
                    self._regress(chunk[0], self._fitted_on(device)),
                ),
            )

    def _probe(self, contrasts: int, parameters: int) -> Any:
        """A closure the calibrator can time, running the real regression."""

        def build(device: torch.device, voxels: int) -> Any:
            generator = torch.Generator(device=device).manual_seed(0)
            signals = torch.randn(
                voxels, contrasts, generator=generator, device=device
            )
            held = self._fitted_on(device)
            return lambda: self._regress(signals, held)

        return build

    def _fitted_on(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The fitted tensors on ``device``.

        They are the same for every chunk, so they cross once per device
        rather than once per chunk.
        """
        key = str(device)
        held = self._replicas.get(key)
        if held is None:
            held = tuple(
                tensor.to(device)
                for tensor in (
                    self.frequency,
                    self.frequency_t,
                    self.phase,
                    self.feature_mean,
                    self.weight,
                    self.parameter_mean,
                )
            )
            self._replicas[key] = held
        return held

    def _regress(
        self, inputs: torch.Tensor, held: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """One chunk, on whichever device its tensors are on."""
        frequency, transposed, phase, feature_mean, weight, parameter_mean = held
        backend = _kernels(inputs.device)
        if backend is not None:
            if inputs.device.type == "cuda":
                return backend.regress(
                    inputs, frequency, phase, feature_mean, weight,
                    parameter_mean,
                )
            return backend.regress(
                inputs, frequency, transposed, phase, feature_mean, weight,
                parameter_mean,
            )
        outputs = []
        for piece in inputs.split(self.chunk_size):
            features = _rff(piece, frequency, phase)
            outputs.append(parameter_mean + (features - feature_mean) @ weight.mT)
        return torch.cat(outputs, dim=0)

    def _fused(self, inputs: torch.Tensor) -> bool:
        """Whether the fused kernel can answer this call.

        It differentiates with respect to the signals, which is what a PERK
        inside a reconstruction network needs. A gradient wanted for one of the
        fitted tensors instead is rare enough to be worth the composed path
        rather than a second adjoint.
        """
        if _kernels(inputs.device) is None:
            return False
        return not any(
            tensor.requires_grad
            for tensor in (
                self.frequency,
                self.phase,
                self.feature_mean,
                self.weight,
                self.parameter_mean,
            )
        )

    def _fit_batches(
        self,
        batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
        sample_count: int,
    ) -> PERK:
        """Fit from a source that can be walked more than once.

        The kernel width is read from the spread of the training inputs, and
        the random features cannot be drawn until it is known -- so estimating
        it costs one pass over the source before the fitting one. Give
        ``length_scale`` and there is only the fitting pass.
        """
        if sample_count < 2:
            raise ValueError("PERK requires at least two training samples")

        length_scale = None
        if self._requested_length_scale is None:
            length_scale = self._estimated_length_scale(batches, sample_count)

        frequency: torch.Tensor | None = None
        phase: torch.Tensor | None = None
        feature_sum = None
        second_moment = None
        cross_moment = None
        parameter_sum = None
        observed = 0
        for inputs, targets in batches():
            inputs = inputs.to(torch.float32)
            targets = targets.to(torch.float32)
            if frequency is None:
                if length_scale is None:
                    length_scale = self._given_length_scale(
                        inputs.shape[-1], inputs.device
                    )
                frequency, phase = self._random_features(length_scale)
                feature_sum = torch.zeros(
                    self.n_features, dtype=torch.float64, device=inputs.device
                )
                second_moment = torch.zeros(
                    self.n_features,
                    self.n_features,
                    dtype=torch.float64,
                    device=inputs.device,
                )
                cross_moment = torch.zeros(
                    targets.shape[-1],
                    self.n_features,
                    dtype=torch.float64,
                    device=inputs.device,
                )
                parameter_sum = torch.zeros(
                    targets.shape[-1], dtype=torch.float64, device=inputs.device
                )
            features = _rff(inputs, frequency, phase).to(torch.float64)
            targets = targets.to(torch.float64)
            feature_sum += features.sum(dim=0)
            parameter_sum += targets.sum(dim=0)
            second_moment += features.mT @ features
            cross_moment += targets.mT @ features
            observed += inputs.shape[0]
        if observed != sample_count or frequency is None:
            raise ValueError("batch source returned an inconsistent sample count")

        # Centred covariances from uncentred moments, so one pass carries both
        # the means and the products they would otherwise have to precede.
        feature_mean = feature_sum / sample_count
        parameter_mean = parameter_sum / sample_count
        covariance = (
            second_moment
            - sample_count * torch.outer(feature_mean, feature_mean)
        ) / (sample_count - 1)
        cross_covariance = (
            cross_moment
            - sample_count * torch.outer(parameter_mean, feature_mean)
        ) / (sample_count - 1)
        covariance.diagonal().add_(self.regularization)
        weight = torch.linalg.solve(covariance, cross_covariance.mT).mT

        self._replicas = {}
        self.frequency = frequency
        self.frequency_t = frequency.mT.contiguous()
        self.phase = phase
        self.feature_mean = feature_mean.to(torch.float32)
        self.parameter_mean = parameter_mean.to(torch.float32)
        self.weight = weight.to(torch.float32)
        self.length_scale = length_scale
        return self

    def _random_features(
        self, length_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fixed frequency and phase of the random Fourier feature map."""
        generator = _generator(length_scale.device, _random_seed(self.seed))
        frequency = torch.randn(
            self.n_features,
            length_scale.numel(),
            dtype=torch.float32,
            device=length_scale.device,
            generator=generator,
        ) / length_scale[None, :]
        phase = 2.0 * math.pi * torch.rand(
            self.n_features,
            dtype=torch.float32,
            device=length_scale.device,
            generator=generator,
        )
        return frequency, phase

    def _estimated_length_scale(
        self,
        batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
        sample_count: int,
    ) -> torch.Tensor:
        """Read the kernel width off the spread of the training inputs."""
        input_sum = None
        input_square_sum = None
        observed = 0
        for inputs, _targets in batches():
            inputs = inputs.to(torch.float32)
            if input_sum is None:
                input_sum = torch.zeros(
                    inputs.shape[-1], dtype=torch.float64, device=inputs.device
                )
                input_square_sum = torch.zeros_like(input_sum)
            input_sum += inputs.sum(dim=0, dtype=torch.float64)
            input_square_sum += inputs.square().sum(dim=0, dtype=torch.float64)
            observed += inputs.shape[0]
        if observed != sample_count or input_sum is None:
            raise ValueError("batch source returned an inconsistent sample count")
        variance = (
            input_square_sum - input_sum.square() / sample_count
        ) / (sample_count - 1)
        floor = torch.finfo(torch.float32).eps
        scale = variance.clamp_min(0.0).sqrt().to(torch.float32)
        # Per-coordinate standard deviations make a high-dimensional Gaussian
        # kernel vanish between almost every pair of samples. This factor
        # implements the usual O(sqrt(d)) median-distance scaling while
        # retaining feature-wise physical units.
        scale *= math.sqrt(input_sum.numel())
        return scale.clamp_min(floor)

    def _given_length_scale(
        self, width: int, device: torch.device
    ) -> torch.Tensor:
        """The kernel width the caller asked for, checked against the inputs."""
        scale = torch.as_tensor(
            self._requested_length_scale, dtype=torch.float32, device=device
        ).flatten()
        if scale.numel() == 1:
            scale = scale.expand(width)
        if scale.numel() != width or torch.any(scale <= 0):
            raise ValueError("length_scale must be positive and match input features")
        return scale


# %% private module subroutines


def _loaded(name: str) -> Any:
    """One kernel backend, or ``None`` where it cannot be imported."""
    try:
        return __import__(f"torchsim.estimators.{name}", fromlist=[name])
    except ImportError:
        return None


_TRITON = _loaded("_perk_triton")
_NATIVE = _loaded("_perk_native")


def _kernels(device: torch.device) -> Any:
    """The fused backend for this device, or ``None``."""
    return _TRITON if device.type == "cuda" else _NATIVE


class _FusedRegression(torch.autograd.Function):
    """The feature map and its regression, without the features in between."""

    @staticmethod
    def forward(
        ctx: Any,
        signals: torch.Tensor,
        frequency: torch.Tensor,
        transposed: torch.Tensor,
        phase: torch.Tensor,
        feature_mean: torch.Tensor,
        weight: torch.Tensor,
        parameter_mean: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(signals, frequency, transposed, phase, weight)
        if signals.device.type == "cuda":
            return _TRITON.regress(
                signals, frequency, phase, feature_mean, weight, parameter_mean
            )
        return _NATIVE.regress(
            signals, frequency, transposed, phase, feature_mean, weight,
            parameter_mean,
        )

    @staticmethod
    def backward(
        ctx: Any, cotangent: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        signals, frequency, transposed, phase, weight = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None,) * 7
        gradient = (
            _TRITON.regress_vjp(cotangent, signals, frequency, phase, weight)
            if signals.device.type == "cuda"
            else _NATIVE.regress_vjp(
                cotangent, signals, frequency, transposed, phase, weight
            )
        )
        return gradient, None, None, None, None, None, None




def _feature_matrix(
    signals: torch.Tensor,
    known: torch.Tensor | None,
    sample_count: int,
    *,
    complex_mode: str,
    normalize: bool,
) -> torch.Tensor:
    signals = signals.reshape(sample_count, -1)
    if normalize:
        norm = torch.linalg.vector_norm(signals, dim=-1).clamp_min(
            torch.finfo(signals.real.dtype).eps
        )
        signals = signals / norm[:, None]
    if torch.is_complex(signals) and complex_mode == "magnitude":
        signals = signals.abs()
    elif torch.is_complex(signals):
        signals = torch.view_as_real(signals).reshape(sample_count, -1)
    signals = signals.to(torch.float32)
    if known is None:
        return signals
    known = torch.as_tensor(known, device=signals.device).reshape(sample_count, -1)
    if torch.is_complex(known):
        known = torch.view_as_real(known).reshape(sample_count, -1)
    return torch.cat((signals, known.to(torch.float32)), dim=-1)


def _rff(
    inputs: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    scale = math.sqrt(2.0 / frequency.shape[0])
    return scale * torch.cos(inputs @ frequency.mT + phase)


def _sample_columns(
    values: Mapping[str, Any], expected: int | None = None
) -> list[torch.Tensor]:
    """One column per named property, all agreeing on the sample count."""
    columns = [
        torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        for value in values.values()
    ]
    counts = {int(column.numel()) for column in columns}
    if expected is not None:
        counts.add(expected)
    if len(counts) > 1:
        raise ValueError(
            f"the properties disagree on the training sample count: "
            f"{sorted(counts)}"
        )
    return columns


def _parameter_matrix(parameters: Any, reference: torch.Tensor) -> torch.Tensor:
    parameters = torch.as_tensor(
        parameters, device=reference.device, dtype=reference.real.dtype
    )
    if parameters.ndim == 1:
        parameters = parameters[:, None]
    return parameters.reshape(-1, parameters.shape[-1]).to(torch.float32)


def _known_matrix(
    known: torch.Tensor | None,
    reference: torch.Tensor,
    sample_count: int,
) -> torch.Tensor | None:
    if known is None:
        return None
    output = torch.as_tensor(known, device=reference.device)
    if output.numel() % sample_count:
        raise ValueError("known parameters have a different sample count")
    return output.reshape(sample_count, -1)


def _random_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return int(torch.empty((), dtype=torch.int64).random_().item())


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed % (2**63 - 1))
    return generator


def _add_noise(
    signals: torch.Tensor,
    noise_std: Any,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    scale = torch.as_tensor(noise_std, dtype=signals.real.dtype, device=signals.device)
    if not torch.any(scale != 0):
        return signals
    shape = signals.shape
    if torch.is_complex(signals):
        noise = torch.complex(
            torch.randn(
                shape,
                dtype=signals.real.dtype,
                device=signals.device,
                generator=generator,
            ),
            torch.randn(
                shape,
                dtype=signals.real.dtype,
                device=signals.device,
                generator=generator,
            ),
        )
    else:
        noise = torch.randn(
            shape,
            dtype=signals.dtype,
            device=signals.device,
            generator=generator,
        )
    return signals + scale * noise
