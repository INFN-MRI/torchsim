"""Scalable parameter estimation via Gaussian random Fourier features."""

from __future__ import annotations

__all__ = ["PERK"]

import math
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch


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
        self.register_buffer("phase", torch.empty(0))
        self.register_buffer("feature_mean", torch.empty(0))
        self.register_buffer("parameter_mean", torch.empty(0))
        self.register_buffer("weight", torch.empty(0))
        self.register_buffer("length_scale", torch.empty(0))

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
        simulator: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        simulation_chunk_size: int | None = None,
        noise_std: float | torch.Tensor = 0.0,
    ) -> PERK:
        """Generate training signals with a vectorized simulator and fit.

        The simulator receives one parameter chunk and the matching known
        parameters, and returns a batch of signals. This keeps sequence-model
        details out of the estimator while allowing large dictionaries to be
        generated within a bounded memory budget.
        """
        parameters = torch.as_tensor(parameters)
        reference = parameters.real if torch.is_complex(parameters) else parameters
        parameters = _parameter_matrix(parameters, reference)
        count = parameters.shape[0]
        chunk_size = simulation_chunk_size or self.chunk_size
        if chunk_size < 1:
            raise ValueError("simulation_chunk_size must be positive")
        known = _known_matrix(known, parameters, count)
        random_seed = _random_seed(self.seed)

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                known_chunk = None if known is None else known[start:stop]
                signals = torch.as_tensor(
                    simulator(parameters[start:stop], known_chunk),
                    device=parameters.device,
                )
                generator = _generator(parameters.device, random_seed + start)
                signals = _add_noise(signals, noise_std, generator=generator)
                yield (
                    _feature_matrix(
                        signals,
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
        outputs = []
        for chunk in inputs.split(self.chunk_size):
            features = _rff(chunk, self.frequency, self.phase)
            outputs.append(
                self.parameter_mean + (features - self.feature_mean) @ self.weight.mT
            )
        return torch.cat(outputs, dim=0).reshape(*sample_shape, -1)

    def _fit_batches(
        self,
        batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
        sample_count: int,
    ) -> PERK:
        if sample_count < 2:
            raise ValueError("PERK requires at least two training samples")

        input_sum = None
        input_square_sum = None
        parameter_sum = None
        observed = 0
        for inputs, targets in batches():
            inputs = inputs.to(torch.float32)
            targets = targets.to(torch.float32)
            if input_sum is None:
                input_sum = torch.zeros(
                    inputs.shape[-1], dtype=torch.float64, device=inputs.device
                )
                input_square_sum = torch.zeros_like(input_sum)
                parameter_sum = torch.zeros(
                    targets.shape[-1], dtype=torch.float64, device=inputs.device
                )
            input_sum += inputs.sum(dim=0, dtype=torch.float64)
            input_square_sum += inputs.square().sum(dim=0, dtype=torch.float64)
            parameter_sum += targets.sum(dim=0, dtype=torch.float64)
            observed += inputs.shape[0]
        if observed != sample_count or input_sum is None:
            raise ValueError("batch source returned an inconsistent sample count")

        length_scale = self._make_length_scale(
            input_sum, input_square_sum, sample_count
        )
        generator = _generator(input_sum.device, _random_seed(self.seed))
        frequency = torch.randn(
            self.n_features,
            input_sum.numel(),
            dtype=torch.float32,
            device=input_sum.device,
            generator=generator,
        ) / length_scale[None, :]
        phase = 2.0 * math.pi * torch.rand(
            self.n_features,
            dtype=torch.float32,
            device=input_sum.device,
            generator=generator,
        )

        feature_sum = torch.zeros_like(frequency[:, 0], dtype=torch.float64)
        for inputs, _ in batches():
            feature_sum += _rff(inputs, frequency, phase).sum(
                dim=0, dtype=torch.float64
            )
        feature_mean = (feature_sum / sample_count).to(torch.float32)
        parameter_mean = (parameter_sum / sample_count).to(torch.float32)

        covariance = torch.zeros(
            self.n_features,
            self.n_features,
            dtype=torch.float64,
            device=input_sum.device,
        )
        cross_covariance = torch.zeros(
            parameter_mean.numel(),
            self.n_features,
            dtype=torch.float64,
            device=input_sum.device,
        )
        for inputs, targets in batches():
            centered_features = (
                _rff(inputs, frequency, phase) - feature_mean
            ).to(torch.float64)
            centered_targets = (targets - parameter_mean).to(torch.float64)
            covariance += centered_features.mT @ centered_features
            cross_covariance += centered_targets.mT @ centered_features
        covariance /= sample_count - 1
        cross_covariance /= sample_count - 1
        covariance.diagonal().add_(self.regularization)
        weight = torch.linalg.solve(covariance, cross_covariance.mT).mT

        self.frequency = frequency
        self.phase = phase
        self.feature_mean = feature_mean
        self.parameter_mean = parameter_mean
        self.weight = weight.to(torch.float32)
        self.length_scale = length_scale
        return self

    def _make_length_scale(
        self,
        input_sum: torch.Tensor,
        input_square_sum: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        if self._requested_length_scale is None:
            variance = (
                input_square_sum - input_sum.square() / sample_count
            ) / (sample_count - 1)
            floor = torch.finfo(torch.float32).eps
            scale = variance.clamp_min(0.0).sqrt().to(torch.float32)
            # Per-coordinate standard deviations make a high-dimensional
            # Gaussian kernel vanish between almost every pair of samples.
            # This factor implements the usual O(sqrt(d)) median-distance
            # scaling while retaining feature-wise physical units.
            scale *= math.sqrt(input_sum.numel())
            return scale.clamp_min(floor)
        scale = torch.as_tensor(
            self._requested_length_scale,
            dtype=torch.float32,
            device=input_sum.device,
        ).flatten()
        if scale.numel() == 1:
            scale = scale.expand(input_sum.numel())
        if scale.numel() != input_sum.numel() or torch.any(scale <= 0):
            raise ValueError("length_scale must be positive and match input features")
        return scale


# %% private module subroutines


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
