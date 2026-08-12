"""Memory-bounded matching against simulated signal dictionaries."""

from __future__ import annotations

__all__ = ["DictionaryMatch", "DictionaryMatcher"]

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DictionaryMatch:
    """Best dictionary atoms and their complex least-squares scales."""

    parameters: torch.Tensor | None
    indices: torch.Tensor
    scores: torch.Tensor
    scales: torch.Tensor


class DictionaryMatcher(torch.nn.Module):
    """Match signals using normalized complex inner products.

    Parameters
    ----------
    dictionary : torch.Tensor
        Simulated atoms shaped ``(n_atoms, n_contrasts)``.
    parameters : torch.Tensor, optional
        Parameter values shaped ``(n_atoms, n_parameters)``. If provided,
        :meth:`forward` returns parameter estimates; otherwise it returns atom
        indices.
    query_chunk_size : int, optional
        Maximum measured signals compared at once.
    dictionary_chunk_size : int, optional
        Maximum dictionary atoms compared at once.
    top_k : int, optional
        Number of candidates retained by :meth:`match`. The first is the
        conventional dictionary-matching estimate.

    Notes
    -----
    The expensive operation is a matrix product. Torch therefore dispatches
    directly to the installed CPU BLAS or cuBLAS implementation; a separate
    C++ or Triton matrix-multiplication kernel would duplicate a faster
    vendor implementation. Chunking bounds the temporary score matrix.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        parameters: torch.Tensor | None = None,
        *,
        query_chunk_size: int = 4096,
        dictionary_chunk_size: int = 16384,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        dictionary = torch.as_tensor(dictionary)
        if dictionary.ndim != 2 or dictionary.shape[0] < 1:
            raise ValueError("dictionary must have shape (atoms, contrasts)")
        if query_chunk_size < 1 or dictionary_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        if top_k < 1 or top_k > dictionary.shape[0]:
            raise ValueError("top_k must be between one and the atom count")
        if not torch.is_floating_point(dictionary) and not torch.is_complex(dictionary):
            dictionary = dictionary.to(torch.float32)
        parameters = _prepare_parameters(parameters, dictionary)

        norm = torch.linalg.vector_norm(dictionary, dim=-1).clamp_min(
            torch.finfo(dictionary.real.dtype).eps
        )
        self.register_buffer("dictionary", dictionary)
        self.register_buffer("normalized_dictionary", dictionary / norm[:, None])
        self.register_buffer("dictionary_power", norm.square())
        self.register_buffer("parameter_values", parameters)
        self.query_chunk_size = int(query_chunk_size)
        self.dictionary_chunk_size = int(dictionary_chunk_size)
        self.top_k = int(top_k)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return best parameter values, or indices if none were supplied."""
        result = self.match(signals)
        if result.parameters is None:
            return result.indices[..., 0]
        return result.parameters[..., 0, :]

    @torch.no_grad()
    def match(self, signals: torch.Tensor) -> DictionaryMatch:
        """Return the top matching atoms, scores, scales, and parameters."""
        signals = torch.as_tensor(signals, device=self.dictionary.device)
        if signals.shape[-1] != self.dictionary.shape[-1]:
            raise ValueError("signal and dictionary contrast counts differ")
        sample_shape = signals.shape[:-1]
        signals = signals.reshape(-1, signals.shape[-1]).to(self.dictionary.dtype)
        signal_norm = torch.linalg.vector_norm(signals, dim=-1).clamp_min(
            torch.finfo(signals.real.dtype).eps
        )
        normalized_signals = signals / signal_norm[:, None]

        score_chunks = []
        index_chunks = []
        for query in normalized_signals.split(self.query_chunk_size):
            best_scores = torch.empty(
                (query.shape[0], 0), dtype=query.real.dtype, device=query.device
            )
            best_indices = torch.empty(
                (query.shape[0], 0), dtype=torch.int64, device=query.device
            )
            for start in range(0, self.dictionary.shape[0], self.dictionary_chunk_size):
                stop = min(start + self.dictionary_chunk_size, self.dictionary.shape[0])
                scores = torch.abs(query @ self.normalized_dictionary[start:stop].mH)
                local_count = min(self.top_k, scores.shape[-1])
                local_scores, local_indices = torch.topk(scores, local_count, dim=-1)
                local_indices += start
                candidates = torch.cat((best_scores, local_scores), dim=-1)
                candidate_indices = torch.cat((best_indices, local_indices), dim=-1)
                keep = min(self.top_k, candidates.shape[-1])
                best_scores, selection = torch.topk(candidates, keep, dim=-1)
                best_indices = torch.gather(candidate_indices, -1, selection)
            score_chunks.append(best_scores)
            index_chunks.append(best_indices)

        scores = torch.cat(score_chunks, dim=0)
        indices = torch.cat(index_chunks, dim=0)
        atoms = self.dictionary[indices]
        scales = (
            torch.sum(atoms.conj() * signals[:, None, :], dim=-1)
            / self.dictionary_power[indices]
        )
        matched_parameters = (
            None
            if self.parameter_values.numel() == 0
            else self.parameter_values[indices]
        )
        output_shape = (*sample_shape, self.top_k)
        return DictionaryMatch(
            parameters=None
            if matched_parameters is None
            else matched_parameters.reshape(*output_shape, -1),
            indices=indices.reshape(output_shape),
            scores=scores.reshape(output_shape),
            scales=scales.reshape(output_shape),
        )


# %% private module subroutines


def _prepare_parameters(
    parameters: torch.Tensor | None,
    dictionary: torch.Tensor,
) -> torch.Tensor:
    if parameters is None:
        return torch.empty(0, dtype=torch.float32, device=dictionary.device)
    output = torch.as_tensor(
        parameters, dtype=dictionary.real.dtype, device=dictionary.device
    )
    if output.ndim == 1:
        output = output[:, None]
    if output.ndim != 2 or output.shape[0] != dictionary.shape[0]:
        raise ValueError("parameters must have shape (atoms, parameters)")
    return output
