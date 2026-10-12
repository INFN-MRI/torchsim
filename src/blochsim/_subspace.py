"""Low-rank temporal basis fitted to a set of signals."""

from __future__ import annotations

__all__ = ["Subspace"]

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Subspace:
    """Orthonormal temporal basis of a set of signals, for subspace reconstruction.

    The basis holds the leading left singular vectors of a dictionary of
    simulated signals (for example an MR fingerprinting dictionary), so that a
    series of ``contrasts`` images is described by ``rank`` coefficient images.
    :meth:`fit` builds one from signals; :meth:`project` and :meth:`expand`
    move between the contrast and coefficient domains. Instances are immutable.

    Parameters
    ----------
    basis : torch.Tensor
        Orthonormal basis of shape ``(contrasts, rank)``; complex when the
        signals are.
    singular_values : torch.Tensor
        All singular values of the fitted signals, including those not kept.
    dictionary : torch.Tensor, default=None
        Signals the basis was fitted to, of shape ``(atoms, contrasts)``. Set
        by ``blochsim.sequence.simulate_subspace``, not by :meth:`fit`.
    simulation : SimulationResult, default=None
        The recorded simulation the dictionary came from, with its sample
        labels. Set by ``blochsim.sequence.simulate_subspace``.
    """

    basis: torch.Tensor
    singular_values: torch.Tensor
    dictionary: torch.Tensor | None = None
    simulation: Any | None = None

    @property
    def rank(self) -> int:
        """Number of basis directions kept."""
        return int(self.basis.shape[-1])

    @property
    def contrasts(self) -> int:
        """Number of contrasts, the length of the time axis the basis spans."""
        return int(self.basis.shape[0])

    @property
    def retained(self) -> float:
        """Fraction of the fitted signals' energy the basis keeps.

        One minus this is the relative squared error of projecting the fitted
        signals onto the basis and back.
        """
        energy = self.singular_values.to(torch.float64).square()
        total = float(energy.sum())
        return 1.0 if total == 0.0 else float(energy[: self.rank].sum()) / total

    @property
    def modes(self) -> torch.Tensor:
        """The basis transposed, ``(rank, contrasts)``.

        This is the layout of mri-nufft's ``MRISubspace`` and BART's
        ``pics -B``. It is a plain transpose: expansion is
        ``image_t = sum_k conj(modes[k, t]) c_k``, as in :meth:`expand`.
        """
        return self.basis.mT

    def project(self, signals: torch.Tensor) -> torch.Tensor:
        """Project ``signals`` onto the basis, giving ``signals @ basis``.

        Parameters
        ----------
        signals : torch.Tensor
            Signals of shape ``(..., contrasts)``. The result is computed in
            the wider of the signals' and the basis's dtypes, on the signals'
            device.

        Returns
        -------
        torch.Tensor
            ``(..., rank)`` coefficients.
        """
        dtype = torch.promote_types(signals.dtype, self.basis.dtype)
        return signals.to(dtype) @ self.basis.to(dtype).to(signals.device)

    def expand(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Expand coefficients to contrasts, giving ``coefficients @ basis.mH``.

        Parameters
        ----------
        coefficients : torch.Tensor
            ``(..., rank)``.

        Returns
        -------
        torch.Tensor
            ``(..., contrasts)``, on the coefficients' device.
        """
        dtype = torch.promote_types(coefficients.dtype, self.basis.dtype)
        return coefficients.to(dtype) @ self.basis.mH.to(dtype).to(coefficients.device)

    @classmethod
    def fit(cls, signals: torch.Tensor, rank: int) -> Subspace:
        """Fit the leading ``rank`` singular directions of ``signals``.

        The basis is the first ``rank`` left singular vectors of the matrix
        whose columns are the signals; no mean is removed.

        Parameters
        ----------
        signals : torch.Tensor
            Signals of shape ``(..., contrasts)``; leading axes are flattened
            into atoms.
        rank : int
            Number of directions to keep.

        Returns
        -------
        Subspace
            The fitted basis.

        Raises
        ------
        ValueError
            If ``rank`` is not positive, or exceeds what the signals span.
        """
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
        flat = torch.as_tensor(signals).reshape(-1, signals.shape[-1])
        if rank > min(flat.shape):
            raise ValueError(
                f"rank={rank} exceeds the signals' dimensions {tuple(flat.shape)}"
            )
        basis, singular_values, _ = torch.linalg.svd(flat.mT, full_matrices=False)
        return cls(basis=basis[:, :rank], singular_values=singular_values)
