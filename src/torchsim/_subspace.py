"""A low-rank temporal basis, and what it keeps.

The signals a sequence can produce do not fill the space its contrasts span.
A few hundred echoes of a relaxation-driven train lie close to a subspace of a
dozen or so directions, and every dimension dropped is arithmetic that neither
a dictionary match nor a kernel regression has to do.

What makes the trade safe is that the error is known. The singular values the
basis is read from say exactly how much of the signal energy is left outside
it, so a rank is chosen against a number rather than a hope.
"""

from __future__ import annotations

__all__ = ["Subspace"]

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Subspace:
    """A temporal basis fitted to a set of signals.

    Attributes
    ----------
    basis : torch.Tensor
        ``(contrasts, rank)``, orthonormal. Complex when the signals are.
    singular_values : torch.Tensor
        Every singular value of the signals it was fitted from, not only the
        ones kept, so :attr:`retained` can be read and another rank costed.
    """

    basis: torch.Tensor
    singular_values: torch.Tensor

    @property
    def rank(self) -> int:
        """How many directions the basis keeps."""
        return int(self.basis.shape[-1])

    @property
    def contrasts(self) -> int:
        """How many contrasts it was fitted over."""
        return int(self.basis.shape[0])

    @property
    def retained(self) -> float:
        """The fraction of the fitted signals' energy the basis keeps.

        One minus this is the relative squared error of projecting those
        signals onto it and back, so it is the approximation, not a proxy
        for it.
        """
        energy = self.singular_values.to(torch.float64).square()
        total = float(energy.sum())
        return 1.0 if total == 0.0 else float(energy[: self.rank].sum()) / total

    @property
    def modes(self) -> torch.Tensor:
        """The basis with the rank axis first: ``(rank, contrasts)``.

        A reconstruction library's subspace operator takes the basis this way
        round -- mri-nufft's ``MRISubspace``, BART's ``pics -B``. It is a
        plain transpose and not a conjugate one: expanding coefficients back
        into contrasts is ``image_t = sum_k conj(modes[k, t]) c_k``, which is
        what :meth:`expand` does and what those operators do.
        """
        return self.basis.mT

    def project(self, signals: torch.Tensor) -> torch.Tensor:
        """Return ``signals`` in the subspace: ``(..., contrasts)`` to ``(..., rank)``.

        A basis fitted to real signals is real, and a complex signal projected
        onto it keeps both its parts -- the arithmetic is promoted to whichever
        of the two is wider, never narrowed to the basis. The basis follows the
        signals to whichever device they are on, rather than the other way
        round: a volume is large and a basis is not.
        """
        dtype = torch.promote_types(signals.dtype, self.basis.dtype)
        return signals.to(dtype) @ self.basis.to(dtype).to(signals.device)

    def expand(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Return subspace coefficients as contrasts again."""
        dtype = torch.promote_types(coefficients.dtype, self.basis.dtype)
        return coefficients.to(dtype) @ self.basis.mH.to(dtype).to(
            coefficients.device
        )

    @classmethod
    def fit(cls, signals: torch.Tensor, rank: int) -> Subspace:
        """Fit the leading ``rank`` directions of ``signals``.

        Parameters
        ----------
        signals:
            ``(..., contrasts)``. Every leading axis is flattened, so a
            simulated dictionary and a training set are the same input.
        rank:
            How many directions to keep.

        Returns
        -------
        Subspace

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
