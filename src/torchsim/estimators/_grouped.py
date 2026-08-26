"""Clustering a dictionary so most of it can be ruled out per voxel.

Bloch simulations over a parameter grid are not spread evenly through signal
space: neighbouring tissues make nearly parallel signals, so a dictionary is
already clustered before anything is done to it. Grouping it and giving each
group one representative signal turns matching into two much smaller problems
-- which groups could this voxel be in, and which atom inside them -- and the
first answer rules out almost all of the second.

Compression comes first and is global. A dictionary is projected onto one
temporal basis, stored as ``(atoms, rank)``, and clustered in that basis; the
signals arrive in the same basis, whether projected on the way in or solved
for there by a subspace reconstruction. One basis for everything is what lets
a group be entered without leaving the space the measurement is in. The two
savings are then independent and multiply: the basis shortens every inner
product, the grouping cuts how many are taken.

The grouping and pruning are those of Cauley et al., *Fast group matching for
MR fingerprinting reconstruction*, Magn Reson Med 74:523 (2015).
"""

from __future__ import annotations

__all__ = ["Grouping", "correlate"]

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grouping:
    """A dictionary split into clusters of nearly parallel signals.

    Attributes
    ----------
    members:
        One tensor of atom indices per group.
    representative:
        ``(groups, contrasts)``, the normalized mean signal of each group.
        Matching against these is what prunes.
    atoms:
        Each group's atoms, normalized, in the basis the dictionary arrived
        in. No group carries a basis of its own -- see the module docstring.
    """

    members: tuple[torch.Tensor, ...]
    representative: torch.Tensor
    atoms: tuple[torch.Tensor, ...]

    @property
    def count(self) -> int:
        """How many groups the dictionary was split into."""
        return len(self.members)

    @property
    def sizes(self) -> tuple[int, ...]:
        """How many atoms are in each group."""
        return tuple(int(index.numel()) for index in self.members)

    @property
    def condition(self) -> float:
        """The condition number of the representative signals.

        The guide to whether there are too many groups. Representatives that
        start to look like one another leave nothing to prune with, and this
        is what says so before any data is matched: add groups while it stays
        flat, stop when it climbs.
        """
        values = torch.linalg.svdvals(self.representative.to(torch.complex64))
        return float(
            values[0] / values[-1].clamp_min(torch.finfo(torch.float32).tiny)
        )

    @classmethod
    def fit(cls, dictionary: torch.Tensor, count: int) -> Grouping:
        """Split a dictionary into ``count`` clusters of similar signals.

        Greedily: take an unassigned atom, give it the atoms most correlated
        with it, and repeat. The seed is free to choose -- what matters is
        that a group holds signals nearly parallel to one another, which is
        what makes their mean a useful stand-in for all of them.

        Parameters
        ----------
        dictionary:
            ``(atoms, contrasts)``, already in whatever basis the matching
            will happen in.
        count:
            How many groups to make. Read :attr:`condition` afterwards to see
            whether it was too many.

        Returns
        -------
        Grouping

        Raises
        ------
        ValueError
            If ``count`` is not between one and the number of atoms.
        """
        total = int(dictionary.shape[0])
        if not 1 <= count <= total:
            raise ValueError(
                f"count must be between one and the {total} atoms, got {count}"
            )
        normalized = _normalized(dictionary)
        size = -(-total // count)

        unassigned = torch.arange(total, device=dictionary.device)
        members: list[torch.Tensor] = []
        while unassigned.numel():
            pool = normalized[unassigned]
            alike = (pool @ pool[0].conj()).abs()
            take = min(size, int(unassigned.numel()))
            chosen = torch.topk(alike, take).indices
            members.append(unassigned[chosen])
            keep = torch.ones(
                unassigned.numel(), dtype=torch.bool, device=unassigned.device
            )
            keep[chosen] = False
            unassigned = unassigned[keep]

        representative = torch.stack(
            [_normalized(normalized[index].mean(0)[None])[0] for index in members]
        )
        return cls(
            members=tuple(members),
            representative=representative,
            atoms=tuple(normalized[index] for index in members),
        )

    def survivors(self, signals: torch.Tensor, prune: float) -> torch.Tensor:
        """Which groups each voxel could still be matched in.

        Parameters
        ----------
        signals:
            ``(voxels, contrasts)``, already normalized.
        prune:
            How far below its best group score a voxel still considers a
            group, as a fraction of that best score.

        Returns
        -------
        torch.Tensor
            ``(voxels, groups)`` of bool.
        """
        scores = correlate(signals, self.representative)
        best = scores.amax(-1, keepdim=True)
        return scores >= best * (1.0 - prune)

    def to(self, device: torch.device) -> Grouping:
        """This grouping, with everything on ``device``."""
        if self.representative.device == device:
            return self
        return Grouping(
            members=tuple(index.to(device) for index in self.members),
            representative=self.representative.to(device),
            atoms=tuple(value.to(device) for value in self.atoms),
        )


def match_in_groups(
    signals: torch.Tensor,
    grouping: Grouping,
    top_k: int,
    prune: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the best ``top_k`` atom indices and scores for each voxel.

    Parameters
    ----------
    signals:
        ``(voxels, contrasts)``, already normalized and in the dictionary's
        basis.
    grouping:
        The clustered dictionary, on the same device.
    top_k:
        How many candidates to keep per voxel.
    prune:
        The relative threshold below the best group score.

    Returns
    -------
    tuple
        ``(indices, scores)``, each ``(voxels, top_k)``.
    """
    voxels = int(signals.shape[0])
    scores = torch.full(
        (voxels, top_k), -1.0, dtype=signals.real.dtype, device=signals.device
    )
    indices = torch.zeros(
        (voxels, top_k), dtype=torch.int64, device=signals.device
    )
    keep = grouping.survivors(signals, prune)

    # One pass to find the groups anybody kept, so a group nobody did costs a
    # lookup rather than a gather over every voxel.
    for group in keep.any(0).nonzero(as_tuple=True)[0].tolist():
        rows = keep[:, group].nonzero(as_tuple=True)[0]
        atoms = grouping.atoms[group]
        local = correlate(signals[rows], atoms)
        take = min(top_k, int(atoms.shape[0]))
        found, where = torch.topk(local, take, dim=-1)
        candidates = torch.cat((scores[rows], found), dim=-1)
        named = torch.cat((indices[rows], grouping.members[group][where]), dim=-1)
        kept, choice = torch.topk(candidates, top_k, dim=-1)
        scores[rows] = kept
        indices[rows] = torch.gather(named, -1, choice)
    return indices, scores


def correlate(signals: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """The match score of every signal against every atom.

    ``|<s, d>|``, which is what makes matching blind to the complex scale a
    voxel's proton density and receive phase put on the signal.

    A dictionary simulated from a model that is real -- an SSFP fingerprint,
    an exponential decay -- is stored real, and half the arithmetic of a
    complex one. A measurement of it is still complex, and the two parts
    correlate against the real atoms separately: taking only the real part
    would score ``Re(rho)`` times the true correlation, which ranks the atoms
    correctly right up until a voxel whose phase is near a quarter turn, where
    it collapses to noise.

    Parameters
    ----------
    signals:
        ``(voxels, contrasts)``, normalized.
    atoms:
        ``(atoms, contrasts)``, normalized.

    Returns
    -------
    torch.Tensor
        ``(voxels, atoms)``, real.
    """
    if torch.is_complex(signals) and not torch.is_complex(atoms):
        return torch.hypot(signals.real @ atoms.mT, signals.imag @ atoms.mT)
    return (signals @ atoms.mH).abs()


# %% private module subroutines


def _normalized(signals: torch.Tensor) -> torch.Tensor:
    """Unit-norm rows, leaving an all-zero row alone rather than dividing."""
    norm = torch.linalg.vector_norm(signals, dim=-1).clamp_min(
        torch.finfo(signals.real.dtype).eps
    )
    return signals / norm[..., None]
