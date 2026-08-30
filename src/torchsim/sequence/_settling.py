"""Reading the settled signal off a few playings instead of running thousands.

A description played over and over is an affine recursion on its states,
``x(n) = T x(n-1) + b``. Subtracting the fixed point leaves ``e(n) = T e(n-1)``,
so expanding in the eigenvectors of ``T`` and reading the sample as a linear
functional of the state gives

    s(n) = s(inf) + sum_j a_j lambda_j**n

exactly, for as long as the recursion is linear -- which it is. A sequence of
that form has its limit determined by finitely many of its terms: ``2m+1`` of
them fix ``s(inf)`` together with ``m`` amplitudes and ``m`` eigenvalues. That
is the Shanks transform of order ``m``, evaluated here by Wynn's epsilon
algorithm.

So this is not a fit or a guess at a trend. It solves for the limit of a
sequence whose form is known, and it recovers the eigenvalues of the transition
operator on the way -- without ever forming that operator, which for an
unbalanced train would be a dense matrix per voxel.

Its weakness is conditioning: the transform divides by differences of
differences, and once the terms are close together those are round-off. The
order is therefore chosen by how little the answer moves when it is raised,
rather than taken as high as the terms allow.
"""

from __future__ import annotations

__all__: list[str] = []

import torch

# Below this the playings have stopped moving on their own and the limit is the
# last of them; the transform would be dividing round-off by round-off.
SETTLED = 1e-6


def settled(playings: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The limit of a sequence of playings, and how far it is trusted.

    ``playings`` is indexed by playing on its leading axis. Returns the settled
    samples and the relative distance between the last two orders tried, which
    is what says whether raising the order is still finding anything.

    Wynn's recurrence is walked once and read at every even column, each of
    which is the Shanks estimate of one order higher than the last. Restarting
    it per order would pass over the whole set of playings a dozen times for
    each, which at volume scale is where all the time would go.
    """
    # Read for the decisions only -- whether to transform at all, and which
    # order to stop at -- so detached: none of it reaches the answer.
    watched = playings.detach()
    scale = float(watched[-1].abs().max())
    if scale == 0.0:
        return playings[-1], 0.0
    drift = float((watched[-1] - watched[-2]).abs().max()) / scale
    if drift < SETTLED:
        # It arrived by itself; there is nothing left for the transform to do.
        return playings[-1], drift

    working = playings.to(torch.complex128 if playings.is_complex() else torch.float64)
    previous = torch.zeros_like(working[:1]).expand_as(working)
    current = working
    best, residual = playings[-1], drift
    # The last entry of a column is the estimate reading the most recent terms,
    # which are the ones closest to the limit.
    reached = working[-1]
    for column in range(1, working.shape[0]):
        step = current[1:] - current[:-1]
        # A step of zero is a term that has already arrived, whose reciprocal
        # the recurrence would take; left alone it carries through unchanged.
        safe = torch.where(step.abs() > 0, step, torch.ones_like(step))
        current, previous = previous[1 : current.shape[0]] + 1.0 / safe, current
        if column % 2 or not torch.isfinite(current[-1]).all():
            continue
        estimate = current[-1]
        moved = float((estimate - reached).detach().abs().max()) / scale
        if moved < residual:
            best, residual = estimate.to(playings.dtype), moved
        reached = estimate
    return best, residual
