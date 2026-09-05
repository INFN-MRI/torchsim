"""The two-pool exchange matrix, built independently of the kernels.

A pool leaves at a rate weighted by the share of the pool it leaves for, and
each diagonal entry is whatever makes its row conserve magnetization. Written
out here so a test has a generator to compare against that the simulator had
no hand in.
"""

from __future__ import annotations

__all__ = ["build_two_pool_exchange_matrix"]

import torch


def build_two_pool_exchange_matrix(
    weight: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Return the exchange generator of a two-pool system.

    Parameters
    ----------
    weight:
        The fractional share of each pool, along a trailing axis of two.
    k:
        The non-directional exchange rate between them, in 1/s.

    Returns
    -------
    torch.Tensor
        The exchange matrix, with a trailing ``(2, 2)``.
    """
    kab = k * weight[..., 1]
    kba = k * weight[..., 0]
    rows = torch.stack(
        (
            torch.stack((-kab, kba), dim=-1),
            torch.stack((kab, -kba), dim=-1),
        ),
        dim=-2,
    )
    # Each row conserves what it moves, so the diagonal is the negated sum of
    # the rest of that row.
    for pool in range(2):
        rows[..., pool, pool] = 0.0
        rows[..., pool, pool] = -rows[..., pool].sum(dim=-1)
    return rows
