"""Solving the linearized problem a Gauss-Newton step leaves behind.

Every Newton-type method reduces to the same question at each iterate: given
the Jacobian ``J`` there, the residual ``r``, and however much damping the
outer loop has decided on, what step ``d`` minimizes ``||J d + r||^2`` with
that damping? Which method it is -- Levenberg-Marquardt, an iteratively
regularized Gauss-Newton -- is a matter of where the damping comes from and
whether the step is then accepted, not of how this question is answered.

Answering it depends only on what ``J`` is. Where the operator is
voxel-diagonal -- a signal model with no encoding in front of it -- the
Jacobian is a stack of small independent blocks and the normal equations are
solved outright, one factorization per voxel, all of them at once. Where an
encoding operator has mixed the voxels together there is no such block
structure and the step is taken iteratively, from products with ``J`` alone.
"""

from __future__ import annotations

__all__ = ["normal_equations"]

import torch


def normal_equations(
    curvature: torch.Tensor, gradient: torch.Tensor, damping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``(J J' + mu I) d = -g`` per voxel, flagging the ones that cannot.

    A Cholesky reports failure per voxel rather than raising, so one flat
    voxel in a volume does not stop the others. A voxel it fails on is not a
    converged voxel -- it is one whose damping is still too small to make its
    system definite, and the caller should raise the damping and try again.

    Parameters
    ----------
    curvature:
        ``(voxels, parameters, parameters)``, ``J J'`` per voxel.
    gradient:
        ``(voxels, parameters)``, ``J r`` per voxel.
    damping:
        ``(voxels,)``, added to the diagonal.

    Returns
    -------
    tuple
        ``(step, singular)`` -- the step, zero where the system was singular,
        and the flag saying where that was.
    """
    size = curvature.shape[-1]
    eye = torch.eye(size, dtype=curvature.dtype, device=curvature.device)
    system = curvature + damping[:, None, None] * eye
    factor, info = torch.linalg.cholesky_ex(system)
    singular = info != 0
    safe = torch.where(singular[:, None, None], eye.expand_as(factor), factor)
    step = torch.cholesky_solve((-gradient)[..., None], safe).squeeze(-1)
    return torch.where(singular[:, None], torch.zeros_like(step), step), singular
