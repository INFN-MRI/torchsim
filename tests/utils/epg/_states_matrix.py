"""Generate EPG states matrix."""

__all__ = ["states_matrix"]

from types import SimpleNamespace

import numpy.typing as npt
import torch


def states_matrix(
    device: torch.device,
    nstates: int,
    nlocs: int = 1,
    ntrans_pools: int = 1,
    nlong_pools: int = 1,
    weight: float | npt.ArrayLike = 1.0,
) -> SimpleNamespace:
    """
    Generate EPG states matrix.

    Parameters
    ----------
    device : torch.device
        Computational device.
    nstates : int
        Numer of EPG states.
    nlocs : int, optional
        Number of spatial locations. The default is 1.
    ntrans_pools : int, optional
        Number of pools for transverse magnetization. The default is 1.
    nlong_pools : int, optional
        Number of pools for longitudinal magnetization. The default is 1.
    weight : float | npt.ArrayLike, optional
        Fractional weight for the different pools of shape ``(nlong_pools,)``.
        The default is ``1.0``.

    Returns
    -------
    states : SimpleNamespace
        EPG states matrix of with fields:

            * Fplus: transverse F+ states of shape (nstates, nlocs, ntrans_pools)
            * Fminus: transverse F- states of shape (nstates, nlocs, ntrans_pools)
            * Z: longitudinal Z states of shape (nstates, nlocs, nlong_pools)

    """
    Fplus = torch.zeros(
        (nstates, nlocs, ntrans_pools), dtype=torch.complex64, device=device
    )
    Fminus = torch.zeros(
        (nstates, nlocs, ntrans_pools), dtype=torch.complex64, device=device
    )
    Z = torch.zeros((nstates, nlocs, nlong_pools), dtype=torch.complex64, device=device)
    Z[0] = 1.0
    Z = Z * weight

    return SimpleNamespace(Fplus=Fplus, Fminus=Fminus, Z=Z)
