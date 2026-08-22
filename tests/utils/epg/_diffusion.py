"""Diffusion damping operator."""

__all__ = ["diffusion_op", "diffusion"]

from types import SimpleNamespace

import torch


def diffusion_op(
    D: torch.Tensor,
    time: torch.Tensor,
    nstates: int,
    total_dephasing: float,
    voxelsize: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare longitudinal and transverse diffusion damping operators.

    Parameters
    ----------
    D : torch.Tensor
        Apparent diffusion coefficient in ``[m**2 / s]``.
    time : torch.Tensor
        Time interval in ``[s]``.
    nstates : int
        Number of EPG states
    total_dephasing : float
        Total dephasing induced by gradient in ``[rad]``.
    voxelsize : float, optional
        Voxel thickness along unbalanced direction in ``[m]``.
        The default is 1.0.

    Returns
    -------
    D1 : torch.Tensor
        Diffusion damping operator for longitudinal states.
    D2 : torch.Tensor
        Diffusion damping operator for transverse states.

    """
    k0_2 = (total_dephasing / voxelsize) ** 2

    # actual operator calculation
    b_prime = k0_2 * time * 1e-3

    # calculate dephasing order
    l = torch.arange(nstates, dtype=torch.float32, device=D.device)[:, None, None]
    lsq = l**2

    # calculate b-factor
    b1 = b_prime * lsq
    b2 = b_prime * (lsq + l + 1.0 / 3.0)

    # actual operator calculation
    D1 = torch.exp(-b1 * D * 1e-9)
    D2 = torch.exp(-b2 * D * 1e-9)

    return D1, D2


def diffusion(
    states: SimpleNamespace,
    D1: torch.Tensor,
    D2: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply diffusion damping.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    D1 : torch.Tensor
        Diffusion damping operator for longitudinal states.
    D2 : torch.Tensor
        Diffusion damping operator for transverse states.

    Returns
    -------
    states : SimpleNamespace
        Output EPG states.

    """
    Fplus = states.Fplus
    Fminus = states.Fminus
    Z = states.Z

    # apply
    Fplus = Fplus.clone() * D2  # Transverse damping
    Fminus = Fminus * D2  # Transverse damping
    Z = Z.clone() * D1  # Longitudinal damping

    # prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus
    states.Z = Z

    return states
