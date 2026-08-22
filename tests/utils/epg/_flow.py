"""Flow dephasing, inflow and washout operators."""

__all__ = ["flow_op", "washout_op", "flow", "washout"]

from types import SimpleNamespace

import torch


def flow_op(
    v: torch.Tensor,
    time: torch.Tensor,
    nstates: int,
    total_dephasing: float,
    voxelsize: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare longitudinal and transverse flow dephasing operators.

    Parameters
    ----------
    v : torch.Tensor
        Apparent spin velocity in ``[m / s]``.
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
    J1 : torch.Tensor
        Flow dephasing operator for longitudinal states.
    J2 : torch.Tensor
        Flow dephasing operator for transverse states.

    """
    dk = total_dephasing / voxelsize

    # calculate dephasing order
    l = torch.arange(nstates, dtype=torch.float32, device=v.device)[:, None, None]
    k0 = dk * l

    # actual operator calculation
    J1 = torch.exp(-1j * k0 * v * time)
    J2 = torch.exp(-1j * (k0 + 0.5 * dk) * v * time)

    return J1, J2


def flow(
    states: SimpleNamespace,
    J1: torch.Tensor,
    J2: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply flow dephasing.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    J1 : torch.Tensor
        Flow dephasing operator for longitudinal states.
    J2 : torch.Tensor
        Flow dephasing operator for transverse states.

    Returns
    -------
    states : SimpleNamespace
        Output EPG states.

    """
    Fplus = states.Fplus
    Fminus = states.Fminus
    Z = states.Z

    # apply
    Fplus = Fplus.clone() * J2  # Transverse dephasing
    Fminus = Fminus.clone() * J2.conj()  # Transverse dephasing
    Z = Z.clone() * J1  # Longitudinal dephasing

    # prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus
    states.Z = Z

    return states


def washout_op(
    v: torch.Tensor,
    time: torch.Tensor,
    voxelsize: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare flow wash-out and in-flow operators.

    Parameters
    ----------
    v : torch.Tensor
        Apparent spin velocity in ``[m / s]``.
    time : torch.Tensor
        Time interval in ``[s]``.
    voxelsize : float, optional
        Voxel thickness along unbalanced direction in ``[m]``.
        The default is 1.0.

    Returns
    -------
    Win : torch.Tensor
        In-flow magnetization operator
    Wout : torch.Tensor
        Magnetization wash-out operator.

    """
    R = torch.abs(v / voxelsize)  # [1 / s]

    # flow wash-in/out
    Win = R * time
    Wout = 1 - Win

    # erase unphysical entries
    Win = (
        1.0
        - torch.heaviside(
            Win - 1.0, torch.as_tensor(1.0, dtype=R.dtype, device=R.device)
        )
    ) * Win + torch.heaviside(
        Win - 1.0, torch.as_tensor(1.0, dtype=R.dtype, device=R.device)
    )
    Wout = (
        torch.heaviside(Wout, torch.as_tensor(1.0, dtype=R.dtype, device=R.device))
        * Wout
    )

    return Win, Wout


def washout(
    states: SimpleNamespace,
    moving_states: SimpleNamespace,
    Win: torch.Tensor,
    Wout: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply magnetization wash-out/inflow.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    moving_states : SimpleNamespace
        In-flow EPG states.
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

    FplusMoving = moving_states.Fplus
    FminusMoving = moving_states.Fminus
    Zmoving = moving_states.Z

    # apply
    Fplus = Wout * Fplus.clone() + Win * FplusMoving
    Fminus = Wout * Fminus.clone() + Win * FminusMoving
    Z = Wout * Z.clone() + Win * Zmoving

    # prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus
    states.Z = Z

    return states
