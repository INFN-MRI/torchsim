"""Longitudinal Relaxation operator."""

__all__ = [
    "longitudinal_relaxation_op",
    "longitudinal_relaxation_exchange_op",
    "longitudinal_relaxation",
    "longitudinal_relaxation_exchange",
]

from types import SimpleNamespace

import torch

from ._utils import matrix_exp


def longitudinal_relaxation_op(
    R1: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare longitudinal relaxation and recovery operators.

    Parameters
    ----------
    R1 : torch.Tensor
        Longitudinal relaxation rate in ``[1/s]``.
    time : torch.Tensor
        Time interval in ``[s]``.

    Returns
    -------
    E1 : torch.Tensor
        Longitudinal relaxation operator.
    rE1 : torch.Tensor
        Longitudinal recovery operator.

    """
    E1 = torch.exp(-R1 * time)
    rE1 = 1 - E1

    return E1, rE1


def longitudinal_relaxation_exchange_op(
    weight: torch.Tensor, k: torch.Tensor, R1: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare longitudinal relaxation and recovery operators in presence of exchange.

    Parameters
    ----------
    weight : torch.Tensor
        Fractional weight for the different pools of shape ``(npools,)``.
    k : torch.Tensor
        Directional exchange rate matrix in ``[1/s]`` of shape ``(npools, npools)``.
    R1 : torch.Tensor
        Longitudinal relaxation rate in ``[1/s]`` of shape ``(npools,)``.
    time : torch.Tensor
        Time interval in ``[s]``.

    Returns
    -------
    E1 : torch.Tensor
        Longitudinal relaxation and exchange operator.
    rE1 : torch.Tensor
        Longitudinal recovery and exchange operator.

    """
    # get npools
    npools = R1.shape[-1]

    # cast to complex
    R1 = R1.to(torch.complex64)

    # recovery
    Id = torch.eye(npools, dtype=R1.dtype, device=R1.device)
    C = weight * R1

    # coefficients
    lambda1 = k - R1 * Id

    # actual operators
    E1 = matrix_exp(lambda1 * time)
    rE1 = torch.einsum("...ij,...j->...i", (E1 - Id), torch.linalg.solve(lambda1, C))

    return E1, rE1


def longitudinal_relaxation(
    states: SimpleNamespace,
    E1: torch.Tensor,
    rE1: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply longitudinal relaxation and recovery.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    E1 : torch.Tensor
        Longitudinal relaxation operator.
    rE1 : torch.Tensor
        Longitudinal recovery operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    # parse
    Z = states.Z.clone()

    # apply
    Z = Z.clone() * E1  # decay
    Z[0] = Z[0].clone() + rE1  # regrowth

    # prepare for output
    states.Z = Z
    return states


def longitudinal_relaxation_exchange(
    states: SimpleNamespace,
    E1: torch.Tensor,
    rE1: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply longitudinal relaxation and recovery in presence of exchange.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    E1 : torch.Tensor
        Longitudinal relaxation exchange operator.
    rE1 : torch.Tensor
        Longitudinal recovery exchange operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    # parse
    Z = states.Z.clone()

    # apply
    Z = torch.einsum("...ij,...j->...i", E1, Z.clone())
    Z[0] = Z[0].clone() + rE1

    # prepare for output
    states.Z = Z
    return states
