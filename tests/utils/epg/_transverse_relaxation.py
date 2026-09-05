"""Transverse Relaxation operator."""

__all__ = [
    "transverse_relaxation_op",
    "transverse_relaxation_exchange_op",
    "transverse_relaxation",
    "transverse_relaxation_exchange",
]

from types import SimpleNamespace

import torch

from ._utils import matrix_exp


def transverse_relaxation_op(R2: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """
    Prepare transverse relaxation operator.

    Parameters
    ----------
    R2 : torch.Tensor
        Transverse relaxation rate in ``[1/s]``.
    time : torch.Tensor
        Time interval in ``[s]``.

    Returns
    -------
    E2 : torch.Tensor
        Transverse relaxation operator.

    """
    E2 = torch.exp(-R2 * time)

    return E2


def transverse_relaxation_exchange_op(
    k: torch.Tensor, R2: torch.Tensor, time: torch.Tensor, df: torch.Tensor = 0.0
) -> torch.Tensor:
    """
    Prepare transverse relaxation operator in presence of exchange.

    Parameters
    ----------
    k : torch.Tensor
        Directional exchange rate matrix in ``[1/s]`` of shape ``(npools, npools)``.
    R2 : torch.Tensor
        Transverse relaxation rate in ``[1/s]`` of shape ``(npools,)``.
    time : torch.Tensor
        Time interval in ``[s]``.
    df : torch.Tensor, optional
        Chemical exchange in ``[rad/s]`` of shape ``(npools,)``.

    Returns
    -------
    E2 : torch.Tensor
        Transverse relaxation operator.

    """
    R2tot = R2 + 1j * 2 * torch.pi * df

    # get npools
    npools = R2tot.shape[-1]

    # recovery
    Id = torch.eye(npools, dtype=R2tot.dtype, device=R2tot.device)

    # coefficients
    lambda2 = k[:npools, :npools] - R2tot[:, None] * Id  # assume MT pool is the last

    # actual operators
    E2 = matrix_exp(lambda2 * time)

    return E2


def transverse_relaxation(
    states: SimpleNamespace,
    E2: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply transverse relaxation.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    E2 : torch.Tensor
        Transverse relaxation operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    Fplus = states.Fplus
    Fminus = states.Fminus

    # apply
    Fplus = Fplus.clone() * E2  # F+
    Fminus = Fminus.clone() * E2  # F-

    # prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus
    return states


def transverse_relaxation_exchange(
    states: SimpleNamespace,
    E2: torch.Tensor,
) -> SimpleNamespace:
    """
    Apply transverse relaxation in presence of exchange.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    E2 : torch.Tensor
        Transverse relaxation exchange operator.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    Fplus = states.Fplus
    Fminus = states.Fminus

    # apply
    Fplus = torch.einsum("...ij,...j->...i", E2, Fplus.clone())
    Fminus = torch.einsum("...ij,...j->...i", E2.conj(), Fminus.clone())

    # prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus
    return states
