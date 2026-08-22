"""Adiabatic Inversion operator."""

__all__ = ["adiabatic_inversion"]

from types import SimpleNamespace

import torch


def adiabatic_inversion(
    states: SimpleNamespace,
    inv_efficiency: float | torch.Tensor = 1.0,
) -> SimpleNamespace:
    """
    Apply adiabatic inversion to EPG states.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG states.
    inv_efficiency : float | torch.Tensor, optional
        Inversion efficiency. The default is 1.0.

    Returns
    -------
    SimpleNamespace
        Output EPG states.

    """
    states.Z = -inv_efficiency * states.Z.clone()
    return states
