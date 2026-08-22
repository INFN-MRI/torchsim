"""EPG states shift operator."""

__all__ = ["shift"]

from types import SimpleNamespace

import torch


def shift(states: SimpleNamespace, delta: int = 1) -> SimpleNamespace:
    """
    Shift transverse EPG states due to gradient dephasing.

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG matrix.
    delta : int, optional
        Integer states shift. The default is 1.

    Returns
    -------
    SimpleNamespace
        Output EPG matrix.

    """
    Fplus = states.Fplus
    Fminus = states.Fminus

    # Apply shift
    Fminus = torch.roll(Fminus, -delta, -3)  # Shift F- states
    Fplus = torch.roll(Fplus, delta, -3)  # Shift F+ states
    Fminus[-1] = 0.0  # Zero highest F- state
    Fplus[0] = Fminus[0].conj()  # Fill in lowest F+ state

    # Prepare for output
    states.Fplus = Fplus
    states.Fminus = Fminus

    return states
