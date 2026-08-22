"""Perfect spoiling operator."""

__all__ = ["spoil"]

from types import SimpleNamespace


def spoil(states: SimpleNamespace) -> SimpleNamespace:
    """
    Null transverse magnetization

    Parameters
    ----------
    states : SimpleNamespace
        Input EPG matrix.

    Returns
    -------
    SimpleNamespace
        Output EPG matrix.

    """
    states.Fplus[:] = 0.0
    states.Fminus[:] = 0.0
    return states
