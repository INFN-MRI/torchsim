"""Lining a closed form's properties up against the contrasts it evaluates."""

from __future__ import annotations

__all__ = ["across_contrasts"]

from collections.abc import Mapping
from typing import Any

import torch

from ..sequence._array import as_torch


def across_contrasts(properties: Mapping[str, Any], *sequence: Any) -> dict[str, Any]:
    """Give every property a trailing axis where the sequence has contrasts.

    A closed form is elementwise in the tissue, so one evaluation covers every
    voxel and every contrast at once -- provided the two occupy different axes.
    A sequence of scalars declares no contrast axis at all, and the properties
    then keep their own shape, which is what makes a single-contrast call
    return one value per voxel rather than a column of one.

    Parameters
    ----------
    properties:
        The declared values the caller passed.
    sequence:
        The sequence values the signal is evaluated at.

    Returns
    -------
    dict
        The properties, ready to broadcast against the sequence.
    """
    contrasts = torch.broadcast_shapes(*(as_torch(value).shape for value in sequence))
    if not contrasts:
        return dict(properties)
    return {name: value[..., None] for name, value in properties.items()}
