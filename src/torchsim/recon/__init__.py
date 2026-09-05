"""Physics-based reconstruction: the signal model, as an operator.

A quantitative reconstruction solves for parameter maps directly from k-space
rather than reconstructing one image per contrast and fitting afterwards. Its
forward operator is a chain of sampling, Fourier encoding, coil sensitivities
and the signal model, and only the last of those changes with the sequence.

This package supplies that last one and the outer loop that inverts the chain.
The encoding comes from elsewhere -- mri-nufft, or anything else exposing a
linear operator and its adjoint -- and is composed with, never reimplemented
here.
"""

from __future__ import annotations

__all__ = [
    "GaussNewton",
    "ModelOperator",
    "TrustRegion",
    "direct",
    "iterative",
]

from ._gauss_newton import (
    GaussNewton,
    LeastSquares,  # noqa: F401
    Linearization,  # noqa: F401
    Schedule,  # noqa: F401
    Solution,  # noqa: F401
    TrustRegion,
    direct,
    iterative,
)
from ._operator import ModelOperator
