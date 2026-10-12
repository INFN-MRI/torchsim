"""Model-based reconstruction: the signal model as an operator, and a Gauss-Newton solver.

The Fourier encoding is supplied externally (mri-nufft, deepinv) and composed
with the operator.
"""

from __future__ import annotations

__all__ = [
    "GaussNewton",
    "ModelOperator",
    "Schedule",
    "TrustRegion",
    "direct",
    "iterative",
]

from ._gauss_newton import (
    GaussNewton,
    LeastSquares,  # noqa: F401
    Linearization,  # noqa: F401
    Schedule,
    Solution,  # noqa: F401
    TrustRegion,
    direct,
    iterative,
)
from ._operator import ModelOperator
