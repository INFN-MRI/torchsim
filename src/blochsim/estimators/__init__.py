"""Quantitative-MRI estimators built on BlochSim signal models.

An estimator is made from the acquisition it inverts; ``fit`` names the
unknowns and their ranges and draws the training set from that acquisition::

    fitter = PERK(acquisition, n_features=1000)
    fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
    maps = fitter.map(volume)
"""

from __future__ import annotations

__all__ = [
    "DictionaryMatcher",
    "LookupTable",
    "PERK",
]

from ._dictionary import DictionaryMatcher, MatchResult  # noqa: F401
from ._lookup import LookupTable
from ._perk import PERK
