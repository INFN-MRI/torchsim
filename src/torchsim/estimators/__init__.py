"""Quantitative-MRI estimators built around TorchSim signal models.

An estimator is made from the acquisition it inverts and whatever settings the
method itself has. ``fit`` states which tissue properties are unknown and over
what range, which are measured separately, and how noisy the measurement is,
and draws its training set from that acquisition -- so what it is trained on
cannot drift from what the scanner will produce::

    fitter = PERK(acquisition, n_features=1000)
    fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
    maps = fitter.map(volume)

A ``rank`` fits a temporal basis over the training set and leaves it on the
estimator; ``subspace=`` works in one fitted elsewhere, which is how a
reconstruction and the estimator reading its coefficients come to agree on the
basis by construction.

Handing arrays in instead -- a dictionary that came from somewhere else -- is
``fit(signals=..., parameters=...)``, and then ``map`` returns the parameter
columns as a tensor rather than named maps.
"""

from __future__ import annotations

__all__ = [
    "DictionaryMatcher",
    "Estimator",
    "LookupTable",
    "MatchResult",
    "NonlinearLeastSquares",
    "PERK",
]

from ._dictionary import DictionaryMatcher, MatchResult
from ._lookup import LookupTable
from ._mapping import Estimator
from ._nlls import NonlinearLeastSquares
from ._perk import PERK
