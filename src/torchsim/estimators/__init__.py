"""Quantitative-MRI estimators built around TorchSim signal models.

A mapping problem is stated once, with :class:`ParameterMapping`, and filled
in by any :class:`Estimator`: the two that ship here, or one a user writes.
"""

from __future__ import annotations

__all__ = [
    "DictionaryMatch",
    "DictionaryMatcher",
    "Estimator",
    "LookupTable",
    "PERK",
    "ParameterMapping",
]

from ._dictionary import DictionaryMatch, DictionaryMatcher
from ._lookup import LookupTable
from ._mapping import Estimator, ParameterMapping
from ._perk import PERK
