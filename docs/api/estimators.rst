Parameter estimation
====================

.. currentmodule:: torchsim

Estimating tissue properties from a measured volume. A :class:`ParameterMapping`
states the problem over an :class:`Acquisition` -- what is unknown and over what
range, what is measured separately, how noisy -- and any :class:`Estimator`
fills it in, returning one named map per unknown.

:class:`DictionaryMatcher` and :class:`PERK` honour :func:`execution`, so a
volume larger than a card is streamed through it and a second card halves the
work.

Where a model has a single unknown its atoms lie on a curve rather than filling
a space, and :class:`LookupTable` reads the answer off that curve by
interpolating between them -- which is how an MP2RAGE T1 map is made, and what
removes the grid spacing from the estimate.

A dictionary is already clustered before anything is done to it -- neighbouring
tissues make nearly parallel signals -- so :class:`DictionaryMatcher` can be
given a ``groups=`` count and match against one representative signal per group
first, ruling out whole groups before any atom in them is scored. Compression
comes first and is global, so the clustering happens inside the one basis the
signals are in and the two savings multiply. :class:`Grouping` reports the
condition number that says whether there are too many groups to prune with.

:class:`NonlinearLeastSquares` fits the model itself rather than a sampling of
it, so a third parameter costs a third column of the Jacobian instead of
multiplying a grid. Every voxel steps together and carries its own damping.
Bounds are given as ``{name: (low, high)}`` and kept by fitting a transformed
variable, so no iterate ever leaves the interval; an equality constraint is
written into the model instead, by leaving out the degree of freedom it
removes.

The problem
-----------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ParameterMapping
   Estimator

Methods
-------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   DictionaryMatcher
   DictionaryMatch
   Grouping
   LookupTable
   NonlinearLeastSquares
   PERK
