Parameter estimation
====================

.. currentmodule:: torchsim

Estimating tissue properties from a measured volume. An estimator is made from
the acquisition it inverts and whatever settings the method itself has;
:meth:`~PERK.fit` states what is unknown and over what range, draws the
training set from that acquisition, and :meth:`~PERK.map` returns one named
map per unknown::

    fitter = PERK(acquisition, n_features=1000)
    fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
    maps = fitter.map(volume)

The sampling belongs to the fit rather than to the estimator, so one estimator
can be fitted over a different range without being rebuilt. The properties are
named as keywords; a mapping given positionally is the same thing, and is the
way to name one that collides with a keyword ``fit`` itself takes. Handing
arrays in instead -- a dictionary that came from somewhere else -- is
``fit(signals=..., parameters=...)``, and then :meth:`~PERK.map` returns the
parameter columns as a tensor rather than named maps. Every method is
an :class:`Estimator`, which is where all of that lives; a method itself is
only what its two tensor steps do, so writing another one is subclassing it.

:class:`DictionaryMatcher` and :class:`PERK` honour :func:`execution`, so a
volume larger than a card is streamed through it and a second card halves the
work. ``PERK(..., stream=True)`` does the same for training: the dictionary is
simulated a chunk at a time and the covariances accumulate as it goes, so
memory follows the feature order rather than the number of training samples.
The dictionary is then never held, and so cannot have a basis read off it --
a ``rank`` needs the default.

A ``rank`` given to :meth:`~PERK.fit` puts the estimator to work in a temporal
:class:`Subspace`, and leaves that basis on it as :attr:`~PERK.subspace`
whichever method fitted it. That is what a subspace reconstruction is handed;
what it returns is already in the basis, so :meth:`~PERK.from_coefficients`
reads the maps off it without projecting a second time.

``subspace=`` is the other direction: work in a basis fitted elsewhere -- the
one another estimator carries, or one :func:`simulate_subspace` produced --
rather than fitting one here. That is how the reconstruction and the estimator
reading its coefficients come to hold the same basis by construction rather
than by both being asked for the same rank. A borrowed basis streams, too,
since each chunk can be projected as it is simulated.

Where a model has a single unknown its atoms lie on a curve rather than filling
a space, and :class:`LookupTable` reads the answer off that curve by
interpolating between them -- which is how an MP2RAGE T1 map is made, and what
removes the grid spacing from the estimate.

A dictionary is already clustered before anything is done to it -- neighbouring
tissues make nearly parallel signals -- so :class:`DictionaryMatcher` can be
given a ``groups=`` count and match against one representative signal per group
first, ruling out whole groups before any atom in them is scored. Compression
comes first and is global, so the clustering happens inside the one basis the
signals are in and the two savings multiply.

:class:`NonlinearLeastSquares` fits the model itself rather than a sampling of
it, so a third parameter costs a third column of the Jacobian instead of
multiplying a grid. Every voxel steps together and carries its own damping.
Bounds are given as ``{name: (low, high)}`` and kept by fitting a transformed
variable, so no iterate ever leaves the interval; an equality constraint is
written into the model instead, by leaving out the degree of freedom it
removes.

The base class
--------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Estimator

Methods
-------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   DictionaryMatcher
   MatchResult
   LookupTable
   NonlinearLeastSquares
   PERK
