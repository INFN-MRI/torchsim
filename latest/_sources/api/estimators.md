# Parameter estimation

```{eval-rst}
.. currentmodule:: blochsim
```

Estimators map a measured volume to tissue parameter maps. An estimator is
made from the acquisition it inverts, {meth}`~Estimator.fit` states the
unknowns and their ranges and draws the training set from the acquisition, and
{meth}`~Estimator.map` returns one named map per unknown:

```python
fitter = PERK(acquisition, n_features=1000)
fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
maps = fitter.map(volume)
```

`fit(signals=..., parameters=...)` takes arrays instead, and `map` then returns
the parameter columns as a tensor. A `rank` or `subspace` given to `fit` works
in a temporal {class}`Subspace`, left on the estimator as
{attr}`~Estimator.subspace`. {class}`DictionaryMatcher` and {class}`PERK`
honour {func}`execution`. `map(volume, uncertainty=True)` also returns an
uncertainty map; its meaning depends on the method and is stated at
{meth}`~Estimator.uncertainty_of`. See the
{doc}`../explanations/parameter-estimation` page for the methods.

## The base class

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Estimator
```

## Methods

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   DictionaryMatcher
   LookupTable
   NonlinearLeastSquares
   PERK
```
