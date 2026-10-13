# Parameter estimation

```{eval-rst}
.. currentmodule:: blochsim.estimators
```

Estimators map a measured volume to tissue parameter maps. An estimator is
made from the acquisition it inverts, `fit` states the unknowns and their
ranges and draws the training set from the acquisition, and `map` returns one
named map per unknown:

```python
fitter = PERK(acquisition, n_features=1000)
fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
maps = fitter.map(volume)
```

`fit(signals=..., parameters=...)` takes arrays instead, and `map` then returns
the parameter columns as a tensor. A `rank` or `subspace` given to `fit` works
in a temporal {class}`~blochsim.Subspace`, left on the estimator as its
`subspace` attribute. `map(volume, execution="auto")` streams a volume held on
the host through the CUDA devices. `map(volume, uncertainty=True)` also
returns an uncertainty map; its meaning depends on the method and is stated at
{meth}`PERK.uncertainty_of`. A per-voxel Levenberg-Marquardt fit of the
same model is bartorch's `apps.mobafit`, over a
{class}`~blochsim.ModelOperator`. See the
{doc}`../explanations/parameter-estimation` page for the methods.

## Methods

| Object | Description |
| --- | --- |
| {obj}`~blochsim.estimators.DictionaryMatcher` | Match signals to dictionary atoms by normalized complex inner product. |
| {obj}`~blochsim.estimators.LookupTable` | Read one unknown back from a monotonic signal curve. |
| {obj}`~blochsim.estimators.PERK` | Parameter estimation via regression with kernels (PERK). |
