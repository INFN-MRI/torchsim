# Parameter estimation

```{admonition} TL;DR
:class: tldr

- An estimator is fitted from the simulator of the acquisition it inverts, and returns one named map per unknown.
- Dictionary matching and lookup tables search simulated signals; nonlinear least squares fits the model; PERK regresses on a prior.
- A grid grows with the product of the parameter ranges; a fit grows by one Jacobian column per parameter.
```

You give an estimator the simulator of your acquisition and the range of each
unknown, and it returns parameter maps from a measured volume.

## Capabilities

::::{container} capabilities

- **Draws the training set from the simulator being inverted, and returns one named map per unknown.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.Estimator`. Tests: *a mapping recovers a tissue it was trained over*; *the maps come back named and shaped like the volume* (`test_mapping.py`).
  :::
- **Matches by normalized inner products, and returns M0 from the score and the stored atom norm.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.DictionaryMatcher`. Tests: *dictionary matcher recovers complex scaled atoms* (`test_dictionary.py`); *a match answers with the density without touching an atom* (`test_uncertainty.py`).
  :::
- **Works in a low-rank temporal subspace, fitted here or passed in, and reports the signal energy it retains.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.Subspace`. Tests: *retained is the error it will cost* (`test_subspace.py`); *a subspace maps as well as the contrasts it replaces* (`test_mapping.py`).
  :::
- **Clusters the dictionary and excludes whole groups before scoring their atoms.**

  :::{dropdown} Show code and tests
  Code: `DictionaryMatcher(groups=...)`. Tests: *grouped matching finds what direct matching finds*; *most of the dictionary is ruled out* (`test_grouped.py`).
  :::
- **Reads one unknown off a monotonic signal curve by interpolation.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.LookupTable`. Tests: *interpolating beats the grid it was built on*; *a t1 map is read from the two blocks* (`test_lookup.py`).
  :::
- **Fits every voxel at once by Levenberg-Marquardt, inside box bounds.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.NonlinearLeastSquares`. Tests: *it finds what scipy finds*; *no iterate ever leaves the bounds* (`test_nlls.py`).
  :::
- **Trains a kernel ridge regression on random Fourier features.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.PERK`. Tests: *streaming reaches what holding the dictionary reaches* (`test_perk.py`); *perk is comparable to dictionary matching under noise* (`test_mapping_comparison.py`).
  :::
- **Takes separately measured maps, such as B1, in PERK and nonlinear least squares.**

  :::{dropdown} Show code and tests
  Code: `fit(known=...)`. Tests: *a known property reaches the simulator and is asked for again* (`test_mapping.py`); *a property measured separately reaches the model* (`test_nlls.py`).
  :::
- **Returns an uncertainty map where the method defines one.**

  :::{dropdown} Show code and tests
  Code: `map(volume, uncertainty=True)`. Tests: *least squares states its own standard error*; *perk states a spread without repeating itself* (`test_uncertainty.py`).
  :::

::::

## Division of responsibility

- **Spatial regularization.** Every estimator is voxel-wise; a spatial prior belongs to {doc}`model-based-reconstruction`.
- **Neural-network estimators.** PERK's estimate is differentiable with respect to the signal, so it can sit inside one.
- **A global minimum** for nonlinear least squares, whose answer depends on its starting point.
- **Declared equality constraints.** Two fractions summing to one are written into the model, with one of them as the unknown.

## Method

```{figure} /generated/figures/parameter_estimation.png
:width: 100%
:alt: Relative T2 error across the T2 range for six estimators, and median error against mapping time per voxel.

T2 and M0 from a 16-echo spin echo at 1 % noise. The lookup table reads the
ratio of late to early echoes. On the same 24-point grid the dictionary is
limited by the spacing and the table is not. Nonlinear least squares matches
the fine dictionary at one to two orders of magnitude more time. PERK is least
accurate at the short-T2 end of its prior. Times are measured at build time.
```

### Data requirements

- **Dictionary matching:** a grid. A subspace shortens each inner product to the rank; grouping reduces how many are taken. A `prune` set too small can exclude the group holding the match.
- **Lookup table:** one unknown, and a `combine` that is monotonic in it, such as the MP2RAGE unified image.
- **Nonlinear least squares:** a starting point strictly inside the bounds, by default the median of the training draws. Each iteration costs one forward-mode pass per unknown.
- **PERK:** a prior and the noise level of the scan. Mapping costs one feature projection per voxel, whatever the number of unknowns.

### Nuisance parameters

Normalization removes M0 and the receive phase from a match. M0 is recovered as
the measurement norm times the score divided by the atom norm;
`PERK(normalize=True)` regresses the reciprocal signal norm instead. A nuisance
parameter that survives normalization, such as the noise floor of a magnitude
image, has to be put on the grid.

### Uncertainty

Nonlinear least squares reports its standard error, from the inverse Fisher
matrix at the solution and the noise level given to `fit`. PERK regresses the absolute error of its training estimates on
the same features, so its uncertainty includes the bias toward the prior; it
holds at the signal amplitude it was trained at. Both are read against
{func}`~blochsim.crlb`, the bound for an unbiased estimator. Grid methods raise
`NotImplementedError` when asked for one.

## Examples

- {doc}`../generated/autoexamples/02-parameter-inference/01-dictionary-matching`: an MRF train matched in full, in a rank-4 subspace, and in 32 groups.
- {doc}`../generated/autoexamples/02-parameter-inference/02-lookup-table`: MP2RAGE T1 by lookup table and by dictionary, against the number of grid points.
- {doc}`../generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares`: multi-echo T2 with a noise-floor offset, fitted and matched.
- {doc}`../generated/autoexamples/02-parameter-inference/04-perk`: PERK at three regression sizes, and its uncertainty against the Cramér-Rao bound.
