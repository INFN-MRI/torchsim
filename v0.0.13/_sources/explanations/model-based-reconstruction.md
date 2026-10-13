# Model-based reconstruction

```{admonition} TL;DR
:class: tldr

- blochsim supplies the signal model of the forward operator; the encoding and the Gauss-Newton loop that inverts it come from bartorch or another library.
- The linear route reconstructs coefficients in a temporal subspace; the nonlinear route solves for the parameter maps themselves.
- Jacobian products cost one forward-mode or one reverse-mode pass, and the Jacobian is never formed.
```

You reconstruct parameter maps directly from k-space, with the signal model
inside the forward operator rather than fitted to images afterwards.

## Capabilities

::::{container} capabilities

- **Wraps any simulator as an operator from parameter maps to contrast images, with a complex amplitude for proton density and receive phase.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.ModelOperator`. Tests: *the operator is the model*; *the amplitude costs no pass* (`test_operator.py`).
  :::
- **Applies the Jacobian and its adjoint without forming it.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.ModelOperator.A_jvp`, {meth}`~blochsim.ModelOperator.A_vjp`. Tests: *the directional derivative is the jacobian contracted*; *the adjoint is the adjoint* (`test_operator.py`).
  :::
- **Keeps box bounds at every point it is evaluated at, by solving for a transformed variable, and imposes an equality constraint by writing the model on what is free.**

  :::{dropdown} Show code and tests
  Code: `ModelOperator(bounds=...)`. Tests: *a bound holds however far the variable goes* (`test_operator.py`); *an equality constraint is the model written on what is free* (`test_constraints.py`).
  :::
- **Hands the temporal basis to a subspace encoding operator, and reads maps from the coefficients it returns.**

  :::{dropdown} Show code and tests
  Code: {attr}`~blochsim.Subspace.modes`, {meth}`~blochsim.estimators.DictionaryMatcher.from_coefficients`. Tests: *the basis goes out in the layout the encoding reads*; *one mapping supplies the basis and reads what comes back* (`test_subspace_route.py`).
  :::
- **Converts to a deepinv physics, so deepinv's optimizers and priors apply to the composed chain.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.ModelOperator.physics`. Tests: *it composes with a linear encoding operator*; *a deepinv optimizer drives the composed operator* (`test_deepinv.py`).
  :::

::::

## Division of responsibility

- **The encoding**: sampling, non-uniform FFT, coil sensitivities, density compensation. bartorch's operators, mri-nufft's and deepinv's `LinearPhysics` all compose with the model operator.
- **The solver.** bartorch's `nlop.ModelOperator` turns the model operator into a BART nonlinear operator; `nlop.IRGNM` inverts it under an encoding, `apps.moba` from k-space and `apps.mobafit` voxel by voxel from images.
- **Regularizers**, which enter through the linear solver of each Gauss-Newton step (`IRGNM(inner=...)`).

## Method

```{figure} /generated/figures/model_based_reconstruction.png
:width: 100%
:alt: Four T2 maps of a disc phantom: the truth, zero-filled images matched to a dictionary, a rank-3 subspace reconstruction, and a Gauss-Newton reconstruction.

An 8-echo spin echo (TE 10 to 150 ms) on a 48 × 48 phantom, sampled at one
phase-encode line in three per echo, shifted from echo to echo, plus six
central lines. blochsim supplies the dictionary, the basis and the model
operator; the masked FFT is bartorch's, the conjugate-gradient solve of the
subspace route is written in the figure's code, and the nonlinear route is
bartorch's `nlop.IRGNM`. Errors are the mean relative T2 error inside the phantom.
```

### Forward operator

The forward operator is the chain $F = P\,\mathcal{F}\,C\,M$ of sampling,
Fourier encoding, coil sensitivities and the signal model. Only $M$ depends on
the sequence. {class}`~blochsim.ModelOperator` is $M$: maps in `(..., channels)`,
images out in `(..., contrasts)`. The encoding receives the contrast axis at
axis 1, as deepinv and mri-nufft lay it out.

### Linear route

An estimator fitted with `rank` carries a {class}`~blochsim.Subspace`.
{attr}`~blochsim.Subspace.modes` gives the basis as `(rank, contrasts)`, the
layout mri-nufft's `MRISubspace` and BART's `pics -B` read. The reconstruction
is linear in the coefficients, has one minimum and needs no starting point.
{meth}`~blochsim.estimators.DictionaryMatcher.from_coefficients` then maps the coefficients
without projecting them again. A signal that needs many components, such as a
phase-modulated train or a model with several parameters, gives no small
basis.

### Nonlinear route

Each Gauss-Newton step linearizes $F$ at the current maps and solves the
damped least-squares problem for the step. With an encoding, the step is
solved by an iterative solver from products alone: one encoding application
and one `A_jvp` forward, one adjoint encoding and one `A_vjp` back. These are
the three products bartorch's `nlop.ModelOperator` hands to BART, so the loop
is BART's iteratively regularized Gauss-Newton method. Composing through
`physics()` instead lets deepinv differentiate the whole chain by autograd,
which gives up these analytic products.

## Examples

- {doc}`../generated/autoexamples/04-model-based-imaging/01-linear-subspace`: radial multi-echo T2 by gridding, per-echo CG, and a rank-3 subspace with mri-nufft.
- {doc}`../generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion`: the same data reconstructed by bartorch's Gauss-Newton method with the model inside the operator.
