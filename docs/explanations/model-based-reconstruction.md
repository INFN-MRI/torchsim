# Model-based reconstruction

```{admonition} TL;DR
:class: tldr

- blochsim supplies the signal model of the forward operator and the Gauss-Newton loop that inverts it; the Fourier encoding comes from another library.
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
- **Keeps box bounds at every iterate, under an encoding as in a voxel-wise fit.**

  :::{dropdown} Show code and tests
  Code: `ModelOperator(bounds=...)`. Tests: *a bound holds at every iterate under an encoding* (`test_constraints.py`); *a bound holds however far the variable goes* (`test_operator.py`).
  :::
- **Solves through the encoding by Gauss-Newton, with the linear solve passed in as a callable.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.GaussNewton`, {func}`~blochsim.iterative`. Tests: *solving through an encoding beats reconstructing first*; *a solver object needs no deepinv* (`test_gauss_newton.py`).
  :::
- **Damps by a geometric schedule (iteratively regularized Gauss-Newton) or a per-voxel trust region (Levenberg-Marquardt).**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.Schedule`, {class}`~blochsim.TrustRegion`, {func}`~blochsim.direct`. Tests: *the two policies land in the same place*; *a per voxel damping is refused under an encoding* (`test_gauss_newton.py`).
  :::
- **Hands the temporal basis to a subspace encoding operator, and reads maps from the coefficients it returns.**

  :::{dropdown} Show code and tests
  Code: {attr}`~blochsim.Subspace.modes`, {meth}`~blochsim.Estimator.from_coefficients`. Tests: *the basis goes out in the layout the encoding reads*; *one mapping supplies the basis and reads what comes back* (`test_subspace_route.py`).
  :::
- **Converts to a deepinv physics, so deepinv's optimizers and priors apply to the composed chain.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.ModelOperator.physics`. Tests: *it composes with a linear encoding operator*; *a deepinv optimizer drives the composed operator* (`test_deepinv.py`).
  :::

::::

## Division of responsibility

- **The encoding**: sampling, non-uniform FFT, coil sensitivities, density compensation. Any object with `A` and `A_adjoint` composes: an mri-nufft operator through its deepinv bridge, a deepinv `LinearPhysics`, or a bartorch operator through `bartorch.interop.to_deepinv`.
- **A general linear solver.** {func}`~blochsim.iterative` calls deepinv's `least_squares` or any callable with its signature; {func}`~blochsim.direct` serves only a model with no encoding in front of it.
- **Regularizers**, which enter as a closure around a proximal solver passed as `solve`.

## Method

```{figure} /generated/figures/model_based_reconstruction.png
:width: 100%
:alt: Four T2 maps of a disc phantom: the truth, zero-filled images matched to a dictionary, a rank-3 subspace reconstruction, and a Gauss-Newton reconstruction.

An 8-echo spin echo (TE 10 to 150 ms) on a 48 × 48 phantom, sampled at one
phase-encode line in three per echo, shifted from echo to echo, plus six
central lines. The masked FFT and the conjugate-gradient solver are written in
the figure's code; blochsim supplies the dictionary, the basis, the model
operator and the loop. Errors are the mean relative T2 error inside the
phantom.
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
{meth}`~blochsim.Estimator.from_coefficients` then maps the coefficients
without projecting them again. A signal that needs many components, such as a
phase-modulated train or a model with several parameters, gives no small
basis.

### Nonlinear route

Each Gauss-Newton step linearizes $F$ at the current maps and solves the
damped least-squares problem for the step. With an encoding, the step is
solved by an iterative solver from products alone: one encoding application
and one `A_jvp` forward, one adjoint encoding and one `A_vjp` back.
{class}`~blochsim.Schedule` lowers one damping weight geometrically and pulls each step toward
the starting maps; a per-voxel {class}`~blochsim.TrustRegion` needs
independent voxels and applies only without an encoding, where the same loop
with {func}`~blochsim.direct` is {class}`~blochsim.NonlinearLeastSquares`. Composing through
`physics()` instead lets deepinv differentiate the whole chain by autograd,
which gives up these analytic products.

## Examples

- {doc}`../generated/autoexamples/04-model-based-imaging/01-linear-subspace`: radial multi-echo T2 by gridding, per-echo CG, and a rank-3 subspace with mri-nufft.
- {doc}`../generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion`: the same data reconstructed by Gauss-Newton with the model inside the operator.
