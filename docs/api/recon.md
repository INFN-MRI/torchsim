# Model-based reconstruction

```{eval-rst}
.. currentmodule:: blochsim
```

Parameter maps solved from k-space with the signal model inside the forward
operator. The concepts are in {doc}`../explanations/model-based-reconstruction`.

{class}`ModelOperator` is the signal model as an operator: parameter maps in,
one image per contrast out, with analytic Jacobian-vector products, a complex
amplitude and box bounds. It honours {func}`execution`, and `physics()` wraps
it for deepinv. {class}`GaussNewton` inverts it by repeated linearization; its
damping policy ({class}`Schedule` for an iteratively regularized
Gauss-Newton, {class}`TrustRegion` for Levenberg-Marquardt) selects the method,
and its linear solve is {func}`direct` or {func}`iterative`. The Fourier
encoding is external: any object with `A` and `A_adjoint` composes.

## Operator

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ModelOperator
```

## Solver

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   GaussNewton
   Schedule
   TrustRegion
```

## Linear solves

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   direct
   iterative
```

## Temporal basis

{attr}`Subspace.modes` gives the basis in the layout a subspace operator reads.
The same basis serves the estimators on {doc}`estimators`.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Subspace
```
