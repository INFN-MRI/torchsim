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

| Object | Description |
| --- | --- |
| {obj}`~blochsim.ModelOperator` | A signal model over parameter maps, with its derivatives. |

## Solver

| Object | Description |
| --- | --- |
| {obj}`~blochsim.GaussNewton` | Solve a nonlinear inverse problem by repeated linearization. |
| {obj}`~blochsim.Schedule` | One damping for the whole volume, lowered geometrically. |
| {obj}`~blochsim.TrustRegion` | Levenberg-Marquardt: per-voxel damping, raised where a step does not pay. |

## Linear solves

| Object | Description |
| --- | --- |
| {obj}`~blochsim.direct` | Solve every voxel's damped linearized system by batched least squares. |
| {obj}`~blochsim.iterative` | Build an inner solve from a regularized least-squares routine. |

## Temporal basis

{attr}`Subspace.modes` gives the basis in the layout a subspace operator reads.
The same basis serves the estimators on {doc}`estimators`.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.Subspace` | Orthonormal temporal basis of a set of signals, for subspace reconstruction. |
