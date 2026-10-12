# Applications

Standalone examples built on the {doc}`course`. Each names the lessons it
assumes.

## Parameter mapping

T1, T2 and M0 estimation, compared on one BrainWeb slice.

| Application | Contents |
| --- | --- |
| {doc}`/generated/autoexamples/02-parameter-inference/01-dictionary-matching` | MRF dictionary matching, low-rank and clustered. |
| {doc}`/generated/autoexamples/02-parameter-inference/02-lookup-table` | MP2RAGE T1 lookup table. |
| {doc}`/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares` | Multi-echo spin echo T2 fit by nonlinear least squares. |
| {doc}`/generated/autoexamples/02-parameter-inference/04-perk` | PERK estimation of T1, T2 and M0, with uncertainty. |

```{toctree}
:hidden:

/generated/autoexamples/02-parameter-inference/01-dictionary-matching
/generated/autoexamples/02-parameter-inference/02-lookup-table
/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares
/generated/autoexamples/02-parameter-inference/04-perk
```

## Sequence design

Sequence parameters optimized by gradient descent through the simulation.

| Application | Contents |
| --- | --- |
| {doc}`/generated/autoexamples/03-sequence-optimization/01-echo-train-design` | Fast spin echo refocusing flip angles for sharpness. |
| {doc}`/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry` | DESPOT flip angles by Cramér-Rao bound. |
| {doc}`/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design` | B1-robust slice-selective RF pulse by optimal control. |

```{toctree}
:hidden:

/generated/autoexamples/03-sequence-optimization/01-echo-train-design
/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry
/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design
```

## Model-based reconstruction

Parameter maps reconstructed from k-space with the signal model in the forward
operator.

| Application | Contents |
| --- | --- |
| {doc}`/generated/autoexamples/04-model-based-imaging/01-linear-subspace` | Linear subspace reconstruction of radial multi-echo spin echo. |
| {doc}`/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion` | T2 mapping by nonlinear model-based inversion. |

```{toctree}
:hidden:

/generated/autoexamples/04-model-based-imaging/01-linear-subspace
/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion
```

## Synthetic data

Training data for a learned reconstruction.

| Application | Contents |
| --- | --- |
| {doc}`/generated/autoexamples/05-misc/01-synthetic-data` | Synthetic MRF training data. |

```{toctree}
:hidden:

/generated/autoexamples/05-misc/01-synthetic-data
```
