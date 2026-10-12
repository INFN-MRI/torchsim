# Applications

Complete worked examples built on the {doc}`course`: mapping tissue
parameters, designing sequences, reconstructing parameter maps from k-space,
and making synthetic data. Each stands alone and says which lessons it assumes.

## Parameter mapping

Four ways to turn a measured signal into T1, T2 and M0 maps, all built from
the same simulator and compared on the same BrainWeb slice.

| Application | What it establishes |
| --- | --- |
| {doc}`/generated/autoexamples/02-parameter-inference/01-dictionary-matching` | T1, T2 and proton density from an MRF train by dictionary matching, and the cost of the match reduced by a low-rank subspace and by clustering the dictionary. |
| {doc}`/generated/autoexamples/02-parameter-inference/02-lookup-table` | T1 from a two-block MP2RAGE acquisition by interpolation along the signal curve and by dictionary matching, against the number of points on the curve. |
| {doc}`/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares` | T2 from a multi-echo spin echo by nonlinear least squares, compared with dictionary matching on the same slice. |
| {doc}`/generated/autoexamples/02-parameter-inference/04-perk` | T1, T2 and proton density by kernel ridge regression, against the size of the regression, with its uncertainty read against the Cramér-Rao bound. |

```{toctree}
:hidden:

/generated/autoexamples/02-parameter-inference/01-dictionary-matching
/generated/autoexamples/02-parameter-inference/02-lookup-table
/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares
/generated/autoexamples/02-parameter-inference/04-perk
```

## Sequence design

Sequence parameters chosen by gradient descent through the simulation.

| Application | What it establishes |
| --- | --- |
| {doc}`/generated/autoexamples/03-sequence-optimization/01-echo-train-design` | The refocusing flip angles of a fast spin echo designed for image sharpness, for one echo train and for a segmented 3D protocol. |
| {doc}`/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry` | The flip angles of a DESPOT protocol chosen to minimize the Cramér-Rao bound of a joint T1 and T2 fit. |
| {doc}`/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design` | The samples of a slice-selective RF pulse designed by optimal control to hold the flip angle over a range of B1. |

```{toctree}
:hidden:

/generated/autoexamples/03-sequence-optimization/01-echo-train-design
/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry
/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design
```

## Model-based reconstruction

Parameter maps reconstructed from k-space, with the signal model inside the
forward operator: linear, in a subspace, or nonlinear.

| Application | What it establishes |
| --- | --- |
| {doc}`/generated/autoexamples/04-model-based-imaging/01-linear-subspace` | An undersampled radial multi-echo spin echo reconstructed by gridding, by conjugate gradients per echo and in a linear subspace, compared in cost and T2 error. |
| {doc}`/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion` | T2 maps reconstructed directly from k-space with the signal model inside the forward operator, compared with gridding followed by a fit. |

```{toctree}
:hidden:

/generated/autoexamples/04-model-based-imaging/01-linear-subspace
/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion
```

## Synthetic data

Training data for a learned reconstruction.

| Application | What it establishes |
| --- | --- |
| {doc}`/generated/autoexamples/05-misc/01-synthetic-data` | A training pair for MR fingerprinting: the undersampled and fully sampled image series and the T1, T2 and M0 maps they come from. |

```{toctree}
:hidden:

/generated/autoexamples/05-misc/01-synthetic-data
```
