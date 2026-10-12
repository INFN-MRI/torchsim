# Sequence design

```{admonition} TL;DR
:class: tldr

- A design is a simulator with the tissue fixed on it, a cost you write as a function of the protocol parameters, and those parameters with their limits.
- {class}`~blochsim.optim.SequenceDesign` minimizes the cost by gradient descent; the gradient is reverse-mode autograd through the simulator.
- A precision design minimizes a Cramér-Rao bound; an image-quality design minimizes a property of the signal itself.
```

You choose protocol parameters, such as flip angles, repetition times or echo
train lengths, by minimizing a cost computed from what the sequence records.

## Capabilities

::::{container} capabilities

- **Minimizes a cost over named protocol parameters, each passed to the cost by keyword.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.optim.SequenceDesign`. Tests: *a design lowers the cost it was given*; *several parameters are designed together* (`test_design.py`).
  :::
- **Keeps each parameter strictly between its limits at every step.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.optim.Bounded`. Test: *the limits hold at every step* (`test_design.py`).
  :::
- **Computes the Cramér-Rao bound from a signal Jacobian, and refuses parameters the sequence cannot separate.**

  :::{dropdown} Show code and tests
  Code: {func}`~blochsim.optim.crlb`. Tests: *the bound is the variance a fit actually reaches*; *parameters no sequence can tell apart are refused* (`test_design.py`).
  :::
- **Designs one train per shot in a single batched simulation.**

  :::{dropdown} Show code and tests
  Code: {class}`~blochsim.optim.SequenceDesign`, {class}`~blochsim.optim.Bounded`. Test: *a batch of trains is designed in one simulation* (`test_design.py`).
  :::
- **Reuses the event structure across the calls of a design loop, rebuilding only the values that change.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.Simulator.simulate`. Tests: *a resolved simulator answers what the plain one does* (`test_binding.py`); *an image quality design fits in its budget* (`test_design.py`).
  :::
- **Returns the cost at every step, and stops early when a callback returns `True`.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.optim.SequenceDesign.minimize`. Test: *a callback can stop early* (`test_design.py`).
  :::

::::

## Division of responsibility

- **Hardware and safety limits** other than box limits on a parameter: gradient, PNS and SAR constraints enter as penalty terms you write, or are checked by sequence design tools such as pypulseqpp.
- **Global optimization.** The optimizer descends from the starting values, and the result depends on them.
- **Predefined costs** other than {func}`~blochsim.optim.crlb`. A blur, contrast or power term is a function you write.
- **Writing the designed protocol** to a Pulseq file.

## Method

```{figure} /generated/figures/sequence_design.png
:width: 100%
:alt: Cost against step, the start and designed flip angles of four SPGR and four bSSFP scans, and the relative Cramér-Rao bound on T1 and T2 before and after.

Four SPGR and four bSSFP flip angles designed for a joint fit of T1, T2, M0
and B0 in white and grey matter at 0.5 % noise. The cost is the log of the
summed relative Cramér-Rao bounds of T1 and T2. The angles collapse onto a few
values; one bSSFP angle ends at its 70° limit.
```

### Cost function

The cost takes the designed parameters by keyword and returns one number. A
precision design asks its simulators for {meth}`~blochsim.Simulator.jacobian`
with respect to the estimated tissue properties and passes the result to
{func}`~blochsim.optim.crlb`, which returns the diagonal of the inverse Fisher matrix,
counting real and imaginary channels as separate measurements. Dividing each
bound by its parameter squared makes the terms dimensionless; taking the
logarithm makes the gradient independent of the noise level. A block that is
blind to a parameter contributes a zero row, and the Fisher matrix sums the
information of all blocks. An image-quality design reads the signal alone, for
example the width of the point spread function an echo train produces. How the
gradient of either cost is computed is in {doc}`derivatives`.

### Limits and parameterization

{class}`~blochsim.optim.Bounded` optimizes an unconstrained variable whose scaled
sigmoid is the parameter, so no iterate leaves the limits; a bare tensor is
left unconstrained. A limit the scanner cannot exceed belongs in `Bounded`; a
preference such as RF power or scan time belongs in the cost as a penalty. The
number of free variables is a choice of parameterization: a 120-echo train can
be written as a function of three control angles and designed over those.

### Design result

{meth}`~blochsim.optim.SequenceDesign.minimize` runs Adam (200 steps, learning rate
0.05 by default; `optimizer_factory` replaces it) and returns a
`SequenceOptimization`: `parameters`, a dictionary of detached values inside
their limits, and `loss`, the cost at every step, from which convergence is
read.

## Examples

- {doc}`../generated/autoexamples/03-sequence-optimization/01-echo-train-design`: refocusing angles of a fast spin echo designed for sharpness, contrast and RF power, then a segmented 3D protocol.
- {doc}`../generated/autoexamples/03-sequence-optimization/02-joint-relaxometry`: a DESPOT protocol designed against the Cramér-Rao bound and checked by fitting a BrainWeb slice.
- {doc}`../generated/autoexamples/03-sequence-optimization/03-rf-pulse-design`: the samples of a slice-selective pulse designed through a Bloch simulation over a range of B1.
