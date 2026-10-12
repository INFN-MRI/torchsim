# Examples

Every page here runs as you read it: the figures and numbers are computed by
the code on the page, and each page has an *Open in Colab* button that runs it
in your browser.

The concepts these pages rely on are in {doc}`/explanations/index`, and the
interfaces they call are documented in {doc}`/api/index`.

## Course

Five lessons that take you from a sequence blochsim ships to a sequence and an
operator of your own. Read them in order.

| Lesson | The question it answers |
| --- | --- |
| {doc}`1. Your first simulation </generated/autoexamples/01-framework/01-first-simulation>` | How do I simulate a sequence and get its derivatives? |
| {doc}`2. Advanced physics </generated/autoexamples/01-framework/02-advanced-physics>` | How do I add B1, off resonance, magnetization transfer, diffusion or flow? |
| {doc}`3. Custom signal model </generated/autoexamples/01-framework/03-custom-signal-model>` | How do I simulate a sequence blochsim does not ship? |
| {doc}`4. Description-based simulation </generated/autoexamples/01-framework/04-description-based-simulation>` | How do I simulate a Pulseq file, or the sequence a scanner is running? |
| {doc}`5. Custom operator </generated/autoexamples/01-framework/05-custom-operator>` | How do I add a preparation or a readout of my own? |

## Applications

Complete worked examples built on the Course, each standalone.

| Application | What it establishes |
| --- | --- |
| {doc}`/generated/autoexamples/02-parameter-inference/01-dictionary-matching` | T1, T2 and proton density from an MRF train by dictionary matching, with a low-rank subspace and a clustered dictionary. |
| {doc}`/generated/autoexamples/02-parameter-inference/02-lookup-table` | T1 from MP2RAGE by interpolation along the signal curve and by dictionary matching. |
| {doc}`/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares` | T2 from a multi-echo spin echo by nonlinear least squares. |
| {doc}`/generated/autoexamples/02-parameter-inference/04-perk` | T1, T2 and proton density by kernel ridge regression, and its uncertainty. |
| {doc}`/generated/autoexamples/03-sequence-optimization/01-echo-train-design` | Fast spin echo refocusing angles designed for image sharpness. |
| {doc}`/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry` | DESPOT flip angles chosen against the Cramér-Rao bound of a joint T1 and T2 fit. |
| {doc}`/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design` | A slice-selective RF pulse designed by optimal control over a range of B1. |
| {doc}`/generated/autoexamples/04-model-based-imaging/01-linear-subspace` | Gridding, per-echo conjugate gradients and a linear subspace on undersampled radial data. |
| {doc}`/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion` | T2 maps reconstructed directly from k-space by nonlinear inversion. |
| {doc}`/generated/autoexamples/05-misc/01-synthetic-data` | A training pair for MR fingerprinting, from tissue maps to undersampled images. |

```{toctree}
:hidden:

course
applications
```
