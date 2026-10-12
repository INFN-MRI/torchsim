# Examples

Executed examples: every figure and number is computed by the code on its
page, and each page opens in Colab.

The concepts these pages rely on are in {doc}`/explanations/index`, and the
interfaces they call are documented in {doc}`/api/index`.

## Course

Five lessons, read in order, from a shipped sequence to a custom signal model
and operator.

| Lesson | Topic |
| --- | --- |
| {doc}`1. Your first simulation </generated/autoexamples/01-framework/01-first-simulation>` | Simulation of a shipped sequence and its derivatives. |
| {doc}`2. Advanced physics </generated/autoexamples/01-framework/02-advanced-physics>` | B1, off resonance, magnetization transfer, diffusion and flow. |
| {doc}`3. Custom signal model </generated/autoexamples/01-framework/03-custom-signal-model>` | A signal model for a new sequence. |
| {doc}`4. Description-based simulation </generated/autoexamples/01-framework/04-description-based-simulation>` | Simulation of a Pulseq file or an MRD stream. |
| {doc}`5. Custom operator </generated/autoexamples/01-framework/05-custom-operator>` | A new preparation or readout operator. |

## Applications

Standalone examples built on the Course.

| Application | Topic |
| --- | --- |
| {doc}`/generated/autoexamples/02-parameter-inference/01-dictionary-matching` | MRF dictionary matching, low-rank and clustered. |
| {doc}`/generated/autoexamples/02-parameter-inference/02-lookup-table` | MP2RAGE T1 lookup table. |
| {doc}`/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares` | Multi-echo spin echo T2 fit by nonlinear least squares. |
| {doc}`/generated/autoexamples/02-parameter-inference/04-perk` | PERK estimation of T1, T2 and M0, with uncertainty. |
| {doc}`/generated/autoexamples/03-sequence-optimization/01-echo-train-design` | Fast spin echo refocusing flip angles for sharpness. |
| {doc}`/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry` | DESPOT flip angles by Cramér-Rao bound. |
| {doc}`/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design` | B1-robust slice-selective RF pulse by optimal control. |
| {doc}`/generated/autoexamples/04-model-based-imaging/01-linear-subspace` | Linear subspace reconstruction of radial multi-echo spin echo. |
| {doc}`/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion` | T2 mapping by nonlinear model-based inversion. |
| {doc}`/generated/autoexamples/05-misc/01-synthetic-data` | Synthetic MRF training data. |

```{toctree}
:hidden:

course
applications
```
