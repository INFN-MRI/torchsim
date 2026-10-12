# Signal models

```{eval-rst}
.. currentmodule:: blochsim
```

{class}`Simulator` is the base class of a pulse sequence protocol and the only
model interface that estimators, reconstruction and sequence design consume.

A subclass binds a handler to each kind of event it plays and implements
{meth}`~Simulator.layout`, which returns the operators of one repetition in
playing order. A protocol with a closed form implements
{meth}`~Simulator.evaluate` instead. The same operators interpret a description
read by {meth}`~Simulator.from_description` or {meth}`~Simulator.from_pulseq`,
in which case the layout is not called; a pulse is played by the handler its
Pulseq RF use names. The constructor takes the keywords
{meth}`~Simulator.simulate` takes, {meth}`~Simulator.bind` fixes more on a
copy, and a call overrides both. See {doc}`../explanations/signal-model`.

```python
class SSFPMRF(Simulator):
    excitation = Excitation
    inversion = Inversion
    readout = SSFPFidReadout
    states = 10

    def layout(self, *, flip, TR, TI=0.0):
        parts = [self.operators.inversion(duration_s=TI * 1e-3)]
        for angle in torch.deg2rad(torch.as_tensor(flip)):
            parts += [
                self.operators.excitation(angle),
                self.operators.readout(duration_s=TR * 1e-3),
            ]
        return parts
```

A model composing other simulators lists its properties in the class body and
implements {meth}`~Simulator.evaluate`:

```python
class JointRelaxometry(Simulator):
    properties = ("T1", "T2", "M0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence): ...
```

| Object | Description |
| --- | --- |
| {obj}`~blochsim.Simulator` | Base class of a pulse sequence protocol and its signal model. |

## Handlers

A class attribute named after a kind of event binds the operator that plays
it. A handler is a built-in operator from {doc}`sequence` or a function
composing them with `@`.

| Attribute | Event | Default |
| --- | --- | --- |
| `excitation` | RF, use *excitation* or none | `Excitation` |
| `refocusing` | RF, use *refocusing* | `Refocusing` |
| `inversion` | RF, use *inversion* | `Inversion` |
| `saturation` | RF, use *saturation* | `Saturation` |
| `preparation` | RF, use *preparation* | `Excitation` |
| `other` | RF, use *other* | `Excitation` |
| `readout` | ADC, with the gradients after it | `Readout` |
| `delay` | wait, with or without a gradient | `Delay` |

## Model operator

{class}`ModelOperator` is a signal model as an operator: parameter maps in,
one image per contrast out, with analytic Jacobian-vector and adjoint
products, a complex amplitude and box bounds. It is what a model-based
reconstruction composes with an encoding: bartorch's `nlop.ModelOperator`
takes it as a BART nonlinear operator, and `physics()` wraps it for deepinv.
See {doc}`../explanations/model-based-reconstruction`.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.ModelOperator` | A signal model over parameter maps, with its derivatives. |

## Temporal basis

{attr}`Subspace.modes` gives the basis in the layout a subspace operator reads.
The same basis serves the estimators on {doc}`estimators`.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.Subspace` | Orthonormal temporal basis of a set of signals, for subspace reconstruction. |
