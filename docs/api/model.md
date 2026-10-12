# Signal models

```{eval-rst}
.. currentmodule:: blochsim.model
```

{class}`Simulator` is the base class of a pulse sequence protocol and the only
model interface that estimators, reconstruction and sequence design consume.

A subclass names an operator for each kind of event (`excitation`,
`refocusing`, `inversion`, `saturation`, `readout`, `delay`) and implements
{meth}`~Simulator.layout`, which returns the operators of one repetition in
playing order. A protocol with a closed form implements
{meth}`~Simulator.evaluate` instead. The same operators interpret a description
read by {meth}`~Simulator.from_description` or {meth}`~Simulator.from_pulseq`,
in which case the layout is not called. The constructor takes the keywords
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

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Simulator
```

## Spin physics and event operators

{class}`SpinPhysics` declares the tissue properties of a voxel and the
{class}`EventOperators` that realize each kind of event. Any tissue property
may be given to any simulator; giving one enables its term.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SpinPhysics
   EventOperators
```

## Operator tables

A `SpinPhysics` takes one of four predefined {class}`EventOperators`.

| | Readout | Refocusing |
| --- | --- | --- |
| {data}`SPOILED` | ideal transverse spoiling after the sample | crushed |
| {data}`UNBALANCED` | one unbalanced gradient after the sample | crushed |
| {data}`BALANCED` | the repetition rewinds after the sample | uncrushed |
| {data}`REFOCUSED` | the sample at the echo centre | crushed |

```{eval-rst}
.. autodata:: blochsim.model.SPOILED
   :no-value:
.. autodata:: blochsim.model.UNBALANCED
   :no-value:
.. autodata:: blochsim.model.BALANCED
   :no-value:
.. autodata:: blochsim.model.REFOCUSED
   :no-value:
```
