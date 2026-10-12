# Signal models

```{eval-rst}
.. currentmodule:: blochsim.model
```

**There is one user-facing base class: {class}`Simulator`.**

{class}`Simulator` is the sequence abstraction and the only model interface
anything downstream consumes. Parameter estimators, model-based
reconstruction and sequence design take a simulator and do not need to know
whether its sequence was built offline or arrived from a running scanner.

For a state-machine sequence, a subclass supplies two complementary pieces:

1. **Command handlers.** The class attributes `excitation`, `refocusing`,
   `inversion`, `saturation`, `readout` and `delay` say how the RF and
   ADC commands of an incoming sequence description are interpreted. This is
   the scanner-facing path: {meth}`~Simulator.from_description` re-emits an
   MRD/Pulseq-derived event stream through those handlers.
2. **An offline layout.** {meth}`~Simulator.layout` returns the same
   operators in the order one repetition plays them. This is the
   design/offline path, when no scanner description already exists.

Both routes produce the same sequence description before the state machine
runs. The EPG engine, derivatives, device placement and memory policy are
therefore shared and are not part of a sequence implementation.

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

The six slots a class body may name are `excitation`, `refocusing`,
`inversion`, `saturation`, `readout` and `delay`; each is one of the operators
on {doc}`sequence`. Naming a different readout is the whole of the difference
between a spoiled, an unbalanced, a balanced and a refocused train, so a
variant is a subclass with one line in it. Naming one is also what says how a
stream arriving from a scanner is to be read, since
{meth}`~Simulator.from_description` re-emits its events through these same
operators.

Nothing is declared about the tissue. Every property a voxel can have may be
given to any simulator, and giving one is what turns its term on.

A sequence that came from somewhere else is read through those same handlers.
{func}`~blochsim.sequence.read_mrd_description` decodes the description
carried ahead of the acquisitions on an MRD stream, and
{meth}`~Simulator.from_description` turns one of those descriptions into the
chosen simulator. {meth}`~Simulator.from_pulseq` does the same from a Pulseq
`.seq` file, or from a sequence object held in memory. None of these routes
walks `layout()`: the incoming description already supplies the layout, while
the simulator class supplies its interpretation.

Implement {meth}`~Simulator.evaluate` instead when the signal has a closed
form -- a mono-exponential decay, an inversion-recovery curve, an Ernst
steady state. There is nothing to play and no state to carry, so there is no
`layout` and the `SpinPhysics` carries only the property declaration.
{class}`~blochsim.simulators.SPGRSimulator` is written this way, and
{class}`~blochsim.simulators.MP2RAGESimulator` carries both: the closed form a
lookup table is built from, and the layout a description arriving from a
scanner is compared against.

A model that composes others rather than declaring physics of its own names
its properties in the class body and writes whatever constructor suits it:

```python
class JointRelaxometry(Simulator):
    properties = ("T1", "T2", "M0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence): ...
```

Either way it fixes its arguments the same way: a constructor takes the
keywords {meth}`~Simulator.simulate` takes, {meth}`~Simulator.bind` adds more
to a copy, and a call overrides either.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Simulator
```

## The physics behind it

{class}`SpinPhysics` is the other half: which tissue properties a voxel has,
and so which terms the kernels carry, together with what each kind of event is
realized as. {class}`EventOperators` holds one slot per role a sequence is
written in terms of, and a class body that names `excitation`, `readout` or
any of the other four is assigning into it.

The two are separate so either can change without the other -- an MRF timing
given a selective excitation, or a refocused train whose readout spoils rather
than winds, is an assignment and not a new model.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SpinPhysics
   EventOperators
```

Four tables are supplied, and a `SpinPhysics` names one rather than filling
the slots itself:

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
