# Simulators

The sequences that ship with BlochSim, as {class}`~blochsim.model.Simulator`
subclasses. A constructor fixes protocol arguments, the keywords of
{meth}`~blochsim.model.Simulator.simulate`; tissue properties are given at the
call. See the {doc}`../explanations/signal-model` page.

## Closed form

Signals that have an analytic expression, evaluated in one pass.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.simulators.bSSFPSimulator
   blochsim.simulators.SPGRSimulator
   blochsim.simulators.MP2RAGESimulator
   blochsim.simulators.InversionRecoverySimulator
   blochsim.simulators.MultiEchoSimulator
   blochsim.simulators.DoubleAngleSimulator
   blochsim.simulators.SpinEchoSimulator
   blochsim.simulators.LookLockerSimulator
   blochsim.simulators.MOLLISimulator
   blochsim.simulators.IRbSSFPSimulator
   blochsim.simulators.HSFPSimulator
   blochsim.simulators.MultiGradientEchoSimulator
   blochsim.simulators.IRMultiGradientEchoSimulator
   blochsim.simulators.DiffusionSimulator
   blochsim.simulators.LorentzianSimulator
   blochsim.simulators.ASLSimulator
```

## State machine

Trains that are played out event by event on the extended phase graph engine ({doc}`../explanations/epg`).

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.simulators.FSESimulator
   blochsim.simulators.HyperechoSimulator
   blochsim.simulators.StimulatedEchoSimulator
   blochsim.simulators.MPRAGESimulator
   blochsim.simulators.MPnRAGESimulator
   blochsim.simulators.MRFSimulator
   blochsim.simulators.FLASHSimulator
   blochsim.simulators.TrueFISPSimulator
   blochsim.simulators.fmSSFPSimulator
   blochsim.simulators.CESTSimulator
```

## Functional wrappers

The `*_sim` functions evaluate a subset of the simulators in one call, with an
optional Jacobian. They are not an extension API: new sequence families are
{class}`~blochsim.model.Simulator` subclasses. The classes are preferred when a
sequence is reused, bound to a protocol, or passed to an estimator,
reconstruction or design object.

### Closed-form wrappers

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.bssfp_sim
   blochsim.spgr_sim
```

### State-machine wrappers

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.fse_sim
   blochsim.mprage_sim
   blochsim.mp2rage_sim
   blochsim.mpnrage_sim
   blochsim.mrf_sim
```
