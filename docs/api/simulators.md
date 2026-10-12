# Simulators

```{eval-rst}
.. currentmodule:: blochsim
```

The sequences that ship with BlochSim, as {class}`~blochsim.Simulator`
subclasses. A constructor fixes protocol arguments, the keywords of
{meth}`~blochsim.Simulator.simulate`; tissue properties are given at the
call. See the {doc}`../explanations/signal-model` page.

## Closed form

Signals that have an analytic expression, evaluated in one pass.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.simulators.bSSFPSimulator` | Balanced steady-state free precession (bSSFP) signal at the echo time. |
| {obj}`~blochsim.simulators.SPGRSimulator` | Spoiled gradient echo (SPGR, FLASH) steady state, in closed form. |
| {obj}`~blochsim.simulators.MP2RAGESimulator` | MP2RAGE: an inversion followed by two spoiled gradient-echo blocks. |
| {obj}`~blochsim.simulators.InversionRecoverySimulator` | Inversion-recovery T1 relaxometry, in closed form. |
| {obj}`~blochsim.simulators.MultiEchoSimulator` | Multi-echo T2 relaxometry, in closed form. |
| {obj}`~blochsim.simulators.DoubleAngleSimulator` | Double-angle B1 mapping, in closed form. |
| {obj}`~blochsim.simulators.SpinEchoSimulator` | Spin echo with partial longitudinal recovery, in closed form. |
| {obj}`~blochsim.simulators.LookLockerSimulator` | Look-Locker inversion recovery under a spoiled train of small flip angles. |
| {obj}`~blochsim.simulators.MOLLISimulator` | Modified Look-Locker inversion recovery (MOLLI): readout blocks separated by free recovery. |
| {obj}`~blochsim.simulators.IRbSSFPSimulator` | Inversion-recovery balanced SSFP (IR-TrueFISP) signal, in closed form. |
| {obj}`~blochsim.simulators.HSFPSimulator` | Hybrid-state free precession (HSFP) of a balanced train, in closed form. |
| {obj}`~blochsim.simulators.MultiGradientEchoSimulator` | Multi-gradient-echo signal of water and fat, in closed form. |
| {obj}`~blochsim.simulators.IRMultiGradientEchoSimulator` | Inversion-recovery multi-gradient-echo signal of water and fat, in closed form. |
| {obj}`~blochsim.simulators.DiffusionSimulator` | Diffusion-weighted signal attenuation by Gaussian diffusion, in closed form. |
| {obj}`~blochsim.simulators.LorentzianSimulator` | Z-spectrum (CEST) as equilibrium minus one Lorentzian line per pool. |
| {obj}`~blochsim.simulators.ASLSimulator` | Arterial spin labelling (ASL) control-label difference signal, in closed form. |

## State machine

Trains that are played out event by event on the extended phase graph engine ({doc}`../explanations/epg`).

| Object | Description |
| --- | --- |
| {obj}`~blochsim.simulators.FSESimulator` | Fast (turbo) spin echo (FSE, RARE) train, sampled at every echo. |
| {obj}`~blochsim.simulators.HyperechoSimulator` | Hyperecho train: refocusing pulses mirrored about a central pulse. |
| {obj}`~blochsim.simulators.StimulatedEchoSimulator` | Stimulated-echo sequence: three pulses, sampled at the spin echo and the stimulated echo. |
| {obj}`~blochsim.simulators.MPRAGESimulator` | MPRAGE: an inversion followed by a spoiled gradient-echo train. |
| {obj}`~blochsim.simulators.MPnRAGESimulator` | MPnRAGE: an inversion and a spoiled gradient-echo train, every shot read. |
| {obj}`~blochsim.simulators.MRFSimulator` | MR fingerprinting with an unbalanced SSFP (FISP) train. |
| {obj}`~blochsim.simulators.FLASHSimulator` | Spoiled gradient-echo (FLASH, SPGR) train from equilibrium, optionally inverted first. |
| {obj}`~blochsim.simulators.TrueFISPSimulator` | Balanced SSFP (TrueFISP, FIESTA) train from equilibrium, optionally inverted first. |
| {obj}`~blochsim.simulators.fmSSFPSimulator` | Frequency-modulated balanced SSFP (fmSSFP) train. |
| {obj}`~blochsim.simulators.CESTSimulator` | Chemical exchange saturation transfer (CEST) Z-spectrum by Bloch-McConnell simulation. |

## Functional wrappers

The `*_sim` functions evaluate a subset of the simulators in one call, with an
optional Jacobian. They are not an extension API: new sequence families are
{class}`~blochsim.Simulator` subclasses. The classes are preferred when a
sequence is reused, bound to a protocol, or passed to an estimator,
reconstruction or design object.

### Closed-form wrappers

| Object | Description |
| --- | --- |
| {obj}`~blochsim.bssfp_sim` | Simulate the balanced SSFP steady-state signal. |
| {obj}`~blochsim.spgr_sim` | Simulate the spoiled gradient-echo steady-state signal. |

### State-machine wrappers

| Object | Description |
| --- | --- |
| {obj}`~blochsim.fse_sim` | Simulate a fast spin echo train. |
| {obj}`~blochsim.mprage_sim` | Simulate the sampled MPRAGE shot in closed form. |
| {obj}`~blochsim.mp2rage_sim` | Simulate the two MP2RAGE blocks in closed form. |
| {obj}`~blochsim.mpnrage_sim` | Simulate an MPnRAGE train, every shot read. |
| {obj}`~blochsim.mrf_sim` | Simulate an unbalanced SSFP MR fingerprinting train. |
