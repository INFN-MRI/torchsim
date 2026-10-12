# Sequences

```{eval-rst}
.. currentmodule:: blochsim
```

Device-agnostic description of an acquisition: a list of events with the
timing and tissue interaction each carries, shared by the interpreter, the
estimators and the sequence optimizer.

Simulators are written on {doc}`model`, run on {doc}`execution`, and the
temporal basis a run spans is on {doc}`recon`. Concepts are in
{doc}`../explanations/description`.

## Operators

Modules a sequence is assembled from. An operator is a function returning
the events it plays and the time it holds the timeline; `@` composes two
operators into one. {func}`SampledPulse` plays a waveform as one hard pulse per
sample, with relaxation, precession and exchange between samples.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.Operator` | One module of a sequence: the events it plays and the time it spans. |
| {obj}`~blochsim.Excitation` | Excitation pulse. |
| {obj}`~blochsim.Refocusing` | Refocusing pulse, between crusher gradients by default. |
| {obj}`~blochsim.Inversion` | Ideal inversion pulse. |
| {obj}`~blochsim.Saturation` | Off-resonance saturation pulse that deposits power in a semisolid pool. |
| {obj}`~blochsim.SampledPulse` | Shaped pulse played as one hard pulse per sample, with free evolution between. |
| {obj}`~blochsim.Readout` | One ADC sample of the transverse magnetization. |
| {obj}`~blochsim.Delay` | Delay, optionally carrying a gradient action. |
| {obj}`~blochsim.Dephase` | Unbalanced gradient of zero duration. |
| {obj}`~blochsim.Spoil` | Ideal transverse spoiling of zero duration. |
| {obj}`~blochsim.bSSFPReadout` | Balanced SSFP readout: a sample, then the rest of the repetition time. |
| {obj}`~blochsim.SSFPFidReadout` | Unbalanced SSFP readout (FISP): a sample, then one unbalanced gradient. |
| {obj}`~blochsim.SSFPEchoReadout` | Reversed SSFP readout (PSIF): one unbalanced gradient, then a sample. |
| {obj}`~blochsim.SPGRReadout` | Spoiled gradient-echo readout: a sample, then ideal transverse spoiling. |
| {obj}`~blochsim.FSEReadout` | Fast spin-echo readout: the sample at a spin echo, marked as the echo centre. |

## Pulses and shims

{func}`rf_definition` builds an RF definition from a complex envelope, one row
per transmit channel. {class}`ShimDefinition` holds the amplitude and phase of
each channel. A simulator takes them as `pulse=` and `shims=`.
{func}`compose_spinor` composes the rotation a shaped pulse leaves on each
spin, differentiably with respect to the samples.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.rf_definition` | Return the RF definition of a pulse envelope. |
| {obj}`~blochsim.ShimDefinition` | Transmit shim: a complex weight per channel for static parallel transmit. |
| {obj}`~blochsim.compose_spinor` | Return the Cayley-Klein pair of a shaped pulse, composed over its samples. |

## Description

The event stream a layout composes to, or a scanner sequence is read into.
{meth}`SequenceDescription.from_operators` builds one from operators,
{meth}`SequenceDescription.from_pulseq` reads a Pulseq `.seq` file or an
in-memory sequence, {func}`read_mrd_description` reads those an MRD stream
carries ahead of its first acquisition, and
{meth}`~blochsim.model.Simulator.from_description` simulates any of them.

Reading Pulseq needs pypulseq: `pip install blochsim[pulseq]`.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.SequenceDescription` | Event stream and RF resources of one repetition of a sequence. |
| {obj}`~blochsim.read_mrd_description` | Read an MRD stream up to its sequence description. |
