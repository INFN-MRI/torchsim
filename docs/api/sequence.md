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

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Operator
   Excitation
   Refocusing
   Inversion
   Saturation
   SampledPulse
   Readout
   Delay
   Dephase
   Spoil
   bSSFPReadout
   SSFPFidReadout
   SSFPEchoReadout
   SPGRReadout
   FSEReadout
```

## Pulses and shims

{func}`rf_definition` builds an RF definition from a complex envelope, one row
per transmit channel. {class}`ShimDefinition` holds the amplitude and phase of
each channel. A simulator takes them as `pulse=` and `shims=`.
{func}`compose_spinor` composes the rotation a shaped pulse leaves on each
spin, differentiably with respect to the samples.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   rf_definition
   ShimDefinition
   compose_spinor
```

## Description

The event stream a layout composes to, or a scanner sequence is read into.
{meth}`SequenceDescription.from_operators` builds one from operators,
{meth}`SequenceDescription.from_pulseq` reads a Pulseq `.seq` file or an
in-memory sequence, {func}`read_mrd_description` reads those an MRD stream
carries ahead of its first acquisition, and
{meth}`~blochsim.model.Simulator.from_description` simulates any of them.

Reading Pulseq needs pypulseq: `pip install blochsim[pulseq]`.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceDescription
   read_mrd_description
```
