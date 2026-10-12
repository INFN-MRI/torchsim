# Sequence description

```{admonition} TL;DR
:class: tldr

- A sequence description is one repetition as WAIT, RF and ADC events, timestamped in microseconds, with the RF definitions its pulses name.
- A Pulseq file and the MRD stream pulserver sends are read into the same {class}`~blochsim.SequenceDescription`.
- Gradients are not events: the handlers of the simulator a description is given to supply the dephasing.
```

You can simulate a sequence written elsewhere, from its Pulseq file or from
the stream a scanner sends during the acquisition, without restating its
parameters. This page states what such a description gives a simulator. How
the handlers then play it is on {doc}`signal-model`; the reading rules and the
wire format are in {doc}`../internals/description`.

## Capabilities

::::{container} capabilities

- **Reads one repetition of a Pulseq file, as many blocks as its `TRSize` definition states, one event per block.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.SequenceDescription.from_pulseq`. Tests: *the stream is the blocks the file lists*; *a file without a declared repetition is refused* (`test_pulseq_reader.py`).
  :::
- **Reads a sequence held in memory by pypulseq or pypulseqpp as it reads the file that sequence writes.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.SequenceDescription.from_pulseq`. Test: *a sequence in memory reads as the file it writes* (`test_pulseq_reader.py`).
  :::
- **Takes the repetition named by `tr_index` or by the file's `TRRef`, and otherwise the first that acquires, provided every repetition plays the same pulses.**

  :::{dropdown} Show code and tests
  Code: `read_pulseq_description`. Test: *naming a repetition out of range is refused* (`test_pulseq_reader.py`).
  :::
- **Stamps an RF event at the isocentre of its pulse, with the flip angle the envelope integrates to on the declared RF raster.**

  :::{dropdown} Show code and tests
  Code: {func}`~blochsim.rf_definition`. Tests: *the pulse reads back at the angle it was written at* (`test_pulseq_reader.py`); *the dwell need not be the raster* (`test_rf_definition.py`).
  :::
- **Makes a pulse played under a slice-select gradient slice selective, with slice positions in units of the excitation's thickness.**

  :::{dropdown} Show code and tests
  Code: `read_pulseq_description`. Tests: *a pulse under a slice select gradient selects the slice it was designed for*; *a slice selective pulse from a file excites only its slice* (`test_pulseq_reader.py`).
  :::
- **Stamps an ADC event at the sample where its readout crosses the k-space centre, found on the gradient trajectory.**

  :::{dropdown} Show code and tests
  Code: `read_pulseq_description`. Test: *the echo sits where the trajectory crosses zero* (`test_pulseq_reader.py`).
  :::
- **Flags the readouts that pass through $k = 0$, which `record="echo"` keeps.**

  :::{dropdown} Show code and tests
  Code: `read_pulseq_description`, `EpgEngine.simulate`. Tests: *only the readout that crosses zero is an echo*; *recording echoes keeps the central readout alone* (`test_pulseq_reader.py`).
  :::
- **Decodes the description pulserver sends on an MRD stream, one per file of the sequence chain, in play order.**

  :::{dropdown} Show code and tests
  Code: {func}`~blochsim.read_mrd_description`. Tests: *a written stream reads back into the rows it was built from*; *every file of a chain is read in play order*; *a stream whose data arrives first is refused* (`test_mrd_reader.py`).
  :::
- **Reproduces the spoiled gradient-echo steady state from a description read either way.**

  :::{dropdown} Show code and tests
  Code: {meth}`~blochsim.model.Simulator.from_description`. Tests: *the steady state is the spoiled gradient echo closed form* (`test_pulseq_reader.py`); *a described sequence drives the spoiled gradient echo steady state* (`test_mrd_reader.py`).
  :::

::::

## Division of responsibility

- **Gradient moments.** A description holds one crusher moment for the whole sequence, `crusher_dephasing_rad` across `voxel_size_m`, given by the caller; neither file nor stream supplies it.
- **A slice-select gradient that varies under the pulse**, as in VERSE or ramp sampling: a file's pulse is selected by the gradient at its centre.
- **Pulses with a time shape of their own**, sampled unevenly in a Pulseq file, which are refused.
- **A gradient that belongs to no pulse or readout**, such as a preparation's own spoiler, which no handler reinstates.

## Method

```{figure} /generated/figures/sequence_description.png
:width: 100%
:alt: Thirteen Pulseq blocks above the events they are read as, and the readout k-space coordinate crossing zero at each ADC timestamp.

Above, the first thirteen blocks of a spin-echo train and the events they are
read as; blocks holding only gradients become WAITs. Below, the readout
coordinate $k_x$ at every ADC sample: each ADC event is stamped at its zero
crossing.
```

### Event classification

A block holding a pulse becomes an RF event, one holding only an ADC an ADC
event, and any other a WAIT; a block holding both is read as the pulse. An RF
event carries the Pulseq use tag, which selects the handler: refocusing,
inversion and saturation pulses reach their own, and any other pulse the
excitation handler. An ADC event carries a role among the repetition's samples
and the echo flag; `record="acquired"` keeps every sample whose role is not
`NON_ACQUIRED`.

### RF shapes

Events name a definition rather than carry a waveform. Rows of the Pulseq RF
library that differ only in amplitude, phase or frequency, as RF spoiling
writes them, share one definition. The flip angle is $2\pi$ times the
magnitude of the event amplitude times the envelope's integral, taken on the
`RadiofrequencyRasterTime` the description carries. Which rotation the
kernels perform with the definition is on {doc}`epg`.

### Timing

Timestamps count from the start of the repetition read, not of the scan, so a
description does not depend on where in the file it was taken. The repetition
lasts the sum of its block durations.

## Examples

- {doc}`../generated/autoexamples/01-framework/04-description-based-simulation`: a description inspected, then simulated from a Pulseq file and through two sets of handlers.
- {doc}`../generated/autoexamples/01-framework/02-advanced-physics`: a shaped pulse from an RF definition, integrated across the slice.
