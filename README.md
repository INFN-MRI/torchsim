[![Tests](https://github.com/pulserver/blochsim/actions/workflows/test.yml/badge.svg)](https://github.com/pulserver/blochsim/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/blochsim.svg)](https://pypi.org/project/blochsim/)
[![Docs](https://img.shields.io/badge/docs-latest-2b76ad)](https://pulserver.github.io/blochsim/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffbd28.svg)](https://github.com/pulserver/blochsim/blob/main/LICENSE.txt)

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/blochsim/main/docs/_static/blochsim-logo-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/blochsim/main/docs/_static/blochsim-logo.svg" alt="blochsim" width="420">
</picture></p>

blochsim simulates the signal of an MR sequence, and its derivatives, in
PyTorch. You describe your sequence once, as a **signal model**, and blochsim
plays it over thousands of voxels at once, on the CPU or on an NVIDIA GPU.

Everything else is written once against that signal model: the derivative
with respect to tissue that a parameter fit needs, the derivative with
respect to the sequence that a protocol design needs, dictionary matching,
model-based reconstruction, and reading the same sequence back from a Pulseq
file or from the stream a scanner sends. Change the signal model and all of
them follow.

blochsim ships signal models for 26 sequences (spin echo, FSE, SPGR, bSSFP,
MPRAGE, MP2RAGE, MR fingerprinting, Look-Locker, CEST, ASL...), so for most
work you write nothing at all.

## How it works

1. You construct a simulator with the sequence parameters: the flip angles,
   the TR, the echo spacing.
2. You call `simulate` with the tissue: T1, T2, and any other property you
   care about (B1, off-resonance, diffusion, a second exchanging pool...).
   A scalar is one voxel, an array is a map, and every voxel runs at once.
   Naming a property is what turns its physics on; what you leave out costs
   nothing.
3. blochsim turns the sequence into a stream of events (pulses, delays,
   readouts) once, and plays it through an extended phase graph (EPG)
   state machine, compiled for the CPU and the GPU. Calling again with new
   numbers reuses the stream.
4. `jacobian` gives you the derivative with respect to tissue properties,
   which is what a fit descends. The flip angles are ordinary PyTorch
   tensors, so `loss.backward()` gives you the derivative with respect to the
   sequence, which is what a design optimizes.

## Shipped sequences

```python
import torch

from blochsim.simulators import FSESimulator

fse = FSESimulator(ESP=5.0, TR=3000.0)  # ms
flip = torch.full((48,), 180.0)  # degrees, one per echo

signal = fse.simulate(flip=flip, T1=1000.0, T2=torch.tensor([80.0, 110.0, 2000.0]))
signal, dT2 = fse.jacobian("T2", flip=flip, T1=1000.0, T2=80.0)
```

- `signal` has one row per voxel and one column per echo.
- Times are in milliseconds and angles in degrees, as on a scanner console.
- Every shipped simulator also has a function form for one-off calls, such
  as `blochsim.fse_sim(...)`.

## Custom signal models

A signal model is two things:

- **handlers**: which operator plays each kind of event. A sequence has
  excitations, readouts and waits, and may have refocusing, inversion,
  saturation, preparation and other pulses, one handler for each Pulseq RF
  use. Each is a class attribute; one you do not set keeps its default (a
  wait is `Delay`: free precession, no RF and no ADC).
- **layout**: the events of one repetition, in order, written with those
  handlers. It is the sequence's default description: the constructor builds
  it from the parameters you pass. `from_pulseq` and `from_description` read
  the description from a `.seq` file or a scanner's stream instead, and skip
  the layout.

Here is saturation recovery, which saturates, waits, and reads what came back:

```python
import torch

from blochsim import Delay, Excitation, SPGRReadout, Spoil
from blochsim import Simulator


def Saturate(flip_rad=torch.pi / 2, phase_rad=0.0):
    """A 90-degree pulse, then a spoiler."""
    return Excitation(flip_rad, phase_rad) @ Spoil()


class SaturationRecovery(Simulator):
    saturation = Saturate  # plays a saturation pulse
    excitation = Excitation  # plays an excitation pulse
    delay = Delay  # plays a wait
    readout = SPGRReadout  # plays a readout, then spoils
    states = 1

    def layout(self, *, TS, flip):
        angle = torch.deg2rad(torch.as_tensor(flip))
        parts = []
        for wait in torch.as_tensor(TS) * 1e-3:  # ms to s
            parts += [self.operators.saturation(), self.operators.delay(wait)]
            parts += [self.operators.excitation(angle), self.operators.readout()]
        return parts


recovery = SaturationRecovery(TS=[100.0, 400.0, 1600.0], flip=10.0)
signal = recovery.simulate(T1=830.0, T2=80.0)
```

- `layout` uses only `self.operators`, the handlers you set. You never
  write timestamps; each operator holds the timeline for as long as it lasts.
- The handlers play the events whichever way the description came, so a
  sequence read with `from_pulseq` or `from_description` still uses them. A Pulseq file or a scanner stream carries pulses, ADC
  windows and timing but no gradients, so the handlers are where the
  dephasing lives: the readout decides whether the states are spoiled,
  wound on or rewound.
- `@` composes two operators into one, so a pulse and its spoiler read as the
  single thing you would name.

Derivatives, devices, extra physics and the estimators all work on your class
without another line.

## Estimation, design and reconstruction

- **Parameter maps**: dictionary matching, lookup tables and PERK, all built
  from a simulator (`blochsim.estimators`).
- **Model-based reconstruction**: the signal model as a factor of the forward
  operator, linear (`blochsim.Subspace`) or nonlinear
  (`blochsim.ModelOperator`). [bartorch](https://github.com/pulserver/bartorch)
  supplies the encoding and the Gauss-Newton solvers, and fits the same model
  voxel by voxel with `apps.mobafit`.
- **Sequence design**: flip angles, timings or RF pulse samples optimized by
  gradient descent through the simulation, within the scanner's limits
  (`blochsim.optim.SequenceDesign`).
- **Sequences you did not write**: `Simulator.from_pulseq` reads a `.seq`
  file, and `Simulator.from_description` reads the sequence description
  pulserver streams from a running scanner.

## Installation

Install the PyTorch build for your machine first, then blochsim:

```bash
pip install blochsim
pip install "blochsim[cu12]"  # the GPU kernels, beside a CUDA 12 build of torch
```

The [User guide](https://pulserver.github.io/blochsim/latest/user-guide/index.html)
covers CUDA versions, macOS and building from source.

## Learn more

- [Course](https://pulserver.github.io/blochsim/latest/examples/course.html):
  five lessons, from a shipped simulator to your own signal model and your own
  operator.
- [Applications](https://pulserver.github.io/blochsim/latest/examples/applications.html):
  parameter mapping, sequence design, model-based reconstruction and
  synthetic data, each a complete worked example.
- [Documentation](https://pulserver.github.io/blochsim/latest/), for the
  physics and every API detail.

## How to cite

If you use blochsim in your work, please cite:

```bibtex
@inproceedings{cencini2025pulserver,
  title     = {Pulserver: an open-source Pulseq-based client-server framework for vendor agnostic, interactive {MR} sequence design},
  author    = {Cencini, Matteo and Wang, Kang and Huang, Sherry and Schulte, Rolf F. and Sprenger, Tim
               and Noll, Douglas C. and Tosetti, Michela and Nielsen, Jon-Fredrik},
  booktitle = {Proceedings of the International Society for Magnetic Resonance in Medicine},
  pages     = {1275},
  year      = {2025}
}
```

blochsim's state machine is the extended phase graph formalism
([Weigel, 2015](https://doi.org/10.1002/jmri.24619)); please cite it too.

## License

MIT, see [LICENSE](https://github.com/pulserver/blochsim/blob/main/LICENSE.txt).
