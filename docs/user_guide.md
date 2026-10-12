# User Guide

Installation, a first simulation, and where the rest of this documentation is.

```{contents}
:local:
:depth: 2
```

## Supported platforms

BlochSim supports Python 3.10 and later; CI tests 3.10 and 3.14 on Linux,
macOS and Windows. One wheel per platform serves every supported interpreter,
because the kernels are built against the CPython stable ABI.

| Platform | Wheels | GPU kernels |
| --- | --- | --- |
| Linux | x86-64 and aarch64, glibc and musl | x86-64, through the `cu12` or `cu13` extra |
| macOS | Apple silicon and Intel | none |
| Windows | x86-64 | none |

Elsewhere, `pip` builds the C++ kernels from source, which needs a C++17
compiler and CMake. There is no pure-Python fallback.

## Installation

### Virtual environment

Install into an environment of its own, so the PyTorch, NumPy and SciPy
versions a simulation pins stay apart from the rest of the machine.

::::{tab-set}

:::{tab-item} venv
:sync: venv

```sh
python -m venv ~/envs/blochsim
source ~/envs/blochsim/bin/activate     # Windows: ~\envs\blochsim\Scripts\activate
python -m pip install --upgrade pip
```
:::

:::{tab-item} conda
:sync: conda

```sh
conda create -n blochsim python=3.12
conda activate blochsim
```

Install BlochSim with `pip` inside the environment; there is no conda package.
:::

:::{tab-item} uv
:sync: uv

```sh
uv venv --python 3.12 ~/envs/blochsim
source ~/envs/blochsim/bin/activate
```

Read `uv pip install` for `pip install` below.
:::

::::

### PyTorch

Install PyTorch first. A CPU build and a CUDA build share a name and a
version, and differ only in the index they come from, so `pip` cannot choose
for you.

::::{tab-set}

:::{tab-item} CPU only
:sync: cpu

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Everything in BlochSim runs on the CPU, on threaded C++ kernels.
:::

:::{tab-item} NVIDIA GPU
:sync: cuda

Pick the CUDA tag your driver supports on
[pytorch.org](https://pytorch.org/get-started/locally/); `cu128` stands for it
below:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
:::

:::{tab-item} Apple silicon
:sync: mac

```sh
pip install torch
```

There is no Metal path; the simulation runs on the CPU kernels.
:::

::::

Check the build before installing on top of it:

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CUDA build prints a CUDA version and `True`; a CPU build prints `None` and
`False`.

### BlochSim

```sh
pip install blochsim            # CPU kernels
pip install "blochsim[cu12]"    # plus the GPU kernels, for a CUDA 12 PyTorch
pip install "blochsim[cu13]"    # plus the GPU kernels, for a CUDA 13 PyTorch
```

The GPU kernels are compiled ahead of time and need a driver, not the CUDA
toolkit. Install the extra that matches PyTorch's CUDA major version; BlochSim
refuses a mismatched one and names the extra to install. Other extras:
`pulseq` reads `.seq` files, and `examples` brings the packages the gallery
imports.

Check the installation by simulating a fast spin echo for three tissues:

```python
import torch
from blochsim.simulators import FSESimulator

acquisition = FSESimulator(
    ESP=5.0,                                   # ms
    TR=3000.0,                                 # ms
    T1=torch.tensor([830.0, 1330.0, 4000.0]),  # ms
    T2=torch.tensor([80.0, 110.0, 2000.0]),    # ms
)
signal = acquisition.simulate(flip=torch.full((48,), 60.0))
print(signal.shape)  # torch.Size([3, 48])
```

On a GPU, move the simulator and give it tensors on the same device:

```python
acquisition = acquisition.to("cuda")
signal = acquisition.simulate(flip=torch.full((48,), 60.0, device="cuda"))
```

## First simulation

Every sequence is a {class}`~blochsim.model.Simulator`. A shipped simulator
fixes the sequence; `simulate` takes the tissue and anything you vary, and
`jacobian` returns the signal with its derivatives:

```python
import numpy as np
from blochsim.simulators import MRFSimulator

flip = np.concatenate(
    (np.linspace(5.0, 60.0, 300), np.linspace(60.0, 2.0, 300), np.full(280, 2.0))
)
sequence = MRFSimulator(flip=flip, TR=10.0)
signal, jacobian = sequence.jacobian(("T1", "T2"), T1=1000.0, T2=100.0)
```

`jacobian` holds the derivatives with respect to T1 and T2, one row each. The
signal comes back in the array library you gave it: NumPy here, CuPy or
PyTorch elsewhere. Dictionary matching, nonlinear least squares, model-based
reconstruction and sequence design all take a simulator.

Functions such as {func}`blochsim.mrf_sim` wrap the shipped simulators for a
one-off call. {class}`~blochsim.model.Simulator` is the class to subclass for a
sequence of your own.

## Documentation structure

| Section | Content |
| --- | --- |
| {doc}`explanations/index` | Signal models, extended phase graphs, sequence descriptions, derivatives, execution, estimation, reconstruction and design. |
| {doc}`generated/autoexamples/index` | The Course, from a shipped simulator to a signal model and an operator of your own, and the Applications. |
| {doc}`api/index` | Exact interfaces, units and defaults: {doc}`api/simulators` for what ships, {doc}`api/model` for writing your own. |
| {doc}`developer_guide` | Development install, style, tests and pull requests. |

## Support and bug reports

Ask questions in
[Discussions](https://github.com/pulserver/blochsim/discussions): how to model
a sequence, whether a signal is expected, which estimator suits a problem.

Report bugs in
[Issues](https://github.com/pulserver/blochsim/issues/new/choose). The form
asks for:

- the shortest script that reproduces it;
- what you expected, and the reference you compare against: a closed form, a
  published figure or another simulator;
- the full traceback, if it raises;
- the environment, as the form's command prints it;
- whether it happens on the CPU kernels, the GPU kernels, or both.

Request a sequence, a physical effect or an estimator through the same form,
naming the paper the model comes from and the figure it should reproduce.

Report a vulnerability privately, as a draft advisory from the repository's
[Security tab](https://github.com/pulserver/blochsim/security/advisories/new)
or by email to the address in the
[security policy](https://github.com/pulserver/blochsim/blob/main/.github/SECURITY.md).
The kernels index raw pointers, so an out-of-bounds read or write reachable
from ordinary arguments belongs there. Wrong physics is a bug report.
