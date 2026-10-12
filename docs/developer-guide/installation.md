# Installation

The prerequisites, the editable install, the pre-commit hooks and the
documentation build.

```{contents}
:local:
:depth: 2
```

## Prerequisites

Besides the environment and PyTorch build of the
{doc}`user guide <../user-guide/index>`, you need:

- **A C++17 compiler and CMake.** The CPU kernels (`src/blochsim/_epg_cpu.cpp`,
  `_perk_cpu.cpp`) are compiled at install, and there is no Python fallback.

  ::::{tab-set}

  :::{tab-item} Linux
  :sync: linux

  ```sh
  sudo apt install build-essential      # Debian, Ubuntu
  sudo dnf install gcc-c++ make         # Fedora, RHEL
  ```
  :::

  :::{tab-item} macOS
  :sync: mac

  ```sh
  xcode-select --install
  ```
  :::

  :::{tab-item} Windows
  :sync: windows

  The Visual Studio Build Tools with the "Desktop development with C++"
  workload; build from the *Developer Command Prompt*.
  :::

  ::::

- **Git**, and a fork of <https://github.com/pulserver/blochsim> to open a
  pull request from.
- **The CUDA toolkit**, only to compile the GPU kernels for a card. CMake
  compiles them wherever it finds `nvcc`; on Linux they are also compiled for
  the host, so they can be tested without one (see {ref}`dev-tests`).

## Development installation

```sh
git clone https://github.com/pulserver/blochsim
cd blochsim
pip install -e ".[dev]"
pre-commit install
```

The `dev` extra is `test`, `doc` and `examples` plus ruff, mypy and
pre-commit. Python edits take effect on the next import. A C++ edit needs the
install again, and a failed compile leaves the previous extension importable,
so check the exit status and which file is loaded:

```sh
pip install -e ".[dev]" ; echo "exit: $?"
python -c "import blochsim._epg_cpu as k; print(k.__file__)"
```

### Source layout

| Path | Content |
| --- | --- |
| `src/blochsim/sequence/` | Events, operators and builders, and the dispatch to the kernels. `_parameters.py` is the parameter ABI the Python dispatch, the C++ extension and the GPU kernels share. |
| `src/blochsim/model/` | The signal model: physics, the simulator that orders events, and the binding that resolves a protocol's structure once. |
| `src/blochsim/_epg_cpu.cpp`, `_perk_cpu.cpp` | The CPU kernels. Every GPU path exists here too; the two agree to float32 round-off. |
| `src/blochsim/_epg_kernels.hpp`, `_pools_kernels.hpp` | The GPU kernels. |
| `src/blochsim/simulators/`, `estimators/`, `recon/`, `optim/` | Shipped sequences, and what is built on them. |
| `tests/`, `examples/`, `docs/` | Tests mirrored by subpackage, gallery scripts, Sphinx sources. |

Read {doc}`internals/kernels` before changing any of the kernels.

## Pre-commit hooks

After `pre-commit install`, every commit checks the staged files.
`.pre-commit-config.yaml` pins the ruff version, so your commit and CI are
judged by the same build.

```sh
pre-commit run --all-files      # what CI runs
pre-commit run ruff-format      # one hook
pre-commit autoupdate           # move the pinned versions forward
```

A hook that rewrites a file fails the commit and leaves the fix in the working
tree: review it, `git add` it and commit again.

### Hook bypass

```sh
SKIP=ruff-check git commit -m "..."     # one hook
git commit --no-verify -m "..."         # all hooks
```

CI runs every hook on every branch, so a bypass only defers the failure. A
rule that does not fit this code belongs in `ignore` or `per-file-ignores` in
`pyproject.toml`, with a comment.

## Documentation build

```sh
bash scripts/build_docs.sh              # incremental
bash scripts/build_docs.sh --clean      # re-execute every example
PYTHON_BIN=~/envs/blochsim/bin/python bash scripts/build_docs.sh
python -m http.server --directory docs/build/html 8000
```

The build executes every gallery example and re-renders the explanation
figures from `docs/explanation_figures.py` with the BlochSim in your working
tree. Adding a figure or an example is described in {doc}`documentation`.
