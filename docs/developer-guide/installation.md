# Developer Guide

Setting up to change BlochSim: prerequisites, the editable install, style,
tests, the documentation build and pull requests.

```{contents}
:local:
:depth: 2
```

## Prerequisites

Besides the environment and PyTorch build of the {doc}`user_guide`, you need:

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

(dev-style)=

## Style

`ruff format` formats and ruff's `I` rules sort imports. `ruff check` also
runs pycodestyle, pyflakes, bugbear, pydocstyle (numpydoc convention),
pyupgrade and annotation rules. `pyproject.toml` holds the whole
configuration.

### Pre-commit hooks

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

### Docstrings and comments

- **Docstrings** are numpydoc: a one-line summary, then Parameters, Returns,
  Raises. Each parameter is one or two lines: what it is, its unit, a
  non-obvious constraint. Types go in the docstring, where they can be
  qualified, because the API pages are built with typehints off.
- **Write for the code as it is now.** No "used to", "previously" or "this
  replaces"; no fixed bugs; no restating what the code says.
- **A comment explains a non-obvious algorithm or a choice a reader would
  undo.** Prefer a well-named function, or a test whose name states the
  invariant.
- **Units are public at the edges.** A caller writes milliseconds, degrees and
  Hz; a description carries microseconds and radians. Convert at the boundary
  and name the unit in the identifier (`duration_s`, `flip_rad`, `t1_ms`).

(dev-tests)=

## Tests

```sh
pytest tests/                     # everything, with coverage
pytest tests/epg/test_shift.py -k inversion
pytest tests/ -n auto             # across cores
```

Coverage is on through `addopts` and writes `coverage.xml`. `tests/epg/` pins
each operator of the state machine against a closed form; the other
directories mirror the subpackages.

On Linux the GPU kernels are also compiled for the host, and
`tests/sequence/test_host_kernels.py` and `test_many_pools_host.py` hold them
to the C++ kernels. Tests that need a card skip without one.

A change to the physics comes with a test against something outside BlochSim:
a closed form, a published figure, or an isochromat summation written in the
test.

## Documentation build

```sh
bash scripts/build_docs.sh              # incremental
bash scripts/build_docs.sh --clean      # re-execute every example
PYTHON_BIN=~/envs/blochsim/bin/python bash scripts/build_docs.sh
python -m http.server --directory docs/build/html 8000
```

The build executes every gallery example and re-renders the explanation
figures from `docs/explanation_figures.py` with the BlochSim in your working
tree. To add a figure, write a function returning a Matplotlib figure,
register it in `FIGURES`, and reference the file it writes with a `figure`
directive. To add an example, add a script to its `examples/` section: the
numeric prefix orders it and the module docstring opens the page.

## Pull requests

1. Branch from `main` in your fork, one subject per branch.
2. Make the change with a test that fails without it.
3. Run what CI runs:

   ```sh
   pre-commit run --all-files
   pytest tests/
   bash scripts/build_docs.sh
   ```

4. Open the pull request. The template asks what the change does and how it
   was checked: the command and what it printed.
5. CI runs the hooks, the tests on Linux, macOS and Windows at Python 3.10 and
   3.14 with the kernels compiled fresh, and the documentation build. All of
   it has to pass.
6. Answer review with follow-up commits rather than a force-push, so the
   reviewer can see what changed.

Contributions follow the {doc}`misc/code_of_conduct` and are released under
the {doc}`misc/license`.

## Internals

Derivations, format rules and kernels, for whoever changes them.

```{toctree}
:maxdepth: 1

internals/epg
internals/description
internals/kernels
```
