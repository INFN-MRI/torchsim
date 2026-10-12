# BlochSim, for an agent working on it

BlochSim simulates MR signals in PyTorch, differentiably. A **signal model** is
the only thing a sequence has to supply; differentiation, device placement,
parameter estimation, model-based reconstruction and sequence design are all
written once against that interface.

`AGENTS.md` and `.github/copilot-instructions.md` are symlinks to this file.
Edit this one. If you find yourself editing either of the others, you have a
broken checkout (Windows without `core.symlinks`), not two documents.

## The shape of the package

| Path | What is in it |
| --- | --- |
| `src/blochsim/sequence/` | The description an acquisition is assembled from — events, operators, builders — and the dispatch that turns one into a kernel launch. |
| `src/blochsim/model/` | What a signal model *is*: the physics, the simulator that orders its events, and the binding that resolves a protocol's structure once and rebinds its values per call. |
| `src/blochsim/_epg_cpu.cpp`, `_perk_cpu.cpp` | The CPU kernels. Every path the GPU has exists here too, and the two agree to float32 round-off. |
| `src/blochsim/_epg_kernels.hpp`, `_pools_kernels.hpp`, `_perk_kernels.hpp` | The GPU kernels, written over the tiles of `_tile.hpp`. `_kernels.hpp` is the table a launch reads: each kernel's parameters, and which of them size the block. |
| `src/blochsim/_gpu.cu`, `_gpu_kernel.cu.in`, `_gpu_host.cpp`, `_gpu_launch.py` | The same kernels compiled ahead of time for the card (`_gpu`), and for the host (`_gpu_host`), which is how the suite checks them without one; and the launcher both answer to. |
| `src/blochsim/sequence/_epg_gpu.py`, `_pools_gpu.py`, `estimators/_perk_gpu.py` | What a GPU launch is given: the tiling and the arguments. |
| `src/blochsim/simulators/`, `estimators/`, `recon/`, `optim/` | The sequences that ship, and what is built on top of them. |
| `tests/`, `examples/`, `docs/` | Mirrored by subpackage, executed by the gallery, built by Sphinx. |

The shared parameter ABI — read by the Python dispatch, the C++ extension and
the GPU kernels alike — is `src/blochsim/sequence/_parameters.py`. A
parameter added there is added in all three places or in none.

## Commands

```sh
pip install -e ".[dev]"     # the whole toolchain, and it compiles the kernels
pytest tests/               # the suite, with coverage
pytest tests/ -n auto       # across cores
pre-commit install          # once per clone, or no hook runs on commit
pre-commit run --all-files  # exactly what CI's Lint job runs
bash scripts/build_docs.sh  # HTML into docs/build/html, examples executed
```

`ruff` is the only style tool: `ruff format` is the formatter and the `I` rules
are the import sorter. There is no black and no isort. Do not add a formatting
argument to a command; `pyproject.toml` is the whole configuration, and
`.pre-commit-config.yaml` pins the ruff the hooks read it with.

A hook that rewrites a file fails the commit and leaves the fix in the working
tree — read it, stage it, commit again. `SKIP=ruff-check git commit` steps
around one hook and `git commit --no-verify` around all of them, but CI runs
the same hooks on every branch, so either only defers the failure.

## Traps worth knowing before you spend an hour on one

**A failed extension build leaves the previous `.so` importable.** The suite
then runs green against a kernel that no longer matches the source you are
reading. Read the exit status of the install, not the last lines of its output:

```sh
pip install -e ".[dev]" ; echo "exit: $?"
python -c "import blochsim._epg_cpu as k; print(k.__file__)"
```

**The GPU kernels are compiled where a CUDA compiler is found.** CMake looks
for `nvcc`, and `BLOCHSIM_CUDA=ON` or `OFF` (`--config-settings=cmake.define.BLOCHSIM_CUDA=ON`)
overrides what it finds; `CMAKE_CUDA_ARCHITECTURES` picks the cards. A kernel
compiles in a file of its own, twice -- bounded to 256 threads and to 1024 --
so a build is minutes of `nvcc` spread over as many cores as Ninja is given.
On Linux the same kernels are also compiled for the host (`_gpu_host`), one
program at a time over host tensors; `tests/sequence/test_host_kernels.py` and
`test_many_pools_host.py` hold them to the C++ kernels, which is how the GPU
path is verified on a machine with no card.

**A block is at most 1024 threads.** A tile holds one state order per thread,
so an EPG launch with more than 1024 state orders, or a pooled one whose orders
by pools pass 1024, is refused with the kernel's name rather than launched. The
problems a program carries are another matter: a thread holds `Y_LANES` of
them in registers, set per kernel in `src/blochsim/_lanes.hpp`, so that one
reading of each event serves all of them.

**The EPG kernels are written for their layouts** (`_layout.hpp`), and a
launch whose rows fit a warp runs them ahead of any tile kernel. A layout is
what is compiled: the pools, how a pulse is formed, whether the tissue has
per-voxel maps, the problems a thread holds, one train or several; every other
switch is read at run time and steers whole blocks once per event, so one
compile serves every combination of them. The loops are written once over a
number type (`_layout_numbers.hpp`): at `float` they are the forward
simulation, at `num::Dual` -- a value and its derivative along the direction
-- the Jacobian-vector product. The adjoints are written the same way, so
their derivative along a direction is the same source at `num::Dual`: the
forward sweep keeps the state every few events and the reverse sweep replays
each stretch from it, and an interval's gradient is contracted against its
operator's derivatives -- taken along every tissue input at once by
`num::Multi` -- only when the interval changes. Code that runs only when an
interval changes is out of line and reads its switches at run time, which is
what keeps a layout's compile to seconds. `_gpu_launch.generic_kernels()`
turns the layouts off and `layout_launches()` counts them;
`tests/sequence/test_layout_kernels.py` holds the two to each other. They
exist only on the card: `_gpu_host` compiles the tile kernels, so the host
lane holds those, not these, to the C++ kernels.

**The EPG kernels index in 32 bits** (`bsk::index_t`). An offset that can
pass 2^31 is cast to 64 bits where it is formed, and the launcher refuses an
integer argument that does not fit rather than truncating it.

**`--cov` is on by default** through `addopts`, so a bare `pytest` writes
`coverage.xml`. It is ignored, not tracked.

## Style, and the rule behind it

Formatting is settled by `ruff format`; do not argue with it. What is left is
what a human decides:

- **Write for someone reading the code as it is now.** Never write text whose
  subject is the history of the code — no "used to", no "previously", no "this
  replaces the old X", no naming a bug that has been fixed. Do not justify the
  present shape by contrast with a shape that is gone. Do not restate what the
  code plainly says. This is enforced in review, and it binds prose docs
  exactly as hard as it binds a header comment.
- **A docstring carries what a caller needs to call it**: one line of what it
  does, then Parameters, Returns, Raises, numpydoc. Types belong in the prose,
  where they can be qualified ("array-like, one per echo"); the API pages are
  built with typehints off for that reason, and the annotations stay in the
  source for editors and for mypy.
- A comment earns its place by explaining a non-obvious algorithm, or a choice
  a reader would otherwise undo. Even then, prefer a well-named function, or a
  test whose name states the invariant — those cannot go stale silently. When
  tempted to explain *why not the other way*, **write a test instead**.
- **Units are public at the edges and internal underneath.** A caller writes
  milliseconds, degrees and Hz; a description timestamps in microseconds and
  carries radians. Convert at the boundary, and name the unit in the identifier
  (`duration_s`, `flip_rad`, `t1_ms`) rather than in a comment beside it.

## Changing the physics

A change to what the kernels compute arrives with a test that pins it against
something **outside** BlochSim: a closed form, a published figure, or an
isochromat summation written out in the test itself. The `tests/epg` files are
written that way and each states its invariant in its module docstring. A test
that only compares BlochSim to BlochSim proves the two agree, which was never
in doubt.

Whatever you change in one kernel, change in the other. The C++ and GPU
implementations are held to each other to float32 round-off, and a path that
exists on one side and not the other is a bug in whichever side is missing it.

## Documentation

The pages under `docs/` are **MyST Markdown**, built by Sphinx. One thing stays
reStructuredText because the tooling requires it:

- `docs/_templates/autosummary/*.rst` — `sphinx.ext.autosummary` writes its
  stubs with a hardcoded `.rst` suffix and finds its directives by regex over
  raw source lines, which is also why the API pages hold their `autosummary`
  and `currentmodule` directives inside `{eval-rst}` blocks.

**A gallery header is Markdown behind a one-line shim.** sphinx-gallery pastes
the header *verbatim* into a generated `index.rst`, so a `README.md` handed to
it directly is parsed as reStructuredText: the build succeeds with no warning
and the page silently loses its title, inherits the first subsection's, and
prints MyST labels as literal text. The working arrangement is a `README.rst`
holding nothing but

```rst
.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_
```

with the prose in `_gallery_header.md` beside it. Two settings make that work
and both are load-bearing: `copyfile_regex` in `sphinx_gallery_conf` carries
the `.md` into the output directory so the include resolves, and
`exclude_patterns` keeps Sphinx from also building it as a page of its own,
which would define every label in it twice.

### The gallery's prose

`~/.claude/CLAUDE.md` carries the rules for explanatory prose and they bind
here. These are specific to this gallery.

**`examples/01-framework/01-first-simulation.py` and `03-custom-signal-model.py`
are the register.** Match them: a numbered title for a Course lesson, an
opening that tells you what you will do, a **Learning objectives** list and a
Previous/Next line; then sections each named by the concept it covers, giving
the physics or the mental model first and the code after it, written to
"you". The last section, *Specification*, is a `text` code block stating what the
page did the way you would ask an agent for it. An Application opens with its
objective and a *Prerequisites* line naming the lessons it assumes, and has no
Previous/Next.

**Every notebook has one scope and stays inside it.** A dictionary match is the
subject of `02-parameter-inference/01`; a notebook that needs one as a baseline
fits it in a hidden cell and gives it a row in a table, not a section.

**Figures read in both themes.** Take colours from `docs/figure_style.py`
(`SERIES`, `MUTED`), never a fixed grey, black or white, and put every legend
outside the axes with `legend_outside`. Check each figure on a light and a dark
background before committing.

**Show the API and hide the plotting.** `sphinx_gallery_start_ignore` is for
figure code and print formatting. A cell a reader would type themselves —
`execution(...)`, `stream=`, `budget_bytes=`, a `description(...)` built by hand
— is visible even when its output is not interesting.

The figures on the explanation pages are simulated at build time by
`docs/explanation_figures.py`, so a figure cannot outlive the behaviour it
shows. To add one, write a function that returns a Matplotlib figure, register
it in the `FIGURES` mapping at the bottom, and reference the file it writes
with a `{figure}` directive. To add a gallery example, drop a script into the
right `examples/` section — the numeric prefix orders it and the module
docstring becomes the page.

An example is executed by the gallery when the interpreter can import
everything it imports, so what a build runs follows from the extras installed
into it, and a page already carrying output executed from the source as it
stands is reused rather than re-run.

`.github/workflows/docs.yml` builds the published pages and GitHub Pages
serves them from `gh-pages`: its HTML job checks every branch with the `doc`
extra, and its Site job executes every example with the `examples`
extra and hands the tree to `scripts/publish_docs.py`, which keeps one
directory per version -- `latest` for `main`, its own for each `v*.*.*` tag --
beside the list the version switcher reads.

## Packaging

The build is `scikit-build-core` driving `CMakeLists.txt`; `setup.py` is a shim
for tools that still shell out to it and configures nothing. The kernels are
plain CPython extensions against the **stable ABI** from 3.10 on: they call no
PyTorch API and link no PyTorch library, which is why one `cp310-abi3` wheel
per platform serves every supported interpreter and why that wheel is a couple
of megabytes rather than the size of libtorch. Keep it that way — a `#include
<torch/...>` in any of them ends all of that.

The card's module is a package of its own per CUDA major version,
`blochsim-cuda12` and `blochsim-cuda13` (`src/cuda/12`, `src/cuda/13`), which
`pip install blochsim[cu12]` or `blochsim[cu13]` installs beside a torch of
the same major version. Each is this CMake project with `BLOCHSIM_CUDA_PACKAGE`
set, which builds `_gpu` alone into `blochsim_cudaNN/`; it links the CUDA
runtime dynamically -- torch's nvidia wheel, found by rpath -- carries machine
code for 7.5, 8.0 and 9.0 and PTX for 9.0, and pins the blochsim it was built
with. `_gpu_launch` loads the build for torch's CUDA major version and refuses
one for another major or another release, naming the extra to install; with
none installed it takes the `_gpu` a source build leaves beside the package.

```sh
python -m build --wheel src/cuda/12   # with CUDA 12.6's nvcc as CUDACXX
```

Wheels are built by cibuildwheel, the CUDA packages in a manylinux container
of their own, and published to PyPI by trusted publishing on a `v*.*.*` tag,
each project from its own environment (`pypi`, `pypi-cuda12`, `pypi-cuda13`).
`scripts/check_wheel.py` loads each compiled kernel by path, without importing
the package, and is what every built wheel is tested with.
