---
name: build-and-test
description: Compile the C++ kernels and run the BlochSim suite, including the GPU kernels on a machine with no GPU. Use when asked to build, install, test, or reproduce a failure in blochsim.
---

# Build and test BlochSim

## Install

```sh
pip install -e ".[dev]" ; echo "exit: $?"
```

`dev` is `test` + `doc` + `examples` plus ruff, mypy and pre-commit. For a test
run alone, `pip install -e ".[test]"` is enough and much smaller.

An editable install puts the Python sources on the path, so an edit under
`src/blochsim` takes effect on the next import. The two C++ extensions are
compiled artifacts and do not: re-run the install after editing any `.cpp`,
`.hpp` or `.cu`.

**Read the exit status, not the output.** A failed compile leaves the
previously built `.so` importable, so the suite runs green against a kernel
that no longer corresponds to the source. Confirm the extension is the one you
just built:

```sh
python -c "import blochsim._epg_cpu as k; print(k.__file__)"
```

## Test

```sh
pytest tests/                          # everything, with coverage
pytest tests/epg/                      # one area
pytest tests/epg/test_shift.py -k inversion
pytest tests/ -n auto                  # across cores
```

`tests/` mirrors the source. `epg/` pins the state machine operator by operator
against closed forms — the shift, the RF rotation, relaxation, diffusion, flow,
spoiling, the two-pool and three-pool longitudinal steps — while `sequence/`,
`model/`, `operator/`, `estimators/` and `optim/` cover the layers above.

## The GPU kernels, without a GPU

On Linux the install compiles the GPU kernels for the host as
`blochsim._gpu_host`, and
for the card as `blochsim._gpu` wherever CMake finds `nvcc`
(`--config-settings=cmake.define.BLOCHSIM_CUDA=ON` insists on it).
`tests/sequence/test_host_kernels.py` and `test_many_pools_host.py` run the
host build against the C++ kernels, which is how the GPU kernels are verified
on any machine; the CUDA tests skip themselves without a card.

## Style

```sh
pre-commit install              # once per clone; nothing runs on commit until you do
pre-commit run --all-files      # the whole tree, which is what CI's Lint job runs
pre-commit run --files path/to/one.py
```

`ruff format`, then `ruff check --fix`, plus whitespace and file hygiene. There
is no black and no isort, and `.pre-commit-config.yaml` pins the ruff version
so a local run and CI agree on it.

A hook that rewrites a file fails the commit and leaves the fix in the working
tree: read it, `git add` it, commit again.

To get a commit through without them — a work-in-progress commit, or a merge
you did not write:

```sh
SKIP=ruff-check git commit -m "..."     # one hook
git commit --no-verify -m "..."         # all of them
```

CI runs the same hooks over the whole tree on every branch, so either one
defers the failure rather than avoiding it.
