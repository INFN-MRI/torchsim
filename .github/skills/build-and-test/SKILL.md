---
name: build-and-test
description: Compile the C++ kernels and run the TorchSim suite, including the Triton paths on a machine with no GPU. Use when asked to build, install, test, or reproduce a failure in torchsim.
---

# Build and test TorchSim

## Install

```sh
pip install -e ".[dev]" ; echo "exit: $?"
```

`dev` is `test` + `doc` + `examples` plus ruff, mypy and pre-commit. For a test
run alone, `pip install -e ".[test]"` is enough and much smaller.

An editable install puts the Python sources on the path, so an edit under
`src/torchsim` takes effect on the next import. The two C++ extensions are
compiled artifacts and do not: re-run the install after editing any `.cpp`.

**Read the exit status, not the output.** A failed compile leaves the
previously built `.so` importable, so the suite runs green against a kernel
that no longer corresponds to the source. Confirm the extension is the one you
just built:

```sh
python -c "import torchsim._epg_cpu as k; print(k.__file__)"
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
`model/`, `estimators/`, `recon/` and `optim/` cover the layers above.

## The Triton paths, without a GPU

The tests carrying the `interpreted` marker are deselected by `addopts`. They
run a Triton kernel through Triton's CPU interpreter — no card, no compile,
about a minute each — and are how the GPU plumbing is verified anywhere:

```sh
pytest tests/ -m interpreted
TRITON_INTERPRET=1 python your_script.py     # the same trick, by hand
```

Triton has no wheel for macOS or Windows. On those platforms the default
deselection is not an optimisation, it is the only thing that runs.

## Before you time anything

Kernel compiles dominate a cold GPU run, not the arithmetic. A suite that takes
minutes on a card is mostly Triton compiling one specialization per feature
combination it meets; the second run of the same suite is a different number
entirely. Run the whole suite at natural boundaries rather than after every
edit.

The cache keys on the *source* of `_epg_triton.py`, not on what it means, so a
formatting pass or a deleted dead local costs the same full recompile as a new
`tl.constexpr`. If the suite suddenly takes an hour where it took minutes,
check whether that file changed before looking for a performance regression.

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
