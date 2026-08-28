---
name: cut-a-release
description: Publish a TorchSim release to PyPI, and diagnose a wheel build. Use when asked about releasing, versioning, cibuildwheel, abi3 wheels, or the packaging setup.
---

# Cut a TorchSim release

The version comes from the git tag through setuptools-scm; nothing in the tree
states it. To release:

```sh
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

The `Wheels` workflow then builds the sdist and every wheel, tests each one,
and publishes to PyPI through trusted publishing — no token is stored
anywhere. The `pypi` environment on the repository gates it.

`workflow_dispatch` builds and tests everything without publishing, which is
how to check a packaging change before tagging. So does any pull request that
touches `pyproject.toml`, `CMakeLists.txt`, the `.cpp` sources, or the workflow.

## What gets built, and why there is so little of it

Both kernels are plain CPython extensions against the **stable ABI** from 3.10
on. They call no PyTorch API and link no PyTorch library. So:

- one `cp310-abi3` wheel per platform serves every supported interpreter —
  there is no per-minor-version build to repeat;
- the wheel is a couple of megabytes rather than the size of libtorch, and
  needs no `auditwheel`/`delocate` repair beyond manylinux compliance.

A `#include <torch/...>` in either `.cpp` ends all of that: the wheel would
become per-version, per-torch-version, and hundreds of megabytes.
`scripts/check_wheel.py` guards the size and is what cibuildwheel tests every
wheel with — it loads each kernel by file path, so it needs no PyTorch.

Platforms: manylinux x86_64 and aarch64 on native runners, macOS arm64 with the
Intel wheel cross-compiled beside it, Windows AMD64.

## The build itself

`scikit-build-core` drives `CMakeLists.txt`. `setup.py` is a shim for tools
that still shell out to it and configures nothing — do not put build logic
there. `[tool.scikit-build]` in `pyproject.toml` sets `wheel.py-api = "cp310"`,
which is what produces the abi3 tag, and excludes the `.cpp` and `.hpp` sources
from the wheel.

CMake 3.26 or newer is required, for `python_add_library(... USE_SABI ...)`.
scikit-build-core installs it as a build-time wheel when the system CMake is
older, so a stale system CMake is not a problem — unless you build with
`--no-build-isolation`, where it is.
