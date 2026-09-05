"""Load the compiled kernels out of an installed wheel, and report their size.

Run by cibuildwheel against every wheel it builds, on the interpreter that
wheel claims to support. It loads each extension by file path rather than by
importing :mod:`torchsim`, so the check needs no PyTorch and says something
about the binary alone: that the stable-ABI module initialises on this
interpreter, and that it carries no vendored library.

    python scripts/check_wheel.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

#: A kernel calls no PyTorch API and links no PyTorch library, so anything
#: approaching this size means something was bundled that should not have been.
LARGEST_REASONABLE_BYTES = 16 * 1024 * 1024


def kernel_directory() -> pathlib.Path:
    """Where the installed package lives, without executing it."""
    spec = importlib.util.find_spec("torchsim")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("torchsim is not installed")
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def main() -> int:
    """Load every compiled kernel in the installed package."""
    root = kernel_directory()
    kernels = sorted(root.glob("_*_cpu.*"))
    kernels = [path for path in kernels if path.suffix in {".so", ".pyd", ".dylib"}]
    if len(kernels) != 2:
        raise SystemExit(f"expected two kernels in {root}, found {kernels}")

    for path in kernels:
        name = path.name.split(".")[0]
        spec = importlib.util.spec_from_file_location(f"torchsim.{name}", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"no loader for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        size = path.stat().st_size
        print(f"{path.name}: {size} bytes, {len(dir(module))} attributes")
        if size > LARGEST_REASONABLE_BYTES:
            raise SystemExit(f"{path.name} is {size} bytes; something was bundled")

    print(f"ok on {sys.implementation.name} {'.'.join(map(str, sys.version_info[:3]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
