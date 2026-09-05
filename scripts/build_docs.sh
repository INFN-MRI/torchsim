#!/usr/bin/env bash
#
# Build the HTML documentation, including the executed examples.
#
#   bash scripts/build_docs.sh              # incremental
#   bash scripts/build_docs.sh --clean      # re-run every example from scratch
#
# The examples are executed by sphinx-gallery, so this needs an interpreter
# that can import torchsim as well as the documentation extensions. Set
# PYTHON_BIN to choose one.
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
clean=0
for argument in "$@"; do
    case "$argument" in
        --clean) clean=1 ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

if ! "$python_bin" - <<'PY'
import importlib, sys


def absent(name):
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:  # the package holding it is not there either
        return True


missing = [
    name
    for name in ("torchsim", "torchsim._epg_cpu", "torch", "matplotlib",
                 "sphinx", "sphinx_gallery", "sphinx_copybutton",
                 "sphinx_exec_directive", "sphinx_book_theme", "myst_parser",
                 "linkify_it", "pypulseq")
    if absent(name)
]
if missing:
    print("missing: " + " ".join(missing), file=sys.stderr)
sys.exit(1 if missing else 0)
PY
then
    echo "install them with: $python_bin -m pip install -e \".[doc]\"" >&2
    exit 1
fi

if [ "$clean" -eq 1 ]; then
    rm -rf "$root/docs/build" "$root/docs/generated"
fi

# The pages are built against the installed TorchSim, kernels and all. Put
# ``src`` on the path instead and the compiled halves of the package are gone:
# a figure the fused kernel draws then fails the build.
cd "$root/docs"
"$python_bin" -m sphinx -b html . build/html

echo
echo "open file://$root/docs/build/html/index.html"
