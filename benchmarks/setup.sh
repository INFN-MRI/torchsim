#!/usr/bin/env bash
# Install everything the dictionary benchmark compares, into one directory that
# can be deleted afterwards.
#
#   bash benchmarks/setup.sh            # everything
#   bash benchmarks/setup.sh python     # just the Python side
#   bash benchmarks/setup.sh julia      # just the Julia side
#
# What it makes, all under $PREFIX (benchmarks/.env by default):
#
#   .env/venv          a virtual environment with PyTorch, TorchSim, epgpy and
#                      sycomore in it
#   .env/micromamba    a package manager, only if conda-forge is needed
#   .env/julia         a Julia, only if there is not one on the path already
#
# Nothing is installed outside $PREFIX except through pip inside that venv, so
# `rm -rf benchmarks/.env` undoes all of it.
#
# Afterwards:
#
#   source benchmarks/.env/activate
#   python benchmarks/run_all.py
#   python benchmarks/validate.py
#   python benchmarks/make_figures.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$ROOT/benchmarks/.env}"
VENV="$PREFIX/venv"
STEP="${1:-all}"
PLATFORM="linux-64"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PLATFORM="osx-arm64" ;;
  Darwin-x86_64) PLATFORM="osx-64" ;;
esac

say() { printf '\n=== %s\n' "$*"; }

# A conda-forge package manager, fetched only if something below needs one.
micromamba() {
  local binary="$PREFIX/micromamba"
  if [ ! -x "$binary" ]; then
    say "fetching micromamba for $PLATFORM"
    local release="2.9.0-0"
    local url="https://github.com/mamba-org/micromamba-releases/releases/download"
    curl -fsSL -o "$binary" "$url/$release/micromamba-$PLATFORM"
    chmod +x "$binary"
  fi
  MAMBA_ROOT_PREFIX="$PREFIX/mamba" "$binary" "$@"
}

setup_python() {
  say "python environment in $VENV"
  # $PYTHON says which interpreter to build on. It matters: an interpreter that
  # ships a C++ runtime of its own -- a conda one does -- loads that runtime
  # ahead of the system's, and TorchSim's kernels are compiled against the
  # system's. The extension then fails to load with a missing GLIBCXX version,
  # every fused kernel is reported absent, and nothing here can run.
  [ -d "$VENV" ] || "${PYTHON:-python3}" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip

  say "PyTorch"
  # On Linux the default wheel carries CUDA and Triton, which is what the
  # `--device cuda` half of the sweep needs. For a machine with no card, the
  # CPU wheel is a much smaller download and runs the rest of it:
  #   pip install torch --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" install --quiet torch

  say "TorchSim, from this checkout"
  "$VENV/bin/pip" install --quiet -e "$ROOT"

  say "epgpy"
  # Not on PyPI; the repository is the distribution.
  "$VENV/bin/pip" install --quiet "git+https://github.com/py-baudin/epgpy"

  say "sycomore"
  # Sycomore's own instructions are conda-forge, and its pip build needs the
  # xsimd headers and pybind11's CMake package, which conda-forge has and PyPI
  # does not. Install it into the venv's site-packages through a conda
  # environment of its own.
  if ! "$VENV/bin/python" -c "import sycomore" 2>/dev/null; then
    micromamba create -y -q -p "$PREFIX/sycomore" -c conda-forge \
      "python=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" sycomore
    local site
    site="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    printf '%s\n' "$PREFIX/sycomore/lib/python"*/site-packages > "$site/sycomore.pth"
  fi

  say "matplotlib, for the figures"
  "$VENV/bin/pip" install --quiet matplotlib

  "$VENV/bin/python" - <<'PY'
for name in ("torch", "torchsim", "epgpy", "sycomore", "matplotlib"):
    try:
        __import__(name)
        print(f"  {name}: ok")
    except Exception as error:  # noqa: BLE001 - a report, not a failure
        print(f"  {name}: MISSING ({error})")
PY
}

setup_julia() {
  say "julia"
  local release="1.12.7"
  if command -v julia >/dev/null 2>&1; then
    echo "  using $(command -v julia)"
    JULIA="$(command -v julia)"
  elif [ -x "$PREFIX/julia/bin/julia" ]; then
    JULIA="$PREFIX/julia/bin/julia"
  elif curl -fsSL -o "$PREFIX/julia.tar.gz" \
      "https://julialang-s3.julialang.org/bin/linux/x64/${release%.*}/julia-$release-linux-x86_64.tar.gz"; then
    echo "  no julia on the path -- unpacking $release"
    mkdir -p "$PREFIX/julia" && tar -xzf "$PREFIX/julia.tar.gz" \
      -C "$PREFIX/julia" --strip-components=1
    rm -f "$PREFIX/julia.tar.gz"
    JULIA="$PREFIX/julia/bin/julia"
  else
    # For a machine that cannot reach the Julia download servers.
    echo "  no julia on the path -- installing one from conda-forge"
    micromamba create -y -q -p "$PREFIX/julia" -c conda-forge julia
    JULIA="$PREFIX/julia/bin/julia"
  fi

  say "BlochSimulators.jl, KomaMRI.jl and CUDA.jl"
  # This resolves against the General registry and downloads a couple of
  # gigabytes, most of it the CUDA toolkit that puts either simulator on a
  # card. On a machine with no driver the CUDA extensions fail to precompile,
  # which is expected and leaves the CPU benchmarks alone.
  JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$PREFIX/juliadepot}" \
    "$JULIA" --project="$ROOT/benchmarks/julia" -e 'using Pkg; Pkg.instantiate()'

  cat > "$PREFIX/julia-env" <<EOF
export JULIA="$JULIA"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$PREFIX/juliadepot}"
EOF
}

write_activate() {
  cat > "$PREFIX/activate" <<EOF
# source this before running the benchmarks
source "$VENV/bin/activate"
[ -f "$PREFIX/julia-env" ] && source "$PREFIX/julia-env"
EOF
  say "done -- source $PREFIX/activate"
}

mkdir -p "$PREFIX"
case "$STEP" in
  all) setup_python; setup_julia; write_activate ;;
  python) setup_python; write_activate ;;
  julia) setup_julia; write_activate ;;
  *) echo "usage: setup.sh [all|python|julia]" >&2; exit 2 ;;
esac
