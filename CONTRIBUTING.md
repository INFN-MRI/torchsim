# Contributing to BlochSim

The [Developer Guide](https://pulserver.github.io/blochsim/latest/developer-guide/index.html)
documents the toolchain, editable installation, coding and documentation
conventions, pre-commit hooks and pull-request workflow.

A quick development setup is:

```bash
git clone https://github.com/YOUR-USER/blochsim.git
cd blochsim
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
pre-commit install
```

Before opening a pull request:

```bash
pre-commit run --all-files
pytest tests/
bash scripts/build_docs.sh      # for documentation changes
```

Open the pull request against [`pulserver/blochsim:main`](https://github.com/pulserver/blochsim/compare).
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md);
security vulnerabilities are reported as described in [SECURITY.md](SECURITY.md).
