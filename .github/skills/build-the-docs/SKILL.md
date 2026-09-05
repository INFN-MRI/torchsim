---
name: build-the-docs
description: Build the TorchSim documentation, add a figure to an explanation page, or add a gallery example. Use when asked about docs, Sphinx, the gallery, MyST, or Read the Docs for torchsim.
---

# Build the TorchSim documentation

```sh
bash scripts/build_docs.sh              # incremental
bash scripts/build_docs.sh --clean      # re-execute every example
PYTHON_BIN=~/envs/torchsim/bin/python bash scripts/build_docs.sh
python -m http.server --directory docs/build/html 8000
```

The script checks that the interpreter it is given can import TorchSim and the
documentation extensions, then builds into `docs/build/html`. Two things make
it slower than a plain Sphinx run and are the point of it: **sphinx-gallery
executes the examples**, and **the explanation pages' figures are re-rendered**
by `docs/explanation_figures.py` with the TorchSim in your working tree.

An example is executed when the interpreter can import everything it imports:
an environment with the `dev` or `examples` extra runs the whole gallery, one
with `doc` alone runs what needs nothing but TorchSim. A page already carrying
output executed from the source as it stands is reused rather than re-run, and
the build names each example it did not run and where its output came from. To
build the pages without running any of them, pass `-D plot_gallery=0`.

## Where the published gallery is executed

Read the Docs builds the published pages from `.readthedocs.yaml`, one version
per release, with the `doc` extra and Sphinx alone: a subject to download, a
segmentation network and a spiral encoding are more than that builder has.

The Gallery job of `.github/workflows/docs.yml` is where the thirteen examples
run. It installs the `examples` extra on a GitHub runner, builds the docs, and
force-pushes `docs/generated` to the `docs-gallery` branch as a single commit.
Read the Docs restores that branch before Sphinx starts, so every page is
published with the figures and printed output it was executed with.

The job runs on `main`, on a `v*.*.*` tag and on demand. An example whose
source has moved on since the branch was written is published without its
output and named in the build log with a warning — push to `main`, or run the
workflow by hand, to refresh it.

## Markdown, and the two places it is not

Pages are **MyST Markdown**. Two file sets stay reStructuredText because the
tooling requires it — converting either breaks the build:

- `examples/**/README.rst` — sphinx-gallery concatenates a gallery header
  verbatim into a generated `index.rst`.
- `docs/_templates/autosummary/*.rst` — `sphinx.ext.autosummary` writes stubs
  with a hardcoded `.rst` suffix and finds its directives by regex over raw
  source lines.

That regex is also why the API pages hold `autosummary` and `currentmodule`
inside `` ```{eval-rst} `` blocks rather than in MyST directive syntax: written
as `` ```{autosummary} `` the stub generator never sees them and every linked
page 404s.

## Add a figure to an explanation page

Write a function in `docs/explanation_figures.py` that returns a Matplotlib
figure, register it in the `FIGURES` mapping at the bottom, and reference the
file it writes:

````markdown
```{figure} /generated/figures/your_figure.png
:width: 100%
:alt: One sentence a screen reader can use.

The caption, which is where the reader is told what to look at.
```
````

Figures are simulated at build time rather than checked in, so one cannot
outlive the behaviour it shows.

## Add a gallery example

Drop a script into the right `examples/` section. The numeric prefix orders it
within the section, and the module docstring — an reStructuredText title block
— becomes the page's introduction. What it imports decides where it is
executed and where it is published as source.

## Writing

The audience is MR scientists: pulse sequences and physics, not software
architecture. Describe what TorchSim does and why that is right on its own
terms. These pages are not a changelog — never justify a design by describing
the design it replaced.
