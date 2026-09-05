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

An example is executed when the interpreter can import everything it imports,
and rendered from its source when it cannot: an environment with the `dev` or
`examples` extra runs the whole gallery, and one with `doc` alone runs what
needs nothing but TorchSim. The build says which ones it did not run, and what
each was missing. To build the pages without running any of them, pass
`-D plot_gallery=0`.

Read the Docs builds the published pages from `.readthedocs.yaml`, one version
per release. It installs the `doc` extra, so the notebooks that fetch a
phantom or a subject are published as source; adding `examples` back there is
what publishes their output too.

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
