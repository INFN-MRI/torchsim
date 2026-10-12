---
name: build-the-docs
description: Build the BlochSim documentation, add a figure to an explanation page, or add a gallery example. Use when asked about docs, Sphinx, the gallery, MyST, or Read the Docs for blochsim.
---

# Build the BlochSim documentation

```sh
bash scripts/build_docs.sh              # incremental
bash scripts/build_docs.sh --clean      # re-execute every example
PYTHON_BIN=~/envs/blochsim/bin/python bash scripts/build_docs.sh
python -m http.server --directory docs/build/html 8000
```

The script checks that the interpreter it is given can import BlochSim and the
documentation extensions, then builds into `docs/build/html`. Two things make
it slower than a plain Sphinx run and are the point of it: **sphinx-gallery
executes the examples**, and **the explanation pages' figures are re-rendered**
by `docs/explanation_figures.py` with the BlochSim in your working tree.

An example is executed when the interpreter can import everything it imports:
an environment with the `dev` or `examples` extra runs the whole gallery, one
with `doc` alone runs what needs nothing but BlochSim. A page already carrying
output executed from the source as it stands is reused rather than re-run, and
the build names each example it did not run and where its output came from. To
build the pages without running any of them, pass `-D plot_gallery=0`.

## Where the published pages are built

`.github/workflows/docs.yml` builds them, and GitHub Pages serves them from
the `gh-pages` branch at <https://pulserver.github.io/blochsim/>.

Its **HTML** job runs on every branch and pull request with the `doc` extra:
every page is written and the examples needing nothing but BlochSim are
executed, in a few minutes, and the tree is uploaded as an artifact to read.

Its **Site** job runs on `main`, on a `v*.*.*` tag and on demand. It installs
the `examples` extra, so every notebook is executed, and hands what
Sphinx wrote to `scripts/publish_docs.py`, which places it in the site:

    site/latest/     the development branch
    site/v1.2.3/     one directory per release, kept as it was published
    site/index.html  a page sending a reader to the newest release
    site/versions.json  the list the switcher in the sidebar reads

`BLOCHSIM_DOCS_VERSION` is what a build calls itself; it names the directory,
the switcher's mark and the Binder path prefix, and defaults to `latest`.

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
architecture. Describe what BlochSim does and why that is right on its own
terms. These pages are not a changelog — never justify a design by describing
the design it replaced.


## Documentation architecture

Keep the public sidebar shallow and in this order:

1. User Guide
2. Developer Guide
3. Explanations
4. Examples
5. API
6. Miscellaneous

The first four Framework examples are the Course. They teach the public
abstraction in order; the remaining example sections are standalone Tours.

For a new sequence, teach {class}`blochsim.Simulator` as the extension
point. A state-machine simulator has two complementary inputs: its class-level
handlers interpret commands from an incoming Pulseq/MRD description, and
`layout()` constructs the same sequence offline. Lower-level
`SequenceDescription`, the private `SpinPhysics` and operator machinery are
reference/developer concepts, not competing user-facing sequence APIs.

Functional `*_sim` calls are conveniences around shipped simulators. Do not
present them as the abstraction a user implements.

Figures must remain readable at final documentation width in both light and
dark mode. Use `docs/figure_style.py` for common typography, transparent
backgrounds and foreground colors; examples set only figure-specific geometry
and scientific encodings.
