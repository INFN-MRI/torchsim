# Documentation

How the pages under `docs/` and the gallery under `examples/` are written and
built. The build itself is in {doc}`installation`.

```{contents}
:local:
:depth: 2
```

## Sources

The pages under `docs/` are MyST Markdown, built by Sphinx. The autosummary
templates in `docs/_templates/autosummary/*.rst` stay reStructuredText:
`sphinx.ext.autosummary` writes its stubs with a hardcoded `.rst` suffix and
finds its directives by regex over raw source lines.

## Gallery headers

A gallery header is Markdown behind a one-line shim. sphinx-gallery pastes the
header verbatim into a generated `index.rst`, so a `README.md` handed to it
directly is parsed as reStructuredText: the build succeeds with no warning and
the page loses its title, inherits the first subsection's, and prints MyST
labels as literal text. Each section therefore has a `README.rst` holding
nothing but

```rst
.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_
```

with the prose in `_gallery_header.md` beside it. Two settings in
`docs/conf.py` make that work: `copyfile_regex` in `sphinx_gallery_conf`
carries the `.md` into the output directory so the include resolves, and
`exclude_patterns` keeps Sphinx from also building it as a page of its own,
which would define every label in it twice.

## Gallery prose

`examples/01-framework/01-first-simulation.py` and `03-custom-signal-model.py`
set the register. A Course lesson has a numbered title, an opening that states
what the lesson does, a **Learning objectives** list and a Previous/Next line;
then sections each named by the concept it covers, giving the physics or the
model first and the code after it, written to "you". The last section,
*Specification*, is a `text` code block stating what the page did. An
Application opens with its objective and a *Prerequisites* line naming the
lessons it assumes, and has no Previous/Next.

Every example has one scope. A dictionary match is the subject of
`02-parameter-inference/01`; an example that needs one as a baseline fits it in
a hidden cell and gives it a row in a table, not a section.

`sphinx_gallery_start_ignore` is for figure code and print formatting. A cell a
reader would type -- `execution="cpu"`, a `description(...)` built by
hand -- is visible even when its output is not
interesting.

## Figures

Figures read in both themes. Colours come from `docs/figure_style.py`
(`SERIES`, `MUTED`), never a fixed grey, black or white, and every legend sits
outside the axes with `legend_outside`. Check each figure on a light and a dark
background before committing.

The figures on the explanation pages are simulated at build time by
`docs/explanation_figures.py`, so a figure cannot outlive the behaviour it
shows. To add one, write a function that returns a Matplotlib figure, register
it in the `FIGURES` mapping at the bottom, and reference the file it writes
with a `{figure}` directive.

## Examples

To add a gallery example, add a script to its `examples/` section: the numeric
prefix orders it and the module docstring becomes the page. Then add a row for
it to {doc}`../examples/index` and to {doc}`../examples/course` or
{doc}`../examples/applications`, whose hidden toctrees place it in the
navigation.

An example is executed by the gallery when the interpreter can import
everything it imports, so what a build runs follows from the extras installed
into it, and a page already carrying output executed from the source as it
stands is reused rather than re-run.

## Publication

`.github/workflows/docs.yml` builds the published pages and GitHub Pages
serves them from `gh-pages`. Its HTML job checks every branch with the `doc`
extra; its Site job executes every example with the `examples` extra and hands
the tree to `scripts/publish_docs.py`, which keeps one directory per version --
`latest` for `main`, its own for each `v*.*.*` tag -- beside the list the
version switcher reads.
