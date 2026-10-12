# Style

`ruff format` formats and ruff's `I` rules sort imports. `ruff check` also
runs pycodestyle, pyflakes, bugbear, pydocstyle (numpydoc convention),
pyupgrade and annotation rules. `pyproject.toml` holds the whole
configuration.

## Docstrings and comments

- **Docstrings** are numpydoc: a one-line summary, then Parameters, Returns,
  Raises. Each parameter is one or two lines: what it is, its unit, a
  non-obvious constraint. Types go in the docstring, where they can be
  qualified, because the API pages are built with typehints off.
- **Write for the code as it is now.** No "used to", "previously" or "this
  replaces"; no fixed bugs; no restating what the code says.
- **A comment explains a non-obvious algorithm or a choice a reader would
  undo.** Prefer a well-named function, or a test whose name states the
  invariant.
- **Units are public at the edges**, as {ref}`dev-units` states.
