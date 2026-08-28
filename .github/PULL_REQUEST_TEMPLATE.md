## What this changes

<!-- What the code does now that it did not do before, in one or two
     sentences. Link the issue or discussion it comes from, if there is one. -->

## How it was checked

<!-- The command you ran and what it printed: a test that fails before the
     change and passes after it, an analytic case the new physics reproduces,
     a benchmark. "Tests pass" on its own does not say which ones. -->

```
pytest tests/
```

## Checklist

- [ ] `pytest tests/` passes locally, and a new test covers what changed.
- [ ] `black .` and `ruff check src` are clean.
- [ ] Public functions and classes carry a numpydoc docstring: one line of
      what, then Parameters, Returns, Raises.
- [ ] Comments and docstrings describe the code as it is now -- no
      "previously", no "this replaces", no naming of a bug that is fixed.
- [ ] The documentation is updated where the change is visible to a caller,
      and `bash scripts/build_docs.sh` still builds.
