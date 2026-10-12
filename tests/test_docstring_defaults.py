"""Consistency checks for defaults in NumPy-style callable docstrings."""

import ast
import importlib
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCES = [
    *ROOT.joinpath("src/blochsim").rglob("*.py"),
]
HEADER = re.compile(r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*(.+)$")
DEFAULT = re.compile(r"(?:^|,\s*)default\s*=\s*(.+)$")


def _callables(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        target = node
        if isinstance(node, ast.ClassDef):
            target = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and item.name in {"__init__", "init_module", "layout", "_signal"}
                ),
                None,
            )
        if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=True)
        if doc and (
            "Parameters\n----------" in doc
            or "Other Parameters\n----------------" in doc
        ):
            yield node, target, doc


def _documented_parameters(doc):
    lines = doc.splitlines()
    out = {}
    section = None
    for index, line in enumerate(lines):
        underline = lines[index + 1] if index + 1 < len(lines) else ""
        if underline and set(underline) == {"-"} and len(underline) >= len(line):
            section = line.strip()
            continue
        if section not in {"Parameters", "Other Parameters"} or set(line) == {"-"}:
            continue
        match = HEADER.match(line)
        if match:
            for name in map(str.strip, match.group(1).split(",")):
                out[name] = match.group(2)
    return out


def _signature_defaults(node):
    positional = node.args.posonlyargs + node.args.args
    args = positional + node.args.kwonlyargs
    defaults = (
        [None] * (len(positional) - len(node.args.defaults))
        + list(node.args.defaults)
        + list(node.args.kw_defaults)
    )
    return {
        arg.arg: None if default is None else ast.unparse(default)
        for arg, default in zip(args, defaults, strict=True)
    }


def _module(path):
    relative = path.relative_to(ROOT / "src").with_suffix("")
    return importlib.import_module(".".join(relative.parts).removesuffix(".__init__"))


def _value(text, namespace):
    try:
        return eval(text, {"pi": math.pi, **vars(namespace)})  # noqa: S307
    except Exception:
        return text


def _same(stated, expected, module):
    """Whether a documented default states the signature's, by value."""
    if stated == expected:
        return True
    actual = _value(expected, module)
    if not isinstance(actual, (type(None), bool, int, float, str, tuple)):
        # A private sentinel: the docstring states the default it stands for.
        return stated is not None
    return stated is not None and _value(stated, module) == actual


CASES = [
    (path, owner, function, doc)
    for path in SOURCES
    for owner, function, doc in _callables(path)
]


@pytest.mark.parametrize(
    "path,owner,function,doc",
    CASES,
    ids=lambda value: str(value) if isinstance(value, Path) else None,
)
def test_documented_defaults_match_python_signatures(path, owner, function, doc):
    """Documented callable parameters state exactly the Python signature default."""
    documented = _documented_parameters(doc)
    defaults = _signature_defaults(function)
    errors = []
    for name, type_spec in documented.items():
        if name not in defaults or name in {"self", "cls"}:
            continue
        match = DEFAULT.search(type_spec)
        stated = match.group(1).strip() if match else None
        expected = defaults[name]
        if expected is None and stated is not None:
            errors.append(f"{name} is required but documents default={stated}")
        elif expected is not None and not _same(stated, expected, _module(path)):
            errors.append(
                f"{name} documents {stated!r}; signature default is {expected!r}"
            )
        if "optional" in type_spec:
            errors.append(f"{name} uses 'optional' instead of an explicit default")
    assert not errors, f"{path}:{owner.lineno}: " + "; ".join(errors)
