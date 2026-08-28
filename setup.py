"""Shim for tools that still invoke ``setup.py`` directly.

The build is declared in ``pyproject.toml`` and carried out by
scikit-build-core; nothing is configured here. Prefer ``pip install .`` or
``python -m build``.
"""

from setuptools import setup

setup()
