"""Setuptools entry point for TorchSim."""

from __future__ import annotations

import os

from setuptools import Extension, setup


def _compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17"]


setup(
    ext_modules=[
        Extension(
            "torchsim._epg_cpu",
            ["src/torchsim/_epg_cpu.cpp"],
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            extra_compile_args=_compile_args(),
            language="c++",
            py_limited_api=True,
        ),
        Extension(
            "torchsim._perk_cpu",
            ["src/torchsim/_perk_cpu.cpp"],
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            extra_compile_args=_compile_args(),
            language="c++",
            py_limited_api=True,
        ),
    ],
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
