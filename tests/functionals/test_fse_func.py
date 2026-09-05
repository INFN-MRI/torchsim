"""FSE tests."""

import numpy as np
from pytest import fixture

from torchsim import (
    fse_sim,
)


@fixture
def flip():
    return np.linspace(5.0, 60.0, 100, dtype=np.float32)


def test_scalar_forward(flip):
    sig = fse_sim(flip, ESP=1.0, T1=1000.0, T2=100.0)
    assert sig.shape == (100,)


def test_multiple_forward(flip):
    sig = fse_sim(flip, ESP=1.0, T1=(200, 500, 1000.0), T2=100.0)
    assert sig.shape == (3, 100)


def test_scalar_derivative(flip):
    _, dsig = fse_sim(flip, ESP=1.0, T1=1000.0, T2=100.0, diff="T1")
    assert dsig.shape == (100,)


def test_multiple_derivative(flip):
    _, dsig = fse_sim(flip, ESP=1.0, T1=(200, 500, 1000.0), T2=100.0, diff="T1")
    assert dsig.shape == (3, 100)


def test_scalar_gradient(flip):
    _, grad = fse_sim(flip, ESP=1.0, T1=1000.0, T2=100.0, diff=("T1", "T2"))
    assert grad.shape == (2, 100)


def test_multiple_gradient(flip):
    _, grad = fse_sim(flip, ESP=1.0, T1=(200, 500, 1000.0), T2=100.0, diff=("T1", "T2"))
    assert grad.shape == (3, 2, 100)
