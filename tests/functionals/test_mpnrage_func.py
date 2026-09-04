"""MPnRAGE tests."""

from torchsim import (
    mpnrage_sim,
)


def test_scalar_forward():
    sig = mpnrage_sim(nshots=100, flip=5.0, TR=10.0, T1=1000.0)
    assert sig.shape == (100,)


def test_multiple_forward():
    sig = mpnrage_sim(nshots=100, flip=5.0, TR=10.0, T1=(200, 500, 1000.0))
    assert sig.shape == (3, 100)


def test_scalar_derivative():
    _, dsig = mpnrage_sim(nshots=100, flip=5.0, TR=10.0, T1=1000.0, diff="T1")
    assert dsig.shape == (100,)


def test_multiple_derivative():
    _, dsig = mpnrage_sim(
        nshots=100, flip=5.0, TR=10.0, T1=(200, 500, 1000.0), diff="T1"
    )
    assert dsig.shape == (3, 100)
