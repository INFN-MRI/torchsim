"""MPRAGE tests."""

from torchsim import mprage_sim


def test_scalar_forward():
    sig = mprage_sim(
        nshots=100,
        TI=600.0,
        flip=5.0,
        TRspgr=10.0,
        T1=1000.0,
    )
    assert sig.shape == ()
    assert sig.abs() > 0


def test_multiple_forward():
    sig = mprage_sim(
        nshots=100,
        TI=600.0,
        flip=5.0,
        TRspgr=10.0,
        T1=(200, 500, 1000.0),
    )
    assert sig.shape == (3,)
    assert sig.abs().min() > 0


def test_multiple_derivative():
    _, derivative = mprage_sim(
        nshots=100,
        TI=600.0,
        flip=5.0,
        TRspgr=10.0,
        T1=(500.0, 1000.0),
        diff="T1",
    )
    assert derivative.shape == (2,)
