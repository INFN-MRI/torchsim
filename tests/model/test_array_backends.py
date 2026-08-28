"""A caller's array library is the caller's, coming in and going back.

Everything inside is torch. What these hold is that the boundary reads a
caller's arrays over the same memory rather than copying them, that the answer
comes back in the library it was asked in, and that going round the loop
changes no number.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsim
from torchsim.sequence._array import as_torch, backend_of, brought, like, matched
from torchsim.simulators import FSESimulator, MRFSimulator, SPGRSimulator

cupy = pytest.importorskip("cupy", reason="no second device backend installed")

FLIP = np.linspace(5.0, 60.0, 12, dtype=np.float32)
T1 = np.array([600.0, 1000.0, 1400.0], dtype=np.float32)
T2 = np.array([40.0, 80.0, 120.0], dtype=np.float32)


def test_reading_a_numpy_array_shares_its_memory() -> None:
    """Zero-overhead means zero copies, which is checkable by writing to it."""
    held = np.arange(6, dtype=np.float32)
    read = as_torch(held)
    read[0] = 42.0
    assert held[0] == 42.0


def test_reading_a_device_array_shares_its_memory() -> None:
    """The same door, without a trip through the host."""
    held = cupy.arange(6, dtype=cupy.float32)
    read = as_torch(held)
    assert read.device.type == "cuda"
    read[0] = 42.0
    assert bool(held[0] == 42.0)


def test_a_read_only_array_is_still_read() -> None:
    """A buffer torch will not share is copied rather than refused."""
    held = np.arange(3, dtype=np.float32)
    held.flags.writeable = False
    assert as_torch(held).tolist() == [0.0, 1.0, 2.0]


def test_a_torch_tensor_is_left_alone() -> None:
    """Nothing is wrapped, copied or moved for a caller already in torch."""
    held = torch.arange(3, dtype=torch.float32)
    assert as_torch(held) is held
    assert backend_of(held) is None
    assert like(held, None) is held


@pytest.mark.parametrize("namespace", [np, cupy], ids=["numpy", "cupy"])
def test_the_signal_comes_back_in_the_library_it_was_asked_in(namespace) -> None:
    """The whole point, on a run that reaches the fused kernels."""
    sequence = MRFSimulator(flip=namespace.asarray(FLIP), TR=10.0)
    signal = sequence.simulate(T1=namespace.asarray(T1), T2=namespace.asarray(T2))
    assert isinstance(signal, namespace.ndarray)
    assert signal.shape == (3, 12)


@pytest.mark.parametrize("namespace", [np, cupy], ids=["numpy", "cupy"])
def test_the_round_trip_changes_no_number(namespace) -> None:
    """A converted call and a torch call are the same simulation.

    The torch arm is put on the device the converted one lands on: a CuPy
    array is device memory, so its run is a CUDA run, and holding it against a
    host run would be comparing two kernels rather than two spellings.
    """
    through = MRFSimulator(flip=namespace.asarray(FLIP), TR=10.0).simulate(
        T1=namespace.asarray(T1), T2=namespace.asarray(T2)
    )
    where = as_torch(through).device
    direct = MRFSimulator(flip=torch.from_numpy(FLIP).to(where), TR=10.0).simulate(
        T1=torch.from_numpy(T1).to(where), T2=torch.from_numpy(T2).to(where)
    )
    assert torch.equal(as_torch(through), direct)


def test_a_closed_form_travels_the_same_way() -> None:
    """Not only the state machine: the analytic sequences take arrays too."""
    signal = SPGRSimulator(flip=FLIP, TR=10.0, TE=2.0).simulate(
        T1=T1, T2star=np.float32(100.0)
    )
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (3, 12)


def test_the_jacobian_comes_back_too() -> None:
    """Forward mode is taken in torch and handed back in the caller's own."""
    signal, jacobian = FSESimulator(
        flip=np.full(8, 180.0, dtype=np.float32), ESP=5.0, TR=3000.0
    ).jacobian(("T1", "T2"), T1=T1, T2=T2)
    assert isinstance(signal, np.ndarray)
    assert isinstance(jacobian, np.ndarray)
    assert jacobian.shape == (3, 2, 8)


def test_the_first_array_decides_and_torch_stops_the_search() -> None:
    """A torch caller is not handed NumPy by a later argument.

    The tissue is torch and the schedule is NumPy here; scanning past the
    tensor for something with a namespace would answer with the schedule's.
    """
    assert brought([torch.zeros(3), FLIP]) is None
    assert brought([FLIP, torch.zeros(3)]) is np
    assert brought([1.0, "all", None]) is None

    signal = MRFSimulator(flip=FLIP, TR=10.0).simulate(
        T1=torch.from_numpy(T1), T2=torch.from_numpy(T2)
    )
    assert isinstance(signal, torch.Tensor)


def test_a_shared_parameter_is_spread_over_the_events() -> None:
    """One value for the whole train, or one apiece, arrive the same shape."""
    angles = torch.zeros(5)
    assert matched(3.0, angles).shape == (5,)
    assert matched(np.arange(5, dtype=np.float32), angles).shape == (5,)
    with pytest.raises(ValueError, match="one value per event"):
        matched(np.arange(4, dtype=np.float32), angles)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: torchsim.spgr_sim(
            5.0, TE=2.0, TR=10.0, T1=1000.0, T2star=100.0, device=d
        ),
        lambda d: torchsim.bssfp_sim(
            5.0, TE=2.0, TR=10.0, T1=1000.0, T2=100.0, device=d
        ),
        lambda d: torchsim.mp2rage_sim(
            TI=(500.0, 1500.0),
            flip=5.0,
            TRspgr=5.0,
            TRmp2rage=3000.0,
            nshots=128,
            T1=1000.0,
            device=d,
        ),
        lambda d: torchsim.mrf_sim(FLIP, TR=10.0, T1=1000.0, T2=80.0, device=d),
        lambda d: torchsim.mprage_sim(
            TI=500.0, flip=5.0, TRspgr=5.0, nshots=128, T1=1000.0, device=d
        ),
        lambda d: torchsim.mpnrage_sim(
            nshots=8, flip=5.0, TR=10.0, T1=1000.0, device=d
        ),
        lambda d: torchsim.fse_sim(
            flip=np.full(4, 120.0, dtype=np.float32),
            ESP=5.0,
            T1=1000.0,
            T2=80.0,
            device=d,
        ),
    ],
    ids=["spgr", "bssfp", "mp2rage", "mrf", "mprage", "mpnrage", "fse"],
)
def test_a_wrapper_takes_a_device_without_it_reaching_the_sequence(call) -> None:
    """``device`` says where to run, not what to play.

    A closed form has no layout to hand it to and a state machine hands it to
    the engine, so neither may see it among its protocol arguments.
    """
    assert call("cpu") is not None


def test_reverse_mode_still_needs_torch() -> None:
    """The one thing the round trip cannot carry, stated rather than implied.

    A gradient belongs to the tensor it was taken with respect to, so a cost
    differentiated with ``backward()`` is built on torch inputs. Forward mode
    is unaffected, which the Jacobian case above shows.
    """
    flip = torch.full((8,), 120.0, requires_grad=True)
    signal = FSESimulator(flip=flip, ESP=5.0, TR=3000.0).simulate(
        T1=torch.from_numpy(T1), T2=torch.from_numpy(T2)
    )
    assert isinstance(signal, torch.Tensor)
    signal.abs().square().sum().backward()
    assert float(flip.grad.abs().max()) > 0.0
