"""The Jacobian shape contract must not drift.

A vectorized engine takes a forward-mode shortcut instead of ``vmap``; both
routes have to expose exactly the same shapes to callers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsim
from torchsim.base import AbstractModel, autocast

FLIP = np.ones(7) * 120.0
ECHOES = 7


def _fse(t1, t2, diff):
    return torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=t1, T2=t2, diff=diff)


VECTOR_T1 = torch.tensor([800.0, 1000.0, 1200.0])
VECTOR_T2 = torch.tensor([50.0, 70.0, 90.0])


@pytest.mark.parametrize(
    ("t1", "t2", "diff", "signal_shape", "jacobian_shape"),
    [
        (1000.0, 80.0, "T2", (ECHOES,), (ECHOES,)),
        (VECTOR_T1, VECTOR_T2, "T2", (3, ECHOES), (3, ECHOES)),
        (1000.0, 80.0, ("T1", "T2"), (ECHOES,), (2, ECHOES)),
        (VECTOR_T1, VECTOR_T2, ("T1", "T2"), (3, ECHOES), (3, 2, ECHOES)),
        (VECTOR_T1, VECTOR_T2, ("T1", "T2", "B1"), (3, ECHOES), (3, 3, ECHOES)),
    ],
)
def test_jacobian_shapes(t1, t2, diff, signal_shape, jacobian_shape) -> None:
    """A bare string ``diff`` collapses the parameter axis; a tuple keeps it."""
    signal, jacobian = _fse(t1, t2, diff)
    assert signal.shape == signal_shape
    assert jacobian.shape == jacobian_shape


def test_jacobian_matches_finite_differences() -> None:
    step = 1e-2
    _, jacobian = _fse(VECTOR_T1, VECTOR_T2, "T2")
    forward = torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=VECTOR_T1, T2=VECTOR_T2 + step)
    backward = torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=VECTOR_T1, T2=VECTOR_T2 - step)
    numerical = (forward - backward) / (2.0 * step)
    assert torch.allclose(jacobian, numerical, atol=1e-4)


def test_signal_matches_undifferentiated_call() -> None:
    """The primal returned alongside the Jacobian is the ordinary signal."""
    signal, _ = _fse(VECTOR_T1, VECTOR_T2, "T2")
    plain = torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=VECTOR_T1, T2=VECTOR_T2)
    assert torch.equal(signal, plain)


class _SingleVoxelModel(AbstractModel):
    """A model written the ergonomic way: one voxel, no batching awareness."""

    @autocast
    def set_properties(self, decay):
        self.properties.decay = decay

    @autocast
    def set_sequence(self, times):
        self.sequence.times = times

    @staticmethod
    def _engine(decay, times):
        return torch.exp(-times / decay)


def test_single_voxel_engine_still_supported() -> None:
    """The vmap route stays available for engines that are not vectorized."""
    model = _SingleVoxelModel(diff="decay")
    model.set_properties(decay=torch.tensor([50.0, 100.0, 200.0]))
    model.set_sequence(times=torch.linspace(0.0, 100.0, ECHOES))
    signal, jacobian = model()

    assert signal.shape == (3, ECHOES)
    assert jacobian.shape == (3, ECHOES)
    times = torch.linspace(0.0, 100.0, ECHOES)
    decay = torch.tensor([50.0, 100.0, 200.0])[:, None]
    expected = times / decay.square() * torch.exp(-times / decay)
    assert torch.allclose(jacobian, expected, atol=1e-4)
