"""Which sequences the fused kernels are allowed to answer for.

An event stream reaches the kernels through ``_pack_events``, which carries
timing, flip angle, phase and a per-event action word. The action word is the
one part not read off the description: it is looked up from the policy's name,
so a policy the lookup does not know would be packed with no crushers and no
spoiling, and would run to a different answer than the operator loop.
"""

import pytest
import torch

from torchsim import TissueProperties, epg, fse_description
from torchsim.sequence._accelerators import _SHIFTS_AND_SPOILS
from torchsim.sequence._description import RfUse
from torchsim.sequence._simulation import EpgSimulator, make_simulator

ECHOES = 8
STATES = 10


def _describe():
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )


def _tissue():
    return TissueProperties(
        t1_ms=torch.tensor([800.0, 1400.0]), t2_ms=torch.tensor([45.0, 120.0])
    )


class _Unknown(EpgSimulator):
    """A sequence of a user's own, crushing around every refocusing pulse."""

    name = "a-policy-of-my-own"

    def before_rf(self, states, event):
        if event.rf_use is RfUse.REFOCUSING:
            return epg.shift(states)
        return states


@pytest.mark.parametrize("name", sorted(_SHIFTS_AND_SPOILS - {"base"}))
def test_every_shipped_policy_reaches_the_kernels(name):
    """A description packs and runs, whichever of these built it."""
    simulator = make_simulator(name)
    loop = simulator.simulate(
        _describe(), _tissue(), backend="torch", nstates=STATES
    ).signal
    fused = simulator.simulate(
        _describe(), _tissue(), backend="native", nstates=STATES
    ).signal

    scale = loop.abs().max()
    assert scale > 0.0
    assert ((loop - fused).abs().max() / scale) < 1e-5


def test_a_policy_the_packing_does_not_know_is_refused():
    """Asked for outright, it must say so rather than answer differently."""
    with pytest.raises(RuntimeError, match="native EPG backend is unavailable"):
        _Unknown().simulate(
            _describe(), _tissue(), backend="native", nstates=STATES
        )


def test_a_policy_the_packing_does_not_know_still_simulates():
    """Left to choose, it takes the path that has the sequence's own state rules."""
    simulator = _Unknown()
    automatic = simulator.simulate(
        _describe(), _tissue(), backend="auto", nstates=STATES
    ).signal
    loop = simulator.simulate(
        _describe(), _tissue(), backend="torch", nstates=STATES
    ).signal

    assert torch.equal(automatic, loop)
