"""The terms a launch carries, chosen per property.

:mod:`test_features` pins what a tissue asks for; this pins what the kernels
then do with that answer. The gated kernel and the full one must give the same
numbers, and the terms that were left out must be the ones whose derivative is
genuinely zero -- so each case here either compares the two kernels or shows a
gradient the gate would have had to lie about.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence import _epg_triton
from torchsim.sequence._epg_triton import _feature_flags
from torchsim.sequence._parameters import Geometry, at_identity
from torchsim.sequence._parameters import TISSUE_NAMES, TISSUE_PARAMETERS

ECHOES = 8
STATES = 8

WINDING = Geometry(flow_scale=120.0, washout_scale=4.0)
STILL = Geometry()


def _parameter(name: str):
    return TISSUE_PARAMETERS[TISSUE_NAMES.index(name)]


def _description():
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )


# --- what the flags say ---


def test_a_caller_who_declares_nothing_keeps_every_term():
    """The reading every existing entry point takes until it declares."""
    flags = _feature_flags(None, WINDING)
    assert all(flags.values())


def test_each_absent_property_takes_its_term_with_it():
    live = frozenset({"T1", "T2"})
    assert _feature_flags(live, WINDING) == {
        "detuned": False,
        "phased": False,
        "flowing": False,
        "washing": False,
    }


def test_a_moving_voxel_in_a_sequence_that_declares_no_geometry_stands_still():
    """Velocity reaches the state machine only through the two scales, so a
    sequence that winds no phase and draws in no fresh spins has nothing to do
    with it however fast the voxel moves.
    """
    flags = _feature_flags(frozenset({"FLOW"}), STILL)
    assert not flags["flowing"]
    assert not flags["washing"]


def test_the_two_velocity_terms_switch_on_one_at_a_time():
    live = frozenset({"FLOW"})
    assert _feature_flags(live, Geometry(flow_scale=120.0)) == {
        "detuned": False, "phased": False, "flowing": True, "washing": False,
    }
    assert _feature_flags(live, Geometry(washout_scale=4.0)) == {
        "detuned": False, "phased": False, "flowing": False, "washing": True,
    }


# --- the trap forward mode sets ---


def test_a_value_carrying_a_forward_direction_asks_for_its_term():
    """A dual number sits at its primal, and its primal may be the identity.

    Dropping the term there would return zero for a directional derivative
    that is not zero, which is the same lie the reverse-mode guard refuses --
    and a forward tangent is not a ``requires_grad`` flag, so it needs asking
    for separately.
    """
    from torch.autograd.forward_ad import dual_level, make_dual

    parameter = _parameter("b1_phase_rad")
    with dual_level():
        held = make_dual(torch.tensor([0.0]), torch.tensor([1.0]))
        assert float(held.item()) == parameter.identity
        assert not held.requires_grad
        assert not at_identity(parameter, held)


def test_the_transmit_phase_derivative_survives_at_zero_phase():
    """The end of that trap: forward mode through the whole simulation, along
    a property sitting exactly where its term would be dropped.
    """
    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    def signal(phase):
        return FSE().simulate(
            _description(),
            TissueProperties(
                t1_ms=torch.tensor([1000.0]),
                t2_ms=torch.tensor([80.0]),
                b1_phase_rad=phase,
            ),
            nstates=STATES,
        ).signal

    with dual_level():
        held = make_dual(torch.tensor([0.0]), torch.tensor([1.0]))
        derivative = unpack_dual(signal(held)).tangent
    assert derivative is not None

    step = 1e-3
    difference = (
        signal(torch.tensor([step])) - signal(torch.tensor([-step]))
    ) / (2.0 * step)
    assert float(derivative.abs().max()) > 0.0
    assert torch.allclose(derivative, difference, rtol=2e-3, atol=1e-5)


# --- the gated kernel against the full one ---


def _adjoint(**properties):
    leaves = {
        name: torch.tensor([value], requires_grad=True)
        for name, value in (("t1_ms", 1000.0), ("t2_ms", 80.0))
    }
    held = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in properties.items()
    }
    signal = FSE().simulate(
        _description(),
        TissueProperties(
            **{name: value.cuda() for name, value in leaves.items()}, **held
        ),
        nstates=STATES,
    ).signal
    signal.abs().square().sum().backward()
    return tuple(leaves[name].grad.clone() for name in leaves)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "properties",
    [
        {},
        {"b0_hz": torch.tensor([30.0])},
        {"b1_phase_rad": torch.tensor([0.4])},
    ],
    ids=["bare", "off-resonance", "transmit-phase"],
)
def test_the_answer_does_not_depend_on_the_gate(monkeypatch, properties) -> None:
    """Whatever the mask leaves out, the numbers it leaves behind are the ones
    the full kernel gives.
    """
    gated = _adjoint(**properties)
    monkeypatch.setattr(
        _epg_triton,
        "_feature_flags",
        lambda features, geometry: _feature_flags(None, geometry),
    )
    whole = _adjoint(**properties)
    for one, other in zip(gated, whole, strict=True):
        assert torch.allclose(one, other, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_an_off_resonance_asked_for_its_gradient_gets_a_real_one():
    """The gate is a choice, not a coincidence: the term it can drop is one
    that moves the answer when it is there.
    """
    b0 = torch.tensor([30.0], device="cuda", requires_grad=True)
    signal = FSE().simulate(
        _description(),
        TissueProperties(
            t1_ms=torch.tensor([1000.0], device="cuda"),
            t2_ms=torch.tensor([80.0], device="cuda"),
            b0_hz=b0,
        ),
        nstates=STATES,
    ).signal
    signal.abs().square().sum().backward()
    assert float(b0.grad.abs().max()) > 0.0
