"""Which terms of the state machine a tissue asks for.

A property left at the value where it has no effect is a property the kernels
need not carry: not its arithmetic, not its buffer, not a row of its gradient.
What decides is what the caller passed, read before broadcasting -- so these
pin the reading rather than the saving, because a wrong reading is a wrong
answer while a missed saving is only a slow one.
"""

from __future__ import annotations

import pytest
import torch

from torchsim.sequence._parameters import (
    TISSUE_NAMES,
    TISSUE_PARAMETERS,
    at_identity,
    features_of,
    wants_bound_pool,
    wants_exchange_pool,
)
from torchsim.sequence._simulation import TissueProperties


def _parameter(name: str):
    return TISSUE_PARAMETERS[TISSUE_NAMES.index(name)]


def _tissue(**overrides) -> TissueProperties:
    return TissueProperties(t1_ms=1000.0, t2_ms=80.0, **overrides)


# --- the floor ---


def test_relaxation_is_not_a_feature_because_it_has_no_identity():
    """There is no value of T1 at which the kernels can skip relaxation, which
    is what makes it the floor every run stands on rather than a term.
    """
    for name in ("t1_ms", "t2_ms"):
        assert _parameter(name).identity is None
        assert not at_identity(_parameter(name), 0.0)
    assert {"T1", "T2"} <= features_of(_tissue())


def test_a_bare_relaxation_tissue_asks_for_nothing_else():
    """The case the whole specialization exists for."""
    assert features_of(_tissue()) == {"T1", "T2"}


# --- what turns a term on ---


@pytest.mark.parametrize(
    ("name", "value", "feature"),
    [
        ("b0_hz", 30.0, "B0"),
        ("diffusion_um2_per_ms", 3.0, "DIFFUSION"),
        ("velocity_m_per_s", 0.1, "FLOW"),
        ("b1", 0.8, "B1"),
        ("b1_phase_rad", 0.3, "B1_PHASE"),
        ("inversion_efficiency", 0.95, "INVERSION"),
        ("m0", 0.5, "M0"),
        ("bound_fraction", 0.1, "MT"),
        ("pool_b_fraction", 0.1, "BM"),
    ],
)
def test_a_property_away_from_its_identity_asks_for_its_term(
    name: str, value: float, feature: str
) -> None:
    assert feature not in features_of(_tissue())
    assert feature in features_of(_tissue(**{name: value}))


def test_a_map_asks_for_its_term_whatever_it_holds():
    """A full tensor is taken to matter rather than reduced over: the reduction
    costs a synchronization on a device, and a caller who built a map of zeros
    meant to vary it.
    """
    zeros = torch.zeros(8)
    assert "B0" in features_of(_tissue(b0_hz=zeros))
    assert "DIFFUSION" in features_of(_tissue(diffusion_um2_per_ms=zeros))


def test_every_feature_has_exactly_one_gate():
    """The rule the mask reads by. A feature with none would never switch on;
    one with two would disagree with itself.
    """
    from collections import Counter

    gates = Counter(
        parameter.feature for parameter in TISSUE_PARAMETERS if parameter.gate
    )
    features = {
        parameter.feature
        for parameter in TISSUE_PARAMETERS
        if parameter.feature is not None
    }
    assert set(gates) == features
    assert all(count == 1 for count in gates.values())


def test_a_term_described_but_not_switched_on_stays_out():
    """A bound pool's exchange rate says what the pool does; its fraction says
    whether there is one. Setting the first without the second must leave the
    single-pool run alone -- which is the reading the whole second pool already
    depends on.
    """
    tissue = _tissue(bound_exchange_hz=25.0, t1_bound_ms=400.0)
    assert "MT" not in features_of(tissue)


def test_the_pool_gates_are_the_mask_read_two_ways():
    """The two predicates that already existed have to keep agreeing with the
    mask that generalizes them, or a run would carry one pool and index the
    other.
    """
    for fraction in (0.0, 0.1, torch.zeros(4)):
        tissue = _tissue(bound_fraction=fraction, pool_b_fraction=fraction)
        assert wants_bound_pool(fraction) == ("MT" in features_of(tissue))
        assert wants_exchange_pool(fraction) == ("BM" in features_of(tissue))


# --- the trap ---


@pytest.mark.parametrize(
    ("name", "feature"),
    [
        ("b0_hz", "B0"),
        ("diffusion_um2_per_ms", "DIFFUSION"),
        ("velocity_m_per_s", "FLOW"),
        ("b1", "B1"),
        ("m0", "M0"),
        ("bound_fraction", "MT"),
    ],
)
def test_a_value_that_carries_a_gradient_asks_for_its_term(
    name: str, feature: str
) -> None:
    """Sitting at the identity is not the same as having no derivative there.

    A B1 map fitted from unity and a diffusion coefficient fitted from zero
    both start at their identity, and both have a gradient there. Leaving the
    term out would return zero for it -- the truth for a property the run does
    not take, and a lie for one it differentiates.
    """
    identity = _parameter(name).identity
    assert identity is not None
    held = torch.tensor(identity, requires_grad=True)

    assert feature not in features_of(_tissue(**{name: identity}))
    assert feature in features_of(_tissue(**{name: held}))


def test_a_python_scalar_cannot_carry_a_gradient():
    """Which is what makes the guard above complete: the only values that reach
    the identity check and could still be differentiated are tensors, and those
    are the ones it asks.
    """
    assert not hasattr(0.0, "requires_grad")
    assert not torch.as_tensor(0.0).requires_grad
