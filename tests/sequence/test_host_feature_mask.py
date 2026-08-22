"""The mask the host kernels read, against the same launch carrying everything.

The Triton backend takes a flag per term because each one compiles a kernel of
its own; the host kernels take one integer and branch on it at run time. These
pin the two ends together -- that the mask says what
:func:`torchsim.sequence._parameters.feature_flags` says, and that a host
kernel told to drop a term gives the answer it gives when told to keep it.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import FSE, TissueProperties, fse_description
from torchsim.sequence import _accelerators
from torchsim.sequence._parameters import (
    FEATURE_BITS,
    Geometry,
    feature_flags,
    feature_mask,
)

ECHOES = 8
STATES = 8
VOXELS = 64

WINDING = Geometry(flow_scale=120.0, washout_scale=4.0)
STILL = Geometry()

# Every bit set, which is what a caller who declares nothing gets and what the
# kernels carried before a mask reached them.
ALL_ON = (1 << len(FEATURE_BITS)) - 1

CRUSHED = 2.0 * torch.pi


def _description(crusher_rad: float = 0.0):
    return fse_description(
        torch.deg2rad(torch.full((ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
        crusher_dephasing_rad=crusher_rad,
        voxel_size_m=None if crusher_rad == 0.0 else 1e-3,
    )


def _spread(low, high, seed):
    generator = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(VOXELS, generator=generator)


# --- the mask against the flags it stands for ---


def test_a_caller_who_declares_nothing_gets_every_bit():
    assert feature_mask(None, WINDING) == ALL_ON


def test_each_bit_stands_for_the_flag_of_the_same_name():
    """One fold, read two ways: a bit is set exactly when its flag is true."""
    for features in (
        None,
        frozenset({"T1", "T2"}),
        frozenset({"T1", "B0"}),
        frozenset({"T1", "FLOW"}),
        frozenset({"T1", "DIFFUSION"}),
        frozenset({"T1", "B1_PHASE", "FLOW", "DIFFUSION"}),
    ):
        for geometry in (STILL, WINDING):
            flags = feature_flags(features, geometry)
            mask = feature_mask(features, geometry)
            for bit, name in enumerate(FEATURE_BITS):
                assert bool(mask & (1 << bit)) is flags[name], (
                    features, geometry, name
                )


# --- the gated host kernels against the full ones ---

CASES = [
    ({}, 0.0),
    ({"b0_hz": _spread(-40.0, 40.0, 1)}, 0.0),
    ({"b1_phase_rad": _spread(-0.5, 0.5, 2)}, 0.0),
    ({"diffusion_um2_per_ms": _spread(0.5, 3.0, 3)}, CRUSHED),
    (
        {
            "b0_hz": _spread(-40.0, 40.0, 4),
            "diffusion_um2_per_ms": _spread(0.5, 3.0, 5),
        },
        CRUSHED,
    ),
]
IDS = ["bare", "off-resonance", "transmit-phase", "diffusion", "both"]


def _run(monkeypatch, extra, crusher_rad, mask, gradients):
    if mask is not None:
        monkeypatch.setattr(
            _accelerators, "feature_mask", lambda features, geometry: mask
        )
    leaves = {
        name: _spread(*bounds, seed).requires_grad_(gradients)
        for seed, (name, bounds) in enumerate(
            (("t1_ms", (600.0, 1800.0)), ("t2_ms", (30.0, 150.0))), start=10
        )
    }
    signal = FSE().simulate(
        _description(crusher_rad),
        TissueProperties(**leaves, **extra),
        nstates=STATES,
    ).signal
    if not gradients:
        return (signal,)
    signal.abs().square().sum().backward()
    return tuple(leaves[name].grad for name in leaves)


@pytest.mark.parametrize(("extra", "crusher_rad"), CASES, ids=IDS)
def test_the_host_forward_does_not_depend_on_the_mask(
    monkeypatch, extra, crusher_rad
) -> None:
    """A term the mask drops is one the answer does not contain, so the two
    launches agree to the bit.
    """
    whole = _run(monkeypatch, extra, crusher_rad, ALL_ON, False)
    monkeypatch.undo()
    gated = _run(monkeypatch, extra, crusher_rad, None, False)
    assert torch.equal(gated[0], whole[0])


@pytest.mark.parametrize(("extra", "crusher_rad"), CASES, ids=IDS)
def test_the_host_adjoint_does_not_depend_on_the_mask(
    monkeypatch, extra, crusher_rad
) -> None:
    """The same for the reverse pass, which walks the interval three times and
    so has three chances to drop a term it still needs.
    """
    whole = _run(monkeypatch, extra, crusher_rad, ALL_ON, True)
    monkeypatch.undo()
    gated = _run(monkeypatch, extra, crusher_rad, None, True)
    for one, other in zip(gated, whole, strict=True):
        assert torch.equal(one, other)


# --- the mask as a choice rather than a coincidence ---

# Each term, the property that drives it, and a value loud enough that a
# kernel still carrying the term could not give the same answer without it.
DROPPED = [
    ("off_axis", {"b0_hz": 500.0}, {"b0_hz": 0.0}, 0.0),
    ("off_axis", {"b1_phase_rad": 1.1}, {"b1_phase_rad": 0.0}, 0.0),
    ("moving", {"velocity_m_per_s": 0.4}, {"velocity_m_per_s": 0.0}, CRUSHED),
    (
        "diffusing",
        {"diffusion_um2_per_ms": 3.0},
        {"diffusion_um2_per_ms": 0.0},
        CRUSHED,
    ),
    ("transmit", {"b1": 0.6}, {"b1": 1.0}, 0.0),
    ("density", {"m0": 0.4}, {"m0": 1.0}, 0.0),
]


def _without(name: str) -> int:
    return ALL_ON & ~(1 << FEATURE_BITS.index(name))


def _signal(monkeypatch, mask, extra, crusher_rad):
    monkeypatch.setattr(
        _accelerators, "feature_mask", lambda features, geometry: mask
    )
    signal = FSE().simulate(
        _description(crusher_rad),
        TissueProperties(
            t1_ms=torch.full((4,), 1000.0),
            t2_ms=torch.full((4,), 80.0),
            **{name: torch.full((4,), value) for name, value in extra.items()},
        ),
        nstates=STATES,
    ).signal
    monkeypatch.undo()
    return signal


@pytest.mark.parametrize(
    ("name", "loud", "silent", "crusher_rad"),
    DROPPED,
    ids=[
        "off-resonance", "transmit-phase", "flow", "diffusion",
        "transmit", "density",
    ],
)
def test_a_dropped_term_gives_the_answer_of_a_tissue_without_it(
    monkeypatch, name, loud, silent, crusher_rad
) -> None:
    """What it means for the mask to have dropped a term, stated exactly.

    A threshold on how far the answer moved would pass on a kernel that never
    read the mask at all -- the recorded echoes of a refocused train barely
    move with off-resonance however large it is. This asks instead that the
    launch told to drop a term land on the answer a tissue that never declared
    the property gets, to the bit, which nothing but the branch can do.
    """
    dropped = _signal(monkeypatch, _without(name), loud, crusher_rad)
    absent = _signal(monkeypatch, ALL_ON, silent, crusher_rad)
    assert torch.equal(dropped, absent)
    # And the property is one the sequence notices at all, so the two are not
    # agreeing because there was nothing to drop. This is only a guard: a
    # refocused train moves very little with off-resonance, and it is the
    # exact match above that says the branch was taken.
    carried = _signal(monkeypatch, ALL_ON, loud, crusher_rad)
    assert not torch.equal(carried, absent)


def test_an_inversion_the_tissue_never_declared_is_left_out(monkeypatch) -> None:
    """The one term a refocused train cannot exercise: it drives no inversion,
    so the gate is held against a sequence that does.
    """
    from torchsim.sequence import SPGR, mprage_description

    def signal(mask, efficiency):
        monkeypatch.setattr(
            _accelerators, "feature_mask", lambda features, geometry: mask
        )
        out = SPGR().simulate(
            mprage_description(2, 4, torch.deg2rad(torch.tensor(12.0)), 5e-3, 20e-3),
            TissueProperties(
                t1_ms=torch.full((4,), 1000.0),
                t2_ms=torch.full((4,), 80.0),
                inversion_efficiency=torch.full((4,), efficiency),
            ),
            nstates=STATES,
        ).signal
        monkeypatch.undo()
        return out

    dropped = signal(_without("inverting"), 0.4)
    absent = signal(ALL_ON, 1.0)
    carried = signal(ALL_ON, 0.4)

    assert torch.equal(dropped, absent)
    assert not torch.equal(carried, absent)
