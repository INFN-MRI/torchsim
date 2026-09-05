"""Clustering a dictionary so most of it can be ruled out per voxel."""

from __future__ import annotations

import pytest
import torch

from torchsim.estimators import DictionaryMatcher
from torchsim.estimators._grouped import Grouping
from torchsim.simulators import MRFSimulator


@pytest.fixture(scope="module")
def fingerprints():
    """A fingerprinting dictionary over a T1/T2 grid, and what each atom means."""
    generator = torch.Generator().manual_seed(0)
    flip = 10.0 + 50.0 * torch.rand(200, generator=generator)
    grid = torch.cartesian_prod(
        torch.linspace(200.0, 3000.0, 60), torch.linspace(20.0, 300.0, 30)
    )
    atoms = MRFSimulator(TR=10.0, TI=20.0, T1=grid[:, 0], T2=grid[:, 1]).simulate(
        flip=flip
    )
    return atoms.reshape(grid.shape[0], -1), grid


@pytest.fixture(scope="module")
def measured(fingerprints):
    """Atoms off the grid, with noise, as a volume to match."""
    atoms, _ = fingerprints
    generator = torch.Generator().manual_seed(7)
    picked = atoms[::37]
    noise = torch.randn(picked.shape, generator=generator, dtype=picked.dtype)
    return picked + 0.01 * float(atoms.abs().max()) * noise


def test_every_atom_lands_in_exactly_one_group(fingerprints) -> None:
    """A grouping is a partition, not a selection."""
    atoms, _ = fingerprints

    grouping = Grouping.fit(atoms, 24)

    assigned = torch.cat(grouping.members).sort().values
    torch.testing.assert_close(assigned, torch.arange(atoms.shape[0]))
    assert grouping.count == 24
    assert sum(grouping.sizes) == atoms.shape[0]


def test_the_groups_come_out_the_size_they_were_asked_for(fingerprints) -> None:
    """All but the last hold a full share; the last holds the remainder."""
    atoms, _ = fingerprints

    grouping = Grouping.fit(atoms, 24)

    share = -(-atoms.shape[0] // 24)
    assert set(grouping.sizes[:-1]) == {share}
    assert 0 < grouping.sizes[-1] <= share


def test_a_group_holds_its_atoms_in_the_basis_they_arrived_in(
    fingerprints,
) -> None:
    """One basis for everything, and the grouping does not add a second.

    Compression is global and comes first, so a group can be entered without
    leaving the space the measurement is in. All a group keeps is its own
    atoms, normalized.
    """
    atoms, _ = fingerprints

    grouping = Grouping.fit(atoms, 24)

    members = atoms[grouping.members[0]]
    torch.testing.assert_close(
        grouping.atoms[0],
        members / torch.linalg.vector_norm(members, dim=-1, keepdim=True),
    )
    assert grouping.atoms[0].shape[-1] == atoms.shape[-1]


def test_grouped_matching_finds_what_direct_matching_finds(
    fingerprints, measured
) -> None:
    """The point of the whole exercise: same answer, less arithmetic.

    Pruning is a claim that the groups ruled out could not have held the
    match. This is that claim tested against the match itself.
    """
    atoms, grid = fingerprints

    direct = DictionaryMatcher(dictionary=atoms, parameters=grid)(measured)
    grouped = DictionaryMatcher(dictionary=atoms, parameters=grid, groups=24)(measured)

    # Where they differ they differ by a grid step, so the relative error is
    # the claim worth making; Cauley et al. report one to two percent.
    agree = (direct == grouped).all(-1).float().mean()
    assert float(agree) > 0.8
    for column in range(grid.shape[-1]):
        error = ((grouped[:, column] - direct[:, column]) / direct[:, column]).abs()
        assert float(error.mean()) < 0.01


def test_most_of_the_dictionary_is_ruled_out(fingerprints, measured) -> None:
    """Pruning that kept every group would be a cost with no saving."""
    atoms, grid = fingerprints
    matcher = DictionaryMatcher(dictionary=atoms, parameters=grid, groups=24)
    normalized = measured / torch.linalg.vector_norm(measured, dim=-1, keepdim=True)

    survivors = matcher.grouping.survivors(normalized, matcher.prune)

    assert float(survivors.sum(-1).float().mean()) < 0.25 * matcher.grouping.count


def test_a_wider_threshold_keeps_more_groups(fingerprints, measured) -> None:
    """The threshold is the accuracy-for-time knob, and it moves both ways."""
    atoms, grid = fingerprints
    normalized = measured / torch.linalg.vector_norm(measured, dim=-1, keepdim=True)
    grouping = DictionaryMatcher(dictionary=atoms, parameters=grid, groups=24).grouping

    tight = grouping.survivors(normalized, 0.0).sum(-1).float().mean()
    loose = grouping.survivors(normalized, 0.2).sum(-1).float().mean()

    assert float(tight) >= 1.0
    assert float(loose) > float(tight)


def test_the_condition_number_is_reported(fingerprints) -> None:
    """What says whether there are too many groups, before any data arrives.

    Representatives that begin to look like one another leave nothing to
    prune with, and this climbs when they do.
    """
    atoms, _ = fingerprints

    few = Grouping.fit(atoms, 4).condition
    many = Grouping.fit(atoms, 64).condition

    assert few > 1.0
    assert many > few


def test_top_k_survives_the_grouping(fingerprints, measured) -> None:
    """Several candidates, gathered across whichever groups they fall in."""
    atoms, grid = fingerprints
    matcher = DictionaryMatcher(dictionary=atoms, parameters=grid, groups=24, top_k=3)

    found = matcher.match(measured)

    assert found.indices.shape == (measured.shape[0], 3)
    # Ranked, and each candidate is a real atom.
    assert bool((torch.diff(found.scores, dim=-1) <= 1e-5).all())
    assert bool((found.indices < atoms.shape[0]).all())


def test_asking_for_more_groups_than_atoms_gives_one_each(
    fingerprints,
) -> None:
    """A group per atom is the limit, not an error."""
    atoms, grid = fingerprints

    matcher = DictionaryMatcher(dictionary=atoms[:8], parameters=grid[:8], groups=99)

    assert matcher.grouping.count == 8
    assert set(matcher.grouping.sizes) == {1}


def test_a_fitted_dictionary_is_grouped_too(fingerprints, measured) -> None:
    """Grouping follows the dictionary however it arrived."""
    atoms, grid = fingerprints

    matcher = DictionaryMatcher(groups=24).fit(signals=atoms, parameters=grid)

    assert matcher.grouping is not None
    assert matcher.grouping.count == 24
    assert matcher(measured).shape == (measured.shape[0], grid.shape[-1])


@pytest.mark.parametrize(
    "settings,complaint",
    [
        (dict(groups=0), "groups"),
        (dict(prune=1.0), "prune"),
        (dict(prune=-0.1), "prune"),
    ],
)
def test_settings_that_make_no_sense(settings, complaint) -> None:
    """Caught where they are written."""
    with pytest.raises(ValueError, match=complaint):
        DictionaryMatcher(**settings)


def test_a_grouping_cannot_have_no_groups(fingerprints) -> None:
    """Zero groups is not a degenerate grouping, it is a mistake."""
    atoms, _ = fingerprints

    with pytest.raises(ValueError, match="between one"):
        Grouping.fit(atoms, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_grouping_moves_with_its_dictionary(fingerprints, measured) -> None:
    """Clustering belongs to the dictionary, not to where it sits."""
    atoms, grid = fingerprints
    matcher = DictionaryMatcher(dictionary=atoms, parameters=grid, groups=24)

    here = matcher(measured)
    there = matcher.to("cuda")(measured.cuda()).cpu()

    torch.testing.assert_close(here, there)


# %% grouping a dictionary that is already compressed


def test_grouping_composes_with_a_temporal_subspace(fingerprints, measured) -> None:
    """A dictionary stored as ``(atoms, rank)`` is grouped like any other.

    The two savings are independent and multiply: a subspace shortens every
    inner product, grouping cuts how many are taken. Neither knows about the
    other, and the answer is the one the compressed dictionary gives without
    grouping -- the grouping must not add error of its own.
    """
    from torchsim import (
        Subspace,
    )

    atoms, grid = fingerprints
    subspace = Subspace.fit(atoms, 8)
    compressed = subspace.project(atoms)
    projected = subspace.project(measured)

    direct = DictionaryMatcher(dictionary=compressed, parameters=grid)(projected)
    grouped = DictionaryMatcher(dictionary=compressed, parameters=grid, groups=24)(
        projected
    )

    assert compressed.shape == (atoms.shape[0], 8)
    for column in range(grid.shape[-1]):
        error = ((grouped[:, column] - direct[:, column]) / direct[:, column]).abs()
        assert float(error.mean()) < 0.01


def test_clustering_happens_inside_the_global_basis(fingerprints) -> None:
    """Compress, then cluster -- in that order and in that one basis.

    A group that carried a basis of its own would have to be entered by
    leaving the space the measurement is in. Everything stays at the rank the
    dictionary was compressed to.
    """
    from torchsim import (
        Subspace,
    )

    atoms, _ = fingerprints
    compressed = Subspace.fit(atoms, 4).project(atoms)

    grouping = Grouping.fit(compressed, 24)

    assert grouping.representative.shape == (24, 4)
    assert all(group.shape[-1] == 4 for group in grouping.atoms)


def test_a_mapping_with_a_rank_reaches_a_grouped_matcher(fingerprints) -> None:
    """Stating the rank on the problem is enough; the matcher follows.

    The matcher fits the subspace, projects the training signals into it,
    matches against a compressed dictionary -- then projects the volume the
    same way when it is called.
    """
    atoms, grid = fingerprints
    generator = torch.Generator().manual_seed(1)
    flip = 10.0 + 50.0 * torch.rand(200, generator=generator)
    acquisition = MRFSimulator(TR=10.0, TI=20.0, flip=flip)
    mapping = DictionaryMatcher(acquisition, groups=24).fit(
        T1=grid[:, 0], T2=grid[:, 1], rank=8, seed=0
    )

    truth = grid[::53]
    volume = MRFSimulator(TR=10.0, TI=20.0, T1=truth[:, 0], T2=truth[:, 1]).simulate(
        flip=flip
    )
    maps = mapping(volume.reshape(truth.shape[0], -1))

    assert mapping.subspace.rank == 8
    assert mapping.grouping.count == 24
    for column, name in enumerate(("T1", "T2")):
        step = float(torch.diff(grid[:, column].unique()).max())
        error = (maps[name] - truth[:, column]).abs()
        assert float(error.mean()) < 0.5 * step, f"{name}: {float(error.mean()):.2f}"
        assert float((error < step).float().mean()) > 0.9


def test_a_tighter_threshold_can_prune_away_the_right_group() -> None:
    """The threshold is a claim, and it can be wrong.

    Pruning asserts that the groups ruled out could not have held the match.
    At the corner of a coarse parameter grid that assertion fails: the right
    group's representative is not within the default fraction of the best, so
    the atom is never scored. The threshold Cauley et al. give was tuned on
    280 groups of 700 atoms, and this is what it does on a small dictionary.

    Widening it recovers the voxel, which is the trade the knob exists for.
    """
    grid = torch.cartesian_prod(
        torch.linspace(200.0, 3000.0, 60), torch.linspace(20.0, 300.0, 30)
    )
    generator = torch.Generator().manual_seed(1)
    flip = 10.0 + 50.0 * torch.rand(200, generator=generator)
    simulator = MRFSimulator(TR=10.0, TI=20.0)
    truth = grid[::53]
    volume = (
        simulator.bind(T1=truth[:, 0], T2=truth[:, 1])
        .simulate(flip=flip)
        .reshape(truth.shape[0], -1)
    )

    def misses(prune):
        mapping = DictionaryMatcher(
            simulator.bind(flip=flip), groups=24, prune=prune
        ).fit(T1=grid[:, 0], T2=grid[:, 1], rank=8, seed=0)
        error = (mapping(volume)["T1"] - truth[:, 0]).abs()
        return int((error > 1.0).sum())

    assert misses(5e-3) > 0
    assert misses(0.05) == 0
