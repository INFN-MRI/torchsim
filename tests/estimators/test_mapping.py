"""What a mapping problem promises, whichever method fills it in.

The point of stating the problem separately from the method is that the two
are exchangeable, so most of these tests are run against both shipped methods
rather than against one.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import (
    Acquisition,
    DictionaryMatcher,
    Estimator,
    PERK,
    ParameterMapping,
)
from torchsim.simulators import FSESimulator, MRFSimulator

ECHOES = 32


def _acquisition() -> Acquisition:
    """An MRF train, which separates T1 from T2 well enough to be mapped."""
    generator = torch.Generator().manual_seed(7)
    flip = 5.0 + 55.0 * torch.rand(ECHOES, generator=generator)
    return Acquisition(MRFSimulator(TR=10.0, TI=20.0, states=10), flip=flip)


def _mapping(**extra) -> ParameterMapping:
    unknown = {"T1": (300.0, 2000.0), "T2": (20.0, 200.0)}
    unknown.update({name: extra.pop(name) for name in list(extra) if name in unknown})
    return ParameterMapping(_acquisition(), seed=0, **unknown, **extra)


METHODS = [
    pytest.param(lambda: PERK(n_features=512, seed=0), id="perk"),
    pytest.param(DictionaryMatcher, id="dictionary"),
]


@pytest.mark.parametrize("method", METHODS)
def test_both_shipped_methods_are_estimators(method) -> None:
    """The protocol is what makes swapping one for the other a word."""
    assert isinstance(method(), Estimator)


def test_a_matcher_returns_the_tissue_its_own_atom_came_from() -> None:
    """The sharp case: mapping the training signals themselves is exact.

    Every training signal is an atom, so its best match is itself and the
    parameters that come back are the ones it was simulated from -- no
    tolerance, no statistics.
    """
    mapping = _mapping()
    signals, parameters, _ = mapping.training_set(256)
    mapping.train(DictionaryMatcher(), samples=256)

    maps = mapping(signals)

    assert torch.allclose(maps["T1"], parameters[:, 0])
    assert torch.allclose(maps["T2"], parameters[:, 1])


@pytest.mark.parametrize("method", METHODS)
def test_the_maps_come_back_named_and_shaped_like_the_volume(method) -> None:
    """A caller reads ``maps["T1"]``, not column zero of a matrix."""
    mapping = _mapping().train(method(), samples=512)
    volume = mapping.acquisition.simulate(
        T1=torch.full((24,), 900.0), T2=torch.full((24,), 70.0)
    ).reshape(2, 3, 4, ECHOES)

    maps = mapping(volume)

    assert set(maps) == {"T1", "T2"}
    assert all(value.shape == (2, 3, 4) for value in maps.values())


@pytest.mark.parametrize("method", METHODS)
def test_a_mapping_recovers_a_tissue_it_was_trained_over(method) -> None:
    """Noise-free, and well inside the range trained over."""
    mapping = _mapping().train(method(), samples=4096)
    truth = {"T1": torch.tensor([700.0, 1300.0]), "T2": torch.tensor([45.0, 110.0])}

    maps = mapping(mapping.acquisition.simulate(**truth))

    for name, value in truth.items():
        assert torch.allclose(maps[name], value, rtol=0.15)


def test_a_subspace_maps_as_well_as_the_contrasts_it_replaces() -> None:
    """The compression is only worth having if it changes the answer little,
    and what it keeps is a number the mapping reports."""
    truth = {"T1": torch.tensor([700.0, 1300.0]), "T2": torch.tensor([45.0, 110.0])}
    full = _mapping().train(DictionaryMatcher(), samples=1024)
    compressed = _mapping(rank=6).train(DictionaryMatcher(), samples=1024)
    volume = full.acquisition.simulate(**truth)

    assert compressed.subspace.retained > 0.999
    assert full.subspace is None
    for name in truth:
        assert torch.allclose(full(volume)[name], compressed(volume)[name], rtol=0.1)


def test_a_known_property_reaches_the_simulator_and_is_asked_for_again() -> None:
    """A transmit map changes the signals trained on, so it has to be given
    again when a volume is mapped."""
    mapping = ParameterMapping(
        Acquisition(
            FSESimulator(ESP=5.0, TR=3000.0, states=10),
            T1=1000.0,
            flip=torch.full((ECHOES,), 150.0),
        ),
        T2=(20.0, 200.0),
        known={"B1": (0.7, 1.3)},
        seed=0,
    )
    signals, _, known = mapping.training_set(64)

    assert known is not None and known.shape == (64, 1)
    assert not torch.allclose(signals[0], signals[1])

    mapping.train(PERK(n_features=128, seed=0), samples=512)
    with pytest.raises(ValueError, match="B1 was not given"):
        mapping(signals, known={})


def test_a_matcher_refuses_a_separately_measured_property() -> None:
    """One dictionary spans one grid, so a per-voxel property would need a
    dictionary per voxel. Saying so beats quietly ignoring it."""
    acquisition = Acquisition(
        MRFSimulator(TR=10.0, TI=20.0, states=10),
        T2=80.0,
        flip=torch.full((ECHOES,), 30.0),
    )
    mapping = ParameterMapping(
        acquisition, T1=(300.0, 2000.0), known={"B1": (0.8, 1.2)}, seed=0
    )
    with pytest.raises(ValueError, match="cannot take a separately measured"):
        mapping.train(DictionaryMatcher(), samples=64)


def test_mapping_before_training_says_so() -> None:
    mapping = _mapping()
    with pytest.raises(RuntimeError, match="must be trained"):
        mapping(torch.zeros(4, ECHOES))


def test_a_property_cannot_be_both_unknown_and_known() -> None:
    with pytest.raises(ValueError, match="both unknown and known"):
        ParameterMapping(_acquisition(), T1=(300.0, 2000.0), known={"T1": (1.0, 2.0)})


def test_a_mapping_needs_something_to_estimate() -> None:
    with pytest.raises(ValueError, match="at least one property"):
        ParameterMapping(_acquisition())


def test_a_range_must_increase() -> None:
    with pytest.raises(ValueError, match="not increasing"):
        _mapping(T2=(200.0, 20.0)).training_set(8)


def test_values_given_for_a_property_are_used_as_given() -> None:
    """A user with their own sampling -- log-spaced, or from a cohort -- hands
    it over instead of a range."""
    values = torch.linspace(400.0, 1800.0, 16)
    mapping = ParameterMapping(_acquisition(), T1=values, T2=(20.0, 200.0), seed=0)

    _, parameters, _ = mapping.training_set(16)

    assert torch.allclose(parameters[:, 0], values)


def test_the_training_set_is_the_same_size_however_it_is_chunked() -> None:
    """Chunking bounds the memory a draw takes, and nothing else."""
    mapping = _mapping()

    whole = mapping.training_set(100, chunk=1000)
    pieces = mapping.training_set(100, chunk=16)

    assert whole[0].shape == pieces[0].shape == (100, ECHOES)
    assert torch.allclose(whole[0], pieces[0])
    assert torch.allclose(whole[1], pieces[1])
