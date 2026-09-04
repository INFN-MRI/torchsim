"""Where a mapping runs, and whether that changes the answer.

Voxels are independent, so the policy may put them anywhere -- on the host, on
a card, through a card a chunk at a time, or across two cards. None of that is
allowed to change what comes back, and a route that quietly did not happen
would pass an agreement test on its own, so the counts are asserted too.
"""

from __future__ import annotations

import pytest
import torch

from torchsim import (
    PERK,
    DictionaryMatcher,
)
from torchsim.estimators import _perk
from torchsim.sequence import execution
from torchsim.simulators import MRFSimulator

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
TWO_CARDS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)
CONTRASTS = 32
VOXELS = 6000


@pytest.fixture
def volume() -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    return torch.randn(VOXELS, CONTRASTS, generator=generator)


@pytest.fixture
def estimator() -> PERK:
    generator = torch.Generator().manual_seed(2)
    signals = torch.randn(3000, CONTRASTS, generator=generator)
    targets = torch.stack((signals.sum(-1), signals.square().mean(-1)), dim=-1)
    return PERK(n_features=256, feature_seed=4).fit(signals=signals, parameters=targets)


@pytest.fixture
def matcher() -> DictionaryMatcher:
    generator = torch.Generator().manual_seed(2)
    atoms = torch.randn(2000, CONTRASTS, generator=generator)
    return DictionaryMatcher(
        dictionary=atoms, parameters=torch.rand(2000, 2, generator=generator)
    )


@CUDA
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param({"target": "cpu"}, id="host"),
        pytest.param({"target": "cuda", "stream": False}, id="resident"),
        pytest.param(
            {"target": "cuda", "stream": True, "budget_bytes": 1 << 16},
            id="streamed",
        ),
        pytest.param(
            {"target": "cuda", "stream": True, "budget_bytes": 1 << 16, "lanes": 2},
            id="streamed-two-lanes",
        ),
    ],
)
def test_a_mapping_is_the_same_wherever_it_runs(estimator, volume, policy) -> None:
    """The policy chooses a place, not an answer."""
    expected = estimator(volume)

    with execution(**policy):
        got = estimator(volume)

    assert got.device == expected.device
    assert torch.allclose(got, expected, rtol=2e-4, atol=1e-5)


@CUDA
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param({"target": "cuda", "stream": False}, id="resident"),
        pytest.param(
            {"target": "cuda", "stream": True, "budget_bytes": 1 << 18},
            id="streamed",
        ),
    ],
)
def test_a_match_is_the_same_wherever_it_runs(matcher, volume, policy) -> None:
    """Indices are exact, so a moved match has nothing to hide behind."""
    expected = matcher.match(volume)

    with execution(**policy):
        got = matcher.match(volume)

    assert torch.equal(got.indices, expected.indices)
    assert torch.allclose(got.scores, expected.scores, atol=1e-5)
    assert torch.allclose(got.parameters, expected.parameters)


@CUDA
def test_a_small_budget_really_does_split_the_volume(
    estimator, volume, monkeypatch
) -> None:
    """Agreement cannot tell one chunk from forty, so count them."""
    chunks: list[int] = []
    original = PERK._regress
    monkeypatch.setattr(
        PERK,
        "_regress",
        lambda self, inputs, held: (
            chunks.append(inputs.shape[0]),
            original(self, inputs, held),
        )[1],
    )

    with execution(target="cuda", stream=True, budget_bytes=1 << 16):
        estimator(volume)

    assert len(chunks) > 1
    assert sum(chunks) == VOXELS


@CUDA
def test_a_volume_that_fits_crosses_in_one_piece(
    estimator, volume, monkeypatch
) -> None:
    """The other half of the same claim: streaming is not the only route."""
    chunks: list[int] = []
    original = PERK._regress
    monkeypatch.setattr(
        PERK,
        "_regress",
        lambda self, inputs, held: (
            chunks.append(inputs.shape[0]),
            original(self, inputs, held),
        )[1],
    )

    with execution(target="cuda", stream=False):
        estimator(volume)

    assert chunks == [VOXELS]


@TWO_CARDS
def test_two_cards_each_take_a_share(estimator, volume, monkeypatch) -> None:
    """Voxels are independent, so a second card is half the work, not none."""
    seen: list[str] = []
    original = PERK._regress
    monkeypatch.setattr(
        PERK,
        "_regress",
        lambda self, inputs, held: (
            seen.append(str(inputs.device)),
            original(self, inputs, held),
        )[1],
    )
    expected = estimator(volume)

    with execution(target=["cuda:0", "cuda:1"], stream=False):
        got = estimator(volume)

    assert len(set(seen)) == 2
    assert torch.allclose(got, expected, rtol=2e-4, atol=1e-5)


@CUDA
def test_a_gradient_keeps_the_ordinary_route(estimator, monkeypatch) -> None:
    """A streamed chunk goes through a pinned buffer that is written again for
    the next chunk, which autograd cannot be walked back through. A call that
    wants a derivative has to stay where it is."""
    monkeypatch.setattr(
        _perk, "per_voxel", lambda *args, **kwargs: pytest.fail("the policy ran")
    )
    measured = torch.randn(64, CONTRASTS, requires_grad=True)

    with execution(target="cuda", stream=True, budget_bytes=1 << 16):
        estimator(measured).sum().backward()

    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()


@CUDA
def test_a_whole_mapping_runs_under_a_policy() -> None:
    """What a user actually writes, with the volume never leaving the host."""
    generator = torch.Generator().manual_seed(5)
    flip = 5.0 + 55.0 * torch.rand(CONTRASTS, generator=generator)
    mapping = PERK(
        MRFSimulator(TR=10.0, TI=20.0, states=10, flip=flip),
        n_features=256,
        feature_seed=0,
    ).fit(T1=(300.0, 2000.0), T2=(20.0, 200.0), seed=0, samples=2048)
    truth = {"T1": torch.full((500,), 900.0), "T2": torch.full((500,), 70.0)}
    measured = mapping.acquisition.simulate(**truth)
    expected = mapping(measured)

    with execution(target="cuda", stream=True, budget_bytes=1 << 16):
        got = mapping(measured)

    for name in truth:
        assert got[name].device.type == "cpu"
        assert torch.allclose(got[name], expected[name], rtol=2e-4, atol=1e-3)


@CUDA
def test_a_fit_gives_the_same_estimator_wherever_it_runs() -> None:
    """Fitting is a reduction, and the policy may put it on a card.

    Torch's random generators do not agree between devices at the same seed,
    so the random Fourier features are drawn in one place whatever the policy
    says -- otherwise where a fit ran would decide which estimator came out
    of it, and that is not a decision about speed.
    """
    generator = torch.Generator().manual_seed(8)
    signals = torch.randn(4000, CONTRASTS, generator=generator)
    targets = torch.stack((signals.sum(-1), signals.square().mean(-1)), dim=-1)

    here = PERK(n_features=256, feature_seed=6).fit(signals=signals, parameters=targets)
    with execution(target="cuda"):
        there = PERK(n_features=256, feature_seed=6).fit(
            signals=signals, parameters=targets
        )

    assert there.weight.device.type == "cpu", "the estimator comes home"
    scale = here.weight.abs().max()
    assert float((there.weight - here.weight).abs().max() / scale) < 5e-6


@CUDA
def test_a_fit_on_a_card_reaches_it(monkeypatch) -> None:
    """The other half: assert the accumulation really moved."""
    generator = torch.Generator().manual_seed(8)
    signals = torch.randn(2000, CONTRASTS, generator=generator)
    targets = signals.sum(-1, keepdim=True)
    seen: list[str] = []
    original = _perk._rff
    monkeypatch.setattr(
        _perk,
        "_rff",
        lambda inputs, frequency, phase: (
            seen.append(inputs.device.type),
            original(inputs, frequency, phase),
        )[1],
    )

    with execution(target="cuda"):
        PERK(n_features=128, feature_seed=6).fit(signals=signals, parameters=targets)

    assert seen and set(seen) == {"cuda"}
