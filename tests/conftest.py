"""Rooted here so ``tests/`` is importable, which is what ``utils`` needs."""

import pytest


@pytest.fixture
def always_worth_detecting(monkeypatch):
    """Reach for the subspace verdict however small the problem looks.

    ``detection`` is measured against the machine rather than fixed, so what
    counts as enough work depends on what else the card has been doing. A test
    whose arms must take the same kernel has to say so: left to the threshold,
    two runs of different sizes -- a batch of trains against the trains one by
    one -- can fall on opposite sides of it and be compared across kernels.
    """
    from torchsim.sequence import _accelerators

    monkeypatch.setattr(
        _accelerators, "detection", lambda kind, device, state_count: 0.0
    )
