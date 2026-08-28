"""Reading one unknown back off a monotonic curve."""

from __future__ import annotations

import pytest
import torch

from torchsim.estimators import LookupTable
from torchsim.simulators import MP2RAGESimulator

PROTOCOL = dict(
    TI=(800.0, 2700.0),
    flip=(4.0, 5.0),
    TRspgr=6.7,
    TRmp2rage=6000.0,
    nshots=128,
)
SPACING_MS = 50.0


def unified(T1_ms):
    """The MP2RAGE ratio, which is what a T1 map is read from."""
    blocks = MP2RAGESimulator(**PROTOCOL).simulate(T1=T1_ms, inv_efficiency=0.96)
    return (blocks[:, 0] * blocks[:, 1]) / blocks.square().sum(-1)


@pytest.fixture
def mp2rage_table():
    """A T1 table over the range a brain spans, on a 50 ms grid."""
    T1 = torch.arange(50.0, 5000.0, SPACING_MS)
    return LookupTable().fit(signals=unified(T1)[:, None], parameters=T1[:, None])


def test_the_table_returns_its_own_knots_exactly(mp2rage_table) -> None:
    """Interpolation at a knot is the knot."""
    got = mp2rage_table(mp2rage_table.intensity[:, None]).flatten()

    torch.testing.assert_close(got, mp2rage_table.parameter)


def test_interpolating_beats_the_grid_it_was_built_on(mp2rage_table) -> None:
    """The point of interpolating rather than matching.

    A nearest-atom match cannot do better than half the grid spacing. Reading
    between the atoms removes the grid from the answer, leaving only the
    curvature the straight line misses.
    """
    low = float(mp2rage_table.parameter.min())
    high = float(mp2rage_table.parameter.max())
    between = torch.arange(low + SPACING_MS / 2, high, SPACING_MS)

    got = mp2rage_table(unified(between)[:, None]).flatten()

    error = (got - between).abs()
    assert float(error.max()) < SPACING_MS / 4
    assert float(error.mean()) < SPACING_MS / 25


def test_the_turning_tails_are_dropped(mp2rage_table) -> None:
    """A curve that turns back has no inverse where it turns.

    The MP2RAGE ratio rises again at very short and very long T1, so the same
    ratio would stand for two different tissues there.
    """
    T1 = torch.arange(50.0, 5000.0, SPACING_MS)
    kept = mp2rage_table.parameter

    assert mp2rage_table.points < len(T1)
    assert float(kept.min()) > float(T1.min())
    assert float(kept.max()) < float(T1.max())
    # What is kept is a single unbroken stretch of the original grid.
    assert torch.allclose(torch.diff(kept.sort().values), torch.tensor(SPACING_MS))


def test_keeping_the_whole_curve_is_a_choice() -> None:
    """``monotonic=False`` keeps every sample, turning points and all."""
    T1 = torch.arange(50.0, 5000.0, SPACING_MS)

    table = LookupTable(monotonic=False).fit(
        signals=unified(T1)[:, None], parameters=T1[:, None]
    )

    assert table.points == len(T1)


def test_a_measurement_off_the_end_reads_as_the_endpoint(mp2rage_table) -> None:
    """Noise puts background voxels outside any curve the model can produce.

    Clamping is what keeps them a number rather than a NaN that spreads.
    """
    low, high = mp2rage_table.span

    got = mp2rage_table(torch.tensor([[low - 10.0], [high + 10.0]])).flatten()

    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got, torch.stack((mp2rage_table.parameter[0], mp2rage_table.parameter[-1]))
    )


def test_a_rising_curve_is_read_the_same_way() -> None:
    """Which way the curve runs is not the caller's to arrange."""
    parameter = torch.linspace(1.0, 4.0, 64)
    rising = parameter.square()

    table = LookupTable().fit(signals=rising[:, None], parameters=parameter[:, None])
    got = table(torch.tensor([[4.0], [9.0]])).flatten()

    torch.testing.assert_close(got, torch.tensor([2.0, 3.0]), atol=2e-3, rtol=0)


def test_a_flat_column_is_refused() -> None:
    """A constant curve carries no information about the parameter."""
    parameter = torch.linspace(1.0, 4.0, 16)

    with pytest.raises(ValueError, match="constant"):
        LookupTable().fit(signals=torch.ones(16, 1), parameters=parameter[:, None])


def test_the_estimate_carries_a_gradient(mp2rage_table) -> None:
    """A table inside a reconstruction still differentiates."""
    measured = torch.tensor([[0.1], [-0.2], [0.3]], requires_grad=True)

    mp2rage_table(measured).sum().backward()

    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()
    assert float(measured.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    "signals,parameters,complaint",
    [
        (torch.zeros(8, 2), torch.zeros(8, 1), "single column"),
        (torch.zeros(8, 1), torch.zeros(8, 2), "single column"),
        (torch.zeros(8, 1), torch.zeros(4, 1), "equal length"),
        (torch.zeros(1, 1), torch.zeros(1, 1), "at least two"),
        (torch.zeros(8, dtype=torch.complex64), torch.zeros(8), "must be real"),
    ],
)
def test_what_a_table_cannot_be_built_from(signals, parameters, complaint) -> None:
    """One curve, one unknown, and a real number per sample."""
    with pytest.raises(ValueError, match=complaint):
        LookupTable().fit(signals=signals, parameters=parameters)


def test_a_separately_measured_property_is_refused(mp2rage_table) -> None:
    """One curve cannot carry a value that differs per voxel."""
    T1 = torch.arange(50.0, 5000.0, SPACING_MS)

    with pytest.raises(ValueError, match="separately measured"):
        LookupTable().fit(
            signals=unified(T1)[:, None],
            parameters=T1[:, None],
            known=torch.zeros(len(T1), 1),
        )
    with pytest.raises(ValueError, match="separately measured"):
        mp2rage_table(torch.zeros(4, 1), torch.zeros(4, 1))


def test_an_unfitted_table_says_so() -> None:
    """Reading a table that holds no curve is a mistake, not an empty answer."""
    table = LookupTable()

    assert not table.fitted
    with pytest.raises(RuntimeError, match="no curve"):
        table(torch.zeros(4, 1))
    with pytest.raises(RuntimeError, match="no curve"):
        _ = table.span


def test_the_shape_of_the_volume_survives(mp2rage_table) -> None:
    """A map comes back shaped like the volume it was read from."""
    volume = torch.full((3, 4, 5, 1), 0.1)

    assert mp2rage_table(volume).shape == (3, 4, 5, 1)
    assert mp2rage_table(volume[..., 0]).shape == (3, 4, 5, 1)


# %% the whole mapping problem, stated once


def test_a_t1_map_is_read_from_the_two_blocks() -> None:
    """What an MP2RAGE study actually does, end to end.

    The table states the combination that makes the curve monotonic; the fit
    states the sequence and what is unknown.
    """
    protocol = dict(PROTOCOL, nshots=(26, 102))
    acquisition = MP2RAGESimulator(**protocol, inv_efficiency=0.96)
    grid = torch.arange(50.0, 5000.0, 25.0)
    table = LookupTable(acquisition, combine=_unified_of)

    # The grid states its own length, so the sample count is not repeated.
    table.fit(T1=grid, seed=0)

    truth = torch.tensor([350.0, 812.0, 1337.0, 2100.0, 3010.0])
    got = table.map(acquisition.simulate(T1=truth))["T1"]

    assert float((got.flatten() - truth).abs().max()) < 1.0


def test_the_map_is_shaped_like_the_volume() -> None:
    """A volume in, a map out, and nothing flattened on the way."""
    protocol = dict(PROTOCOL, nshots=128)
    acquisition = MP2RAGESimulator(**protocol, inv_efficiency=0.96)
    grid = torch.arange(50.0, 5000.0, 25.0)
    table = LookupTable(acquisition, combine=_unified_of).fit(T1=grid, seed=0)
    volume = acquisition.simulate(T1=torch.full((24,), 900.0)).reshape(4, 6, 2)

    got = table.map(volume)["T1"]

    assert got.shape == (4, 6)
    assert float((got - 900.0).abs().max()) < 1.0


def test_several_contrasts_without_a_combination_say_so() -> None:
    """The reduction is the sequence's business, so it has to be stated."""
    T1 = torch.arange(50.0, 5000.0, SPACING_MS)
    blocks = MP2RAGESimulator(**PROTOCOL).simulate(T1=T1, inv_efficiency=0.96)

    with pytest.raises(ValueError, match="single column"):
        LookupTable().fit(signals=blocks, parameters=T1[:, None])


def _unified_of(signals):
    """The MP2RAGE unified image, as a combination a table can be given."""
    return (signals[..., 0] * signals[..., 1]) / signals.square().sum(-1)
