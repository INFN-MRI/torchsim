"""Stating a design problem, and what solving it is allowed to do.

Two families are exercised, because one object claims to carry both: a
precision cost, which reads an acquisition's Jacobian, and an image-quality
cost, which reads the signal alone and never differentiates the tissue at all.
"""

from __future__ import annotations

import time

import pytest
import torch

from torchsim.optim import Acquisition, Bounded, SequenceDesign, crlb
from torchsim.simulators import FSESimulator, SPGRSimulator, bSSFPSimulator

TISSUE = {"T1": [800.0, 1400.0], "T2": [45.0, 120.0]}


# --- the statistic ----------------------------------------------------------


def test_the_bound_is_the_variance_a_fit_actually_reaches() -> None:
    """Checked against estimation rather than against itself.

    For a linear model the least-squares estimate attains the Cramer-Rao
    bound exactly, so the empirical spread of many fits is what the bound
    claims. A test that recomputed the inverse Fisher matrix another way would
    only be checking the algebra.
    """
    generator = torch.Generator().manual_seed(7)
    design = torch.stack(
        (torch.linspace(1.0, 2.0, 40), torch.linspace(2.0, -1.0, 40))
    )
    truth = torch.tensor([0.7, -0.3])
    noise = 0.05
    draws = 20000

    signal = truth @ design
    measured = signal + noise * torch.randn(
        (draws, 40), generator=generator
    )
    fisher = design @ design.T
    estimates = torch.linalg.solve(fisher, design @ measured.T).T

    bound = crlb(design, noise_variance=noise**2)
    torch.testing.assert_close(
        estimates.var(dim=0), bound, rtol=0.05, atol=0.0
    )


def test_a_complex_signal_counts_both_channels() -> None:
    """Noise is independent on real and imaginary, so both carry information."""
    real = torch.linspace(1.0, 2.0, 16).unsqueeze(0)
    together = torch.complex(real, real)

    assert float(crlb(together)[0]) == pytest.approx(
        0.5 * float(crlb(real)[0]), rel=1e-6
    )


def test_parameters_no_sequence_can_tell_apart_are_refused() -> None:
    """A singular Fisher matrix says the design cannot work, not that it is poor."""
    row = torch.linspace(1.0, 2.0, 12)
    with pytest.raises(torch.linalg.LinAlgError):
        crlb(torch.stack((row, 2.0 * row)))


# --- the acquisition --------------------------------------------------------


def test_an_acquisition_answers_what_the_simulator_does() -> None:
    """The tissue moves from the call to the constructor and nothing else."""
    sequence = FSESimulator(ESP=5.0, TR=3000.0, states=10)
    flip = torch.full((16,), 120.0)

    bound = Acquisition(sequence, **TISSUE).simulate(flip=flip)

    torch.testing.assert_close(
        bound, sequence.simulate(flip=flip, **TISSUE), rtol=1e-6, atol=1e-7
    )


def test_an_acquisition_keeps_a_numpy_design_tissue_in_torch() -> None:
    """A property left as NumPy would decide the answer's library.

    That would be right for a simulation and fatal for a design: the gradient
    does not survive the trip out of torch and back, so the cost would have
    nothing to differentiate.
    """
    numpy = pytest.importorskip("numpy")
    shots = Acquisition(
        FSESimulator(ESP=5.0, TR=3000.0, states=10),
        T1=numpy.asarray([800.0, 1400.0]),
        T2=numpy.asarray([45.0, 120.0]),
    )
    flip = torch.full((16,), 120.0, requires_grad=True)

    signal = shots.simulate(flip=flip)
    signal.abs().square().sum().backward()

    assert torch.is_tensor(signal)
    assert flip.grad is not None and float(flip.grad.abs().max()) > 0.0


def test_an_acquisition_reports_what_its_simulator_exposes() -> None:
    """Which is what a joint design reads to know which block is blind."""
    spgr = Acquisition(SPGRSimulator(TE=2.0, TR=10.0), T1=1000.0, T2star=40.0)
    assert "T2star" in spgr.exposes and "T2" not in spgr.exposes


# --- the problem ------------------------------------------------------------


def _precision():
    """A relative-T2-precision cost over a refocused train and two tissues."""
    shots = Acquisition(FSESimulator(ESP=5.0, TR=3000.0, states=10), **TISSUE)

    def cost(flip: torch.Tensor) -> torch.Tensor:
        _, derivative = shots.jacobian("T2", flip=flip)
        variance = crlb(derivative.unsqueeze(-2)).squeeze(-1)
        return (variance / shots.properties["T2"].square()).mean().log()

    return cost


def test_a_design_lowers_the_cost_it_was_given() -> None:
    design = SequenceDesign(
        _precision(), flip=Bounded(torch.full((16,), 120.0), 20.0, 180.0)
    )

    result = design.minimize(iterations=40, learning_rate=0.1)

    assert float(result.loss[-1]) < float(result.loss[0])
    assert result.loss.shape == (40,)


def test_the_limits_hold_at_every_step() -> None:
    """They are enforced by construction, so no iterate can be outside them."""
    design = SequenceDesign(
        _precision(), flip=Bounded(torch.full((16,), 120.0), 60.0, 170.0)
    )
    seen: list[torch.Tensor] = []

    result = design.minimize(
        iterations=15,
        learning_rate=0.5,
        callback=lambda step, values, loss: seen.append(values["flip"]),
    )

    for values in [*seen, result.parameters["flip"]]:
        assert float(values.min()) >= 60.0
        assert float(values.max()) <= 170.0


def test_a_callback_can_stop_early() -> None:
    design = SequenceDesign(_precision(), flip=torch.full((8,), 120.0))

    result = design.minimize(
        iterations=50, callback=lambda step, values, loss: step == 3
    )

    assert result.loss.shape == (4,)


def test_a_cost_returning_more_than_one_number_says_so() -> None:
    design = SequenceDesign(lambda flip: flip.square(), flip=torch.full((4,), 1.0))
    with pytest.raises(ValueError, match="one number"):
        design.minimize(iterations=1)


def test_a_design_with_nothing_to_design_says_so() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        SequenceDesign(lambda: torch.zeros(()))


def test_several_parameters_are_designed_together() -> None:
    """Each reaches the cost by name, with its own limits."""
    spgr = Acquisition(SPGRSimulator(TE=2.0), T1=1000.0, T2star=40.0)
    ssfp = Acquisition(bSSFPSimulator(TE=2.5), T1=1000.0, T2=80.0)

    def contrast(spgr_flip, ssfp_flip):
        return -(
            spgr.simulate(flip=spgr_flip, TR=10.0).abs().sum()
            + ssfp.simulate(flip=ssfp_flip, TR=5.0).abs().sum()
        )

    design = SequenceDesign(
        contrast,
        spgr_flip=Bounded(torch.full((3,), 5.0), 1.0, 30.0),
        ssfp_flip=Bounded(torch.full((3,), 20.0), 5.0, 70.0),
    )

    result = design.minimize(iterations=25, learning_rate=0.2)

    assert set(result.parameters) == {"spgr_flip", "ssfp_flip"}
    assert float(result.loss[-1]) < float(result.loss[0])


# --- the other family, and its budget ---------------------------------------


def _modulation_spread(signal: torch.Tensor) -> torch.Tensor:
    """How far the echo train's k-space modulation spreads a point.

    The second moment of the point spread function, which is what blurring is
    when the modulation is read into k-space in echo order.
    """
    spread = torch.fft.fftshift(
        torch.fft.fft(signal.abs(), dim=-1).abs().square(), dim=-1
    )
    position = torch.arange(spread.shape[-1], dtype=spread.dtype) - (
        spread.shape[-1] // 2
    )
    return (spread * position.square()).sum(-1) / spread.sum(-1)


def test_a_batch_of_trains_is_designed_in_one_simulation() -> None:
    """The shots of one acquisition, each with its own train, at once."""
    shots = Acquisition(FSESimulator(ESP=5.0, TR=1800.0, states=10), **TISSUE)

    def sharpness(flip):
        return _modulation_spread(shots.simulate(flip=flip)).mean()

    initial = torch.full((6, 48), 130.0)
    design = SequenceDesign(sharpness, flip=Bounded(initial, 20.0, 180.0))

    result = design.minimize(iterations=20, learning_rate=0.2)

    assert result.parameters["flip"].shape == (6, 48)
    assert float(result.loss[-1]) < float(result.loss[0])
    # Every shot is designed, not just the one the mean happens to favour.
    assert not torch.allclose(
        result.parameters["flip"], initial, atol=1e-3
    )


def test_an_image_quality_design_fits_in_its_budget() -> None:
    """The reason the structure is resolved once, stated as a number.

    Deliberately loose -- this is a shared machine and the point is the order
    of magnitude, not the millisecond. Unresolved, the same loop takes about
    eight times as long and would not fit.
    """
    shots = Acquisition(FSESimulator(ESP=5.0, TR=1800.0, states=10), **TISSUE)

    def sharpness(flip):
        return _modulation_spread(shots.simulate(flip=flip)).mean()

    design = SequenceDesign(
        sharpness, flip=Bounded(torch.full((8, 64), 130.0), 20.0, 180.0)
    )
    design.minimize(iterations=2, learning_rate=0.2)

    start = time.perf_counter()
    design.minimize(iterations=60, learning_rate=0.2)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, f"60 iterations took {elapsed:.2f} s"
