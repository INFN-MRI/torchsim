"""Reading a Pulseq file, against pypulseq's own reading and a closed form.

The sequences here are built with pypulseq and written to disk, so what the
reader is asked for -- the flip angle, the echo time, which sample sits at
k = 0 -- is checked twice over: against pypulseq's own trajectory, and against
the spoiled gradient-echo steady state, which has never seen a sequence file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchsim import Excitation, SequenceDescription, SPGRReadout
from torchsim.model import Simulator
from torchsim.sequence import AdcRole, EventType, RfUse
from torchsim.simulators import SPGRSimulator

pp = pytest.importorskip("pypulseq", reason="reading a .seq file needs pypulseq")

FLIP_DEG = 30.0
TR_S = 20e-3
FOV_M, SAMPLES, DWELL_S = 0.22, 64, 10e-6


def _system():
    return pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
    )


def _gradient_echo(path, *, readouts: int = 1):
    """Write a spoiled gradient echo, one or two readouts to the repetition.

    With two, the second is played on the same gradient as the first with no
    rewinder between them, so the trajectory runs away from the centre instead
    of turning back through it.
    """
    system = _system()
    sequence = pp.Sequence(system=system)
    pulse = pp.make_block_pulse(
        flip_angle=math.radians(FLIP_DEG),
        duration=200e-6,
        system=system,
        use="excitation",
    )
    readout = pp.make_trapezoid(
        "x", flat_area=SAMPLES / FOV_M, flat_time=SAMPLES * DWELL_S, system=system
    )
    adc = pp.make_adc(
        num_samples=SAMPLES,
        duration=readout.flat_time,
        delay=readout.rise_time,
        system=system,
    )
    prewinder = pp.make_trapezoid(
        "x", area=-readout.area / 2, duration=1e-3, system=system
    )
    played = pp.calc_duration(pulse) + pp.calc_duration(prewinder)
    played += readouts * pp.calc_duration(readout)

    sequence.add_block(pulse)
    sequence.add_block(prewinder)
    for _ in range(readouts):
        sequence.add_block(readout, adc)
    sequence.add_block(pp.make_delay(TR_S - played))

    sequence.set_definition("TRSize", len(sequence.block_durations))
    sequence.write(str(path))
    return sequence


def _echo_times_s(sequence):
    """When each readout passes closest to k = 0, on pypulseq's own trajectory."""
    k_adc, _, t_excitation, _, t_adc = sequence.calculate_kspace()
    k_adc, t_adc = np.asarray(k_adc), np.asarray(t_adc)
    windows = t_adc.reshape(-1, SAMPLES)
    centres = np.argmin((k_adc**2).sum(axis=0).reshape(-1, SAMPLES), axis=1)
    return (
        windows[np.arange(windows.shape[0]), centres] - float(t_excitation[0]),
        centres,
    )


@pytest.fixture
def gradient_echo(tmp_path):
    """One repetition of a spoiled gradient echo, written and read back."""
    path = tmp_path / "gre.seq"
    sequence = _gradient_echo(path)
    return sequence, SequenceDescription.from_pulseq(path)


def test_the_stream_is_the_blocks_the_file_lists(gradient_echo):
    """One event per block, in order, each of the kind the block declares."""
    sequence, described = gradient_echo
    assert [event.type for event in described.events] == [
        EventType.RF,
        EventType.WAIT,
        EventType.ADC,
        EventType.WAIT,
    ]
    assert described.tr_duration_us == pytest.approx(TR_S * 1e6)


def test_the_pulse_reads_back_at_the_angle_it_was_written_at(gradient_echo):
    """The flip is the envelope integrated on the raster the file declares."""
    _sequence, described = gradient_echo
    excitation = described.events[0]
    flip_rad, _phase = described.rf_definitions[0].flip_angle(
        excitation.rf_amplitude_hz, rf_raster_time_s=described.rf_raster_time_s
    )
    assert math.degrees(float(flip_rad)) == pytest.approx(FLIP_DEG, rel=1e-3)
    assert excitation.rf_use is RfUse.EXCITATION


def test_the_echo_sits_where_the_trajectory_crosses_zero(gradient_echo):
    """Against pypulseq's own trajectory, which the description never sees.

    Nothing in the file says where the echo is: the prewinder undoes half the
    readout's area, and the sample the sum passes through zero in is the one
    the timestamp is put on.
    """
    sequence, described = gradient_echo
    expected_s, _centres = _echo_times_s(sequence)
    measured_us = described.events[2].timestamp_us - described.events[0].timestamp_us

    assert measured_us * 1e-6 == pytest.approx(float(expected_s[0]), abs=DWELL_S)
    assert described.events[2].is_echo
    assert described.events[2].adc_role is AdcRole.SINGLE


def test_the_steady_state_is_the_spoiled_gradient_echo_closed_form(gradient_echo):
    """What the file plays, against a formula that has never seen a file."""

    class Gre(Simulator):
        excitation = Excitation
        readout = SPGRReadout
        states = 1

    _sequence, described = gradient_echo
    echo_time_ms = 1e-3 * (
        described.events[2].timestamp_us - described.events[0].timestamp_us
    )
    played = Gre.from_description(described, states=1, repetitions=400).simulate(
        T1=torch.tensor([830.0]), T2=torch.tensor([80.0]), M0=1.0
    )
    closed = SPGRSimulator(flip=FLIP_DEG, TR=TR_S * 1e3, TE=echo_time_ms).simulate(
        T1=torch.tensor([830.0]), T2star=torch.tensor([80.0]), M0=1.0
    )

    assert float(played.abs()) == pytest.approx(float(closed.abs()), rel=1e-3)


def test_only_the_readout_that_crosses_zero_is_an_echo(tmp_path):
    """Both readouts play the same gradient and the same ADC.

    What separates them is where the running area has got to by the time each
    is played, which makes the second a position the trajectory passes through
    rather than one it turns around at.
    """
    path = tmp_path / "two.seq"
    _gradient_echo(path, readouts=2)
    centre, outer = SequenceDescription.from_pulseq(path).adc_events

    assert centre.is_echo and not outer.is_echo
    assert centre.adc_role is AdcRole.ECHO_CENTER
    assert outer.adc_role is AdcRole.NON_CENTER


def test_recording_echoes_keeps_the_central_readout_alone(tmp_path):
    """``record`` reads the flag the trajectory put on each sample."""
    path = tmp_path / "two.seq"
    _gradient_echo(path, readouts=2)

    class Gre(Simulator):
        excitation = Excitation
        readout = SPGRReadout
        states = 1

    played = Gre.from_pulseq(path, states=1)
    tissue = dict(T1=torch.tensor([830.0]), T2=torch.tensor([80.0]), M0=1.0)
    assert played.simulate(**tissue, record="all").shape[-1] == 2
    assert played.simulate(**tissue, record="echo").shape[-1] == 1


def test_a_file_without_a_declared_repetition_is_refused(tmp_path):
    """How many blocks a repetition holds is stated, never searched for."""
    path = tmp_path / "undeclared.seq"
    _gradient_echo(path)
    path.write_text(
        "\n".join(
            line for line in path.read_text().splitlines() if "TRSize" not in line
        )
    )
    with pytest.raises(ValueError, match="TRSize"):
        SequenceDescription.from_pulseq(path)


def test_naming_a_repetition_out_of_range_is_refused(tmp_path):
    path = tmp_path / "gre.seq"
    _gradient_echo(path)
    with pytest.raises(ValueError, match="out of range"):
        SequenceDescription.from_pulseq(path, tr_index=7)
