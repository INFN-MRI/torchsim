"""Reading a Pulseq ``.seq`` file into a sequence description.

The file is parsed and its trajectory computed by :mod:`pypulseq`, which is the
reference implementation of the format. What is added here is the reading a
simulation needs and Pulseq does not write down: which pulse shapes the RF rows
share, and which ADC sample each readout passes through k = 0 in.
"""

from __future__ import annotations

__all__ = ["read_pulseq_description"]

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._description import (
    AdcRole,
    RfDefinition,
    RfUse,
    SequenceDescription,
    SequenceEvent,
    _trapezoid,
)

# Where each event kind sits in a Pulseq block row.
_RF, _ADC = 1, 5

# What Pulseq's ``use`` word is called here.
_USES = {
    "excitation": RfUse.EXCITATION,
    "refocusing": RfUse.REFOCUSING,
    "inversion": RfUse.INVERSION,
    "saturation": RfUse.SATURATION,
    "preparation": RfUse.PREPARATION,
    "other": RfUse.OTHER,
}


def _read(source: Any) -> Any:
    """Return ``source`` if it is a sequence already, or the file it names parsed."""
    if hasattr(source, "block_events"):
        return source
    import pypulseq

    sequence = pypulseq.Sequence()
    sequence.read(str(source))
    return sequence


@dataclass
class _Readout:
    """One ADC window, and where in k-space its samples run."""

    block: int
    trajectory: np.ndarray
    center: int

    @property
    def k_echo(self) -> np.ndarray:
        """Where the trajectory stands at the sample that reads the echo."""
        return self.trajectory[:, self.center]

    @property
    def swept(self) -> np.ndarray:
        """Which axes this readout moves along, as a mask over the three."""
        span = self.trajectory.max(axis=1) - self.trajectory.min(axis=1)
        return span > 1e-6 * np.maximum(1.0, np.abs(self.trajectory).max(axis=1))


def _readouts(sequence: Any) -> list[_Readout]:
    """Every ADC in the sequence, with the trajectory across its samples.

    A readout's centre is the sample where the axes it sweeps come closest to
    zero -- its own zero crossing, independent of any encode carried in from
    elsewhere in the repetition.
    """
    k_adc = np.asarray(sequence.calculate_kspace()[0], dtype=float)
    found: list[_Readout] = []
    taken = 0
    for index in range(1, len(sequence.block_durations) + 1):
        if not sequence.block_events[index][_ADC]:
            continue
        count = int(sequence.get_block(index).adc.num_samples)
        track = k_adc[:, taken : taken + count]
        taken += count
        swept = _Readout(index, track, 0).swept
        centre = (
            int(np.argmin((track[swept] ** 2).sum(axis=0)))
            if swept.any()
            else count // 2
        )
        found.append(_Readout(index, track, centre))
    return found


def _shape(pulse: Any) -> tuple[bytes, bytes]:
    """What a pulse plays, less the amplitude, phase and frequency it plays at."""
    signal = np.asarray(pulse.signal, dtype=np.complex128)
    peak = float(np.abs(signal).max()) or 1.0
    times = np.asarray(pulse.t, dtype=float)
    return np.round(signal / peak, 9).tobytes(), np.round(times, 12).tobytes()


def _number(sequence: Any, name: str) -> float | None:
    """The one number recorded under ``name``, or ``None`` if there is none."""
    declared = np.atleast_1d(sequence.get_definition(name))
    if declared.size == 0 or str(declared.flat[0]) == "":
        return None
    return float(declared.flat[0])


def _tr_window(sequence: Any, tr_size: int, tr_index: int | None) -> int:
    """Return the first block of the repetition to describe, one-based.

    The file declares how many blocks a repetition holds; which instance of it
    to read is the caller's, since a schedule whose repetitions differ -- an
    optimized echo train, a fingerprinting schedule -- has no canonical one.
    """
    blocks = len(sequence.block_durations)
    windows = blocks // tr_size
    if tr_index is not None:
        if not 0 <= tr_index < windows:
            raise ValueError(
                f"tr_index={tr_index} is out of range; the file holds "
                f"{windows} repetitions of {tr_size} blocks"
            )
        return 1 + tr_index * tr_size
    declared = _number(sequence, "TRRef")
    if declared is not None:
        return 1 + int(declared) * tr_size

    rows = np.asarray([sequence.block_events[i + 1] for i in range(windows * tr_size)])
    acquiring = [
        start
        for start in range(0, windows * tr_size, tr_size)
        if rows[start : start + tr_size, _ADC].any()
    ]
    if not acquiring:
        raise ValueError("no repetition of the declared TRSize acquires anything")
    # Compared by shape and flip: RF spoiling writes one library row per
    # repetition of the same pulse, and a phase increment is not a difference
    # in what the sequence plays.
    played = {
        tuple(
            (_shape(pulse), round(_flip(pulse), 9))
            for pulse in (
                sequence.get_block(start + offset + 1).rf
                for offset in range(tr_size)
                if rows[start + offset, _RF]
            )
        )
        for start in acquiring
    }
    if len(played) > 1:
        raise ValueError(
            "this sequence's repetitions do not play the same pulses, so which "
            "one the simulation stands for is a choice: declare it as a TRRef "
            "definition in the file, or pass tr_index="
        )
    return 1 + acquiring[0]


def _echo_and_role(
    readouts: list[_Readout], start: int, tr_size: int, acquired: list[int]
) -> tuple[dict[int, bool], dict[int, AdcRole]]:
    """Classify each ADC position in the repetition, over all its instances.

    Two questions, and they are not the same one. Whether a position *bears an
    echo* is absolute: some instance of it reaches the scan's own closest
    approach to k = 0, which is what separates the line through the centre from
    the phase-encoded ones. What its *role* is, is structural: the norm is taken
    only along the axes the readout itself sweeps, so a CPMG echo is central
    whatever line it encodes.
    """
    absolute: dict[int, float] = {}
    structural: dict[int, float] = {}
    floor, peak = np.inf, 0.0
    for readout in readouts:
        offset = (readout.block - start) % tr_size
        norm = float(np.linalg.norm(readout.k_echo))
        peak = max(peak, float(np.abs(readout.k_echo).max()))
        floor = min(floor, norm)
        absolute[offset] = min(absolute.get(offset, np.inf), norm)
        swept = float(np.linalg.norm(readout.k_echo[readout.swept]))
        structural[offset] = min(structural.get(offset, np.inf), swept)
    if not np.isfinite(floor):
        floor = 0.0
    threshold = floor + 1e-5 * peak + 1e-6
    echo = {offset: value <= threshold for offset, value in absolute.items()}

    role = dict.fromkeys(acquired, AdcRole.SINGLE)
    held = [structural[offset] for offset in acquired if offset in structural]
    if len(held) > 1 and max(held) > 0.0:
        centre_threshold = min(held) + 1e-3 * max(held)
        centred = sum(value <= centre_threshold for value in held)
        central = AdcRole.ECHO_CENTER if centred == 1 else AdcRole.SINGLE
        for offset in acquired:
            if offset in structural:
                role[offset] = (
                    central
                    if structural[offset] <= centre_threshold
                    else AdcRole.NON_CENTER
                )
    return echo, role


def _pulse(
    signal: np.ndarray, times_s: np.ndarray, raster_s: float, bandwidth_hz: float
) -> RfDefinition:
    """Return the shape of one pulse, scaled so an amplitude in radians is a flip.

    Raises
    ------
    ValueError
        If the pulse's samples are not evenly spaced. Pulseq can carry a time
        shape of its own, and a definition is played one sample per dwell, so
        an uneven pulse would be played at the wrong times rather than refused.
    """
    from ._description import rf_definition

    steps = np.diff(np.asarray(times_s, dtype=float))
    if steps.size and not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError(
            "this pulse carries a time shape of its own, and a definition is "
            "played one sample per dwell"
        )
    return rf_definition(
        np.asarray(signal, dtype=np.complex128),
        dwell_s=float(steps[0]) if steps.size else raster_s,
        bandwidth_hz=bandwidth_hz,
        rf_raster_time_s=raster_s,
    )


def _bandwidth_hz(pulse: Any) -> float:
    """The full width at half maximum of a pulse's small-tip profile, in Hz."""
    signal = np.asarray(pulse.signal, dtype=np.complex128)
    times = np.asarray(pulse.t, dtype=float)
    dwell_s = float(times[1] - times[0]) if times.size > 1 else 1.0
    size = 64 * max(signal.size, 64)
    profile = np.abs(np.fft.fftshift(np.fft.fft(signal, size)))
    peak = int(np.argmax(profile))
    above = profile >= 0.5 * profile[peak]
    low, high = peak, peak
    while low > 0 and above[low - 1]:
        low -= 1
    while high < size - 1 and above[high + 1]:
        high += 1
    return (high - low + 1) / (size * dwell_s)


def _rf_definitions(
    sequence: Any, blocks: range, raster_s: float
) -> tuple[dict[int, int], dict[int, RfDefinition]]:
    """Collapse the RF rows onto the shapes they play.

    Rows differing only in amplitude, phase or frequency -- which is what RF
    spoiling writes one of per repetition -- are one pulse played differently,
    so they share a definition.

    A pulse played under a slice-select gradient is given the bandwidth that
    gradient selects across one reference slice thickness: the excitation's,
    read as its small-tip bandwidth over its own gradient. Slice positions are
    then in units of that thickness for every pulse in the repetition, so a
    refocusing pulse that selects a wider slice than the excitation does so in
    the simulation as well.
    """
    played: dict[int, tuple[Any, float]] = {}
    for index in blocks:
        row = int(sequence.block_events[index][_RF])
        if not row or row in played:
            continue
        block = sequence.get_block(index)
        centre_s = block.rf.delay + float(block.rf.center)
        played[row] = (block.rf, _slice_select(block, centre_s))

    selective = [(pulse, gradient) for pulse, gradient in played.values() if gradient]
    reference = next(
        (
            (pulse, gradient)
            for pulse, gradient in selective
            if str(pulse.use) == "excitation"
        ),
        selective[0] if selective else None,
    )
    thickness_m = _bandwidth_hz(reference[0]) / reference[1] if reference else 0.0

    by_shapes: dict[tuple[Any, ...], int] = {}
    of_row: dict[int, int] = {}
    definitions: dict[int, RfDefinition] = {}
    for row, (pulse, gradient) in played.items():
        bandwidth_hz = gradient * thickness_m
        key = (*_shape(pulse), round(bandwidth_hz, 6))
        if key not in by_shapes:
            identifier = len(by_shapes)
            by_shapes[key] = identifier
            shape = _pulse(pulse.signal, np.asarray(pulse.t), raster_s, bandwidth_hz)
            definitions[identifier] = RfDefinition(
                **{**shape.__dict__, "id": identifier}
            )
        of_row[row] = by_shapes[key]
    return of_row, definitions


def read_pulseq_description(
    source: Any,
    *,
    tr_index: int | None = None,
    subsequence_index: int = 0,
    crusher_dephasing_rad: float = 0.0,
    voxel_size_m: float | None = None,
) -> SequenceDescription:
    """Return the event stream one repetition of a Pulseq sequence plays.

    The sequence says how many blocks a repetition holds -- the ``TRSize``
    definition a design writes -- so nothing here searches for the period.

    Parameters
    ----------
    source : str, Path or sequence
        A ``.seq`` file, or a sequence in memory with pypulseq's reading
        interface -- pypulseq's own ``Sequence``, or pypulseqpp's.
    tr_index : int, default=None
        Which repetition to describe, counted in whole ``TRSize`` windows.
        ``None`` is the ``TRRef`` definition when the file carries one, and
        otherwise to the first repetition that acquires -- which is refused
        when the repetitions play different pulses, since a simulation then has
        to be told which one it stands for.
    subsequence_index : int, default=0
        Which subsequence of a scan this is.
    crusher_dephasing_rad : float, default=0.0
        Dephasing wound across one voxel by the unbalanced gradient the
        sequence plays, in radians; diffusion and flow are read off it and are
        zero without it.
    voxel_size_m : float, default=None
        Voxel length the dephasing is wound across, in metres. ``None``
        declares no voxel, so spin velocity has no effect.

    Returns
    -------
    SequenceDescription
        One repetition, with event times in microseconds from its start. It is
        the object a description arriving from a scanner is read into.

    Raises
    ------
    ValueError
        If the file declares no ``TRSize``, if ``tr_index`` is out of range, or
        if the repetitions differ and none was named.
    """
    sequence = _read(source)
    name = source if isinstance(source, str | Path) else "this sequence"
    declared = _number(sequence, "TRSize")
    if declared is None:
        raise ValueError(
            f"{name} declares no TRSize, so how many blocks one repetition "
            "holds is not stated; write the definition from the design side"
        )
    tr_size = int(declared)
    if not 0 < tr_size <= len(sequence.block_durations):
        raise ValueError(f"{name} declares TRSize={tr_size}, which is not a window")

    raster_s = _number(sequence, "RadiofrequencyRasterTime") or 1e-6
    readouts = _readouts(sequence)
    start = _tr_window(sequence, tr_size, tr_index)
    window = range(start, start + tr_size)
    of_row, definitions = _rf_definitions(sequence, window, raster_s)

    durations = np.asarray(
        [sequence.block_durations[index] for index in range(1, start + tr_size)]
    )
    edges = np.concatenate(([0.0], np.cumsum(durations)))
    origin_s = edges[start - 1]

    acquired = [
        offset
        for offset in range(tr_size)
        if sequence.block_events[start + offset][_ADC]
    ]
    echo, role = _echo_and_role(readouts, start, tr_size, acquired)
    centres = {
        readout.block: readout.center
        for readout in readouts
        if start <= readout.block < start + tr_size
    }

    events = []
    for offset in range(tr_size):
        index = start + offset
        block = sequence.get_block(index)
        block_s = edges[index - 1] - origin_s
        if block.rf is not None:
            row = int(sequence.block_events[index][_RF])
            centre_s = block.rf.delay + float(block.rf.center)
            flip_rad = _flip(block.rf)
            events.append(
                SequenceEvent.rf(
                    1e6 * (block_s + centre_s),
                    definition_id=of_row[row],
                    use=_USES.get(str(block.rf.use), RfUse.UNKNOWN),
                    amplitude_hz=flip_rad,
                    phase_rad=float(block.rf.phase_offset),
                    frequency_hz=float(block.rf.freq_offset),
                    slice_select_gradient_hz_per_m=_slice_select(block, centre_s),
                )
            )
        elif block.adc is not None:
            centre = centres.get(index, 0)
            events.append(
                SequenceEvent.adc(
                    1e6 * (block_s + block.adc.delay + centre * block.adc.dwell),
                    role=role.get(offset, AdcRole.SINGLE),
                    phase_rad=float(block.adc.phase_offset),
                    is_echo=echo.get(offset, False),
                )
            )
        else:
            events.append(SequenceEvent.wait(1e6 * block_s))

    return SequenceDescription(
        subsequence_index=subsequence_index,
        tr_duration_us=1e6 * float(edges[start + tr_size - 1] - origin_s),
        events=tuple(events),
        rf_definitions=definitions,
        crusher_dephasing_rad=crusher_dephasing_rad,
        voxel_size_m=voxel_size_m,
        rf_raster_time_s=raster_s,
    )


def _flip(pulse: Any) -> float:
    """Return the angle one occurrence of a pulse turns, in radians.

    Integrated the way a definition reads its own envelope, so the angle stated
    here is the angle the kernels turn a spin through. The shape's own phase
    stays in the definition, and the occurrence carries only its phase offset.
    """
    area = _trapezoid(np.asarray(pulse.signal), np.asarray(pulse.t))
    return float(2.0 * np.pi * abs(area))


def _slice_select(block: Any, when_s: float) -> float:
    """The magnitude of the gradient played under a pulse's centre, in Hz/m.

    An oblique slice is selected along the vector the axes play together, so
    the magnitude of that vector is the gradient the slice is selected by.
    """
    playing = [
        _gradient_at(axis, when_s)
        for axis in (block.gx, block.gy, block.gz)
        if axis is not None
    ]
    magnitude = float(np.linalg.norm(playing)) if playing else 0.0
    return magnitude if magnitude > 1e-3 else 0.0


def _gradient_at(gradient: Any, when_s: float) -> float:
    """What one gradient event is playing at ``when_s`` from its block's start."""
    delay = float(getattr(gradient, "delay", 0.0) or 0.0)
    if gradient.type == "trap":
        rise, flat, fall = gradient.rise_time, gradient.flat_time, gradient.fall_time
        edges = np.array([0.0, rise, rise + flat, rise + flat + fall]) + delay
        return float(
            np.interp(
                when_s,
                np.concatenate(([0.0], edges)),
                np.array([0.0, 0.0, gradient.amplitude, gradient.amplitude, 0.0]),
            )
        )
    return float(np.interp(when_s, delay + np.asarray(gradient.tt), gradient.waveform))
