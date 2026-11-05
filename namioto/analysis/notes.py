# SPDX-License-Identifier: AGPL-3.0-only
"""The note algebra every extractor shares: the grid, the snap, the parts, and the MIDI file.

A transcription answers with notes - onset, offset and pitch, in seconds - and the editor wants them
on the grid it draws and, on request, in a MIDI file. That arithmetic is the same whether the notes
came from GAME's singing voice or MuScriptor's instruments, so it lives here rather than in either
of them: `game` and `muscriptor` name their models, this module names what a model found.

The grid code is GAME's (`quantize_notes` and the two fitters), taken from its MIT-licensed
`infer_onnx.py` like the rest of `namioto.analysis.game`; what is added here is `monophonic`, which
is on for a single voice and off for a polyphonic part, where a chord must keep the onsets it has.

A run answers in `Part`s rather than one flat list: a part is one instrument's share, so a
multi-instrument model has somewhere to put what it heard. GAME answers one unnamed part.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np

Note = tuple[float, float, float]  # onset and offset in seconds, then the pitch


class Part(NamedTuple):
    """One instrument's share of a transcription: its name, its General MIDI program, its notes."""

    name: str
    program: int
    notes: tuple[Note, ...]


def whole(notes: Iterable[Note], name: str = "", program: int = 0) -> list[Part]:
    """A single-voice answer as the one part it is."""
    return [Part(name, program, tuple(notes))]


def flatten(parts: Iterable[Part]) -> list[Note]:
    """Every note of every part, in the order the parts were given."""
    return [note for part in parts for note in part.notes]


def quantize_notes(
    notes: Sequence[Note], unit: float, phase: float = 0.0, monophonic: bool = True
) -> list[Note]:
    """Snap to the grid cells picked around `phase`, output strictly on the absolute grid.

    The phase only influences rounding decisions; it is never baked into the output. A note the grid
    leaves without a cell of its own has nowhere to sit and is dropped, which is what takes GAME's
    sub-grid fragments out of a quantised run.

    `monophonic` makes the line single-voiced: a note the grid put before the one before it ended
    starts where that one ended. A part that may hold a chord is left alone, so its notes keep the
    onsets they were given.
    """
    out = []
    last_time = 0.0
    for onset, oset, pitch in notes:
        onset = unit * round((onset - phase) / unit)
        oset = unit * round((oset - phase) / unit)
        if monophonic:
            onset = max(onset, last_time)
        if oset <= onset:  # what is left is shorter than a cell, so there is no note here
            continue
        out.append((onset, oset, pitch))
        last_time = oset
    return out


def estimate_grid_phase(onsets, unit: float) -> float:
    """Best global phase offset for the grid: the one with the least quantization distortion."""
    o = np.asarray(onsets, dtype=np.float64)
    if o.size == 0:
        return 0.0

    def score(phase):
        off = (o - phase) / unit
        err = off - np.round(off)
        return float(np.sum(err * err))

    coarse = np.linspace(0.0, unit, 65, endpoint=False)
    s = [score(p) for p in coarse]
    k = int(np.argmin(s))
    lo, hi = coarse[(k - 1) % len(coarse)], coarse[(k + 1) % len(coarse)]
    if lo > hi:
        lo, hi = hi, lo
    for _ in range(20):  # golden section on [lo, hi]
        a = lo + (hi - lo) * 0.382
        b = lo + (hi - lo) * 0.618
        if score(a) < score(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def estimate_grid_period(onsets, phase: float, unit: float, max_drift: float = 0.05) -> float:
    """Refine the grid period around `unit` by least squares on the snapped groups."""
    o = np.asarray(onsets, dtype=np.float64)
    if o.size < 4:
        return unit
    k = np.round((o - phase) / unit)
    for _ in range(6):
        k = np.round((o - phase) / unit)
        valid = (k > 0) & (np.abs(o - (phase + k * unit)) < unit * 0.25)
        if valid.sum() < 3:
            break
        period = float(np.sum(k[valid] * (o[valid] - phase)) / np.sum(k[valid] ** 2))
        if unit * (1 - max_drift) <= period <= unit * (1 + max_drift):
            unit = period
        else:
            break
    return unit


def quantized(
    notes: Sequence[Note],
    tempo: float,
    subdivisions: int,
    phase: float | None = None,
    fit_tempo: bool = False,
    offset: float = 0.0,
    monophonic: bool = True,
) -> tuple[list[Note], float, float]:
    """Snap the notes to the beat grid `subdivisions` per quarter note apart, and report what was used.

    `offset` is the grid's own zero - the editor's slid grid - and is written into the notes, so they
    land on the lines that are drawn; `phase`, fitted from the onsets when not given, is retained
    in the output timestamps. Returns the notes, the seconds per cell and the phase.
    """
    unit = (60.0 / tempo) / subdivisions
    onsets = [note[0] for note in notes]
    if offset:
        phase = offset  # the drawn grid is the one to snap to, not one fitted from the onsets
    elif phase is None:
        phase = estimate_grid_phase(onsets, unit)
    if fit_tempo:
        unit = estimate_grid_period(onsets, phase, unit)
    snapped = quantize_notes(notes, unit, phase, monophonic=monophonic)
    snapped = [(onset + phase, oset + phase, pitch) for onset, oset, pitch in snapped]
    return snapped, unit, phase


def quantized_parts(
    parts: Sequence[Part],
    tempo: float,
    subdivisions: int,
    phase: float | None = None,
    fit_tempo: bool = False,
    offset: float = 0.0,
    monophonic: bool = False,
) -> tuple[list[Part], float, float]:
    """`quantized` over a whole answer: one grid for every part, fitted on all of their onsets."""
    unit = (60.0 / tempo) / subdivisions
    onsets = [note[0] for part in parts for note in part.notes]
    if offset:
        phase = offset
    elif phase is None:
        phase = estimate_grid_phase(onsets, unit)
    if fit_tempo:
        unit = estimate_grid_period(onsets, phase, unit)
    snapped = [
        Part(
            part.name,
            part.program,
            tuple(
                (onset + phase, oset + phase, pitch)
                for onset, oset, pitch in quantize_notes(part.notes, unit, phase, monophonic=monophonic)
            ),
        )
        for part in parts
    ]
    return snapped, unit, phase


def save_midi(
    path: pathlib.Path, notes: Sequence[Note], tempo: float, grid: tuple[float, int] | None = None
) -> None:
    """Write the notes as a type 1 MIDI file; `grid` is (seconds per cell, ticks per cell)."""
    save_parts(path, whole(notes), tempo, grid=grid)


def save_parts(
    path: pathlib.Path,
    parts: Sequence[Part],
    tempo: float,
    grid: tuple[float, int] | None = None,
) -> None:
    """Write one track per part as a type 1 MIDI file, each track naming its program.

    A track carries the program change of its part only when the part names one: GAME's single part
    has no program of its own, and leaving the change out keeps that file what it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = []
    for index, part in enumerate(parts):
        conductor = index == 0  # the tempo belongs to the first track, and only to it
        tracks.append(_track(part, tempo, grid, conductor=conductor))
    header = bytes([0x4D, 0x54, 0x68, 0x64])  # MThd
    header += (6).to_bytes(4, "big") + (1).to_bytes(2, "big")
    header += len(tracks).to_bytes(2, "big") + (480).to_bytes(2, "big")
    path.write_bytes(header + b"".join(tracks))


def _track(part: Part, tempo: float, grid: tuple[float, int] | None, conductor: bool) -> bytes:
    track_data = bytearray()
    if conductor:
        track_data += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + round(60_000_000 / tempo).to_bytes(3, "big")
    if part.program:
        track_data += _vlq(0) + bytes([0xC0, part.program & 0x7F])
    last_time = 0
    for onset, oset, pitch in part.notes:
        if grid is not None:
            seconds, ticks = grid
            onset_ticks = round(onset / seconds) * ticks
            offset_ticks = round(oset / seconds) * ticks
        else:
            onset_ticks = round(onset * tempo * 8)
            offset_ticks = round(oset * tempo * 8)
        midi_pitch = round(pitch)
        if offset_ticks <= onset_ticks:
            continue
        track_data += _vlq(onset_ticks - last_time) + bytes([0x90, midi_pitch, 100])
        track_data += _vlq(offset_ticks - onset_ticks) + bytes([0x80, midi_pitch, 64])
        last_time = offset_ticks
    track_data += _vlq(0) + bytes([0xFF, 0x2F, 0x00])
    return bytes([0x4D, 0x54, 0x72, 0x6B]) + len(track_data).to_bytes(4, "big") + bytes(track_data)


def _vlq(value: int) -> bytes:
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(out)