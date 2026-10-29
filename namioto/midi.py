# SPDX-License-Identifier: AGPL-3.0-only
"""MIDI files: reading them into notes and channels, and writing those back out.

Qt-free on purpose, like project.py and settings.py. mido is a file codec here and nothing else: no
port is ever opened, so a machine with no sound card and a test read and write the same.

A file's track chunks are not kept: a channel is what a note plays on, so every channel with notes
becomes one entry, and one MTrk holding several channels is read as several. Names ride on a track
chunk, not on a channel, so none is read or written - they belong to the project file.

Notes are stored in seconds, as everywhere else in the project, so the tempo map of a file is walked
on the way in, kept as the grid's map, and written back out as the same set of tempo changes. A
file with one tempo is a map with one point, so nothing about the single-tempo case changes.

`settings.midi.wavetone` (on by default) drops the leading bar WaveTone's own export writes in front
of every note and adds it back on the way out; a file whose notes start before that bar did not come
from WaveTone and is read as it stands.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mido

from namioto import project
from namioto.channels import CHANNEL_COUNT, Channel
from namioto.tempo_map import TempoMap, TempoPoint

SUFFIXES = (".mid", ".midi")
PPQ = 480  # the beat grid, the standard division every DAW reads
VELOCITY = 100  # a note carries no velocity of its own, so every one is written alike
DEFAULT_VOLUME = 100  # MIDI's own channel volume when the file says nothing
LEAD_IN_BEATS = 4  # WaveTone's export starts one bar late; see `read` and `write`


def looks_like_midi(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUFFIXES


@dataclass(frozen=True)
class Imported:
    """What a MIDI file holds, in the project's terms."""

    channels: tuple[Channel, ...]
    notes: tuple[project.Note, ...]
    bpm: float
    tempo_changes: int = 0  # tempo events past the first
    tempo_map: TempoMap = TempoMap()  # the file's whole tempo map, in the grid's beats
    dropped: int = 0  # note events that never made a note: no pair, or no length
    left_out: tuple[int, ...] = ()  # channels past the limit


def read(path: str | Path, *, wavetone: bool = False, limit: int = CHANNEL_COUNT) -> Imported:
    """Read a MIDI file into channels and notes.

    Raises OSError, EOFError or ValueError for a file that is not MIDI, or is cut short.
    """
    return _read(_load(path), wavetone=wavetone, limit=limit)


def read_bytes(data: bytes, *, wavetone: bool = False, limit: int = CHANNEL_COUNT) -> Imported:
    """Read the same MIDI format from an in-memory export."""
    try:
        mid = mido.MidiFile(file=io.BytesIO(data), charset="utf-8")
    except UnicodeDecodeError:
        mid = mido.MidiFile(file=io.BytesIO(data), charset="latin1")
    return _read(mid, wavetone=wavetone, limit=limit)


def _read(mid: mido.MidiFile, *, wavetone: bool, limit: int) -> Imported:
    ppq = max(1, int(mid.ticks_per_beat))
    tempos = _tempo_map(mid)
    lead_in = LEAD_IN_BEATS * ppq if wavetone else 0

    events: list[tuple[int, int, int, int]] = []  # (tick, channel, pitch, velocity), 0 meaning off
    programs: dict[int, dict[int, int]] = {}
    volumes: dict[int, dict[int, int]] = {}
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type in ("note_on", "note_off"):
                velocity = message.velocity if message.type == "note_on" else 0
                events.append((tick, message.channel, message.note, velocity))
            elif message.type == "program_change":
                programs.setdefault(message.channel, {})[tick] = message.program
            elif message.type == "control_change" and message.control == 7:
                volumes.setdefault(message.channel, {})[tick] = message.value

    # an off sorts before an on of the same tick, so a note ending where the next one starts pairs
    events.sort(key=lambda event: (event[0], event[3]))
    if lead_in and min((tick for tick, _c, _p, velocity in events if velocity > 0), default=0) < lead_in:
        lead_in = 0  # a file whose notes start before that bar is not one WaveTone wrote
    pending: dict[tuple[int, int], list[int]] = {}
    played: list[tuple[int, int, int, int]] = []  # (channel, start, end, pitch) in file ticks
    dropped = 0
    for tick, channel, pitch, velocity in events:
        key = (channel, pitch)
        if velocity > 0:
            pending.setdefault(key, []).append(tick)
            continue
        queue = pending.get(key)
        if not queue:
            dropped += 1
            continue
        start = max(0, queue.pop(0) - lead_in)
        end = max(0, tick - lead_in)
        if end <= start:
            dropped += 1
            continue
        played.append((channel, start, end, pitch))
    dropped += sum(len(queue) for queue in pending.values())
    played.sort(key=lambda note: (note[1], note[3], note[0]))  # by start, then pitch, then channel

    present = sorted({channel for channel, _start, _end, _pitch in played})
    kept, left_out = present[:limit], tuple(present[limit:])

    channels: list[Channel] = []
    for channel in kept:
        first = min(start for source, start, _end, _pitch in played if source == channel)
        channels.append(
            Channel(
                channel=channel,
                program=_before(programs.get(channel, {}), first, 0),
                volume=_before(volumes.get(channel, {}), first, DEFAULT_VOLUME),
            )
        )
    notes = tuple(
        project.Note(
            _seconds(tempos, start, ppq),
            _seconds(tempos, end, ppq) - _seconds(tempos, start, ppq),
            pitch,
            channel,
        )
        for channel, start, end, pitch in played
        if channel in kept
    )
    return Imported(
        channels=tuple(channels),
        notes=notes,
        bpm=_bpm(tempos),
        tempo_changes=len(tempos) - 1,
        tempo_map=_read_tempo_map(tempos, ppq, lead_in),
        dropped=dropped,
        left_out=left_out,
    )


def write(path: str | Path, channels, notes, tempo: TempoMap | float, *, wavetone: bool = False) -> Path:
    """Write channels and notes, both in seconds, out as a MIDI file, on the project's own tempo.

    `tempo` is the beat grid's map, or a single BPM for a score that keeps one. One MTrk per channel,
    the first of them carrying the conductor's tempo changes and time signature, so the notes start
    on track 0; a channel's name is not written, since a name belongs to this project and not to a
    MIDI channel. A note is written nearest the tick its time names, so one drawn on the roll's grid
    lands exactly on the tick that grid stands for and one taken from the audio keeps the time it
    has.
    """
    return _save(path, _build(channels, notes, tempo, wavetone=wavetone))


def to_bytes(channels, notes, tempo: TempoMap | float, *, wavetone: bool = False) -> bytes:
    """Export MIDI without requiring an intermediate file."""
    stream = io.BytesIO()
    _build(channels, notes, tempo, wavetone=wavetone).save(file=stream)
    return stream.getvalue()


def _build(channels, notes, tempo: TempoMap | float, *, wavetone: bool) -> mido.MidiFile:
    ppq = PPQ
    tempo_map = tempo if isinstance(tempo, TempoMap) else TempoMap.single(float(tempo))
    lead_in = LEAD_IN_BEATS * ppq if wavetone else 0

    mid = mido.MidiFile(type=1, ticks_per_beat=ppq, charset="utf-8")
    changes = [(round(point.beat * ppq), mido.bpm2tempo(point.bpm)) for point in tempo_map.points]

    for entry in channels:
        notes_here: list[tuple[int, int, int]] = []  # (tick, 1 for on, pitch)
        for note in notes:
            if note.channel != entry.channel:
                continue
            start = _tick(note.start, tempo_map, ppq)
            end = _tick(note.start + note.duration, tempo_map, ppq)
            notes_here.append((start, 1, note.pitch))
            notes_here.append((max(end, start + 1), 0, note.pitch))
        notes_here.sort()  # a note ending where the next one starts lets go of the pitch first

        # one stream per track, absolute ticks, sorted: the conductor's changes and the notes of
        # this channel share track 0, so they have to be interleaved rather than laid one after the other
        stream: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
        if not mid.tracks:
            stream.extend((tick, 0, mido.MetaMessage("set_tempo", tempo=tempo)) for tick, tempo in changes)
            stream.append((0, 0, mido.MetaMessage("time_signature", numerator=4, denominator=4)))
        stream.append((0, 1, mido.Message("program_change", channel=entry.channel, program=entry.program)))
        stream.append((0, 1, mido.Message("control_change", channel=entry.channel, control=7, value=entry.volume)))
        stream.extend(
            (
                tick + lead_in,
                2,
                mido.Message(
                    "note_on",
                    channel=entry.channel,
                    note=pitch,
                    velocity=VELOCITY if kind else 0,
                ),
            )
            for tick, kind, pitch in notes_here
        )
        stream.sort(key=lambda item: (item[0], item[1]))

        written = mido.MidiTrack()
        previous = 0
        for tick, _order, message in stream:
            message.time = tick - previous
            written.append(message)
            previous = tick
        mid.tracks.append(written)
    if not mid.tracks:
        conductor = mido.MidiTrack()
        previous = 0
        for tick, tempo in changes:
            conductor.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - previous))
            previous = tick
        conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        mid.tracks.append(conductor)  # a file with no channel still needs the tempo
    return mid


def _tick(seconds: float, tempo: TempoMap, ppq: int) -> int:
    """The tick a time names: the beat the map gives it, on the file's own division."""
    return int(round(tempo.beats_at(seconds) * ppq))


def _load(path: str | Path) -> mido.MidiFile:
    try:
        return mido.MidiFile(path, charset="utf-8")
    except UnicodeDecodeError:
        return mido.MidiFile(path, charset="latin1")  # a file whose text predates anything else


def _tempo_map(mid: mido.MidiFile) -> list[tuple[int, int]]:
    """(tick, microseconds per beat) of every tempo the file sets, in order."""
    points: list[tuple[int, int]] = []
    tick = 0
    for message in mido.merge_tracks(mid.tracks):
        tick += message.time
        if message.type == "set_tempo":
            points.append((tick, message.tempo))
    if not points or points[0][0] != 0:
        return [(0, mido.bpm2tempo(120.0)), *points]  # MIDI's own default, for the opening stretch
    return points


def _seconds(tempos: list[tuple[int, int]], tick: int, ppq: int) -> float:
    """Seconds from the file's origin to `tick`, through every tempo change in between."""
    seconds = 0.0
    last, tempo = tempos[0]
    for point, value in tempos[1:]:
        if point >= tick:
            break
        seconds += (point - last) * tempo / 1e6 / ppq
        last, tempo = point, value
    return seconds + (tick - last) * tempo / 1e6 / ppq


def _bpm(tempos: list[tuple[int, int]]) -> float:
    return round(mido.tempo2bpm(tempos[0][1]), 1)


def _read_tempo_map(tempos: list[tuple[int, int]], ppq: int, lead_in: int) -> TempoMap:
    """The file's tempo events as the grid's map, with the lead-in bar taken off the beats."""
    points = []
    for tick, microseconds in tempos:
        beat = max(0.0, (tick - lead_in) / ppq)
        points.append(TempoPoint(beat, mido.tempo2bpm(microseconds)))
    return TempoMap(tuple(points))


def _before(values: dict[int, int], tick: int, default: int) -> int:
    """What a controller or a program was set to just before `tick`."""
    earlier = [point for point in values if point <= tick]
    return values[max(earlier)] if earlier else default


def _save(path: str | Path, mid: mido.MidiFile) -> Path:
    """Write the file in one go, the way a project or a settings file is written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".tmp")
    try:
        os.close(handle)
        mid.save(name)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return target
