# SPDX-License-Identifier: AGPL-3.0-only
"""The project file: the notes, the audio they were drawn over, and the values that go with them.

Plain UTF-8 JSON, so a project can be read, diffed and edited by hand; the only thing that would need
a binary container is the spectrum, and reanalysing a five-minute file takes about a second.

Qt-free on purpose. Notes are stored in seconds and rounded to a tenth of a millisecond: seconds are
what the editor anchors them to, so a different tempo moves the grid, not the notes.

`ProjectSettings` holds the values that describe a song - how it is analysed, drawn and played back -
and the table below is their single source of truth. They are the project's alone: the program's own
preferences are in `namioto.settings`, and the window's state in `namioto.state`. A field marked
`reuse` is the exception - its last value says more about the user than about the song - so the
window keeps every change the user makes to one in `namioto.state`, and the next document starts
from it.

`Lyrics` carries the `.krc` text itself (the baseline), the aligned times keyed by a hash of that
text, and the mode. External text editing uses a temporary working copy; the project remains the
source of truth.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from namioto import params
from namioto.analysis.choices import CHANNEL_MODES
from namioto.channels import CHANNEL_COUNT, Channel, valid_color
from namioto.karaoke.operations import MAPPING_VERSION, Drop, Match, Merge, Operation, SoundRef
from namioto.params import Field, Section
from namioto.utils import write_json

FORMAT = "namioto"
VERSION = 1
SUFFIX = ".nto"
NOTE_DECIMALS = 4
LYRIC_MODES = ("edit", "read")
DIVISIONS = ("beats", "seconds")

PROJECT_SECTIONS: tuple[Section, ...] = (
    Section(
        "analysis",
        "Analysis",
        (
            Field(
                "channels",
                "choice",
                "mono",
                "Channels",
                "Which channels the analysis reads",
                choices=CHANNEL_MODES,
                reuse=True,
            ),
            Field(
                "t_num",
                "float",
                40.0,
                "Frames/s",
                "Analysis frames per second: the time resolution of the spectrum",
                low=1,
                high=200,
                decimals=2,
                reuse=True,
            ),
            Field(
                "fft_points",
                "int",
                8192,
                "FFT points",
                "Window size of the analysis: the frequency resolution",
                low=256,
                high=32768,
                step=256,
                reuse=True,
            ),
            Field(
                "a4",
                "float",
                440.0,
                "A4 (Hz)",
                "Frequency of A4, followed by both the analysis bands and the played notes",
                low=400,
                high=480,
                step=0.5,
                decimals=1,
            ),
        ),
    ),
    Section(
        "spectrum",
        "Spectrum",
        (
            Field(
                "gain",
                "float",
                240.0,
                "Gain",
                "Energy it takes for the spectrum to reach full red",
                low=10,
                high=600,
                step=1,
                reuse=True,
            ),
            Field(
                "contrast",
                "float",
                1.0,
                "Contrast",
                "Exponent applied to the spectrum's energy",
                low=0.2,
                high=4.0,
                step=0.1,
                decimals=1,
                reuse=True,
            ),
        ),
    ),
    Section(
        "playback",
        "Playback",
        (
            Field(
                "audio_volume",
                "int",
                80,
                "Audio volume",
                "Starting volume of the analysed audio",
                low=0,
                high=100,
                suffix="%",
                reuse=True,
            ),
            Field(
                "midi_volume",
                "int",
                80,
                "MIDI volume",
                "Starting volume of the note playback",
                low=0,
                high=100,
                suffix="%",
                reuse=True,
            ),
            Field(
                "speed",
                "float",
                1.0,
                "Speed",
                "Playback speed in 5% steps, 0.10x to 2.00x; the pitch is left alone",
                low=0.1,
                high=2.0,
                step=0.05,
                decimals=2,
                suffix="x",
            ),
        ),
    ),
    Section(
        "editor",
        "Editor",
        (
            Field("snap", "float", 0.5, "Snap", "Snap grid for the pen tool", low=0.0625, high=4.0, reuse=True),
            Field(
                "division",
                "choice",
                "beats",
                "Division",
                "What the ruler's lower row and the drawn grid lines divide by",
                choices=DIVISIONS,
                reuse=True,
            ),
            Field(
                "grid_offset_ms",
                "int",
                0,
                "Grid offset (ms)",
                "Shifts the drawn grid lines by this many ms; - left, + right, playback untouched",
                low=-500,
                high=500,
            ),
            Field(
                "zoom_x",
                "float",
                48.0,
                "Zoom x",
                "Pixels per beat at startup",
                low=12,
                high=900,
                decimals=1,
                reuse=True,
            ),
            Field(
                "zoom_y",
                "float",
                16.0,
                "Zoom y",
                "Pixels per semitone row at startup",
                low=8,
                high=64,
                decimals=1,
                reuse=True,
            ),
        ),
    ),
    Section(
        "tempo",
        "Tempo",
        (
            Field(
                "bpm",
                "float",
                120.0,
                "Tempo",
                "Tempo of the beat grid at startup",
                low=20,
                high=300,
                step=0.1,
                decimals=1,
            ),
        ),
    ),
    Section(
        "view",
        "View",
        (
            Field("center_x", "float", 8.0, "View center x", "", low=0.0, high=10000.0, decimals=1),
            Field("center_y", "float", 48.0, "View center y", "", low=0.0, high=88.0, decimals=1),
        ),
    ),
)

ProjectSettings = params.build(PROJECT_SECTIONS)
FIELD_SPECS = ProjectSettings.__field_specs__


def default_settings(remembered: Mapping | None = None) -> ProjectSettings:
    """The values a document starts from when the file names none of its own.

    `remembered` is what the window kept of the last document; only the fields marked reusable take
    from it, and every other one keeps the table's default.
    """
    settings = ProjectSettings()
    if not isinstance(remembered, Mapping):
        return settings
    for section in PROJECT_SECTIONS:
        fields = tuple(item for item in section.fields if item.reuse)
        values = remembered.get(section.name)
        if not fields or not isinstance(values, Mapping):
            continue
        for name, value in params.coerce_values(fields, values).items():
            setattr(getattr(settings, section.name), name, value)
    return settings


def remember(remembered: dict[str, dict], section: str, name: str, value: Any) -> None:
    """Keep one reusable field the user changed themselves, checked, for the next document."""
    remembered.setdefault(section, {})[name] = params.coerce(FIELD_SPECS[(section, name)], value)


def remember_changes(remembered: dict[str, dict], before: ProjectSettings, after: ProjectSettings) -> bool:
    """Keep every reusable field that differs between two documents, and say whether one did.

    Only what the user changed carries over: a field left at the value an earlier document or a
    dialog filled in says nothing about how they want the next one.
    """
    changed = False
    for section in PROJECT_SECTIONS:
        for item in section.fields:
            if not item.reuse:
                continue
            value = params.get_value(after, section.name, item.name)
            if value != params.get_value(before, section.name, item.name):
                remember(remembered, section.name, item.name, value)
                changed = True
    return changed


def settings_from_dict(data: Any) -> ProjectSettings:
    """The project-scoped values a parsed file holds, with every one checked as it is read."""
    settings = ProjectSettings()
    if not isinstance(data, dict):
        return settings
    for section in PROJECT_SECTIONS:
        for name, value in params.coerce_values(section.fields, data.get(section.name)).items():
            setattr(getattr(settings, section.name), name, value)
    return settings


def to_settings_dict(settings: ProjectSettings) -> dict[str, dict]:
    """The project-scoped values by section, as the project file spreads them over its top level."""
    return {
        section.name: {item.name: getattr(getattr(settings, section.name), item.name) for item in section.fields}
        for section in PROJECT_SECTIONS
    }


class Note(NamedTuple):
    """A note as the file holds it: when it starts and how long it lasts in seconds, its pitch, the
    MIDI channel it plays on, and the stable id a saved mapping names it by (0 for an old file)."""

    start: float
    duration: float
    pitch: int
    channel: int = 0
    id: int = 0


class Lyrics(NamedTuple):
    """A `.krc` as the project keeps it: the text itself, the times made from it, and the mapping.

    The text is the baseline and the source of truth for temporary editing copies and exports.
    `key` is the hash of the text the times were made from, so a changed text invalidates them;
    `raw` holds one `(onset, raw_length)` in seconds per Sound, `None` where none was found; `scores`
    holds the aligner's own score per sound beside it; `flagged` holds one bool per line, true where
    the aligner itself doubted the times it made; `problems` holds the aligner's line-level
    complaints. `mode` is `edit` while the aligner's times lay the sounds out, or `read` while the
    `.krc`'s own `.N` and groups do. Each line's KRC track names its MIDI channel; only channel 0
    has an edit mapping. `operations` is that mapping, and `version` the algorithm that suggested it.
    """

    text: str = ""
    key: str = ""
    model: str = ""
    mode: str = "edit"
    raw: tuple[tuple[tuple[float | None, float | None], ...], ...] = ()
    flagged: tuple[bool, ...] = ()
    scores: tuple[tuple[float | None, ...], ...] = ()
    problems: tuple[tuple[str, ...], ...] = ()
    operations: tuple = ()
    version: int = MAPPING_VERSION
    use_constraints: bool = False


@dataclass(frozen=True)
class Project:
    settings: ProjectSettings = field(default_factory=ProjectSettings)  # how the song is analysed, drawn and played
    audio: str = ""
    channels: tuple[Channel, ...] = ()
    notes: tuple[Note, ...] = ()
    next_id: int = 1
    lyrics: Lyrics | None = None


def looks_like_project(path: str | Path) -> bool:
    return Path(path).suffix.lower() == SUFFIX


def resolve_audio(project_path: str | Path, stored: str) -> Path | None:
    """Where a project's audio is: beside the project file first, then where it was saved from."""
    if not stored:
        return None
    path = Path(stored).expanduser()
    return path if path.is_absolute() else Path(project_path).parent / path


def store_audio(project_path: str | Path, audio: str | Path | None) -> str:
    """How a project should record its audio: relative to the project when that still finds it."""
    if not audio:
        return ""
    target = Path(audio).expanduser().resolve()
    try:
        return str(target.relative_to(Path(project_path).expanduser().resolve().parent))
    except ValueError:
        return str(target)  # another drive or an unrelated tree: the absolute path still works


def to_dict(project: Project) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        **to_settings_dict(project.settings),
        "audio": project.audio,
        "channels": [_channel_dict(channel) for channel in project.channels],
        "notes": [
            {
                "start": round(note.start, NOTE_DECIMALS),
                "duration": round(note.duration, NOTE_DECIMALS),
                "pitch": note.pitch,
                "channel": note.channel,
                **({"id": note.id} if note.id else {}),
            }
            for note in project.notes
        ],
        "note_id": project.next_id,
        "lyrics": _lyrics_dict(project.lyrics),
    }


def _lyrics_dict(lyrics: Lyrics | None) -> dict | None:
    if lyrics is None:
        return None
    return {
        "text": lyrics.text,
        "key": lyrics.key,
        "model": lyrics.model,
        "mode": lyrics.mode,
        "raw": [[list(sound) for sound in line] for line in lyrics.raw],
        "scores": [list(row) for row in lyrics.scores],
        "flagged": list(lyrics.flagged),
        "problems": [list(row) for row in lyrics.problems],
        "operations": [_operation_dict(operation) for operation in lyrics.operations],
        "version": lyrics.version,
        "use_constraints": lyrics.use_constraints,
    }


def _operation_dict(operation: Operation) -> dict:
    if isinstance(operation, Match):
        return {
            "kind": "match",
            "sound": [operation.sound.line, operation.sound.index],
            "notes": list(operation.notes),
            "confirmed": operation.confirmed,
        }
    if isinstance(operation, Merge):
        return {
            "kind": "merge",
            "sounds": [[ref.line, ref.index] for ref in operation.sounds],
            "note": operation.note,
            "confirmed": operation.confirmed,
        }
    return {
        "kind": "drop",
        "sound": [operation.sound.line, operation.sound.index],
        "confirmed": operation.confirmed,
    }


def _operation(value: Any) -> Operation | None:
    """One stored operation, or None when it is malformed; a confirmed one is never padded over."""
    if not isinstance(value, dict):
        return None
    confirmed = value.get("confirmed") is True
    try:
        if value.get("kind") == "match":
            ref = value["sound"]
            return Match(SoundRef(int(ref[0]), int(ref[1])), tuple(int(note) for note in value["notes"]), confirmed)
        if value.get("kind") == "merge":
            refs = tuple(SoundRef(int(ref[0]), int(ref[1])) for ref in value["sounds"])
            return Merge(refs, int(value["note"]), confirmed)
        if value.get("kind") == "drop":
            ref = value["sound"]
            return Drop(SoundRef(int(ref[0]), int(ref[1])), confirmed)
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return None


def _operations(value: Any) -> tuple:
    if not isinstance(value, list):
        return ()
    found = [_operation(entry) for entry in value]
    return tuple(operation for operation in found if operation is not None)


def _channel_dict(channel: Channel) -> dict:
    return {
        "name": channel.name,
        "color": channel.color,
        "channel": channel.channel,
        "program": channel.program,
        "volume": channel.volume,
        "mute": channel.mute,
        "visible": channel.visible,
        "lock": channel.lock,
    }


def _audio(value: Any) -> str:
    return str(value).strip()[: params.TEXT_LIMIT] if isinstance(value, str) else ""


def _note(entry: Any) -> Note | None:
    if not isinstance(entry, dict):
        return None
    start, duration, pitch = entry.get("start"), entry.get("duration"), entry.get("pitch")
    numbers = (start, duration, pitch)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numbers):
        return None
    if not all(math.isfinite(value) for value in numbers) or duration <= 0:
        return None
    channel = entry.get("channel", 0)
    if isinstance(channel, bool) or not isinstance(channel, int):
        channel = 0
    identifier = entry.get("id", 0)
    if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
        identifier = 0
    # a channel past the sixteen is a broken note, not a broken file: it plays on the last channel
    return Note(
        max(0.0, float(start)),
        float(duration),
        int(round(pitch)),
        min(max(channel, 0), CHANNEL_COUNT - 1),
        identifier,
    )


def _channel(entry: Any) -> Channel:
    if not isinstance(entry, dict):
        return Channel()

    def whole(value, low: int, high: int, default: int) -> int:
        ok = not isinstance(value, bool) and isinstance(value, int) and low <= value <= high
        return value if ok else default

    booleans = (
        entry.get("mute", False),
        entry.get("visible", True),
        entry.get("lock", False),
    )
    name = entry.get("name")
    return Channel(
        name=name.strip()[: params.TEXT_LIMIT] if isinstance(name, str) else "",
        color=valid_color(entry.get("color")),
        channel=whole(entry.get("channel", 0), 0, CHANNEL_COUNT - 1, 0),
        program=whole(entry.get("program", 0), 0, 127, 0),
        volume=whole(entry.get("volume", 100), 0, 127, 100),
        mute=booleans[0] if isinstance(booleans[0], bool) else False,
        visible=booleans[1] if isinstance(booleans[1], bool) else True,
        lock=booleans[2] if isinstance(booleans[2], bool) else False,
    )


def _channels(value: Any) -> tuple[Channel, ...]:
    if not isinstance(value, list):
        return ()
    unique: dict[int, Channel] = {}
    for entry in value:
        channel = _channel(entry)
        unique.setdefault(channel.channel, channel)
    return tuple(unique[number] for number in sorted(unique))


def _notes(value: Any) -> tuple[Note, ...]:
    if not isinstance(value, list):
        return ()
    notes: list[Note] = []
    dropped = 0
    for entry in value:
        note = _note(entry)
        if note is None:
            dropped += 1
        else:
            notes.append(note)
    if dropped:
        warnings.warn(f"{dropped} of the notes in the project could not be read and were left out", stacklevel=3)
    return tuple(notes)


def _lyrics(value: Any) -> Lyrics | None:
    if not isinstance(value, dict):
        return None
    key, raw = value.get("key"), value.get("raw")
    model, text, mode = value.get("model"), value.get("text"), value.get("mode")
    flagged = value.get("flagged")
    if not isinstance(key, str) or not isinstance(raw, list):
        return None
    rows: list[tuple[tuple[float | None, float | None], ...]] = []
    for line in raw:
        if not isinstance(line, list):
            return None
        row = []
        for span in line:
            if span is None:
                row.append((None, None))
                continue
            if not isinstance(span, list) or len(span) != 2:
                return None
            if any(
                value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))) for value in span
            ):
                return None
            if any(value is not None and not math.isfinite(value) for value in span):
                return None
            row.append((None if span[0] is None else float(span[0]), None if span[1] is None else float(span[1])))
        rows.append(tuple(row))
    return Lyrics(
        text=text if isinstance(text, str) else "",
        key=key,
        model=model if isinstance(model, str) else "",
        mode=mode if mode in LYRIC_MODES else "edit",
        raw=tuple(rows),
        flagged=tuple(bool(flag) for flag in flagged) if isinstance(flagged, list) else (),
        scores=_scores(value.get("scores")),
        problems=_problems(value.get("problems")),
        operations=_operations(value.get("operations")),
        version=_version(value.get("version")),
        use_constraints=value.get("use_constraints") is True,
    )


def _scores(value: Any) -> tuple[tuple[float | None, ...], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[tuple[float | None, ...]] = []
    for row in value:
        if not isinstance(row, list):
            return ()
        found = []
        for score in row:
            bad = score is None or isinstance(score, bool) or not isinstance(score, (int, float))
            if bad or not math.isfinite(score):
                found.append(None)
            else:
                found.append(float(score))
        rows.append(tuple(found))
    return tuple(rows)


def _problems(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[tuple[str, ...]] = []
    for row in value:
        if not isinstance(row, list):
            return ()
        rows.append(tuple(str(problem) for problem in row))
    return tuple(rows)


def _version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return MAPPING_VERSION
    return value


def from_dict(data: Any) -> Project:
    """Read a project out of a parsed file, with every value checked the way the settings are."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} project")
    channels = _channels(data.get("channels")) or (Channel(),)
    notes = _notes(data.get("notes"))
    # a note on a channel the file never described gets a plain entry back, the way the roll fills one
    missing = sorted({note.channel for note in notes} - {channel.channel for channel in channels})
    if missing:
        filled = (*channels, *(Channel(channel=number) for number in missing))
        channels = tuple(sorted(filled, key=lambda channel: channel.channel))
    return Project(
        settings=settings_from_dict(data),
        audio=_audio(data.get("audio")),
        channels=channels,
        notes=notes,
        next_id=_next_id(data.get("note_id"), notes),
        lyrics=_lyrics(data.get("lyrics")),
    )


def _next_id(value: Any, notes: tuple[Note, ...]) -> int:
    """The id a new note takes: the file's own next free id, else one past the largest it holds."""
    if not isinstance(value, bool) and isinstance(value, int) and value > 0:
        return max(value, max((note.id for note in notes), default=0) + 1)
    return max((note.id for note in notes), default=0) + 1


def load(path: str | Path) -> Project:
    """Read a project. Raises OSError or ValueError, both of which say what was wrong."""
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save(project: Project, path: str | Path) -> Path:
    """Write the project out whole: a half-written file would be read as a broken one."""
    return write_json(to_dict(project), Path(path))
