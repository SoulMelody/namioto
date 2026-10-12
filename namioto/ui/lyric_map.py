# SPDX-License-Identifier: AGPL-3.0-only
"""Map each channel's lyric Sounds independently, retaining global identities off the GUI thread.

Every channel uses the edit solver or faithful NOTE slots according to the session mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from PySide6.QtCore import QThread, Signal

from namioto.karaoke.operations import Match, Merge, Operation, SoundRef
from namioto.lyricmap import confidence, constraints, faithful
from namioto.lyricmap.channels import split
from namioto.lyricmap.notes import TimedNote, resolve
from namioto.lyricmap.problems import MappingError
from namioto.lyricmap.raw import Raw
from namioto.lyricmap.solver import MappingState, solve
from namioto.lyricmap.spans import sound_spans

Span = tuple[float | None, float | None]


@dataclass(frozen=True)
class LyricResult:
    """One mapping: what the strip draws, and the operations and readings it was read from."""

    lines: tuple = ()
    spans: tuple = ()
    red: tuple = ()
    raw: tuple = ()
    zero: tuple = ()
    group: tuple = ()
    mapped: tuple = ()
    operations: tuple = ()
    filtered: tuple = ()
    readings: tuple = ()
    error: str = ""
    state: MappingState | None = None
    channel_errors: tuple = ()
    channel_results: tuple = ()


def map_lyrics(
    text: str,
    lines: Sequence,
    raw: Sequence[Sequence[tuple]],
    scores: Sequence[Sequence[float | None]] = (),
    flagged: Sequence[bool] = (),
    notes: Sequence[tuple] = (),
    mode: str = "edit",
    anchors: Sequence[Operation] = (),
    previous: LyricResult | None = None,
    use_constraints: bool = False,
    notes_by_channel: dict | None = None,
) -> LyricResult:
    """The mapping for the mode in force: the aligner's times in edit, the `.krc`'s own in read.

    `raw` is one `(onset, raw_length)` per Sound, per line; `scores` the alignment confidence;
    `notes` the target channel as `(start, end, pitch, id)` tuples. A mapping that cannot be made -
    missing alignment, no notes, a broken anchor - comes back with `.error` set. `notes_by_channel`
    selects the NOTE stream for each Sound's channel. Failed faithful rows retain
    their raw times for display. `previous` permits local time-edit solves when the lyric and NOTE
    structure and conflict filtering are unchanged.
    """
    if notes_by_channel is not None:
        return _channel_result(
            text, lines, raw, scores, flagged, notes_by_channel, mode, anchors, previous, use_constraints
        )
    lines = list(lines)
    raw = [[(span[0], span[1]) for span in row] for row in raw]
    if mode == "read":
        try:
            return _read_result(text, lines, raw, notes)
        except (MappingError, ValueError) as error:
            return _failed(lines, raw, str(error))
    try:
        return _edit_result(text, lines, raw, scores, flagged, notes, anchors, previous, use_constraints)
    except (MappingError, ValueError) as error:
        return _failed(lines, raw, str(error))
    except Exception as error:  # noqa: BLE001 - the worker reports whatever the mapping raised
        return _failed(lines, raw, f"{type(error).__name__}: {error}")


def _channel_result(text, lines, raw, scores, flagged, notes, mode, anchors, previous, use_constraints):
    try:
        parts = split(text)
        owned = {ref: part.channel for part in parts.values() for refs in part.refs for ref in refs}
        if any(ref not in owned for operation in anchors for ref in operation.sounds):
            raise ValueError("an anchor names a missing lyric Sound")
        if any(len({owned[ref] for ref in operation.sounds}) != 1 for operation in anchors):
            raise ValueError("an anchor cannot cross channels")
    except ValueError as error:
        return replace(_failed(lines, raw, str(error)), operations=tuple(anchors))
    baseline = _failed(lines, raw, "")
    tables = {
        name: [list(row) for row in getattr(baseline, name)]
        for name in ("spans", "red", "raw", "zero", "group", "mapped")
    }
    operations, readings, filtered, errors, results = [], [], [], [], []
    previous_results = dict(previous.channel_results) if previous is not None else {}
    shapes = constraints.read(text) if use_constraints else {}
    group_offset = 0
    for channel, part in parts.items():
        local_anchors = part.operations(anchors)
        local = part.local
        local_shapes = {local[ref]: value for ref, value in shapes.items() if ref in local}
        if mode == "read":
            result = map_lyrics(part.text, part.lines, part.rows(raw), notes=notes.get(channel, ()), mode=mode)
        else:
            try:
                result = _edit_result(
                    part.text,
                    part.lines,
                    part.rows(raw),
                    part.rows(scores),
                    part.flags(flagged),
                    notes.get(channel, ()),
                    local_anchors,
                    previous_results.get(channel),
                    use_constraints,
                    local_shapes,
                )
            except (MappingError, ValueError) as error:
                result = _failed(part.lines, part.rows(raw), str(error))
        for name, rows in tables.items():
            for position, refs in enumerate(part.refs):
                for at, ref in enumerate(refs):
                    value = getattr(result, name)[position][at]
                    if name == "spans" and mode == "read" and value[0] is None:
                        value = rows[ref.line][ref.index]
                    if name == "group" and value >= 0:
                        value += group_offset
                    rows[ref.line][ref.index] = value
        kept = local_anchors if result.error and mode == "edit" else result.operations
        operations.extend(part.restore(op) for op in kept)
        readings.extend(result.readings)
        filtered.extend(result.filtered)
        results.append((channel, result))
        if result.error:
            errors.append((channel, result.error))
        group_offset += len(notes.get(channel, ()))
    return LyricResult(
        lines=tuple(lines),
        **{name: tuple(tuple(row) for row in rows) for name, rows in tables.items()},
        operations=tuple(operations),
        readings=tuple(readings),
        filtered=tuple(filtered),
        error="; ".join(f"Channel {channel}: {error}" for channel, error in errors),
        state=dict(results).get(0, LyricResult()).state,
        channel_errors=tuple(errors),
        channel_results=tuple(results),
    )


def _read_result(text, lines, raw, notes) -> LyricResult:
    resolved = resolve([TimedNote(*note) for note in notes])
    spans, ids = faithful.lay_out(text, resolved.stream)
    rows = _rows(lines, spans)
    empty = tuple(tuple(False for _sound in line.sounds) for line in lines)
    zero = tuple(tuple(span is None or span[0] is None for span in row) for row in rows)
    groups = tuple(tuple(-1 for _sound in line.sounds) for line in lines)
    error = "the target channel has no notes" if not resolved.stream else ""
    dropped = {ref for ref, (_run, slots) in constraints.read(text).items() if slots == 0}
    if not error and any(
        (span is None or span[0] is None) and SoundRef(index, at) not in dropped
        for index, row in enumerate(rows)
        for at, span in enumerate(row)
    ):
        error = "the faithful mapping has more NOTE slots than the target channel has notes"
    return LyricResult(
        lines=tuple(lines),
        spans=_pairs(rows),
        red=empty,
        raw=_pairs(raw),
        zero=zero,
        group=groups,
        mapped=tuple(tuple(tuple(chunk) for chunk in row) for row in ids),
        filtered=tuple(resolved.filtered),
        error=error,
    )


def _edit_result(
    text, lines, raw, scores, flagged, notes, anchors, previous, use_constraints, shapes=None
) -> LyricResult:
    rows = [_raw_row(raw[index], scores[index] if index < len(scores) else ()) for index in range(len(lines))]
    resolved = resolve([TimedNote(*note) for note in notes])
    state = previous.state if previous is not None and previous.filtered == resolved.filtered else None
    shapes = (constraints.read(text) if use_constraints else {}) if shapes is None else shapes
    operations = tuple(solve(lines, rows, resolved.stream, anchors, state, shapes))
    readings = tuple(confidence.read(lines, rows, resolved.stream, operations, flagged))
    spans = sound_spans([len(line.sounds) for line in lines], operations, resolved.stream)
    red, zero, group, mapped = _tables(lines, operations, readings, resolved.stream)
    return LyricResult(
        lines=tuple(lines),
        spans=_pairs(spans),
        red=red,
        raw=_pairs(raw),
        zero=zero,
        group=group,
        mapped=mapped,
        operations=operations,
        filtered=tuple(resolved.filtered),
        readings=readings,
        state=MappingState.capture(lines, rows, resolved.stream, anchors, operations),
    )


def _failed(lines, raw, message: str) -> LyricResult:
    empty = tuple(tuple(False for _sound in line.sounds) for line in lines)
    groups = tuple(tuple(-1 for _sound in line.sounds) for line in lines)
    return LyricResult(
        lines=tuple(lines),
        spans=_pairs(
            [
                [(onset, onset + length if onset is not None and length is not None else None) for onset, length in row]
                for row in raw
            ]
        ),
        red=empty,
        raw=_pairs(raw),
        zero=empty,
        group=groups,
        mapped=tuple(tuple(() for _sound in line.sounds) for line in lines),
        error=message,
    )


def _pairs(rows) -> tuple:
    """Rows of spans as immutable two-tuples, `None` kept as `(None, None)`."""
    return tuple(tuple((None, None) if span is None else (span[0], span[1]) for span in row) for row in rows)


def _raw_row(spans: Sequence[tuple], scores: Sequence[float | None]) -> list[Raw]:
    return [
        Raw(onset, length, scores[index] if index < len(scores) else None)
        for index, (onset, length) in enumerate(spans)
    ]


def _rows(lines, spans) -> list[list[Span]]:
    """The faithful spans as drawable tuples, one row per line, matching the Sounds."""
    rows: list[list[Span]] = []
    for line, row in zip(lines, spans, strict=True):
        found: list[Span] = []
        for index in range(len(line.sounds)):
            span = row[index] if index < len(row) else None
            found.append((None, None) if span is None else (span[0], span[1]))
        rows.append(found)
    return rows


def _tables(lines, operations, readings, stream) -> tuple:
    """The strip's doubt, grey, shared-note and note-id tables, read off the operations and readings."""
    at = {note.id: index for index, note in enumerate(stream)}
    red = [[False] * len(line.sounds) for line in lines]
    zero = [[False] * len(line.sounds) for line in lines]
    group = [[-1] * len(line.sounds) for line in lines]
    mapped: list[list[tuple[int, ...]]] = [[() for _sound in line.sounds] for line in lines]
    for operation, reading in zip(operations, readings, strict=True):
        if isinstance(operation, Match):
            red[operation.sound.line][operation.sound.index] = reading.low
            mapped[operation.sound.line][operation.sound.index] = tuple(operation.notes)
        elif isinstance(operation, Merge):
            note = at[operation.note]
            for ref in operation.sounds:
                red[ref.line][ref.index] = reading.low
                group[ref.line][ref.index] = note
                mapped[ref.line][ref.index] = (operation.note,)
        else:
            red[operation.sound.line][operation.sound.index] = reading.low
            zero[operation.sound.line][operation.sound.index] = True
    return (
        tuple(tuple(row) for row in red),
        tuple(tuple(row) for row in zero),
        tuple(tuple(row) for row in group),
        tuple(tuple(row) for row in mapped),
    )


class LyricMapper(QThread):
    mapped = Signal(object)
    failed = Signal(str)

    def __init__(self, revision, request, parent=None):
        super().__init__(parent)
        self.revision = revision
        self.request = request

    def run(self) -> None:
        try:
            result = map_lyrics(**self.request)
            self.mapped.emit((self.revision, result))
        except Exception as error:  # noqa: BLE001 - the worker reports whatever the mapping raised
            self.failed.emit(f"{type(error).__name__}: {error}")
