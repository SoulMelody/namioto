# SPDX-License-Identifier: AGPL-3.0-only
"""Map each channel's lyric Sounds independently, retaining global identities off the GUI thread.

Every channel reads faithful NOTE slots; explicit automatic mapping requests use the solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from PySide6.QtCore import Signal

from namioto.karaoke.operations import Match, Merge, Operation
from namioto.lyricmap import constraints, faithful
from namioto.lyricmap.channels import split
from namioto.lyricmap.editing import note_runs
from namioto.lyricmap.notes import TimedNote, resolve
from namioto.lyricmap.problems import MappingError
from namioto.lyricmap.raw import Raw
from namioto.lyricmap.solver import MappingState, solve
from namioto.lyricmap.spans import sound_spans
from namioto.ui.loading import BackgroundThread

Span = tuple[float | None, float | None]


@dataclass(frozen=True)
class LyricResult:
    """One mapping: what the strip draws, and the operations it was read from."""

    lines: tuple = ()
    spans: tuple = ()
    red: tuple = ()
    raw: tuple = ()
    zero: tuple = ()
    group: tuple = ()
    mapped: tuple = ()
    operations: tuple = ()
    filtered: tuple = ()
    error: str = ""
    state: MappingState | None = None
    channel_errors: tuple = ()
    channel_results: tuple = ()
    read_key: tuple = ()


def map_lyrics(
    text: str,
    lines: Sequence,
    raw: Sequence[Sequence[tuple]],
    scores: Sequence[Sequence[float | None]] = (),
    flagged: Sequence[bool] = (),
    notes: Sequence[tuple] = (),
    mode: str = "read",
    anchors: Sequence[Operation] = (),
    previous: LyricResult | None = None,
    use_constraints: bool = False,
    notes_by_channel: dict | None = None,
    solve_channels: tuple[int, ...] | None = None,
) -> LyricResult:
    """Read faithful KRC slots, or solve an explicit automatic mapping request.

    `raw` is one `(onset, raw_length)` per Sound, per line; `scores` the alignment confidence;
    `notes` the target channel as `(start, end, pitch, id)` tuples. A mapping that cannot be made -
    missing alignment, no notes, a broken anchor - comes back with `.error` set. `notes_by_channel`
    selects the NOTE stream for each Sound's channel. Uncovered faithful Sounds stay untimed.
    `solve_channels` limits an automatic request to selected channels; the others remain faithful.
    `previous` permits solver reuse when the lyric and NOTE structure are unchanged.
    """
    if notes_by_channel is not None:
        return _channel_result(
            text,
            lines,
            raw,
            scores,
            flagged,
            notes_by_channel,
            mode,
            anchors,
            previous,
            use_constraints,
            solve_channels,
        )
    lines = list(lines)
    raw = [[(span[0], span[1]) for span in row] for row in raw]
    if mode == "read":
        key = (
            text,
            tuple(lines),
            tuple(tuple(row) for row in raw),
            tuple(tuple(row) for row in scores),
            tuple(flagged),
            tuple(notes),
        )
        if previous is not None and previous.read_key == key:
            return previous
        try:
            return replace(
                _read_result(text, lines, raw, notes),
                read_key=key,
            )
        except (MappingError, ValueError) as error:
            return _failed(lines, raw, str(error))
    try:
        return _edit_result(text, lines, raw, scores, flagged, notes, anchors, previous, use_constraints)
    except (MappingError, ValueError) as error:
        return _failed(lines, raw, str(error))
    except Exception as error:  # noqa: BLE001 - the worker reports whatever the mapping raised
        return _failed(lines, raw, f"{type(error).__name__}: {error}")


def _channel_result(
    text,
    lines,
    raw,
    scores,
    flagged,
    notes,
    mode,
    anchors,
    previous,
    use_constraints,
    solve_channels,
):
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
    operations, filtered, errors, results = [], [], [], []
    previous_results = dict(previous.channel_results) if previous is not None else {}
    shapes = constraints.read(text) if use_constraints else {}
    group_offset = 0
    for channel, part in parts.items():
        local_anchors = part.operations(anchors)
        local = part.local
        local_shapes = {local[ref]: value for ref, value in shapes.items() if ref in local}
        automatic = mode != "read" and (solve_channels is None or channel in solve_channels)
        if not automatic:
            result = map_lyrics(
                part.text,
                part.lines,
                part.rows(raw),
                part.rows(scores),
                part.flags(flagged),
                notes=notes.get(channel, ()),
                mode="read",
                previous=previous_results.get(channel),
            )
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
                    if name == "group" and value >= 0:
                        value += group_offset
                    rows[ref.line][ref.index] = value
        kept = local_anchors if result.error and automatic else result.operations
        operations.extend(part.restore(op) for op in kept)
        filtered.extend(result.filtered)
        results.append((channel, result))
        if result.error:
            errors.append((channel, result.error))
        group_offset += len(notes.get(channel, ()))
    if mode == "read":
        tables["group"] = [[-1 for _sound in line.sounds] for line in lines]
        for group, run in enumerate(note_runs(text)):
            if len(run.sounds) > 1 and len({lines[ref.line].sounds[ref.index].channel for ref in run.sounds}) == 1:
                for ref in run.sounds:
                    tables["group"][ref.line][ref.index] = group
    return LyricResult(
        lines=tuple(lines),
        **{name: tuple(tuple(row) for row in rows) for name, rows in tables.items()},
        operations=tuple(operations),
        filtered=tuple(filtered),
        error="; ".join(f"Channel {channel}: {error}" for channel, error in errors),
        state=dict(results).get(0, LyricResult()).state,
        channel_errors=tuple(errors),
        channel_results=tuple(results),
    )


def _read_result(text, lines, raw, notes) -> LyricResult:
    resolved = resolve([TimedNote(*note) for note in notes], include_branches=False)
    spans, ids = faithful.lay_out(text, resolved.stream)
    rows = _rows(lines, spans)
    zero = tuple(tuple(span is None or span[0] is None for span in row) for row in rows)
    error = "the target channel has no notes" if not resolved.stream else ""
    operations = faithful.operations(text, resolved.stream)
    red, _zero, _groups, _mapped = _tables(lines, operations, resolved.stream)
    groups = [[-1 for _sound in line.sounds] for line in lines]
    for group, run in enumerate(note_runs(text)):
        if len(run.sounds) > 1:
            for ref in run.sounds:
                groups[ref.line][ref.index] = group
    return LyricResult(
        lines=tuple(lines),
        spans=_pairs(rows),
        red=red,
        raw=_pairs(raw),
        zero=zero,
        group=tuple(tuple(row) for row in groups),
        mapped=tuple(tuple(tuple(chunk) for chunk in row) for row in ids),
        operations=operations,
        filtered=tuple(resolved.filtered),
        error=error,
    )


def _edit_result(
    text, lines, raw, scores, flagged, notes, anchors, previous, use_constraints, shapes=None
) -> LyricResult:
    rows = [_raw_row(raw[index], scores[index] if index < len(scores) else ()) for index in range(len(lines))]
    resolved = resolve([TimedNote(*note) for note in notes], include_branches=False)
    state = previous.state if previous is not None and previous.filtered == resolved.filtered else None
    shapes = (constraints.read(text) if use_constraints else {}) if shapes is None else shapes
    operations = tuple(solve(lines, rows, resolved.stream, anchors, state, shapes))
    spans = sound_spans([len(line.sounds) for line in lines], operations, resolved.stream)
    red, zero, group, mapped = _tables(lines, operations, resolved.stream)
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


def _tables(lines, operations, stream) -> tuple:
    """The strip's colour, grey, shared-note and note-id tables, read off the operations."""
    at = {note.id: index for index, note in enumerate(stream)}
    red = [[False] * len(line.sounds) for line in lines]
    zero = [[False] * len(line.sounds) for line in lines]
    group = [[-1] * len(line.sounds) for line in lines]
    mapped: list[list[tuple[int, ...]]] = [[() for _sound in line.sounds] for line in lines]
    for operation in operations:
        if isinstance(operation, Match):
            mapped[operation.sound.line][operation.sound.index] = tuple(operation.notes)
        elif isinstance(operation, Merge):
            note = at[operation.note]
            for ref in operation.sounds:
                group[ref.line][ref.index] = note
                mapped[ref.line][ref.index] = (operation.note,)
        else:
            zero[operation.sound.line][operation.sound.index] = True
    return (
        tuple(tuple(row) for row in red),
        tuple(tuple(row) for row in zero),
        tuple(tuple(row) for row in group),
        tuple(tuple(row) for row in mapped),
    )


class LyricMapper(BackgroundThread):
    mapped = Signal(object)

    def __init__(self, revision, request, parent=None):
        super().__init__(parent)
        self.revision = revision
        self.request = request
        self.previous = None

    def job(self) -> None:
        request = dict(self.request, previous=self.previous) if self.request.get("mode") == "read" else self.request
        result = map_lyrics(**request)
        self.mapped.emit((self.revision, result))
