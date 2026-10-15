# SPDX-License-Identifier: AGPL-3.0-only
"""The mapping DP, with local re-solving against a previous mapping for time-only edits.

Every Sound is exactly one `match`, `merge` or `drop`, and every note is consumed exactly once, so
the cost compares each operation's raw onset against its predicted onset, plus the line's last
Sound against its onset plus reference duration. Full solves settle this over the whole song rather
than line by line. Each operation's whole cost takes its pattern weight; no fixed penalty is added. A line
boundary does not cut the note stream, so a line's last Sound may run over the rest between lines;
a `merge` stays within one line but may cross ruby containers. Confirmed operations
are hard anchors: the DP solves the stretches between them and never moves them.

Exact ties are broken the way the spec orders them: less structural complexity first, then the
solution that keeps the earlier Sound's `match`, then `match` over `merge` over `drop`. The tie is
encoded as a base-3 number over the sounds, one digit per Sound, so a whole song's worth of ties still
compares as an integer. Full solves visit `O(sounds * notes)` states, with `O(line length^2)`
merge work per starting Sound and note. Time-only edits reuse operations outside the changed lines
and their neighbours, fixing the previous NOTE boundaries. This local optimum need not be the
whole song's optimum; a full solve remains available. Invalid local boundaries expand the range
before falling back to the whole song. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from namioto.karaoke.operations import Drop, Match, Merge, Operation, SoundRef
from namioto.karaoke.sounds import SoundLine
from namioto.lyricmap.notes import Note, TimedNote
from namioto.lyricmap.problems import MappingError
from namioto.lyricmap.raw import Raw, validate
from namioto.lyricmap.weights import merge_weights


@dataclass(frozen=True)
class MappingState:
    """Immutable inputs and operations of a completed edit mapping, never loaded from disk."""

    lines: tuple[SoundLine, ...]
    raw: tuple[tuple[Raw, ...], ...]
    notes: tuple[TimedNote, ...]
    anchors: tuple[Operation, ...]
    operations: tuple[Operation, ...]

    @classmethod
    def capture(cls, lines, raw, notes, anchors, operations) -> MappingState:
        return cls(
            tuple(lines),
            tuple(tuple(row) for row in raw),
            tuple(TimedNote(note.start, note.end, note.pitch, note.id) for note in notes),
            tuple(anchors),
            tuple(operations),
        )


@dataclass(frozen=True)
class _Context:
    info: list[tuple[int, int]]
    line_start: list[int]
    line_count: list[int]
    raw_start: list[float]
    line_end: list[float]
    is_last: list[bool]
    maxm: list[int]
    merge_weights: list[tuple[float, ...]]
    notes: list
    total: int
    constraints: dict[int, tuple[int, int]]


def solve(
    lines: Sequence[SoundLine],
    raw: Sequence[Sequence[Raw]],
    notes: Sequence[Note],
    anchors: Sequence[Operation] = (),
    previous: MappingState | None = None,
    constraints: dict[SoundRef, tuple[int, int]] | None = None,
) -> list[Operation]:
    """Partition all Sounds and notes, retaining confirmed anchors.

    `constraints` holds explicit KRC run lengths and NOTE counts; session anchors override the
    runs they edit. Constrained solves use the full DP. Without constraints, `previous` permits
    an approximate local solve for time-only changes. Raises `MappingError` for impossible input.
    """
    validate(lines, raw)
    note_list = list(notes)
    context = _context(lines, raw, note_list)
    anchored = {ref for operation in anchors for ref in operation.sounds}
    for ref, shape in (constraints or {}).items():
        if not 0 <= ref.line < len(lines) or not 0 <= ref.index < len(lines[ref.line].sounds):
            raise MappingError("invalid_constraint", "a KRC constraint names a missing sound")
        run, slots = shape
        if run < 1 or slots < 0 or (run > 1 and slots != 1) or ref.index + run > len(lines[ref.line].sounds):
            raise MappingError("invalid_constraint", "a KRC constraint cannot be represented by one operation")
        if any(SoundRef(ref.line, at) in anchored for at in range(ref.index, ref.index + run)):
            continue
        index = context.line_start[ref.line] + ref.index
        context.constraints[index] = shape
        for inner in range(index + 1, index + run):
            context.constraints[inner] = (0, 0)
    if context.constraints:
        previous = None
    if context.total == 0:
        raise MappingError("no_lyric_sounds", "the lyrics have no sounds")
    if not note_list:
        raise MappingError("no_target_notes", "the target channel has no notes")
    pinned, forced = _anchors(anchors, context)
    if previous is not None:
        changed = _changed_lines(lines, raw, note_list, anchors, previous)
        if changed is not None:
            if not changed:
                return list(previous.operations)
            low, high = max(0, min(changed) - 1), min(len(lines), max(changed) + 2)
            while low > 0 or high < len(lines):
                before = [op for op in previous.operations if op.sounds[0].line < low]
                after = [op for op in previous.operations if op.sounds[0].line >= high]
                sa = context.line_start[low]
                sb = context.line_start[high] if high < len(lines) else context.total
                na = sum(op.slots for op in before)
                nb = len(note_list) - sum(op.slots for op in after)
                try:
                    local = _solve_range(context, pinned, forced, sa, sb, na, nb)
                    return before + local + after
                except MappingError:
                    low, high = max(0, low - 1), min(len(lines), high + 1)
    return _solve_range(context, pinned, forced, 0, context.total, 0, len(note_list))


def _changed_lines(lines, raw, notes, anchors, previous: MappingState) -> set[int] | None:
    if tuple(lines) != previous.lines or [(n.id, n.pitch) for n in notes] != [(n.id, n.pitch) for n in previous.notes]:
        return None
    changed = {index for index, row in enumerate(raw) if tuple(row) != previous.raw[index]}
    for operation in set(anchors) ^ set(previous.anchors):
        changed.update(ref.line for ref in operation.sounds)
    moved = {
        note.id: (old, note)
        for old, note in zip(previous.notes, notes, strict=True)
        if old.start != note.start or old.end != note.end
    }
    for operation in previous.operations:
        ids = (
            operation.notes
            if isinstance(operation, Match)
            else (operation.note,)
            if isinstance(operation, Merge)
            else ()
        )
        if any(identifier in moved for identifier in ids):
            changed.update(ref.line for ref in operation.sounds)
    for old, note in moved.values():
        low, high = min(old.start, note.start), max(old.end, note.end)
        changed.update(index for index, row in enumerate(raw) if any(low <= sound.onset <= high for sound in row))
    return changed


def _solve_range(context, pinned, forced, sa, sb, na, nb) -> list[Operation]:
    fixed = set(forced)
    operations: list[Operation] = []
    sound_at, note_at = sa, na
    for low, high, first, last, operation in pinned:
        if high <= sa or low >= sb:
            continue
        if low < sound_at or high > sb or first < note_at or last > nb:
            raise MappingError("invalid_anchor", "a confirmed operation crosses the local boundary")
        operations.extend(_segment(context, sound_at, low, note_at, first, fixed))
        operations.append(operation)
        sound_at, note_at = high, last
    operations.extend(_segment(context, sound_at, sb, note_at, nb, fixed))
    return [forced.get(_position(operation, context), operation) for operation in operations]


def _position(operation: Operation, context: _Context) -> int | None:
    if not isinstance(operation, Drop):
        return None
    return context.line_start[operation.sound.line] + operation.sound.index


def _context(lines: Sequence[SoundLine], rows: Sequence[Sequence[Raw]], notes: list) -> _Context:
    info: list[tuple[int, int]] = []
    weights: list[tuple[float, ...]] = []
    line_start: list[int] = []
    line_count: list[int] = []
    raw_start: list[float] = []
    line_end: list[float] = []
    for line_index, line in enumerate(lines):
        weights.extend(merge_weights(line))
        line_start.append(len(info))
        line_count.append(len(line.sounds))
        line_end.append(rows[line_index][-1].reference_end if line.sounds else 0.0)
        for index in range(len(line.sounds)):
            info.append((line_index, index))
            raw_start.append(rows[line_index][index].onset)
    total = len(info)
    is_last = [index == line_start[info[index][0]] + line_count[info[index][0]] - 1 for index in range(total)]
    maxm = [1] * total
    for index in range(total - 2, -1, -1):
        if info[index][0] == info[index + 1][0]:
            maxm[index] = maxm[index + 1] + 1
    return _Context(info, line_start, line_count, raw_start, line_end, is_last, maxm, weights, notes, total, {})


def _boundary(context: _Context, note_at: int) -> float:
    """Where an operation with no note of its own lands: the next note start, else the last end."""
    if note_at < len(context.notes):
        return context.notes[note_at].start
    return context.notes[-1].end


def _line_end(context: _Context, index: int) -> float:
    return context.line_end[context.info[index][0]]


def _drop_point(context: _Context, index: int, note_at: int) -> float:
    """The cursor boundary a dropped Sound sits on: the nearer of the taken note's end and the next start.

    A drop owns no note, so neither side of the cursor is the next Sound's note in particular.
    """
    next_point = _boundary(context, note_at)
    if note_at == 0:
        return next_point
    return min(next_point, context.notes[note_at - 1].end, key=lambda point: abs(context.raw_start[index] - point))


def _drop_base(context: _Context, index: int, note_at: int) -> float:
    point = _drop_point(context, index, note_at)
    cost = abs(context.raw_start[index] - point)
    if context.is_last[index]:
        cost += abs(_line_end(context, index) - point)
    return cost


def _match_base(context: _Context, index: int, note_at: int, count: int) -> float:
    cost = abs(context.raw_start[index] - context.notes[note_at].start)
    if context.is_last[index]:
        cost += abs(_line_end(context, index) - context.notes[note_at + count - 1].end)
    return cost


def _merge_base(context: _Context, index: int, run: int, note_at: int) -> float:
    start, end = context.notes[note_at].start, context.notes[note_at].end
    cost = sum(abs(context.raw_start[index + p] - (start + p * (end - start) / run)) for p in range(run))
    last = index + run - 1
    if context.is_last[last]:
        cost += abs(_line_end(context, last) - end)
    return cost * context.merge_weights[index][run - 2]


def _global(ref: SoundRef, context: _Context) -> int:
    if not 0 <= ref.line < len(context.line_start):
        raise MappingError("invalid_anchor", f"the mapping names line {ref.line}")
    found = context.line_start[ref.line] + ref.index
    if not 0 <= ref.index < context.line_count[ref.line]:
        raise MappingError("invalid_anchor", f"the mapping names sound {ref}")
    return found


def _anchors(anchors: Sequence[Operation], context: _Context) -> tuple[list[tuple], dict[int, Operation]]:
    """The confirmed operations split into note-pinning ones and forced drops.

    A `match`/`merge` pins both the Sounds and the notes, so it splits the song; a `drop` pins only
    its Sound and is forced inside the stretch it falls in. Anything inconsistent is `invalid_anchor`.
    """
    pinned: list[tuple] = []
    forced: dict[int, Operation] = {}
    for operation in anchors:
        refs = operation.sounds
        sounds = [_global(ref, context) for ref in refs]
        if not isinstance(operation, Drop):
            if sounds != list(range(min(sounds), min(sounds) + len(sounds))):
                raise MappingError("invalid_anchor", "a confirmed operation's sounds do not run on")
            if isinstance(operation, Merge) and len({context.info[g][0] for g in sounds}) != 1:
                raise MappingError("invalid_anchor", "a confirmed merge crosses a line")
        if isinstance(operation, Match):
            notes = _note_indices(operation.notes, context)
            pinned.append((sounds[0], sounds[0] + 1, notes[0], notes[-1] + 1, operation))
        elif isinstance(operation, Merge):
            notes = _note_indices((operation.note,), context)
            pinned.append((sounds[0], sounds[-1] + 1, notes[0], notes[0] + 1, operation))
        else:
            forced[sounds[0]] = operation
    pinned.sort(key=lambda entry: entry[0])
    sound_at, note_at = 0, 0
    for low, high, first, last, _operation in pinned:
        if low < sound_at or first < note_at or last < first:
            raise MappingError("invalid_anchor", "confirmed operations cross or repeat")
        if any(low <= g < high for g in forced):
            raise MappingError("invalid_anchor", "a confirmed drop falls inside another anchor")
        sound_at, note_at = high, last
    return pinned, forced


def _note_indices(ids: Sequence[int], context: _Context) -> list[int]:
    at = {note.id: index for index, note in enumerate(context.notes)}
    found = [at.get(identifier) for identifier in ids]
    if any(index is None for index in found):
        raise MappingError("invalid_anchor", "a confirmed operation names a note that is not in the stream")
    if found != list(range(found[0], found[0] + len(found))):
        raise MappingError("invalid_anchor", "a confirmed operation's notes do not run on")
    return found


def _segment(context: _Context, sa: int, sb: int, na: int, nb: int, forced: set[int]) -> list[Operation]:
    """The cheapest mapping of sounds `[sa, sb)` onto notes `[na, nb)`, with forced drops applied."""
    size = sb - sa
    count = nb - na
    if size == 0:
        if count == 0:
            return []
        raise MappingError("invalid_anchor", "notes are left with no sounds to take them")
    weight = [3 ** (size - 1 - step) for step in range(size)]
    next_forced = [size] * (size + 1)
    following = size
    for step in range(size - 1, -1, -1):
        if sa + step in forced or sa + step in context.constraints:
            following = step
        next_forced[step] = following

    best: list[list[tuple[float, int, int] | None]] = [[None] * (count + 1) for _ in range(size + 1)]
    back: list[list[tuple | None]] = [[None] * (count + 1) for _ in range(size + 1)]
    best[0][0] = (0.0, 0, 0)
    for step in range(size):
        index = sa + step
        forced_here = index in forced
        constraint = context.constraints.get(index)
        for have in range(count + 1):
            here = best[step][have]
            if here is None:
                continue
            if constraint is not None:
                run, slots = constraint
                if run == 0 or step + run > size or have + slots > count or (forced_here and slots):
                    continue
                if slots == 0:
                    base, complexity, rank, pointer = _drop_base(context, index, na + have), 1, 2 * weight[step], ("d",)
                elif run == 1:
                    base = _match_base(context, index, na + have, slots)
                    complexity, rank, pointer = slots - 1, 0, ("m", have)
                else:
                    base = _merge_base(context, index, run, na + have)
                    complexity, rank, pointer = run - 1, sum(weight[step : step + run]), ("g", run)
                _relax(best, back, step + run, have + slots, _add(here, base, complexity, rank), pointer)
                continue
            _relax(
                best,
                back,
                step + 1,
                have,
                _add(here, _drop_base(context, index, na + have), 1, 2 * weight[step]),
                ("d",),
            )
            if forced_here:
                continue
            if have < count:
                top = min(context.maxm[index], next_forced[step] - step)
                for run in range(2, top + 1):
                    _relax(
                        best,
                        back,
                        step + run,
                        have + 1,
                        _add(
                            here, _merge_base(context, index, run, na + have), run - 1, sum(weight[step : step + run])
                        ),
                        ("g", run),
                    )
        if not forced_here and constraint is None:
            _match_run(context, best, back, step, index, na, count, weight)
    found = best[size][count]
    if found is None:
        code = "invalid_constraint" if context.constraints else "invalid_anchor"
        raise MappingError(code, "the sounds and the notes cannot be partitioned")
    return _rebuild(context, back, sa, na, size, count)


def _match_run(context, best, back, step, index, na, count, weight) -> None:
    """Relax every match from the running cheapest earlier note, in one pass over the notes."""
    run: tuple[float, int, int, int] | None = None
    notes = context.notes
    for have in range(1, count + 1):
        before = best[step][have - 1]
        if before is not None:
            candidate = (
                before[0] + abs(context.raw_start[index] - notes[na + have - 1].start),
                before[1] - (have - 1),
                before[2],
                have - 1,
            )
            if run is None or candidate[:3] < run[:3]:
                run = candidate
        if run is None:
            continue
        end_time = notes[na + have - 1].end
        base = run[0] + (abs(_line_end(context, index) - end_time) if context.is_last[index] else 0.0)
        _relax(best, back, step + 1, have, (base, run[1] + (have - 1), run[2]), ("m", run[3]))


def _add(value: tuple[float, int, int], base: float, complexity: int, rank: int) -> tuple[float, int, int]:
    return (value[0] + base, value[1] + complexity, value[2] + rank)


def _relax(best, back, row: int, column: int, value: tuple[float, int, int], pointer: tuple) -> None:
    here = best[row][column]
    if here is None or value < here:
        best[row][column] = value
        back[row][column] = pointer


def _rebuild(context: _Context, back, sa: int, na: int, size: int, count: int) -> list[Operation]:
    operations: list[Operation] = []
    step, have = size, count
    while step or have:
        pointer = back[step][have]
        if pointer is None:
            raise MappingError("invalid_anchor", "the mapping has no complete partition")
        index = sa + step - 1
        line, position = context.info[index]
        if pointer[0] == "d":
            operations.append(Drop(SoundRef(line, position)))
            step -= 1
        elif pointer[0] == "m":
            first = pointer[1]
            ids = tuple(context.notes[na + at].id for at in range(first, have))
            operations.append(Match(SoundRef(line, position), ids))
            step, have = step - 1, first
        else:
            run = pointer[1]
            first = sa + step - run
            note = context.notes[na + have - 1].id
            refs = tuple(SoundRef(context.info[at][0], context.info[at][1]) for at in range(first, first + run))
            operations.append(Merge(refs, note))
            step, have = step - run, have - 1
    operations.reverse()
    return operations
