# SPDX-License-Identifier: AGPL-3.0-only
"""Rebuild a `.krc` from the natural Sounds and a mapping, rather than patching the input's `.N`.

The input's `.N`, `.+` and `(...)` carried a mapping; the rebuild throws them away and starts from
the Sounds the text naturally reads. Each Sound's `.N` lands on the smallest Unit that still writes that
Sound whole - `(しょ).2`, never `し.2ょ`. A merge uses `(...).1` within a container and `.+` to
continue across containers without changing the ruby structure. A Sound that takes one note gets
no `.N` at all. The rebuilt text is parsed and flattened again; if its Sound or token sequence
moved, the rebuild is refused, so nothing that would mis-time a sound is ever written. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

from namioto.karaoke.model import Group, KrcError, Line, Unit, Word
from namioto.karaoke.operations import NoteRun, Operation, partition, remap_operation
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import SoundLine, _base_words, natural_sounds
from namioto.karaoke.writer import dumps


def rebuild(text: str, operations: Sequence[Operation | NoteRun], *, line_indices: Sequence[int] | None = None) -> str:
    """The canonical `.krc` for `text` under `operations`, verified to read back the same Sounds.

    `operations` must partition the natural Sounds of `line_indices`, or all lines when omitted.
    Unselected lines keep their literal KRC structure. An incomplete partition or a written `.N`
    that would move a Sound raises `KrcError` rather than writing a broken file.
    """
    if not text.strip():
        return ""
    lines = natural_sounds(text)
    if line_indices is None:
        plan = dict(enumerate(partition(list(operations), [len(line.sounds) for line in lines])))
    else:
        indices = tuple(line_indices)
        local = {index: position for position, index in enumerate(indices)}
        selected = [op for op in operations if all(ref.line in local for ref in op.sounds)]
        plan = dict(
            zip(
                indices,
                partition(
                    [remap_operation(op, local) for op in selected], [len(lines[index].sounds) for index in indices]
                ),
                strict=True,
            )
        )
    lyrics = parse(text)
    line_index = 0
    for chapter in lyrics.chapters:
        for line in chapter.lines:
            if line_index not in plan:
                line_index += 1
                continue
            pieces = []
            for operation in plan[line_index]:
                for index, (_container, refs) in enumerate(
                    groupby(operation.sounds, key=lambda ref: lines[line_index].sounds[ref.index].container)
                ):
                    count = len(list(refs))
                    shared = len(operation.sounds) > 1
                    joined = shared and index > 0
                    override = None if joined or (shared and count == 1 and operation.slots == 1) else operation.dot
                    pieces.append((count, override, joined))
            _rebuild_line(line, pieces, lines[line_index])
            line_index += 1
    rebuilt = dumps(lyrics, dotted=True)
    _verify(lines, rebuilt)
    return rebuilt


def _rebuild_line(line: Line, pieces: list[tuple[int, int | None, bool]], sounds: SoundLine) -> None:
    """Replace the units of one line with the ones the mapping writes."""
    cursor = 0
    new_units: list[Unit] = []
    run: list[Word] = []
    entries_by_container: list[list[tuple[str, int, int]]] = [[] for _ in sounds.containers]
    for sound in sounds.sounds:
        entries_by_container[sound.container].append((sound.reading, sound.first_atom, sound.last_atom))
    containers = iter(entries_by_container)

    def build(atoms: list[Word]) -> list[Unit]:
        nonlocal cursor
        if not atoms:
            return []
        entries = next(containers)
        taken = _take(pieces, cursor, len(entries))
        cursor += len(taken)
        return _build(atoms, taken, entries)

    def flush() -> None:
        if not run:
            return
        new_units.extend(build(run))
        run.clear()

    for unit in line.units:
        if unit.ruby is None:
            run.extend(_base_words(unit))
            continue
        flush()
        unit.override = None
        unit.join_previous = False
        for part in unit.ruby.parts:
            atoms = [word for inner in part for word in _base_words(inner)]
            part[:] = build(atoms)
        new_units.append(unit)
    flush()
    if cursor != len(pieces):
        raise KrcError("the mapping does not cover this line's sounds")
    line.units = new_units


def _take(pieces: list[tuple[int, int | None, bool]], cursor: int, count: int) -> list[tuple[int, int | None, bool]]:
    taken = []
    found = 0
    at = cursor
    while found < count:
        piece = pieces[at]
        taken.append(piece)
        found += piece[0]
        at += 1
    if found != count:
        raise KrcError("the mapping does not cover this container's sounds")
    return taken


def _build(
    atoms: list[Word], pieces: list[tuple[int, int | None, bool]], entries: list[tuple[str, int, int]]
) -> list[Unit]:
    """The units one container's characters become: an override lands on the Sound's whole range."""
    starts: dict[int, tuple[int, int | None, bool]] = {}
    at = 0
    for count, override, joined in pieces:
        low = entries[at][1]
        high = entries[at + count - 1][2]
        starts[low] = (high, override, joined)
        at += count
    units: list[Unit] = []
    index = 0
    while index < len(atoms):
        if index not in starts:
            units.append(Unit(atoms[index]))
            index += 1
            continue
        high, override, joined = starts[index]
        units.extend(_units(atoms[index : high + 1], override, joined))
        index = high + 1
    return units


def _units(chars: list[Word], override: int | None, joined: bool) -> list[Unit]:
    """A Sound's characters as one unit: a word, or a group when an override must write them whole.

    Without an override or continuation, the characters stay words of their own.
    """
    if override is None and not joined:
        return [Unit(word) for word in chars]
    base = chars[0] if len(chars) == 1 else Group(chars)
    return [Unit(base, override=override, join_previous=joined)]


def _verify(before: list[SoundLine], rebuilt: str) -> None:
    """The rebuilt text must read back as the same Sounds, tokens and readings as the input."""
    try:
        after = natural_sounds(rebuilt)
    except (KrcError, ValueError) as error:
        raise KrcError(f"round_trip_mismatch: the written lyrics do not read back: {error}") from None
    if len(before) != len(after) or any(_shape(one) != _shape(other) for one, other in zip(before, after, strict=True)):
        raise KrcError("round_trip_mismatch: writing the mapping back would move a sound")


def _shape(line: SoundLine) -> list[tuple]:
    return [
        (sound.reading, sound.base, sound.token, sound.rubied, sound.first, sound.channel, sound.color_channel)
        for sound in line.sounds
    ]
