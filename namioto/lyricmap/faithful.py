# SPDX-License-Identifier: AGPL-3.0-only
"""The `.krc`'s own `.N`, groups and `.+` lay the Sounds out, with nothing estimated.

The input's NOTE slots are read in order, never solved for. `note_groups` reads the literal mapping,
including continuations, into runs of Sounds and their NOTE counts. Sounds sharing one NOTE divide
it equally; one Sound taking several holds their whole span. When the notes run out the rest have
no time to render. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from namioto.karaoke.operations import Drop, Match, Merge, Operation, SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds, note_groups
from namioto.lyricmap.notes import Note

Span = tuple[float, float] | None


def read(text: str, notes: Sequence[Note]) -> list[list[Span]]:
    """The input `.krc`'s own NOTE slots, one span per natural Sound, in lines."""
    return lay_out(text, notes)[0]


def lay_out(text: str, notes: Sequence[Note]) -> tuple[list[list[Span]], list[list[tuple[int, ...]]]]:
    """The `.N` spans, and beside them the stable ids of the notes each Sound takes.

    The ids say which notes a Sound landed on where a span alone cannot: a Sound over several notes
    carries several, and Sounds sharing one note each carry that same id.
    """
    lines = natural_sounds(text)
    if not text.strip():
        return [], []
    parsed = [line for chapter in parse(text).chapters for line in chapter.lines]
    rows: list[list[Span]] = []
    found: list[list[tuple[int, ...]]] = []
    at = 0
    for sound_line, line in zip(lines, parsed, strict=True):
        row: list[Span] = [None] * len(sound_line.sounds)
        ids: list[tuple[int, ...]] = [() for _sound in sound_line.sounds]
        for slots, indices in note_groups(line, sound_line):
            taken = list(notes[at : at + slots])
            at += len(taken)
            if not taken:
                continue
            if slots == 1:
                width = (taken[0].end - taken[0].start) / len(indices)
                for position, index in enumerate(indices):
                    start = taken[0].start + position * width
                    row[index] = (start, start + width)
                    ids[index] = (taken[0].id,)
            else:
                row[indices[0]] = (taken[0].start, taken[-1].end)
                ids[indices[0]] = tuple(note.id for note in taken)
        rows.append(row)
        found.append(ids)
    return rows, found


def operations(text: str, notes: Sequence[Note]) -> tuple[Operation, ...]:
    """Derive editable operations from KRC slots; uncovered Sounds have no operation."""
    if not text.strip():
        return ()
    parsed = [line for chapter in parse(text).chapters for line in chapter.lines]
    found: list[Operation] = []
    at = 0
    for row, (sounds, line) in enumerate(zip(natural_sounds(text), parsed, strict=True)):
        for slots, indices in note_groups(line, sounds):
            refs = tuple(SoundRef(row, index) for index in indices)
            if slots == 0:
                found.extend(Drop(ref, confirmed=True) for ref in refs)
                continue
            taken = notes[at : at + slots]
            at += len(taken)
            if len(taken) != slots:
                continue
            if len(refs) > 1:
                found.append(Merge(refs, taken[0].id, confirmed=True))
            else:
                found.append(Match(refs[0], tuple(note.id for note in taken), confirmed=True))
    return tuple(found)


def consumes(text: str, notes: Sequence[Note]) -> int:
    """How many notes the faithful reading asks for; more than `len(notes)` leaves Sounds untimed."""
    if not text.strip():
        return 0
    parsed = [line for chapter in parse(text).chapters for line in chapter.lines]
    return sum(
        slots
        for sound_line, line in zip(natural_sounds(text), parsed, strict=True)
        for slots, _indices in note_groups(line, sound_line)
    )


__all__ = ["Span", "consumes", "lay_out", "operations", "read"]
