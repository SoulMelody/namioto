# SPDX-License-Identifier: AGPL-3.0-only
"""The `.krc`'s own `.N`, groups and `.+` lay the Sounds out, with nothing estimated.

The input's NOTE slots are read in order, never solved for. `note_groups` reads the literal mapping,
including continuations, into runs of Sounds and their NOTE counts. Sounds sharing one NOTE divide
it equally; one Sound taking several holds their whole span. When the notes run out the rest have
no time to render. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from namioto.karaoke.operations import Drop, Match, Merge, NoteRun, Operation, SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import SoundLine, mapping_runs, note_groups, sound_lines
from namioto.lyricmap.notes import Note

Span = tuple[float, float] | None


@dataclass(frozen=True)
class Structure:
    lines: tuple[SoundLine, ...]
    groups: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]
    runs: tuple[NoteRun, ...]


@lru_cache(maxsize=8)
def structure(text: str) -> Structure:
    """The immutable lyric structure shared by faithful reads and hand edits."""
    if not text.strip():
        return Structure((), (), ())
    lyrics = parse(text)
    lines = tuple(sound_lines(lyrics))
    parsed = [line for chapter in lyrics.chapters for line in chapter.lines]
    groups, runs = [], []
    for row, (sounds, line) in enumerate(zip(lines, parsed, strict=True)):
        groups.append(tuple((slots, tuple(indices)) for slots, indices in note_groups(line, sounds)))
        runs.extend(
            NoteRun(tuple(SoundRef(row, index) for index in indices), slots)
            for slots, indices in mapping_runs(line, sounds)
        )
    return Structure(lines, tuple(groups), tuple(runs))


def read(text: str, notes: Sequence[Note]) -> list[list[Span]]:
    """The input `.krc`'s own NOTE slots, one span per natural Sound, in lines."""
    return lay_out(text, notes)[0]


def lay_out(text: str, notes: Sequence[Note]) -> tuple[list[list[Span]], list[list[tuple[int, ...]]]]:
    """The `.N` spans, and beside them the stable ids of the notes each Sound takes.

    The ids say which notes a Sound landed on where a span alone cannot: a Sound over several notes
    carries several, and Sounds sharing one note each carry that same id.
    """
    layout = structure(text)
    rows: list[list[Span]] = []
    found: list[list[tuple[int, ...]]] = []
    at = 0
    for sound_line, groups in zip(layout.lines, layout.groups, strict=True):
        row: list[Span] = [None] * len(sound_line.sounds)
        ids: list[tuple[int, ...]] = [() for _sound in sound_line.sounds]
        for slots, indices in groups:
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
    found: list[Operation] = []
    at = 0
    for row, groups in enumerate(structure(text).groups):
        for slots, indices in groups:
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
    return sum(slots for groups in structure(text).groups for slots, _indices in groups)


__all__ = ["Span", "consumes", "lay_out", "operations", "read"]
