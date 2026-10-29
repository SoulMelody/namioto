# SPDX-License-Identifier: AGPL-3.0-only
"""The time each Sound is drawn at, derived from the mapping - never stored as fact.

A match spans its first note's start to its last note's end; a merge splits its one note evenly
between its Sounds, so the members tile the note instead of each covering the whole of it; a drop has
no block at all. This is the only place the derived span is computed, so the roll and the lyric map
read the same times. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from namioto.karaoke.operations import Drop, Match, Operation
from namioto.lyricmap.notes import Note

Span = tuple[float, float] | None


def sound_spans(
    sounds_per_line: Sequence[int], operations: Sequence[Operation], notes: Sequence[Note]
) -> list[list[Span]]:
    """Per line, per Sound, the span the mapping derives, or `None` where the Sound is dropped."""
    at = {note.id: index for index, note in enumerate(notes)}
    rows: list[list[Span]] = [[None] * count for count in sounds_per_line]
    for operation in operations:
        if isinstance(operation, Drop):
            continue
        if isinstance(operation, Match):
            ref = operation.sound
            indices = [at[identifier] for identifier in operation.notes]
            rows[ref.line][ref.index] = (notes[indices[0]].start, notes[indices[-1]].end)
            continue
        index = at[operation.note]
        start, end = notes[index].start, notes[index].end
        total = len(operation.sounds)
        for member, ref in enumerate(operation.sounds):
            rows[ref.line][ref.index] = (
                start + member * (end - start) / total,
                start + (member + 1) * (end - start) / total,
            )
    return rows


__all__ = ["Span", "sound_spans"]
