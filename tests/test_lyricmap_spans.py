# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the derived spans the roll and the lyric map read - never persisted as fact."""

from __future__ import annotations

from namioto.document import Note
from namioto.karaoke.operations import Drop, Match, Merge, SoundRef
from namioto.lyricmap.spans import sound_spans


def _notes(*spans):
    return [Note(60, start, end - start, id=index + 1) for index, (start, end) in enumerate(spans)]


def test_a_match_spans_its_first_to_its_last_note():
    notes = _notes((0.0, 1.0), (2.0, 3.0))
    assert sound_spans([1], [Match(SoundRef(0, 0), (1, 2))], notes) == [[(0.0, 3.0)]]


def test_merge_members_tile_the_note_without_repeating_it():
    notes = _notes((0.0, 3.0))
    rows = sound_spans([3], [Merge((SoundRef(0, 0), SoundRef(0, 1), SoundRef(0, 2)), 1)], notes)
    assert rows == [[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]]


def test_a_drop_has_no_block():
    notes = _notes((0.0, 1.0))
    assert sound_spans([1], [Drop(SoundRef(0, 0))], notes) == [[None]]


def test_spans_follow_lines_and_sounds():
    notes = _notes((0.0, 1.0), (1.0, 2.0))
    operations = [Match(SoundRef(0, 0), (1,)), Match(SoundRef(1, 0), (2,))]
    assert sound_spans([1, 1], operations, notes) == [[(0.0, 1.0)], [(1.0, 2.0)]]
