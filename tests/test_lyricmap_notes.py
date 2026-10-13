# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the target NOTE stream: its order, its conflicts, and the path that survives them."""

from __future__ import annotations

from random import Random

from namioto.document import Note
from namioto.lyricmap import notes as note_stream
from namioto.lyricmap.notes import components, ordered, overlaps, resolve


def _note(identifier: int, start: float, end: float, pitch: int = 60):
    return Note(pitch, start, end - start, id=identifier)


def test_notes_are_ordered_by_time_then_pitch_then_id():
    notes = [_note(3, 0.0, 1.0, 62), _note(1, 0.0, 1.0, 60), _note(2, 2.0, 3.0)]
    assert [note.id for note in ordered(notes)] == [1, 3, 2]


def test_a_touch_inside_the_slack_is_not_a_conflict():
    assert not overlaps(_note(1, 0.0, 1.0), _note(2, 1.0, 2.0))
    assert not overlaps(_note(1, 0.0, 1.0), _note(2, 0.999, 2.0))
    assert overlaps(_note(1, 0.0, 1.0), _note(2, 0.998, 2.0))


def test_any_pitch_overlaps_another():
    assert overlaps(_note(1, 0.0, 1.0, 60), _note(2, 0.5, 1.5, 72))


def test_components_join_every_note_that_reaches_the_region():
    notes = [_note(1, 0.0, 1.0), _note(2, 0.5, 1.5), _note(3, 1.4, 2.0), _note(4, 3.0, 4.0)]
    regions = components(notes)
    assert [[note.id for note in region] for region in regions] == [[1, 2, 3], [4]]


def test_best_only_resolution_matches_exhaustive_paths():
    random = Random(734)
    for _ in range(300):
        identifiers = list(range(1, 10))
        random.shuffle(identifiers)
        notes = [
            note_stream.TimedNote(start, start + random.choice((0.0005, 0.001, 0.25, 0.5, 1.0)), 60, identifier)
            for identifier in identifiers
            for start in [random.randrange(20) * 0.125]
        ]
        expected = resolve(notes)
        actual = resolve(notes, include_branches=False)
        assert actual.stream == expected.stream
        assert actual.filtered == expected.filtered
        assert actual.branches == ()


def test_long_overlap_chains_do_not_enumerate_paths(monkeypatch):
    def forbidden(*args):
        raise AssertionError("best-only resolution must not enumerate paths")

    monkeypatch.setattr(note_stream, "_maximal_paths", forbidden)
    notes = [_note(index + 1, float(index), index + 1.5) for index in range(3000)]
    found = resolve(notes, include_branches=False)
    assert found.stream == tuple(notes[::2])
    assert found.filtered == tuple(notes[1::2])
    assert found.branches == ()


def test_best_only_resolution_retains_custom_costs():
    notes = [_note(1, 0.0, 1.0), _note(2, 0.5, 1.5), _note(3, 1.2, 2.0)]
    found = resolve(notes, lambda path: 0.0 if len(path) == 1 else 5.0, include_branches=False)
    assert found.stream == (notes[1],)
    assert found.filtered == (notes[0], notes[2])
    assert found.branches == ()


def test_a_note_that_never_conflicts_is_always_kept():
    notes = [_note(1, 0.0, 1.0), _note(2, 2.0, 3.0)]
    found = resolve(notes)
    assert [note.id for note in found.stream] == [1, 2]
    assert found.filtered == ()
    assert found.branches == ()


def test_the_path_that_keeps_more_notes_wins():
    notes = [_note(1, 0.0, 1.0), _note(2, 0.5, 1.5), _note(3, 1.2, 2.0)]
    found = resolve(notes)
    assert [note.id for note in found.stream] == [1, 3]
    assert [note.id for note in found.filtered] == [2]


def test_the_path_with_smaller_ids_wins_a_exact_tie():
    # both paths keep two notes; the one with the smaller first id is chosen
    notes = [_note(1, 0.0, 1.0), _note(4, 0.5, 1.5), _note(2, 2.0, 3.0), _note(3, 2.5, 3.5)]
    found = resolve(notes)
    kept = [note.id for note in found.stream]
    assert kept in ([1, 2], [1, 3])
    assert kept == [1, 2]


def test_a_lower_cost_beats_more_notes():
    notes = [_note(1, 0.0, 1.0), _note(2, 0.5, 1.5), _note(3, 1.2, 2.0)]

    def cost(path):
        return 0.0 if [note.id for note in path] == [2] else 5.0

    found = resolve(notes, cost)
    assert [note.id for note in found.stream] == [2]
    assert found.filtered == (notes[0], notes[2])


def test_the_branches_offer_every_maximal_path():
    notes = [_note(1, 0.0, 1.0), _note(2, 0.5, 1.5), _note(3, 1.2, 2.0)]
    found = resolve(notes)
    offered = {tuple(note.id for note in branch.notes) for branch in found.branches[0]}
    assert offered == {(1, 3), (2,)}


def test_a_path_that_could_grow_is_not_maximal():
    # 1 and 3 do not touch, so dropping 1 to keep only 2 would leave 1 or 3 addable
    notes = [_note(1, 0.0, 1.0), _note(2, 0.5, 1.5), _note(3, 1.2, 2.0)]
    found = resolve(notes)
    for branch in found.branches[0]:
        ids = {note.id for note in branch.notes}
        for note in notes:
            if note.id in ids:
                continue
            assert any(overlaps(note, other) for other in branch.notes)
