# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the global mapping DP: what it chooses, what it never crosses, and what it refuses."""

from __future__ import annotations

from functools import cache

import pytest

from namioto.document import Note
from namioto.karaoke.canonical import rebuild
from namioto.karaoke.operations import Drop, Match, Merge, SoundRef, partition
from namioto.karaoke.sounds import natural_sounds
from namioto.lyricmap.problems import MappingError
from namioto.lyricmap.raw import Raw
from namioto.lyricmap.solver import _context, _drop_base, _match_base, _merge_base, solve


def _notes(*spans):
    return [Note(60, start, end - start, id=index + 1) for index, (start, end) in enumerate(spans)]


def _cost(context, operations):
    sound_at = note_at = 0
    cost = 0.0
    for operation in operations:
        if isinstance(operation, Match):
            cost += _match_base(context, sound_at, note_at, operation.slots)
        elif isinstance(operation, Merge):
            cost += _merge_base(context, sound_at, len(operation.sounds), note_at)
        else:
            cost += _drop_base(context, sound_at, note_at)
        sound_at += len(operation.sounds)
        note_at += operation.slots
    return cost


def _minimum(context):
    @cache
    def visit(sound, note):
        if sound == context.total:
            return 0.0 if note == len(context.notes) else float("inf")
        costs = [_drop_base(context, sound, note) + visit(sound + 1, note)]
        for count in range(1, len(context.notes) - note + 1):
            costs.append(_match_base(context, sound, note, count) + visit(sound + 1, note + count))
        if note < len(context.notes):
            for run in range(2, context.maxm[sound] + 1):
                costs.append(_merge_base(context, sound, run, note) + visit(sound + run, note + 1))
        return min(costs)

    return visit(0, 0)


def _solve(text, raw, notes, anchors=()):
    lines = natural_sounds(text)
    operations = solve(lines, raw, notes, anchors)
    ordered = partition(operations, [len(line.sounds) for line in lines])
    return lines, operations, ordered


def test_one_note_a_sound():
    # a sound whose raw time is on a note starts its own match
    lines, _operations, rows = _solve("あい", [[Raw(0.0, 1.0), Raw(1.0, 1.0)]], _notes((0.0, 1.0), (1.0, 2.0)))
    assert [type(operation).__name__ for operation in rows[0]] == ["Match", "Match"]


def test_a_match_holds_more_than_one_note():
    lines, _operations, rows = _solve("あ", [[Raw(0.0, 2.0)]], _notes((0.0, 1.0), (1.0, 2.0)))
    assert rows[0] == [Match(SoundRef(0, 0), (1, 2))]


def test_a_merge_takes_the_note_the_sounds_evenly_share():
    lines, _operations, rows = _solve("あい", [[Raw(0.0, 0.5), Raw(0.5, 0.5)]], _notes((0.0, 1.0)))
    assert rows[0] == [Merge((SoundRef(0, 0), SoundRef(0, 1)), 1)]


@pytest.mark.parametrize("onset", [0.8, 0.9])
def test_match_and_drop_beats_a_merge_when_the_times_disagree(onset):
    lines, operations, rows = _solve("あい", [[Raw(0.0, onset), Raw(onset, 1.0 - onset)]], _notes((0.0, 1.0)))
    assert [type(operation).__name__ for operation in rows[0]] == ["Match", "Drop"]
    assert rebuild("あい", operations) == "あい.0"


def test_a_drop_takes_the_cursor_boundary_nearest_its_onset():
    # B sits just past A's note across a rest: its own boundary is that note's end, not C's note start
    lines, _operations, rows = _solve(
        "あいか", [[Raw(0.0, 1.0), Raw(1.05, 0.15), Raw(3.0, 1.0)]], _notes((0.0, 1.0), (3.0, 4.0))
    )
    assert [type(operation).__name__ for operation in rows[0]] == ["Match", "Drop", "Match"]


@pytest.mark.parametrize("rest", [0.0, 0.1, 8.0435, 40.0])
def test_a_line_end_drop_does_not_pay_for_the_following_rest(rest):
    next_start = 65.3254 + rest
    raw = [[Raw(64.62, 0.14), Raw(65.22, 0.135)], [Raw(next_start, 0.2174)]]
    notes = _notes((64.6732, 65.3254), (next_start, next_start + 0.2174))
    lines, operations, rows = _solve("ぱい\nあ", raw, notes)
    assert rows == [[Match(SoundRef(0, 0), (1,)), Drop(SoundRef(0, 1))], [Match(SoundRef(1, 0), (2,))]]
    assert rebuild("ぱい\nあ", operations) == "ぱい.0\nあ"
    assert _cost(_context(lines, raw, notes), operations) == pytest.approx(0.1882)


def test_a_drop_consumes_no_note():
    lines, _operations, rows = _solve("あか", [[Raw(0.0, 1.0), Raw(5.0, 1.0)]], _notes((0.0, 1.0)))
    assert any(isinstance(operation, Drop) for operation in rows[0])


def test_every_sound_and_note_is_consumed_once():
    text = "胡椒[こ,しょう]は"
    lines, operations, rows = _solve(
        text,
        [[Raw(0.0, 0.3), Raw(0.3, 0.3), Raw(0.6, 0.3), Raw(0.9, 0.3)]],
        _notes((0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2)),
    )
    assert rebuild(text, operations) == "胡椒[こ,しょう]は"
    assert sorted(ref.index for operation in rows[0] for ref in operation.sounds) == [0, 1, 2, 3]


@pytest.mark.parametrize("text", ["胡椒[こ,(しょう)]", "世界[せ,かい]", "泣[な]いちゃ"])
def test_a_merge_can_cross_ruby_containers(text):
    lines, operations, _rows = _solve(
        text, [[Raw(0.0, 1 / 3), Raw(1 / 3, 1 / 3), Raw(2 / 3, 1 / 3)]], _notes((0.0, 1.0))
    )
    assert operations == [Merge(tuple(SoundRef(0, index) for index in range(3)), 1)]
    assert natural_sounds(rebuild(text, operations)) == lines


def test_a_merge_never_crosses_a_line():
    _lines, operations, _rows = _solve("あ\nい", [[Raw(0.0, 0.5)], [Raw(0.5, 0.5)]], _notes((0.0, 1.0)))
    assert not any(isinstance(operation, Merge) for operation in operations)
    with pytest.raises(MappingError, match="crosses a line"):
        _solve(
            "あ\nい",
            [[Raw(0.0, 0.5)], [Raw(0.5, 0.5)]],
            _notes((0.0, 1.0)),
            [Merge((SoundRef(0, 0), SoundRef(1, 0)), 1, confirmed=True)],
        )


def test_a_confirmed_merge_can_cross_containers():
    anchor = Merge((SoundRef(0, 0), SoundRef(0, 1)), 1, confirmed=True)
    _lines, operations, _rows = _solve("泣[な]い", [[Raw(0.0, 0.5), Raw(0.5, 0.5)]], _notes((0.0, 1.0)), [anchor])
    assert operations == [anchor]


def test_edit_mapping_ignores_input_continuations():
    _lines, operations, _rows = _solve("泣[な]い.+", [[Raw(0.0, 1.0), Raw(1.0, 1.0)]], _notes((0.0, 1.0), (1.0, 2.0)))
    assert operations == [Match(SoundRef(0, 0), (1,)), Match(SoundRef(0, 1), (2,))]
    assert rebuild("泣[な]い.+", operations) == "泣[な]い"


def test_no_character_is_banned_from_a_merge():
    # a sokuon, a long vowel and a moraic n may all be merged when the times have them share a note
    lines, _operations, rows = _solve(
        "がっこう",
        [[Raw(0.0, 0.25), Raw(0.25, 0.25), Raw(0.5, 0.25), Raw(0.75, 0.25)]],
        _notes((0.0, 1.0)),
    )
    assert rows[0] == [Merge(tuple(SoundRef(0, index) for index in range(4)), 1)]


def test_an_exact_tie_prefers_the_earlier_match():
    # merge(A, B) and match(A)+drop(B) cost the same 0.25 and have the same complexity, so the
    # solution keeping A's match (rank 0) beats the one merging (rank 1) on the earlier sound
    lines, _operations, rows = _solve("あか", [[Raw(0.0, 0.75), Raw(0.75, 0.25)]], _notes((0.0, 1.0)))
    assert [type(operation).__name__ for operation in rows[0]] == ["Match", "Drop"]


def test_an_empirical_pair_can_outweigh_match_and_drop():
    raw = [[Raw(0.0, 0.76), Raw(0.76, 0.24)]]
    notes = _notes((0.0, 1.0))
    assert _solve("もー", raw, notes)[1] == [Merge((SoundRef(0, 0), SoundRef(0, 1)), 1)]
    assert _solve("もか", raw, notes)[1] == [Match(SoundRef(0, 0), (1,)), Drop(SoundRef(0, 1))]


def test_a_pair_without_a_long_vowel_mark_falls_to_the_earlier_match():
    assert _solve("あい", [[Raw(0.0, 0.75), Raw(0.75, 0.25)]], _notes((0.0, 1.0)))[1] == [
        Match(SoundRef(0, 0), (1,)),
        Drop(SoundRef(0, 1)),
    ]


def test_the_size_weight_can_prefer_a_smaller_merge_and_drop():
    assert _solve("あかさ", [[Raw(0.0, 0.5), Raw(0.5, 0.25), Raw(0.75, 0.25)]], _notes((0.0, 1.0)))[1] == [
        Merge((SoundRef(0, 0), SoundRef(0, 1)), 1),
        Drop(SoundRef(0, 2)),
    ]


def test_an_edge_sokuon_weight_can_prefer_match_and_drop():
    raw = [[Raw(0.0, 0.72), Raw(0.72, 0.28)]]
    notes = _notes((0.0, 1.0))
    assert _solve("きっ", raw, notes)[1] == [Match(SoundRef(0, 0), (1,)), Drop(SoundRef(0, 1))]
    assert _solve("きか", raw, notes)[1] == [Merge((SoundRef(0, 0), SoundRef(0, 1)), 1)]


def test_a_confirmed_merge_is_not_overridden_by_pattern_weights():
    anchor = Merge((SoundRef(0, 0), SoundRef(0, 1)), 1, confirmed=True)
    assert _solve("きっ", [[Raw(0.0, 0.72), Raw(0.72, 0.28)]], _notes((0.0, 1.0)), [anchor])[1] == [anchor]


def test_a_leading_sokuon_can_still_drop_before_a_match():
    assert _solve("っか", [[Raw(0.0, 0.29), Raw(0.29, 0.71)]], _notes((0.0, 1.0)))[1] == [
        Drop(SoundRef(0, 0)),
        Match(SoundRef(0, 1), (1,)),
    ]


def test_multiplicative_weights_preserve_a_zero_cost_merge():
    lines, operations, _rows = _solve("あっ", [[Raw(0.0, 0.5), Raw(0.5, 0.5)]], _notes((0.0, 1.0)))
    assert operations == [Merge((SoundRef(0, 0), SoundRef(0, 1)), 1)]
    assert _cost(_context(lines, [[Raw(0.0, 0.5), Raw(0.5, 0.5)]], _notes((0.0, 1.0))), operations) == 0.0


def test_the_whole_merge_cost_including_first_onset_and_line_end_is_weighted():
    lines = natural_sounds("もー")
    context = _context(lines, [[Raw(0.1, 0.5), Raw(0.6, 0.6)]], _notes((0.0, 1.0)))
    assert _merge_base(context, 0, 2, 0) == pytest.approx(0.32)
    assert _match_base(context, 0, 0, 1) == pytest.approx(0.1)
    assert _drop_base(context, 1, 1) == pytest.approx(0.6)


@pytest.mark.parametrize("text", ["ない", "きっ", "きっちく", "もーす"])
@pytest.mark.parametrize("notes", [_notes((0.0, 1.0)), _notes((0.0, 0.5), (0.6, 1.0))])
def test_forward_finds_the_minimum_of_all_weighted_candidates(text, notes):
    lines = natural_sounds(text)
    raw = [[Raw(index * 0.3 + 0.03, 0.2) for index in range(len(lines[0].sounds))]]
    context = _context(lines, raw, notes)
    operations = solve(lines, raw, notes)
    sound_at = note_at = 0
    cost = 0.0
    for operation in operations:
        if isinstance(operation, Match):
            cost += _match_base(context, sound_at, note_at, len(operation.notes))
        elif isinstance(operation, Merge):
            cost += _merge_base(context, sound_at, len(operation.sounds), note_at)
        else:
            cost += _drop_base(context, sound_at, note_at)
        sound_at += len(operation.sounds)
        note_at += operation.slots
    assert sound_at == context.total
    assert note_at == len(notes)
    assert cost == pytest.approx(_minimum(context))


def test_a_confirmed_drop_is_forced_and_kept():
    lines, _operations, rows = _solve(
        "あい", [[Raw(0.0, 1.0), Raw(1.0, 1.0)]], _notes((0.0, 1.0)), [Drop(SoundRef(0, 0), confirmed=True)]
    )
    assert rows[0] == [Drop(SoundRef(0, 0), confirmed=True), Match(SoundRef(0, 1), (1,))]


def test_a_confirmed_match_is_a_hard_anchor():
    lines, _operations, _rows = _solve(
        "あい",
        [[Raw(0.0, 1.0), Raw(1.0, 1.0)]],
        _notes((0.0, 1.0), (1.0, 2.0)),
        [Match(SoundRef(0, 1), (2,), confirmed=True)],
    )
    operations = _rows[0]
    assert operations[-1] == Match(SoundRef(0, 1), (2,), confirmed=True)
    assert operations[0] == Match(SoundRef(0, 0), (1,))


def test_an_anchor_with_an_unknown_note_is_refused():
    with pytest.raises(MappingError):
        _solve("あ", [[Raw(0.0, 1.0)]], _notes((0.0, 1.0)), [Match(SoundRef(0, 0), (99,), confirmed=True)])


def test_an_anchor_that_crosses_another_is_refused():
    with pytest.raises(MappingError):
        _solve(
            "あい",
            [[Raw(0.0, 1.0), Raw(1.0, 1.0)]],
            _notes((0.0, 1.0), (1.0, 2.0)),
            [
                Match(SoundRef(0, 0), (1,), confirmed=True),
                Match(SoundRef(0, 0), (2,), confirmed=True),
            ],
        )


def test_missing_evidence_refuses_the_whole_mapping():
    with pytest.raises(MappingError):
        _solve("あい", [[Raw(0.0, 1.0), Raw(None, None)]], _notes((0.0, 1.0)))


def test_no_target_notes_is_refused():
    with pytest.raises(MappingError):
        _solve("あ", [[Raw(0.0, 1.0)]], [])


def test_no_lyric_sounds_is_refused():
    with pytest.raises(MappingError):
        _solve("", [], _notes((0.0, 1.0)))


def test_reversed_raw_times_are_refused():
    with pytest.raises(MappingError):
        _solve("あい", [[Raw(1.0, 1.0), Raw(0.0, 1.0)]], _notes((0.0, 1.0), (1.0, 2.0)))
