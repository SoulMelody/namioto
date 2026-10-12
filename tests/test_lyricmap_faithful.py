# SPDX-License-Identifier: AGPL-3.0-only
"""Checks the KRC's faithful NOTE-slot reading without estimation."""

from __future__ import annotations

import pytest

from namioto.document import Note
from namioto.lyricmap.faithful import consumes, lay_out, read
from namioto.lyricmap.verify import faithful_gate


def _notes(*spans):
    return [Note(60, start, end - start, id=index + 1) for index, (start, end) in enumerate(spans)]


def test_a_dot_holds_one_sound_over_several_notes():
    assert read("あ.2", _notes((0.0, 1.0), (1.0, 2.0))) == [[(0.0, 2.0)]]


def test_lay_out_names_the_notes_each_sound_takes():
    spans, ids = lay_out("あ.2", _notes((0.0, 1.0), (1.0, 2.0)))
    assert spans == [[(0.0, 2.0)]]
    assert ids == [[(1, 2)]]


def test_lay_out_shares_one_note_between_the_sounds_that_tile_it():
    spans, ids = lay_out("(あい).1", _notes((0.0, 1.0)))
    assert spans == [[(0.0, 0.5), (0.5, 1.0)]]
    assert ids == [[(1,), (1,)]]


def test_a_group_slots_several_sounds_onto_one_note():
    assert read("(あい).1", _notes((0.0, 1.0))) == [[(0.0, 0.5), (0.5, 1.0)]]


def test_small_kana_stays_one_sound():
    assert read("京[きょう]", _notes((0.0, 1.0), (1.0, 2.0))) == [[(0.0, 1.0), (1.0, 2.0)]]


def test_a_grouped_ruby_keeps_its_own_dot():
    rows = read("青色[あ,(お).2]", _notes((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)))
    assert rows == [[(0.0, 1.0), (1.0, 3.0)]]


def test_notes_running_out_leave_no_time():
    assert read("あい", _notes((0.0, 1.0))) == [[(0.0, 1.0), None]]


def test_a_multi_note_sound_displays_remaining_slots_without_reducing_its_request():
    notes = _notes((0.0, 1.0))
    spans, ids = lay_out("あ.2い", notes)
    assert spans == [[(0.0, 1.0), None]]
    assert ids == [[(1,), ()]]
    assert consumes("あ.2い", notes) == 3
    assert faithful_gate(spans, notes, required_slots=3).counts() == {"incomplete_alignment": 1}


def test_an_input_dot_is_respected_even_where_edit_mode_ignores_it():
    assert read("あ.0", _notes((0.0, 1.0))) == [[None]]


def test_consumes_counts_the_slots_the_reading_asks_for():
    assert consumes("あ.2", _notes((0.0, 1.0), (1.0, 2.0))) == 2
    assert consumes("(あい).1", _notes((0.0, 1.0))) == 1
    assert consumes("あい", _notes((0.0, 1.0), (1.0, 2.0))) == 2
    assert consumes("", []) == 0


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("泣[な]い.+", 2),
        ("あい.+う.+", 3),
        ("あ(いう).+", 3),
        ("あ字[いう].+", 3),
        ("文字[い,う.+]", 2),
        ("泣[な]、い.+", 2),
        ("(あい).1う.+", 3),
    ],
)
def test_continuations_share_one_note_and_redivide_the_whole_group(text, count):
    spans, ids = lay_out(text, _notes((0.0, 1.0)))
    assert ids == [[(1,)] * count]
    assert len(spans[0]) == count
    for index, span in enumerate(spans[0]):
        assert span == pytest.approx((index / count, (index + 1) / count))
    assert consumes(text, []) == 1
    assert read(text, []) == [[None] * count]


def test_a_continuation_targets_the_last_sound_not_the_whole_ruby():
    spans, ids = lay_out("字[いう]え.+", _notes((0.0, 1.0), (1.0, 2.0)))
    assert spans == [[(0.0, 1.0), (1.0, 1.5), (1.5, 2.0)]]
    assert ids == [[(1,), (2,), (2,)]]
    assert consumes("字[いう]え.+", []) == 2


def test_a_continuation_can_follow_a_single_note_sound_inside_a_multi_note_ruby():
    spans, ids = lay_out("字[い.2う]え.+", _notes((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)))
    assert spans == [[(0.0, 2.0), (2.0, 2.5), (2.5, 3.0)]]
    assert ids == [[(1, 2), (3,), (3,)]]
    assert consumes("字[い.2う]え.+", []) == 3


def test_ruby_and_top_level_runs_keep_independent_character_offsets():
    assert read("青[あお]いう", _notes((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0))) == [
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    ]


def test_notes_running_out_inside_a_shared_unit_leave_the_rest_untimed():
    assert read("(あいう).2", _notes((0.0, 1.0))) == [[(0.0, 1.0), None, None]]


def test_the_read_only_subtitle_gate():
    notes = _notes((0.0, 1.0))
    assert faithful_gate(read("あ", notes), notes).open()
    assert faithful_gate(read("あい", notes), notes).counts() == {"incomplete_alignment": 1}
    assert faithful_gate(read("あ", []), []).counts() == {"no_target_notes": 1, "incomplete_alignment": 1}
    assert faithful_gate(read("あ", notes), notes, filtered=notes).counts() == {"filtered_note": 1}
