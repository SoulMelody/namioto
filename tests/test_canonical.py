# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the canonical `.krc` rebuild: what a mapping writes back, and what it refuses."""

from __future__ import annotations

import pytest

from namioto.karaoke.canonical import rebuild
from namioto.karaoke.model import KrcError
from namioto.karaoke.operations import Drop, Match, Merge, SoundRef, partition
from namioto.karaoke.sounds import natural_sounds, natural_tokens


def _match(line, index, notes):
    return Match(SoundRef(line, index), tuple(notes))


def _merge(line, *indices):
    return Merge(tuple(SoundRef(line, index) for index in indices), 0)


def _drop(line, index):
    return Drop(SoundRef(line, index))


def test_a_match_writes_the_notes_it_takes():
    assert rebuild("あ", [_match(0, 0, [1, 2])]) == "あ.2"
    assert rebuild("あ", [_match(0, 0, [1])]) == "あ"


def test_a_multi_character_sound_is_grouped_before_its_dot():
    assert rebuild("しょ", [_match(0, 0, [1, 2])]) == "(しょ).2"
    assert rebuild("きゃ", [_match(0, 0, [1, 2, 3])]) == "(きゃ).3"
    assert rebuild("しょう", [_match(0, 0, [1, 2]), _match(0, 1, [3])]) == "(しょ).2う"
    assert rebuild("しょう", [_match(0, 0, [1]), _match(0, 1, [2])]) == "しょう"


def test_a_merge_writes_a_group_of_one():
    assert rebuild("あい", [_merge(0, 0, 1)]) == "(あい).1"
    assert rebuild("胡椒[こ,しょう]", [_match(0, 0, [1]), _merge(0, 1, 2)]) == "胡椒[こ,(しょう).1]"


def test_a_drop_writes_zero_on_the_whole_sound():
    assert rebuild("い", [_drop(0, 0)]) == "い.0"
    assert rebuild("しょ", [_drop(0, 0)]) == "(しょ).0"


def test_the_input_dot_and_group_do_not_survive():
    ops = [_match(0, 0, [1]), _match(0, 1, [2])]
    assert rebuild("(しょう).1", ops) == "しょう"
    assert rebuild("(あい).1", ops) == "あい"


def test_a_ruby_gets_its_dot_on_the_part_it_reads():
    assert rebuild("青[あお]", [_match(0, 0, [1, 2]), _match(0, 1, [3])]) == "青[あ.2お]"
    assert rebuild("世界[せ,かい]", [_match(0, 0, [1]), _merge(0, 1, 2)]) == "世界[せ,(かい).1]"


def test_punctuation_stays_between_the_sounds():
    assert rebuild("わ、を", [_match(0, 0, [1]), _match(0, 1, [2])]) == "わ、を"
    assert rebuild("わ、を", [_merge(0, 0, 1)]) == "(わ、を).1"


def test_a_track_and_a_chapter_are_kept():
    text = "あ\n---\n@2{い}"
    assert rebuild(text, [_match(0, 0, [1]), _match(1, 0, [2, 3])]) == "あ\n---\n@2{い.2}"


def test_a_latin_run_is_written_whole():
    assert rebuild("hello", [_match(0, 0, [1, 2])]) == "(hello).2"
    assert rebuild("hello world", [_merge(0, 0, 1)]) == "(hello world).1"


def test_the_rebuilt_text_reads_back_the_same_sounds():
    text = "胡椒[こ,しょう]は"
    rebuilt = rebuild(text, [_drop(0, 0), _merge(0, 1, 2), _match(0, 3, [1, 2])])
    assert natural_tokens(natural_sounds(rebuilt)) == natural_tokens(natural_sounds(text))


@pytest.mark.parametrize(
    ("text", "operations", "expected"),
    [
        ("泣[な]い", [_merge(0, 0, 1)], "泣[な]い.+"),
        ("あ字[いう]", [_merge(0, 0, 1, 2)], "あ字[(いう).+]"),
        ("胡椒[こ,(しょう)]", [_merge(0, 0, 1), _match(0, 2, [2])], "胡椒[こ,(しょ).+う]"),
        ("胡椒[こ,しょう]は", [_match(0, 0, [1]), _match(0, 1, [2]), _merge(0, 2, 3)], "胡椒[こ,しょう]は.+"),
        ("世界[せ,かい]", [_merge(0, 0, 1, 2)], "世界[せ,(かい).+]"),
        ("あ字[いう]え", [_merge(0, 0, 1, 2, 3)], "あ字[(いう).+]え.+"),
        ("字[いう]え", [_match(0, 0, [1]), _merge(0, 1, 2)], "字[いう]え.+"),
        ("字[いう]、え", [_match(0, 0, [1]), _merge(0, 1, 2)], "字[いう]、え.+"),
    ],
)
def test_cross_container_merges_write_continuations(text, operations, expected):
    assert rebuild(text, operations) == expected
    assert natural_sounds(expected) == natural_sounds(text)
    assert rebuild(expected, operations) == expected


def test_input_continuations_are_discarded_when_rebuilding():
    assert rebuild("泣[な]い.+", [_match(0, 0, [1]), _match(0, 1, [2])]) == "泣[な]い"
    assert rebuild("あ字[いう].+", [_match(0, 0, [1]), _match(0, 1, [2]), _match(0, 2, [3])]) == "あ字[いう]"
    assert rebuild("字[いう].1", [_match(0, 0, [1]), _match(0, 1, [2])]) == "字[いう]"


def test_a_mapping_that_leaves_a_sound_out_is_refused():
    with pytest.raises(KrcError):
        rebuild("あい", [_match(0, 0, [1])])


def test_a_mapping_that_names_a_sound_twice_is_refused():
    with pytest.raises(KrcError):
        rebuild("あい", [_match(0, 0, [1]), _match(0, 0, [1])])


def test_a_line_with_no_sound_stays_and_writes_nothing():
    text = "わ、を\n！？"
    assert rebuild(text, [_match(0, 0, [1]), _match(0, 1, [2])]) == "わ、を\n！？"
    assert rebuild(text, [_match(0, 0, [1, 2]), _match(0, 1, [3])]) == "わ.2、を\n！？"


def test_empty_lyrics_rebuild_to_nothing():
    assert rebuild("", []) == ""


def test_a_match_takes_at_least_one_note():
    with pytest.raises(KrcError):
        Match(SoundRef(0, 0), ())


def test_a_merge_takes_at_least_two_sounds():
    with pytest.raises(KrcError):
        Merge((SoundRef(0, 0),), 0)


def test_partition_sorts_by_line_and_checks_the_whole_song():
    operations = [_match(1, 0, [1]), _match(0, 0, [1]), _match(0, 1, [2])]
    rows = partition(operations, [2, 1])
    assert rows[0] == operations[1:]
    assert rows[1] == operations[:1]


def test_partition_refuses_a_merge_that_holds_non_consecutive_sounds():
    with pytest.raises(KrcError):
        partition([Merge((SoundRef(0, 0), SoundRef(0, 2)), 0), _match(0, 1, [1]), _match(0, 3, [1])], [4])


def test_keep_operations_trusts_suggestions_only_at_the_same_version():
    from namioto.karaoke.operations import keep_operations

    operations = [_match(0, 0, [1]), Match(SoundRef(0, 1), (2,), confirmed=True)]
    assert keep_operations(operations, 1, 1) == tuple(operations)
    assert keep_operations(operations, 1, 2) == (Match(SoundRef(0, 1), (2,), confirmed=True),)
