# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the `.krc` parser: chapters, tracks, ruby, mora and the errors it refuses."""

from __future__ import annotations

import pytest

from namioto.karaoke import Group, Lyrics, Unit, Word, flatten_ruby, merge_words, parse

SAMPLE = """青[あお]い(星)[ほし]見[み]つめ
ねえ# comment
@2{(手)[て]を伸[の]ばす}

---

(幾千)[いくせん].1の(瞬)[またた]き
何度目[なん,ど,め]の(朝)[あさ]
"""


def test_parse_splits_the_file_into_chapters_and_lines():
    lyrics = parse(SAMPLE)
    assert isinstance(lyrics, Lyrics)
    assert [len(chapter.lines) for chapter in lyrics.chapters] == [3, 2]
    assert [line.track for line in lyrics.chapters[0].lines] == [0, 0, 2]
    assert [line.total_mora() for line in lyrics.chapters[0].lines] == [8, 2, 5]


def test_a_ruby_is_attached_to_the_run_of_kanji_before_it():
    word = parse("何度目[なん,ど,め]").chapters[0].lines[0].words[0]
    assert word.text == "何度目"
    assert word.mora == 4
    assert [sum(word.mora for word in part) for part in word.ruby.parts] == [2, 1, 1]


def test_parens_attach_a_ruby_to_a_multi_character_word():
    word = parse("(星)[ほし]").chapters[0].lines[0].words[0]
    assert word.text == "星"
    assert word.mora == 2
    assert word.is_ruby_mora


def test_a_dot_overrides_the_mora_count():
    word = parse("(幾千)[いくせん].1").chapters[0].lines[0].words[0]
    assert word.base_mora == 4
    assert word.mora == 1
    assert not word.is_ruby_mora


def test_a_comment_runs_to_the_end_of_its_line():
    lyrics = parse("ねえ# comment\nきみ")
    assert [word.text for word in lyrics.chapters[0].lines[0].words] == ["ね", "え"]
    assert [word.text for word in lyrics.chapters[0].lines[1].words] == ["き", "み"]


def test_a_ruby_with_the_wrong_number_of_parts_is_rejected():
    with pytest.raises(ValueError):
        parse("(幾千)[いく,せん,の]")


def test_a_nested_ruby_is_rejected():
    with pytest.raises(ValueError):
        parse("見[み[た]]")


def test_a_malformed_file_is_rejected():
    with pytest.raises(ValueError):
        parse("見[み")


def test_overriding_a_ruby_to_the_same_mora_warns():
    with pytest.warns(UserWarning):
        parse("(幾千)[いくせん].4")


def test_flatten_ruby_gives_one_word_per_mora():
    original = parse("何度目[なん,ど,め]")
    lyrics = flatten_ruby(original)
    words = lyrics.chapters[0].lines[0].words
    assert [word.text for word in words] == ["何", "#", "#度", "#目"]
    assert all(word.mora == 1 and word.ruby is not None for word in words)
    assert [word.text for word in original.chapters[0].lines[0].words] == ["何度目"]


def test_merge_folds_the_runs_unless_it_is_turned_off():
    assert [word.text for word in parse("何度目[なん,ど,め]").chapters[0].lines[0].words] == ["何度目"]
    assert [word.text for word in parse("hello").chapters[0].lines[0].words] == ["hello"]

    raw = parse("何度目[なん,ど,め] hello", merge=False).chapters[0].lines[0].words
    assert [word.text for word in raw] == ["何", "度", "目", " ", "h", "e", "l", "l", "o"]

    merged = merge_words(parse("何度目[なん,ど,め] hello", merge=False))
    assert [word.text for word in merged.chapters[0].lines[0].words] == ["何度目", " ", "hello"]


def test_a_run_of_latin_letters_is_one_word_of_one_mora():
    words = parse("hello世界").chapters[0].lines[0].words
    assert [word.text for word in words] == ["hello", "世", "界"]
    assert words[0].mora == 1


def test_full_width_latin_letters_merge_too():
    words = parse("Ｈｅｌｌｏ").chapters[0].lines[0].words
    assert [word.text for word in words] == ["Ｈｅｌｌｏ"]
    assert words[0].mora == 1


def test_a_dot_overrides_a_latin_run():
    word = parse("(hello).3").chapters[0].lines[0].words[0]
    assert word.text == "hello"
    assert word.base_mora == 5
    assert word.mora == 3


def test_a_dot_after_a_bare_run_overrides_the_whole_run():
    word = parse("hello.3").chapters[0].lines[0].words[0]
    assert word.text == "hello"
    assert word.base_mora == 5
    assert word.mora == 3


def test_a_grouped_word_stops_the_latin_merge():
    words = parse("(F)(LO)(WER)").chapters[0].lines[0].words
    assert [word.text for word in words] == ["F", "LO", "WER"]
    assert [word.mora for word in words] == [1, 1, 1]


def test_an_overridden_grouped_word_keeps_its_base_mora():
    words = parse("(F).1(LO).1(WER).1").chapters[0].lines[0].words
    assert [word.text for word in words] == ["F", "LO", "WER"]
    assert [word.base_mora for word in words] == [1, 2, 3]
    assert [word.mora for word in words] == [1, 1, 1]


def test_a_word_is_one_character_and_parentheses_make_an_explicit_group():
    units = parse("(AB)c").chapters[0].lines[0].units
    assert isinstance(units[0], Unit)
    assert isinstance(units[0].base, Group) and units[0].base.explicit
    assert [word.char for word in units[0].base.words] == ["A", "B"]
    assert isinstance(units[1].base, Word) and units[1].base.char == "c"


def test_a_kanji_run_with_a_ruby_is_an_auto_group():
    unit = parse("季節[き,せつ]").chapters[0].lines[0].units[0]
    assert isinstance(unit.base, Group) and not unit.base.explicit
    assert unit.text == "季節"


@pytest.mark.parametrize(
    "text",
    ["泣[な]い.+", "あい.+う.+", "あ(いう).+", "あ字[いう].+", "文字[い,う.+]", "泣[な]、い.+", "あhello.+"],
)
def test_continuations_reuse_one_note_without_changing_natural_mora(text):
    line = parse(text).chapters[0].lines[0]
    assert line.total_mora() == 1
    assert sum(unit.natural_mora for unit in line.units) >= 2
    assert parse(text, merge=False)


def test_a_continuation_suffix_binds_only_the_preceding_unit():
    units = parse("あいう.+").chapters[0].lines[0].units
    assert [unit.join_previous for unit in units] == [False, False, True]
    assert [unit.mora for unit in units] == [1, 1, 0]
    unit = parse("字[いう.+]").chapters[0].lines[0].units[0]
    assert unit.natural_mora == 2
    assert unit.mora == 1


@pytest.mark.parametrize(
    "text",
    [
        "い.+",
        "あ\nい.+",
        "あ\n---\nい.+",
        "あ.0い.+",
        "あ.2い.+",
        "(あい).3う.+",
        "あい.1.+",
        "あい.+.1",
        "あい.+.+",
        "あ字[い.0う].+",
        "あ字[いう.+].+",
        "字[いう.+].1",
        "字[いう.+].2",
        "字[いう.+].0",
        "あ、.+",
        "あしょ.+",
        "あ(いき).+ゃ",
    ],
)
def test_invalid_continuations_are_rejected(text):
    with pytest.raises(ValueError):
        parse(text)


def test_parentheses_around_one_word_change_nothing():
    plain = parse("あ").chapters[0].lines[0].units[0]
    grouped = parse("(あ)").chapters[0].lines[0].units[0]
    assert grouped.text == plain.text
    assert grouped.mora == plain.mora
