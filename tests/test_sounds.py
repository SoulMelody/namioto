# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the natural Sounds of a `.krc`: what reads as one Sound, and where it is written."""

from __future__ import annotations

import pytest

from namioto.karaoke.model import KrcError
from namioto.karaoke.sounds import natural_sounds, natural_tokens
from namioto.utils import kana_tokens


def _labels(text):
    return [sound.label for line in natural_sounds(text) for sound in line.sounds]


def _tokens(text):
    return natural_tokens(natural_sounds(text))


def test_plain_kana_is_one_sound_per_character():
    assert _tokens("しんぶん") == ["shi", "n", "bu", "n"]


def test_a_long_vowel_is_a_sound_of_its_own():
    assert _tokens("コーヒー") == ["ko", "o", "hi", "i"]
    assert _tokens("ローラ") == ["ro", "o", "ra"]


def test_a_sokuon_is_a_sound_of_its_own():
    assert _tokens("がっこう") == ["ga", "k", "ko", "u"]
    assert _tokens("ちょっと") == ["cho", "t", "to"]


def test_a_lone_long_vowel_or_sokuon_still_reads():
    assert _tokens("んー") == ["n", "n"]
    assert _tokens("ーあ") == ["-", "a"]
    assert _tokens("あっっ") == ["a", "'", "'"]


def test_a_small_kana_rides_the_sound_before_it():
    assert _tokens("きょう") == ["kyo", "u"]
    assert _tokens("ヴァイオリン") == ["va", "i", "o", "ri", "n"]
    assert _labels("きょう") == ["きょ", "う"]
    assert _labels("ワイドショー") == ["ワ", "イ", "ド", "ショ", "ー"]


def test_a_leading_small_kana_stands_alone():
    assert _tokens("ゃあ") == ["ya", "a"]
    assert _labels("ゃあ") == ["ゃ", "あ"]


def test_a_run_of_latin_letters_is_one_sound():
    assert _tokens("hello") == ["hello"]
    assert _tokens("hello world") == ["hello", "world"]
    assert _labels("hello") == ["hello"]


def test_a_digit_is_a_sound_of_its_own():
    assert _tokens("123") == ["1", "2", "3"]
    assert _tokens("abc123") == ["abc", "1", "2", "3"]


def test_punctuation_and_symbols_carry_no_sound():
    assert _tokens("わ、を") == ["wa", "o"]
    assert _tokens("あ！い") == ["a", "i"]
    assert _labels("わ、を") == ["わ", "を"]


def test_an_input_dot_changes_nothing():
    assert _tokens("あ.2") == _tokens("あ")
    assert _tokens("(あい).1") == _tokens("あい")


@pytest.mark.parametrize(
    ("plain", "linked"),
    [
        ("泣[な]い", "泣[な]い.+"),
        ("あいう", "あい.+う.+"),
        ("あいう", "あ(いう).+"),
        ("あ字[いう]", "あ字[いう].+"),
        ("文字[い,う]", "文字[い,う.+]"),
        ("あhello", "あ(hello).+"),
    ],
)
def test_input_continuations_do_not_change_sounds_or_tokens(plain, linked):
    assert natural_sounds(linked) == natural_sounds(plain)
    assert _tokens(linked) == _tokens(plain)


def test_an_input_group_changes_nothing_but_its_readings():
    # `(しょう)` is one Sound, `しょう` is two: the parentheses only carry a reading or a mapping
    assert _tokens("(しょ)") == ["sho"]
    assert _tokens("(しょう)") == ["sho", "u"]
    assert _tokens("しょう") == ["sho", "u"]
    assert _tokens("胡椒[こ,(しょう)]") == ["ko", "sho", "u"]
    assert _tokens("胡椒[こ,しょう]") == ["ko", "sho", "u"]


def test_a_label_puts_the_reading_beside_the_base_unit():
    assert _labels("世界[せ,かい]") == ["せ(世)", "か[界]", "い[界]"]
    assert _labels("百合[ゆり]") == ["ゆ(百合)", "り(百合)"]


def test_a_sound_keeps_the_container_and_characters_it_is_written_from():
    line = natural_sounds("胡椒[こ,(しょう)]")[0]
    readings = [sound.reading for sound in line.sounds]
    assert readings == ["こ", "しょ", "う"]
    assert [line.containers[sound.container].key for sound in line.sounds] == [
        ("part", 0, 0),
        ("part", 0, 1),
        ("part", 0, 1),
    ]
    first, second, third = line.sounds
    assert (first.first_atom, first.last_atom) == (0, 0)
    assert (second.first_atom, second.last_atom) == (0, 1)
    assert (third.first_atom, third.last_atom) == (2, 2)


def test_top_level_runs_are_containers_of_their_own():
    line = natural_sounds("きょう")[0]
    assert line.containers[0].key == ("top", 0)
    assert [sound.container for sound in line.sounds] == [0, 0]


def test_a_ruby_part_is_a_container_of_its_own():
    line = natural_sounds("世界[せ,かい]")[0]
    assert [container.key for container in line.containers] == [("part", 0, 0), ("part", 0, 1)]
    assert [sound.container for sound in line.sounds] == [0, 1, 1]


def test_no_character_is_banned_from_being_a_sound():
    # a sokuon, a long vowel and a moraic n all read as Sounds of their own; nothing is filtered
    assert _tokens("っッーん") == ["'", "'", "'", "n"]
    assert len(_tokens("ーっん")) == 3


def test_a_kanji_without_a_ruby_is_refused():
    with pytest.raises(KrcError):
        natural_sounds("世界")


def test_the_track_and_every_chapter_become_lines():
    lines = natural_sounds("あ\n---\n@2{い}")
    assert [line.text for line in lines] == ["あ", "い"]
    assert natural_tokens(lines) == ["a", "i"]
    assert [sound.line for line in lines for sound in line.sounds] == [0, 1]


def test_a_line_with_no_sound_is_kept_and_maps_nothing():
    lines = natural_sounds("わ、を\n\n！？")
    assert [len(line.sounds) for line in lines] == [2, 0]


CASES = [
    "しんぶん",
    "コーヒー",
    "ラーメン",
    "がっこう",
    "きょう",
    "ちょっと",
    "ヴァイオリン",
    "季節[き,せつ]は移[うつ]ろい",
    "明日[あした]を描[えが]く",
    "映[うつ]し出[だ]す",
    "hello world",
    "abc123",
    "123",
    "One two",
    "わ、を",
    "あ！い",
    "ローラ",
    "ブランケット",
    "今日[きょう]は",
    "胡椒[こ,(しょう)]",
]


@pytest.mark.parametrize("text", CASES)
def test_the_tokens_are_the_folded_readings(text):
    lines = natural_sounds(text)
    kana = "".join(sound.reading for line in lines for sound in line.sounds)
    assert "".join(natural_tokens(lines)) == "".join(kana_tokens(kana))


def test_split_tokens_cuts_the_aligner_stream_into_lines():
    from namioto.karaoke.sounds import split_tokens

    class _Token:
        def __init__(self, start, end):
            self.start, self.end = start, end

    lines = natural_sounds("あい\nう")
    tokens = [_Token(0.0, 1.0), _Token(1.0, 2.0), _Token(2.0, 3.0)]
    assert split_tokens(tokens, lines) == [[(0.0, 1.0), (1.0, 2.0)], [(2.0, 3.0)]]
