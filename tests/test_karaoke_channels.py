# SPDX-License-Identifier: AGPL-3.0-only
"""Channel wrappers annotate characters without changing the original KRC grammar."""

from dataclasses import asdict

import pytest

from namioto.karaoke import dumps, flatten_ruby, parse
from namioto.karaoke.canonical import rebuild
from namioto.karaoke.model import unit_words
from namioto.karaoke.operations import Match, SoundRef
from namioto.karaoke.sounds import natural_sounds


def structure(value):
    if isinstance(value, dict):
        return {key: structure(item) for key, item in value.items() if key != "channel"}
    if isinstance(value, list):
        return [structure(item) for item in value]
    return value


@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize(
    "text",
    [
        "あいう",
        "季節[き,せつ]",
        "(あいう).2",
        "あい.+",
        "文字[い,う.+]",
        "hello.3 world",
        "あ\nい\n---\nう",
        "あ # ignored @99{\nい",
    ],
)
def test_wrapping_changes_only_ownership(text, merge):
    plain = parse(f"\n{text}\n", merge=merge)
    owned = parse(f"@3{{\n{text}\n}}", merge=merge)
    assert structure(asdict(owned)) == structure(asdict(plain))
    assert {
        word.channel
        for chapter in owned.chapters
        for line in chapter.lines
        for unit in line.units
        for word in unit_words(unit)
    } == {3}


@pytest.mark.parametrize(
    "owned,plain",
    [
        ("@1{hel}@2{lo}", "hello"),
        ("@1{季}節[き,せつ]", "季節[き,せつ]"),
        ("(@1{あ}@2{い}).1", "(あい).1"),
        ("字[@1{あ}@2{い}]", "字[あい]"),
        ("@1{あ}@2{い.+}", "あい.+"),
        ("あ@1{\nい\n}う", "あ\nい\nう"),
    ],
)
def test_boundaries_do_not_split_or_rebind_original_units(owned, plain):
    assert structure(asdict(parse(owned))) == structure(asdict(parse(plain)))


@pytest.mark.parametrize(
    "text",
    [
        "@1{あ}@2{い}う",
        "@1{\nあ\n---\nい\n}",
        "@1{季}節[き,せつ]",
        "(@1{あ}@2{い}).1",
        "字[@1{あ}@2{い}]",
        "@1{hel}@2{lo}",
        "@1{あ}@2{い.+}",
    ],
)
def test_writer_preserves_structure_and_character_ownership(text):
    lyrics = parse(text)
    written = dumps(lyrics, dotted=True)
    assert parse(written) == lyrics
    assert dumps(parse(written), dotted=True) == written


def test_multiline_scope_preserves_channel_on_each_sound():
    lines = natural_sounds("@1{\nあ\nい\n---\nう\n}\nえ@2{お}")
    assert [[sound.channel for sound in line.sounds] for line in lines] == [[1], [1], [1], [0, 2]]
    assert [line.chapter for line in lines] == [0, 0, 1, 1]


def test_a_sound_spanning_channels_retains_the_conflict_without_changing_parsing():
    assert [sound.channel for sound in natural_sounds("@1{し}@2{ょ}")[0].sounds] == [-1]


def test_canonical_rebuild_preserves_ownership():
    text = "@1{あ}@2{い}\n@3{字[うえ]}"
    operations = [
        Match(SoundRef(0, 0), (10, 20)),
        Match(SoundRef(0, 1), (30,)),
        Match(SoundRef(1, 0), (40,)),
        Match(SoundRef(1, 1), (50, 60)),
    ]
    written = rebuild(text, operations)
    assert [[sound.channel for sound in line.sounds] for line in natural_sounds(written)] == [[1, 2], [3, 3]]
    assert written == "@1{あ.2}@2{い}\n@3{字[うえ.2]}"


@pytest.mark.parametrize("text", ["@16{あ}", "@1{あ", "@1{@2{あ}}", "{1}あ"])
def test_invalid_channel_syntax_is_rejected(text):
    with pytest.raises(ValueError):
        parse(text)


def test_comments_and_ordinary_at_signs_keep_their_original_meaning():
    assert parse("a@b") == parse("@0{a@b}")
    assert parse("あ # @99{ignored}\nい") == parse("あ \nい")


def test_flattening_ruby_preserves_channel_ownership():
    lyrics = flatten_ruby(parse("@2{何度目[なん,ど,め]}"))
    assert {
        word.channel
        for chapter in lyrics.chapters
        for line in chapter.lines
        for unit in line.units
        for word in unit_words(unit)
    } == {2}


def test_error_location_refers_to_wrapped_source():
    with pytest.raises(ValueError, match="line 1, column 7"):
        parse("@1{あ} .")
