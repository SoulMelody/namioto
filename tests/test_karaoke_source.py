# SPDX-License-Identifier: AGPL-3.0-only
"""Concrete KRC syntax survives ordinary edits and structural changes."""

from copy import deepcopy

import pytest

from namioto.karaoke import dumps, parse
from namioto.karaoke.model import Chapter, Line, Lyrics, Unit, Word
from namioto.karaoke.transforms import merge_words


@pytest.mark.parametrize(
    "text",
    [
        "# header\n\nあ  い # inline\n\n-----\n# chapter\nう\n# footer\n",
        "# header\r\n@00{あ  い}\r\n\r\n------\r\nう\r\n",
        "@2{\n# comment\nあ\nい\n}\n",
        "あ\f\xa0い\n",
        "あ# no final newline",
        "(Turn).1",
        "@0{@c02{あ}}",
    ],
)
def test_unchanged_source_is_exact(text):
    model = parse(text)
    assert dumps(model) == text
    assert dumps(model, dotted=True) == text
    assert dumps(deepcopy(model)) == text
    assert dumps(merge_words(model)) == text


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_mora_edit_only_changes_its_annotation(newline):
    text = newline.join(["# あ.2", "@02{あ.2  い} # あ.2", "", "-----", "う", ""])
    model = parse(text)
    model.chapters[0].lines[0].units[0].override = 3
    expected = text.replace("@02{あ.2", "@02{あ.3")
    assert dumps(model) == expected
    assert dumps(parse(expected)) == expected


def test_edit_inside_a_multiline_owner_keeps_the_wrapper():
    text = "@02{\n# comment\nあ\nい\n}\n"
    model = parse(text)
    model.chapters[0].lines[0].units[0].override = 2
    assert dumps(model) == text.replace("あ\n", "あ.2\n")


def test_ruby_edit_keeps_comments_and_spacing():
    text = "# reading\n星[ほし]  あ # note\n"
    model = parse(text)
    model.chapters[0].lines[0].units[0].ruby.parts[0][0].base.char = "ぼ"
    assert dumps(model) == text.replace("ほし", "ぼし")


def test_deleting_a_line_keeps_its_inline_comment():
    model = parse("# header\nあ # keep\n\nい\n# footer\n")
    del model.chapters[0].lines[0]
    assert dumps(model) == "# header\n# keep\n\nい\n# footer\n"


def test_deleting_all_lyrics_keeps_comments():
    model = parse("# header\nあ # keep\n")
    model.chapters.clear()
    assert dumps(model) == "# header\n# keep\n"


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_inserted_line_uses_default_syntax_and_preserves_final_newline(ending):
    model = parse("# header\nあ" + ending)
    model.chapters[0].lines.append(Line([Unit(Word("い"))]))
    newline = "\r\n" if ending == "\r\n" else "\n"
    assert dumps(model) == "# header\nあ" + newline + "い" + ending


def test_split_chapter_keeps_comments_and_untouched_separator():
    model = parse("あ # first\nい\n\n-----\n# second\nう\n")
    line = model.chapters[0].lines.pop()
    model.chapters.insert(1, Chapter([line]))
    result = dumps(model)
    assert result == "あ # first\n---\nい\n\n-----\n# second\nう\n"
    assert len(parse(result).chapters) == 3


def test_channel_rewrite_keeps_comments_when_multiline_wrappers_must_change():
    model = parse("# header\n@2{\nあ # first\n\n# between\nい\n}\n# footer\n")
    model.chapters[0].lines[0].track = 3
    result = dumps(model)
    assert [line.track for line in parse(result).chapters[0].lines] == [3, 2]
    assert result == "# header\n# first\n@3{あ }\n\n# between\n@2{い}\n# footer\n"


def test_new_models_still_use_canonical_syntax():
    model = Lyrics([Chapter([Line([Unit(Word("あ"), override=1)])])])
    assert dumps(model) == "あ"
    assert dumps(model, dotted=True) == "あ.1"
