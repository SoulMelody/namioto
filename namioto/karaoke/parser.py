# SPDX-License-Identifier: AGPL-3.0-only
"""The parser for a `.krc`: its chapters, lines, words, ruby and mora.

`@N{...}` adds channel ownership; `@cN{...}` changes only its colour. `---` splits chapters,
`漢字[かんじ]` and `(word)[ruby]` attach a ruby, a comma splits that ruby per character,
`.N` overrides a word's mora count, `.+` joins its Sounds to the preceding mapping unit, and `#` starts a
comment. A word carries the mora its ruby gives it; a run of kanji before a ruby is merged into
the word the ruby belongs to, so a bare
`季節[き,せつ]` reads the same as `(季節)[き,せつ]`, and a run of Latin letters (half- or full-width)
is one word of one mora.

`parse()` builds the model in `model.py` and validates it. Qt-free.
"""

from __future__ import annotations

import re

from lark import Lark, Transformer
from lark.exceptions import LarkError, VisitError

from namioto.karaoke.model import Chapter, Group, KrcError, Line, Lyrics, Ruby, Unit, Word, _has_join
from namioto.karaoke.transforms import merge_words

GRAMMAR = r"""
start: _NEWLINE? chapter (_SEP chapter)*
chapter: chapter_line (_NEWLINE chapter_line)* _NEWLINE?
chapter_line: words
words: (word|word_ruby|word_mora|word_join)+
ruby_line: words ("," words)*

word: WORD | OPEN WORD+ CLOSE
word_ruby: word "[" ruby_line "]"
word_mora: (word|word_ruby) "." INTEGER
word_join: (word|word_ruby) "." "+"
OPEN: "("
CLOSE: ")"

COMMENT: "#" /[^\n]/*
WORD: /[^#\n\[\]\(\)\.\-\,{}]/
INTEGER: /\d+/

_NEWLINE: /\n+/
# one token for a chapter break, its leading newline included, so the grammar stays LALR: split off,
# the newline is the lookahead for both ending a chapter and continuing it
_SEP: /\n+-{3,}-*\n*/

%ignore /[\f\r\xa0]/
%ignore COMMENT
"""


class LyricsTransformer(Transformer):
    """Builds the model out of the parse tree."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def chapter_line(self, items):
        return Line(items[0])

    def words(self, items):
        return items

    def ruby_line(self, items):
        return Ruby(items)

    def word(self, items):
        words = [
            Word(str(item), *self.channels[item.start_pos]) for item in items if item.type not in ("OPEN", "CLOSE")
        ]
        if any(item.type in ("OPEN", "CLOSE") for item in items):
            return Unit(Group(words, explicit=True))
        return Unit(words[0])

    def chapter(self, items):
        return Chapter(items)

    def word_ruby(self, items):
        word, ruby = items
        word.set_ruby(ruby)
        return word

    def word_mora(self, items):
        word, mora = items
        word.override_mora(int(mora.value))
        return word

    def word_join(self, items):
        word = items[0]
        word.join_previous = True
        return word

    def start(self, items):
        return Lyrics(items)


parser = Lark(GRAMMAR, start="start", parser="lalr")
CHANNEL_BLOCK = re.compile(r"@(c?)(\d+)\{")


def _unwrap(text: str) -> tuple[str, list[tuple[int, int | None]], list[int]]:
    chars, channels, positions = [], [], []
    channel = (0, None)
    opened = []
    at = 0
    while at < len(text):
        if text[at] == "#":
            end = text.find("\n", at)
            end = len(text) if end < 0 else end
            line_start = text.rfind("\n", 0, at) + 1
            if not text[line_start:at].strip():
                at = end + (end < len(text))
                continue
            chars.extend(text[at:end])
            channels.extend([channel] * (end - at))
            positions.extend(range(at, end))
            at = end
            continue
        match = CHANNEL_BLOCK.match(text, at) if text[at] == "@" else None
        if match:
            kind = bool(match[1])
            if opened and (not kind or any(previous_kind for previous_kind, _ in opened)):
                raise KrcError("only a color block inside an ownership block may be nested")
            number = int(match[2])
            if not 0 <= number < 16:
                raise KrcError("a channel must be between 0 and 15")
            opened.append((kind, channel))
            channel = (channel[0], number) if kind else (number, channel[1])
            at = match.end()
            continue
        if text[at] == "}" and opened:
            _, channel = opened.pop()
            at += 1
            continue
        if text[at] not in "\r\f\xa0":
            chars.append(text[at])
            channels.append(channel)
            positions.append(at)
        at += 1
    if opened:
        raise KrcError("unclosed channel block")
    positions.append(len(text))
    return "".join(chars), channels, positions


def parse(text: str, merge: bool = True) -> Lyrics:
    """Parse a `.krc` into its model; a malformed file raises `KrcError`."""
    unwrapped, channels, positions = _unwrap(text)
    try:
        tree = parser.parse(unwrapped)
    except LarkError as error:
        offset = getattr(error, "pos_in_stream", None)
        if offset is not None:
            original = positions[min(offset, len(positions) - 1)] if offset >= 0 else positions[-1]
            line = text.count("\n", 0, original) + 1
            column = original - text.rfind("\n", 0, original)
            raise KrcError(f"line {line}, column {column}: {error}") from None
        raise KrcError(str(error)) from None
    try:
        lyrics = LyricsTransformer(channels).transform(tree)
    except VisitError as error:
        if isinstance(error.orig_exc, ValueError):
            raise error.orig_exc from None
        raise
    if merge:
        lyrics = merge_words(lyrics)
    elif any(_has_join(unit) for chapter in lyrics.chapters for line in chapter.lines for unit in line.units):
        merge_words(lyrics)
    if not merge:
        return lyrics
    from namioto.karaoke.source import Source
    from namioto.karaoke.writer import _canonical

    starts = [
        next(line.scan_values(lambda value: hasattr(value, "start_pos"))).start_pos
        for line in tree.find_data("chapter_line")
    ]
    starts.extend(match.start() + match[0].index("-") for match in re.finditer(r"\n+-{3,}\n*", unwrapped))
    line_numbers = tuple(text.count("\n", 0, positions[start]) for start in sorted(starts))
    lyrics.source = Source(text, _canonical(lyrics, dotted=True), line_numbers)
    return lyrics
