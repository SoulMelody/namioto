# SPDX-License-Identifier: AGPL-3.0-only
"""The `.krc` model: words, groups, units, rubies, lines, chapters and the mora they add up to.

A `Word` is a single character, a `Group` is a run of them - the `(...)` the syntax writes, or the
run a normalization pass folds together. A `Unit` carries a ruby and a `.N` or `.+`: a base (a
`Word` or a `Group`) with an optional reading, mora override or `.+` continuation. Lines hold units,
chapters hold lines, and `Lyrics` holds chapters.

Mora is never stored - `Unit.mora`, `Unit.base_mora` and `Ruby.total_mora` are derived - so a change
to the text, the ruby or the override is reflected everywhere at once.

Qt-free.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass

KANJI = re.compile(r"[\u4e00-\u9faf々]+$")
LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]+")
IGNORED = "ャュョァィゥェォゃゅょぁぃぅぇぉゎヮ 「」『』、。．・♥☆※；…‥？！：（）〔〕“”‘’"


class KrcError(ValueError):
    """A `.krc` that cannot be parsed or does not hold together."""


def calc_mora(text: str) -> int:
    """One mora per printable character that carries a sound; punctuation and symbols carry none."""
    mora = 0
    for char in text:
        if not char.isprintable() or char in IGNORED:
            continue
        if unicodedata.category(char)[0] in ("P", "S"):
            continue
        mora += 1
    return mora


@dataclass
class Word:
    """One character: the atom a group is made of."""

    char: str
    channel: int = 0
    color_channel: int | None = None

    @property
    def text(self) -> str:
        return self.char

    def is_kanji(self) -> bool:
        return bool(KANJI.match(self.char))

    def is_latin(self) -> bool:
        return bool(LATIN.fullmatch(self.char))


@dataclass
class Group:
    """A run of words: the syntax's `(...)`, or what a normalization pass folds into one base.

    `explicit` marks the ones the file wrote between parentheses, which `dumps` writes back; a group
    a pass folded together is written plain whenever its text does not need the parentheses.
    """

    words: list[Word]
    explicit: bool = False

    @property
    def text(self) -> str:
        return "".join(word.char for word in self.words)

    def is_kanji(self) -> bool:
        return bool(self.words) and all(word.is_kanji() for word in self.words)

    def is_latin(self) -> bool:
        return bool(self.words) and all(word.is_latin() for word in self.words)


@dataclass
class Unit:
    """A base with its reading, `.N` override or `.+` continuation of the preceding mapping unit."""

    base: Word | Group
    ruby: Ruby | None = None
    override: int | None = None
    join_previous: bool = False

    @property
    def text(self) -> str:
        return self.base.text

    @property
    def grouped(self) -> bool:
        return isinstance(self.base, Group)

    @property
    def base_mora(self) -> int:
        return self.ruby.total_mora(base=True) if self.ruby is not None else calc_mora(self.text)

    @property
    def natural_mora(self) -> int:
        """The mora before mapping annotations, with a Latin run counting one."""
        if self.ruby is not None:
            return sum(unit.natural_mora for part in self.ruby.parts for unit in part)
        return 1 if self.base.is_latin() else calc_mora(self.text)

    @property
    def mora(self) -> int:
        if self.join_previous:
            return 0
        if self.override is not None:
            return self.override
        return self.ruby.total_mora() if self.ruby is not None else self.natural_mora

    @property
    def is_ruby_mora(self) -> bool:
        return self.ruby is not None and self.override is None and not self.join_previous

    def is_kanji(self) -> bool:
        return self.base.is_kanji()

    def is_latin(self) -> bool:
        return self.base.is_latin()

    def set_ruby(self, ruby: Ruby) -> None:
        if any(unit.ruby is not None for part in ruby.parts for unit in part):
            raise KrcError("a ruby may not carry a ruby of its own")
        self.ruby = ruby

    def override_mora(self, mora: int) -> None:
        if self.ruby is not None and self.base_mora != self.mora:
            raise KrcError("a ruby's mora may not be overridden twice")
        if self.base_mora == mora:
            warnings.warn(f"overriding mora to the same value {mora} for word '{self.text}'", stacklevel=2)
        self.override = mora

    def __str__(self) -> str:
        text = self.text
        if self.ruby is not None:
            text += f"[{self.ruby}]"
        if self.join_previous:
            text += ".+"
        elif self.mora > 2:
            text += f".{self.mora}"
        return text


@dataclass
class Ruby:
    """The ruby of a unit: its comma-separated parts, each a run of units."""

    parts: list[list[Unit]]

    def total_mora(self, base: bool = False) -> int:
        return sum(unit.base_mora if base else unit.mora for part in self.parts for unit in part)

    def __str__(self) -> str:
        return "".join(str(unit) for part in self.parts for unit in part)


@dataclass
class Line:
    """One display line; channel ownership belongs to its characters, not its layout."""

    units: list[Unit]

    @property
    def track(self) -> int:
        channels = {word.channel for unit in self.units for word in unit_words(unit)}
        return next(iter(channels)) if len(channels) == 1 else -1 if channels else 0

    @track.setter
    def track(self, channel: int) -> None:
        for unit in self.units:
            for word in unit_words(unit):
                word.channel = channel

    @property
    def words(self) -> list[Unit]:
        return self.units

    @words.setter
    def words(self, units: list[Unit]) -> None:
        self.units = units

    def total_mora(self, base: bool = False) -> int:
        return sum(unit.base_mora if base else unit.mora for unit in self.units)

    def __str__(self) -> str:
        text = "".join(str(unit) for unit in self.units)
        return f"{text} | {self.total_mora()}"


@dataclass
class Chapter:
    """A chapter: its lines, each with the track it is sung on."""

    lines: list[Line]

    def __str__(self) -> str:
        return "\n".join(str(line) for line in self.lines)


@dataclass
class Lyrics:
    """A whole `.krc`: its chapters."""

    chapters: list[Chapter]

    def __str__(self) -> str:
        return "\n----\n".join(str(chapter) for chapter in self.chapters)


def unit_words(unit: Unit) -> list[Word]:
    words = list(unit.base.words) if isinstance(unit.base, Group) else [unit.base]
    if unit.ruby is not None:
        words.extend(word for part in unit.ruby.parts for inner in part for word in unit_words(inner))
    return words


def validate(lyrics: Lyrics) -> None:
    for chapter in lyrics.chapters:
        for line in chapter.lines:
            for unit in line.units:
                _validate_unit(unit)
                if unit.ruby is not None and len(unit.ruby.parts) > 1 and len(unit.ruby.parts) != len(unit.text):
                    raise KrcError(
                        f"the ruby of '{unit.text}' has {len(unit.ruby.parts)} parts for {len(unit.text)} characters"
                    )
            if any(_has_join(unit) for unit in line.units):
                from namioto.karaoke.sounds import _line_sounds, note_groups

                note_groups(line, _line_sounds(line, 0, 0))


def _has_join(unit: Unit) -> bool:
    return unit.join_previous or (
        unit.ruby is not None and any(_has_join(inner) for part in unit.ruby.parts for inner in part)
    )


def _validate_unit(unit: Unit) -> None:
    if unit.join_previous and unit.override is not None:
        raise KrcError("a unit cannot carry both .N and .+")
    if unit.ruby is None:
        return
    for part in unit.ruby.parts:
        for inner in part:
            _validate_unit(inner)
            if unit.join_previous and (inner.override is not None or _has_join(inner)):
                raise KrcError("an outer .+ cannot overlap an inner mapping annotation")
            if unit.override is not None and _has_join(inner):
                raise KrcError("an outer .N cannot overlap an inner .+")
