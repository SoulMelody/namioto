# SPDX-License-Identifier: AGPL-3.0-only
"""Write a `Lyrics` model back out as `.krc` text.

A group is written between `(...)`, a single word plain; the model is what survives, and a word
that cannot be written raises `KrcError`. `.+` keeps its continuation relation. Comments, blank
lines and the original spacing are gone.

Qt-free.
"""

from __future__ import annotations

from itertools import groupby

from namioto.karaoke.model import Chapter, Group, KrcError, Line, Lyrics, Unit, _has_join, unit_words, validate

SEPARATOR = "\n---\n"
META = set("#\n[]().-,{}\r\f\xa0")


def dumps(lyrics: Lyrics, *, dotted: bool = False) -> str:
    """Serialise the model to `.krc` text; a word that cannot be written raises `KrcError`.

    With `dotted`, every unit that carries an override is written with its `.N`, even where the
    override happens to equal the reading's own mora count - the mapping's canonical writer sets the
    override it wants written and leaves it off where it wants none. Without it, a `.N` equal to the
    reading is dropped, which is what an ordinary edit of the file wants.
    """
    if any(_has_join(unit) for chapter in lyrics.chapters for line in chapter.lines for unit in line.units):
        validate(lyrics)
    return SEPARATOR.join(_chapter_text(chapter, dotted) for chapter in lyrics.chapters)


def _chapter_text(chapter: Chapter, dotted: bool = False) -> str:
    return "\n".join(_line_text(line, dotted) for line in chapter.lines)


def _line_text(line: Line, dotted: bool = False) -> str:
    return _owned_text(
        (_unit_text(unit, dotted, ownership=_channel(unit) is None), _channel(unit)) for unit in line.units
    )


def _channel(unit: Unit) -> int | None:
    channels = {word.channel for word in unit_words(unit)}
    return next(iter(channels)) if len(channels) == 1 else None


def _owned_text(parts) -> str:
    result = []
    for channel, run in groupby(parts, key=lambda part: part[1]):
        text = "".join(part[0] for part in run)
        if channel:
            if not 0 <= channel < 16:
                raise KrcError("a channel must be between 0 and 15")
            text = f"@{channel}{{{text}}}"
        result.append(text)
    return "".join(result)


def _unit_text(unit: Unit, dotted: bool = False, *, ownership: bool = True) -> str:
    _check_text(unit.text)
    if ownership and _channel(unit) is not None:
        return _owned_text([(_unit_text(unit, dotted, ownership=False), _channel(unit))])
    words = unit.base.words if isinstance(unit.base, Group) else [unit.base]
    base = _owned_text((word.char, word.channel) for word in words) if ownership else unit.text
    text = f"({base})" if _needs_group(unit) else base
    if unit.ruby is not None:
        ruby = ",".join(_ruby_text(part, dotted, ownership=ownership) for part in unit.ruby.parts)
        text += f"[{ruby}]"
    if unit.join_previous:
        text += ".+"
    elif unit.override is not None and (dotted or unit.mora != unit.natural_mora):
        text += f".{unit.mora}"
    return text


def _needs_group(unit: Unit) -> bool:
    """Whether the text must be parenthesised to come back as one unit."""
    if isinstance(unit.base, Group) and unit.base.explicit:
        return True
    if len(unit.text) <= 1:
        return False
    if unit.override is not None or unit.join_previous:
        return True
    if unit.is_latin():
        return False
    return not (unit.ruby is not None and unit.is_kanji())


def _ruby_text(part: list[Unit], dotted: bool = False, *, ownership: bool = True) -> str:
    if not part:
        raise KrcError("a ruby part may not be empty")
    return "".join(_unit_text(inner, dotted, ownership=ownership) for inner in part)


def _check_text(text: str) -> None:
    if not text:
        raise KrcError("a word's text may not be empty")
    forbidden = META.intersection(text)
    if forbidden:
        raise KrcError(f"a word's text may not hold {sorted(forbidden)!r}")
