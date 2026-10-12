# SPDX-License-Identifier: AGPL-3.0-only
"""Optional passes over a parsed `.krc`, each one `Lyrics -> Lyrics` and free of side effects.

A pass returns a new model and leaves its argument alone, so the caller decides whether to use it.
`merge_words` is the normalization pass: it folds each run of kanji, or of Latin letters, into a
`Group`, and recurses into rubies, which is the "auto `()`" the syntax would otherwise have to spell
out. `flatten_ruby` expands a ruby into one unit per mora, for a reader that wants one unit a mora.
"""

from __future__ import annotations

from namioto.karaoke.model import Chapter, Group, Line, Lyrics, Ruby, Unit, Word, validate


def merge_words(lyrics: Lyrics) -> Lyrics:
    """Fold each run of kanji, or of Latin letters, into the group that carries the ruby."""
    chapters = []
    for chapter in lyrics.chapters:
        lines = [Line([_merge_ruby(unit) for unit in _fold(line.units)]) for line in chapter.lines]
        chapters.append(Chapter(lines))
    merged = Lyrics(chapters)
    validate(merged)
    return merged


def _merge_ruby(unit: Unit) -> Unit:
    if unit.ruby is None:
        return unit
    return Unit(unit.base, Ruby([_fold(part) for part in unit.ruby.parts]), unit.override, unit.join_previous)


def _fold(units: list[Unit]) -> list[Unit]:
    merged: list[Unit] = []
    for unit in _fold_kanji(units):
        if (
            merged
            and not _explicit(unit)
            and not _explicit(merged[-1])
            and unit.is_latin()
            and merged[-1].is_latin()
            and merged[-1].ruby is None
            and not merged[-1].join_previous
        ):
            previous = merged.pop()
            override = unit.override if unit.override is not None else previous.override
            unit = Unit(
                Group([*_base_words(previous.base), *_base_words(unit.base)]), unit.ruby, override, unit.join_previous
            )
        merged.append(unit)
    return merged


def _explicit(unit: Unit) -> bool:
    return isinstance(unit.base, Group) and unit.base.explicit


def _fold_kanji(units: list[Unit]) -> list[Unit]:
    pending = list(units)
    folded: list[Unit] = []
    while pending:
        unit = pending.pop()
        if unit.ruby is not None and unit.is_kanji():
            explicit = _explicit(unit)
            words = _base_words(unit.base)
            grew = False
            while pending and pending[-1].ruby is None and pending[-1].is_kanji():
                words = _base_words(pending.pop().base) + words
                grew = True
            if grew:
                unit = Unit(Group(words, explicit=explicit), unit.ruby, unit.override, unit.join_previous)
        folded.insert(0, unit)
    return folded


def _base_words(base: Word | Group) -> list[Word]:
    return list(base.words) if isinstance(base, Group) else [base]


def flatten_ruby(lyrics: Lyrics) -> Lyrics:
    """Give every ruby word one mora, by writing one unit per mora of its ruby."""
    chapters = [Chapter([_flatten_line(line) for line in chapter.lines]) for chapter in lyrics.chapters]
    return Lyrics(chapters)


def _flatten_line(line: Line) -> Line:
    units: list[Unit] = []
    for unit in line.units:
        if unit.ruby is None or not unit.is_ruby_mora:
            units.append(unit)
            continue

        if unit.mora == 1:
            part = unit.ruby.parts[0]
            if len(part) == 1:
                units.append(unit)
                continue

        for part_idx, part in enumerate(unit.ruby.parts):
            for i, inner in enumerate(part):
                words = _base_words(unit.base)
                base_words = words if len(unit.ruby.parts) == 1 else words[part_idx : part_idx + 1]
                placeholder = Word("#", _base_words(inner.base)[0].channel)
                flat_words = ([placeholder] if part_idx != 0 else []) + base_words if i == 0 else [placeholder]
                flat = Unit(flat_words[0] if len(flat_words) == 1 else Group(flat_words))
                flat.set_ruby(Ruby([[inner]]))
                units.append(flat)
    return Line(units)
