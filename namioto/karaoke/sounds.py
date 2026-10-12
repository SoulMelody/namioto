# SPDX-License-Identifier: AGPL-3.0-only
"""The natural Sounds of a `.krc`: one pronunciation atom each, before any mapping is read.

A Sound is what the lyrics are actually sung as: one kana, or a kana with the small kana riding it,
a long vowel or sokuon of its own, one run of Latin letters, one digit. Punctuation and whitespace
carry no Sound. The input `.N` and the input `(...)` are ignored here - they say how the file was
mapped to notes, not how it reads - so a `(しょう)` reads `しょ`, `う` and a `(しょ)` reads `しょ`.
`.+` also leaves the natural stream unchanged; `note_groups` reads the literal mapping separately.

Each Sound keeps the minimal writable container it came from - a run of the line's top-level units,
or one part of a ruby - and the range of source characters it covers there, so a later pass can write
a `.N` back onto exactly the characters it reads. The token is the one hepburn token the aligner
reads, and the whole line's tokens are the token stream the aligner is given. Qt-free.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from namioto.karaoke.model import LATIN, Group, KrcError, Line, Unit
from namioto.karaoke.parser import parse
from namioto.utils import kana_tokens

SMALL_KANA = frozenset("ャュョァィゥェォゃゅょぁぃぅぇぉゎヮ")
# what the aligner folds away from its neighbour instead of reading as kana: a long vowel and a
# sokuon are Sounds of their own, but their token comes from the vowel or the consonant they lean on
OWN_SOUND = frozenset("ーっッ")


@dataclass(frozen=True)
class Container:
    """One writable level a Sound's characters sit in: a line's top-level run, or one ruby part.

    `key` is `("top", run)` for a run of the line's top-level units and `("part", top, part)` for one
    ruby part, `top` the unit that carries the ruby; it tells two containers apart where the
    characters alone would not. `atoms` are the single characters of that level, in reading order.
    """

    key: tuple
    atoms: tuple[str, ...]
    channels: tuple[int, ...] = ()


@dataclass(frozen=True)
class Sound:
    """One natural Sound: what it reads, where it is written, and the token the aligner reads.

    `index` counts the Sound within its line and `line` the line within the `.krc`; a later pass
    names a Sound by that pair. `container` indexes `SoundLine.containers`; `first_atom`/`last_atom`
    are the character range it covers there. `base` is the surface it is read from - the word itself
    without a ruby, the base unit beside a reading with one.
    """

    line: int
    index: int
    token: str
    reading: str
    base: str
    rubied: bool = False
    first: bool = True
    container: int = 0
    first_atom: int = 0
    last_atom: int = 0
    channel: int = 0

    @property
    def label(self) -> str:
        """What a block shows: the word itself without a ruby, else the reading beside its base."""
        if not self.rubied:
            return self.reading
        left, right = ("(", ")") if self.first else ("[", "]")
        return f"{self.reading}{left}{self.base}{right}"


@dataclass(frozen=True)
class SoundLine:
    """One lyric line as its Sounds, the containers their characters sit in, and its track/chapter."""

    text: str
    sounds: tuple[Sound, ...]
    containers: tuple[Container, ...]
    track: int = 0
    chapter: int = 0


def natural_sounds(text: str) -> list[SoundLine]:
    """Every line of a `.krc` as its natural Sounds, with the input `.N` and groups ignored.

    A kanji with no ruby has no reading to align, so it raises `KrcError`: the file cannot be
    normalised into Sounds. Empty lyrics have no lines and no Sounds.
    """
    if not text.strip():
        return []
    lyrics = parse(text)
    lines: list[SoundLine] = []
    line_index = 0
    for chapter_index, chapter in enumerate(lyrics.chapters):
        for line in chapter.lines:
            lines.append(_line_sounds(line, line_index, chapter_index))
            line_index += 1
    return lines


def natural_tokens(lines: list[SoundLine]) -> list[str]:
    """The token stream the aligner reads: every line's Sounds, in order, one token each."""
    return [sound.token for line in lines for sound in line.sounds]


def split_tokens(tokens: Sequence, lines: list[SoundLine]) -> list[list[tuple[float | None, float | None]]]:
    """The aligner's flat tokens, one per Sound in order, cut back into the lines' rows."""
    rows = []
    at = 0
    for line in lines:
        row = tokens[at : at + len(line.sounds)]
        rows.append([(token.start, token.end) for token in row])
        at += len(row)
    return rows


def note_groups(line: Line, sound_line: SoundLine, *, explicit_only: bool = False) -> list[tuple[int, list[int]]]:
    """NOTE slots and Sound indices; optionally only explicit overrides and continuations."""
    groups: list[tuple[int, list[int]]] = []
    explicit: list[bool] = []
    for unit, indices in _leaves(line, sound_line):
        if unit.join_previous:
            if not groups or groups[-1][0] != 1:
                raise KrcError(".+ requires a preceding Sound mapped to exactly one NOTE in the same line")
            groups[-1][1].extend(indices)
            explicit[-1] = True
            continue
        slots = unit.override if unit.override is not None else len(indices)
        count = len(indices)
        before = len(groups)
        if slots == 0:
            groups.append((0, indices))
        elif slots >= count:
            for position, index in enumerate(indices):
                size = (position + 1) * slots // count - position * slots // count
                groups.append((size, [index]))
        else:
            for slot in range(slots):
                groups.append((1, indices[slot * count // slots : (slot + 1) * count // slots]))
        explicit.extend([unit.override is not None] * (len(groups) - before))
    return [group for group, marked in zip(groups, explicit, strict=True) if marked or not explicit_only]


def _leaves(line: Line, sound_line: SoundLine) -> list[tuple[Unit, list[int]]]:
    index_of = {container.key: index for index, container in enumerate(sound_line.containers)}
    by_container: dict[int, list[Sound]] = {}
    for sound in sound_line.sounds:
        by_container.setdefault(sound.container, []).append(sound)
    leaves: list[tuple[Unit, list[int]]] = []
    runs = 0
    run_units: list[tuple[int, int, Unit]] = []
    offset = 0

    def append(unit: Unit, sounds: Sequence[Sound], low: int, high: int) -> None:
        owned = [sound for sound in sounds if low <= sound.last_atom <= high]
        if unit.join_previous and (
            not owned
            or any(
                sound.first_atom < low or sound.last_atom > high
                for sound in sounds
                if sound.first_atom <= high and sound.last_atom >= low
            )
        ):
            raise KrcError(".+ must cover one or more complete Sounds")
        if owned:
            leaves.append((unit, [sound.index for sound in owned]))

    def flush() -> None:
        nonlocal runs, run_units, offset
        if not run_units:
            return
        container = index_of.get(("top", runs))
        if container is not None:
            for low, high, unit in run_units:
                append(unit, by_container.get(container, ()), low, high)
        runs += 1
        run_units = []
        offset = 0

    for top, unit in enumerate(line.units):
        if unit.ruby is None:
            chars = _chars(unit)
            run_units.append((offset, offset + len(chars) - 1, unit))
            offset += len(chars)
            continue
        flush()
        if unit.override is not None or unit.join_previous:
            owned = []
            for part in range(len(unit.ruby.parts)):
                container = index_of.get(("part", top, part))
                if container is not None:
                    owned.extend(sound.index for sound in by_container.get(container, ()))
            if unit.join_previous and not owned:
                raise KrcError(".+ must cover one or more complete Sounds")
            if owned:
                leaves.append((unit, owned))
            continue
        for part_index, part in enumerate(unit.ruby.parts):
            container = index_of.get(("part", top, part_index))
            if container is None:
                continue
            part_offset = 0
            for inner in part:
                chars = _chars(inner)
                append(inner, by_container.get(container, ()), part_offset, part_offset + len(chars) - 1)
                part_offset += len(chars)
    flush()
    return leaves


def _line_sounds(line: Line, line_index: int, chapter_index: int) -> SoundLine:
    text = "".join(unit.text for unit in line.units)
    containers: list[Container] = []
    sounds: list[Sound] = []
    run: list[str] = []
    ownership: list[int] = []
    runs = 0

    def flush() -> None:
        nonlocal runs
        if not run:
            return
        containers.append(Container(("top", runs), tuple(run), tuple(ownership)))
        runs += 1
        _collect(sounds, containers, run, line_index, rubied=False, first=True, base="")
        run.clear()
        ownership.clear()

    for top, unit in enumerate(line.units):
        if unit.ruby is None:
            if unit.is_kanji():
                raise KrcError(f"'{text}' has kanji with no ruby to align: {unit.text}")
            run.extend(_chars(unit))
            ownership.extend(word.channel for word in _base_words(unit))
            continue
        flush()
        for part_index, part in enumerate(unit.ruby.parts):
            words = [word for inner in part for word in _base_words(inner)]
            atoms = [word.char for word in words]
            if not atoms:
                continue
            base = unit.text if len(unit.ruby.parts) == 1 else unit.text[part_index : part_index + 1]
            containers.append(Container(("part", top, part_index), tuple(atoms), tuple(word.channel for word in words)))
            _collect(sounds, containers, atoms, line_index, rubied=True, first=part_index == 0, base=base)
    flush()
    channels = {sound.channel for sound in sounds}
    track = next(iter(channels)) if len(channels) == 1 else -1 if channels else 0
    return SoundLine(text, tuple(sounds), tuple(containers), track, chapter_index)


def _collect(
    sounds: list[Sound],
    containers: list[Container],
    atoms: list[str],
    line: int,
    *,
    rubied: bool,
    first: bool,
    base: str,
) -> None:
    entries = _entries(atoms)
    tokens = _tokens([reading for reading, _lo, _hi in entries])
    container = len(containers) - 1
    for (reading, low, high), token in zip(entries, tokens, strict=True):
        channels = set(containers[container].channels[low : high + 1])
        channel = next(iter(channels)) if len(channels) == 1 else -1
        sounds.append(
            Sound(
                line=line,
                index=len(sounds),
                token=token,
                reading=reading,
                base=base if rubied else reading,
                rubied=rubied,
                first=first,
                container=container,
                first_atom=low,
                last_atom=high,
                channel=channel,
            )
        )


def set_channel(line: Line, index: int, channel: int) -> None:
    sounds = _line_sounds(line, 0, 0)
    sound = sounds.sounds[index]
    key = sounds.containers[sound.container].key
    if key[0] == "part":
        unit = line.units[key[1]]
        words = [word for inner in unit.ruby.parts[key[2]] for word in _base_words(inner)]
    else:
        runs, words = [], []
        for unit in line.units:
            if unit.ruby is not None:
                if words:
                    runs.append(words)
                    words = []
            else:
                words.extend(_base_words(unit))
        if words:
            runs.append(words)
        words = runs[key[1]]
    for word in words[sound.first_atom : sound.last_atom + 1]:
        word.channel = channel


def _base_words(unit: Unit):
    return unit.base.words if isinstance(unit.base, Group) else [unit.base]


def _chars(unit: Unit) -> list[str]:
    """The single characters a unit is written as: a group's members, or the word itself."""
    if isinstance(unit.base, Group):
        return [word.char for word in unit.base.words]
    return [unit.base.char]


def _entries(atoms: list[str]) -> list[tuple[str, int, int]]:
    """`(reading, first atom, last atom)` per natural Sound of one container, in reading order.

    A small kana rides the Sound before it, a run of Latin letters is one Sound, and a character that
    carries no sound is skipped. A Latin letter joins the run before it only when the two are
    neighbours, so `hello world` reads two Sounds and not one.
    """
    entries: list[tuple[str, int, int]] = []
    for at, char in enumerate(atoms):
        if not _sounds(char):
            continue
        if char in SMALL_KANA and entries:
            entries[-1] = (entries[-1][0] + char, entries[-1][1], at)
            continue
        if _is_latin(char) and entries and _is_latin(entries[-1][0][-1]) and entries[-1][2] == at - 1:
            entries[-1] = (entries[-1][0] + char, entries[-1][1], at)
            continue
        entries.append((char, at, at))
    return entries


def _tokens(readings: list[str]) -> list[str]:
    """One token per reading, cut from the whole container's folded token stream."""
    folded = [char for token in kana_tokens("".join(readings)) for char in token]
    sizes = [_size(reading) for reading in readings]
    if sum(sizes) != len(folded):
        raise KrcError("a sound does not fold into the token stream it reads as")
    tokens: list[str] = []
    at = 0
    for size in sizes:
        tokens.append("".join(folded[at : at + size]))
        at += size
    return tokens


def _size(reading: str) -> int:
    """How many characters a reading owns in the folded token stream; a long vowel or sokuon one."""
    if reading in OWN_SOUND:
        return 1
    return len("".join(kana_tokens(reading)))


def _is_latin(char: str) -> bool:
    return bool(LATIN.fullmatch(char))


def _sounds(char: str) -> bool:
    """Whether a character carries a sound: kana, Latin, a digit, but never punctuation or space.

    A small kana does carry one - it rides the Sound before it - so it is not filtered out here.
    """
    if not char.isprintable() or char.isspace():
        return False
    return unicodedata.category(char)[0] not in ("P", "S")
