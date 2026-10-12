# SPDX-License-Identifier: AGPL-3.0-only
"""Single-channel views retaining global Sound identities and display lines. Qt-free.

Each view's text encodes literal NOTE slots with one ruby per Sound; it is only input to the
single-voice reader and verification gate. Exports rebuild the original text, never this projection.
Contiguous channel runs stay separate so a solver cannot merge across another voice's lyrics.
"""

from dataclasses import dataclass, replace
from itertools import groupby

from namioto.karaoke.model import Chapter, Group, KrcError, Line, Lyrics, Ruby, Unit, Word
from namioto.karaoke.operations import SoundRef, map_operation
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds, note_groups
from namioto.karaoke.writer import dumps


@dataclass(frozen=True)
class ChannelSlice:
    channel: int
    text: str
    lines: tuple
    refs: tuple

    @property
    def local(self):
        return {ref: SoundRef(row, at) for row, refs in enumerate(self.refs) for at, ref in enumerate(refs)}

    def rows(self, rows, default=None):
        return [
            [
                rows[ref.line][ref.index] if ref.line < len(rows) and ref.index < len(rows[ref.line]) else default
                for ref in refs
            ]
            for refs in self.refs
        ]

    def flags(self, flags):
        return [flags[refs[0].line] if refs[0].line < len(flags) else False for refs in self.refs]

    def operations(self, operations):
        local = self.local
        return [
            map_operation(operation, local.__getitem__)
            for operation in operations
            if all(ref in local for ref in operation.sounds)
        ]

    def restore(self, operation):
        return map_operation(operation, lambda ref: self.refs[ref.line][ref.index])


def split(text: str) -> dict[int, ChannelSlice]:
    lines = natural_sounds(text)
    parsed = [line for chapter in parse(text).chapters for line in chapter.lines]
    chunks = {}
    for row, (line, sounds) in enumerate(zip(parsed, lines, strict=True)):
        annotations = {}
        for slots, indices in note_groups(line, sounds):
            channels = {sounds.sounds[index].channel for index in indices}
            if slots and (len(channels) != 1 or -1 in channels):
                raise KrcError("one NOTE group cannot belong to multiple channels")
            for at, index in enumerate(indices):
                annotations[index] = (slots if at == 0 or slots == 0 else None, at > 0 and slots == 1)
        for channel, run in groupby(sounds.sounds, key=lambda sound: sound.channel):
            if channel < 0:
                raise KrcError("one Sound cannot belong to multiple channels")
            selected = list(run)
            chunks.setdefault(channel, []).append((row, sounds, selected, annotations))
    result = {}
    for channel, segments in chunks.items():
        projected, refs, literal = [], [], []
        for row, source, selected, annotations in segments:
            local_row = len(projected)
            projected.append(
                replace(
                    source, sounds=tuple(replace(sound, line=local_row, index=at) for at, sound in enumerate(selected))
                )
            )
            refs.append(tuple(SoundRef(row, sound.index) for sound in selected))
            units = []
            for sound in selected:
                slots, joined = annotations[sound.index]
                reading = Unit(Group([Word(char) for char in sound.reading]))
                units.append(
                    Unit(Word("字"), Ruby([[reading]]), override=slots if slots != 1 else None, join_previous=joined)
                )
            literal.append(Line(units))
        result[channel] = ChannelSlice(
            channel, dumps(Lyrics([Chapter(literal)]), dotted=True), tuple(projected), tuple(refs)
        )
    return result
