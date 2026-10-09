# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit KRC NOTE counts and shared Sounds, independent of note identity and alignment. Qt-free."""

from namioto.karaoke.operations import SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds, note_groups


def read(text: str) -> dict[SoundRef, tuple[int, int]]:
    """Map each constrained run's first Sound to its Sound count and NOTE count."""
    parsed = [line for chapter in parse(text).chapters for line in chapter.lines]
    result = {}
    for row, (line, sounds) in enumerate(zip(parsed, natural_sounds(text), strict=True)):
        for slots, indices in note_groups(line, sounds, explicit_only=True):
            if slots == 0:
                result.update((SoundRef(row, index), (1, 0)) for index in indices)
            else:
                result[SoundRef(row, indices[0])] = (len(indices), slots)
    return result
