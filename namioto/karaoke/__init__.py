# SPDX-License-Identifier: AGPL-3.0-only
"""Karaoke: the `.krc` lyric model, and what reads, writes and times it.

`model.py` holds the tree (`Word`/`Ruby`/`Line`/`Chapter`/`Lyrics`), `parser.parse` reads a `.krc`
into it, `writer.dumps` writes an edited tree back out, `transforms` holds the optional passes over
one, `sounds.py` reads a lyric as its natural Sounds, `canonical.py` rebuilds a `.krc` from a
mapping, and `operations.py` holds the mapping's operations. The whole package is Qt-free; the names
it re-exports are its public surface.
"""

from namioto.karaoke.canonical import rebuild
from namioto.karaoke.model import Chapter, Group, KrcError, Line, Lyrics, Ruby, Unit, Word
from namioto.karaoke.operations import Drop, Match, Merge, Operation, SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import (
    Container,
    Sound,
    SoundLine,
    natural_sounds,
    natural_tokens,
    split_tokens,
)
from namioto.karaoke.transforms import flatten_ruby, merge_words
from namioto.karaoke.writer import dumps

__all__ = [
    "Chapter",
    "Container",
    "Drop",
    "Group",
    "KrcError",
    "Line",
    "Lyrics",
    "Match",
    "Merge",
    "Operation",
    "Ruby",
    "Sound",
    "SoundLine",
    "SoundRef",
    "Unit",
    "Word",
    "dumps",
    "flatten_ruby",
    "merge_words",
    "natural_sounds",
    "natural_tokens",
    "parse",
    "rebuild",
    "split_tokens",
]
