# SPDX-License-Identifier: AGPL-3.0-only
"""The word sung on a note, and how a text becomes one word per note.

A lyric lives on the note that sings it: `document.Note.lyric` is the whole of the model, so
nothing here keeps a second copy of the words. `split` turns a text into the words a note takes -
whitespace separates words, and text without whitespace is read one character at a time, which is
how Japanese lyrics are typed - and `distribute` puts them on the notes in the order they sound.
`join` reads them back for the window that edits them.

Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from namioto.document import Note


def split(text: str) -> list[str]:
    """One word per note: whitespace separates words, text without whitespace reads per character."""
    words = text.split()
    if len(words) <= 1:
        return [char for char in text if not char.isspace()]
    return words


def distribute(notes: Sequence[Note], text: str) -> None:
    """Put one word on each note, in the order the notes sound; a note past the text keeps none."""
    words = split(text)
    for index, note in enumerate(notes):
        note.lyric = words[index] if index < len(words) else ""


def join(notes: Sequence[Note]) -> str:
    """The words the notes carry, in their order, as the text the lyrics window shows."""
    return " ".join(note.lyric for note in notes if note.lyric)


def has_lyrics(notes: Sequence[Note]) -> bool:
    return any(note.lyric for note in notes)