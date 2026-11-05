# SPDX-License-Identifier: AGPL-3.0-only
"""Every key and pointer gesture the editor answers to, as one table and nothing else.

This is the one place the answer lives. The status bar used to carry a single sentence listing
them, and a sentence cannot be true for long: it had to be read against `app.py`, `roll.py` and
`controls.py` to see whether it still held, it could not fit a narrow window, and it sat over
every message the window wanted to say. `namioto.ui.shortcuts_dialog` draws this table; nothing
here knows what a widget is, so a test can check it without Qt.

`keys` stays in English and is never handed to `tr()`: a key cap reads `Ctrl` on a Chinese desktop
too, and translating it would turn the one string a user matches against their keyboard into a
word. Only `action` is translated.

What is listed here was read off the code, not off the sentence it replaces:

* the keyboard bindings are the `QShortcut`s in `namioto.ui.app` and the key branch of
  `PianoRollView.keyPressEvent`;
* the pointer gestures are the branches of `PianoRollView.mousePressEvent`, its `wheelEvent`, and
  `TimelineRuler`'s press/release/move;
* the button tooltips in `namioto.ui.controls` name the file shortcuts.

Anything the window only offers through a menu or a button (Import MIDI, Stop, the A-B loop, Fit,
the overtone switches) has no key of its own and is not listed: a shortcut table that names a
shortcut which does not exist is the same lie as the sentence was.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shortcut:
    """One way to ask for something: the keys as they are written on the cap, and the action."""

    keys: str
    action: str


@dataclass(frozen=True)
class Group:
    """A page of the help: a title, and the shortcuts under it in the order they are asked about."""

    title: str
    shortcuts: tuple[Shortcut, ...]


GROUPS: tuple[Group, ...] = (
    Group(
        "Files",
        (
            Shortcut("Ctrl+N", "New project"),
            Shortcut("Ctrl+O", "Open a project"),
            Shortcut("Ctrl+S", "Save the project"),
            Shortcut("Ctrl+Shift+S", "Save the project under another name"),
        ),
    ),
    Group(
        "Transport",
        (
            Shortcut("Space", "Play or pause"),
            Shortcut("Click the ruler", "Move the playhead, also while playing"),
            Shortcut("Drag the ruler", "Scroll the roll"),
        ),
    ),
    Group(
        "Navigation",
        (
            Shortcut("Click", "Move the playhead while stopped"),
            Shortcut("Middle-drag", "Pan the roll"),
            Shortcut("Wheel", "Scroll horizontally"),
            Shortcut("Shift+Wheel", "Scroll vertically"),
            Shortcut("Ctrl+Wheel", "Zoom horizontally"),
            Shortcut("Ctrl+Shift+Wheel", "Zoom vertically"),
        ),
    ),
    Group(
        "Editing",
        # the pen, the box and the trims are the branches of one press handler; the tool in hand
        # decides whether an empty row draws or frames, so both entries stay
        (
            Shortcut("Pen-drag", "Draw a note on an empty row"),
            Shortcut("Select-drag", "Draw a box around notes"),
            Shortcut("Drag a note", "Move it with the rest of the selection"),
            Shortcut("Drag a note's edge", "Trim that end"),
            Shortcut("Shift-drag a note", "Trim its start from the left half or its end from the right"),
            Shortcut("Right-click", "Move the selection to another channel"),
            Shortcut("Escape", "Cancel the drag in progress"),
        ),
    ),
    Group(
        "Selection",
        (
            Shortcut("Ctrl-click a note", "Add it to the selection or take it out"),
            Shortcut("Ctrl+A", "Select every note that is not locked"),
            Shortcut("Ctrl+C", "Copy the selection, in edit mode"),
            Shortcut("Ctrl+V", "Paste it at the playhead, in edit mode"),
            Shortcut("Ctrl+D, Delete", "Delete the selection, in edit mode"),
        ),
    ),
    Group(
        "History",
        (
            Shortcut("Ctrl+Z", "Undo"),
            Shortcut("Ctrl+Shift+Z", "Redo"),
        ),
    ),
)
