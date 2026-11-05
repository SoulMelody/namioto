# SPDX-License-Identifier: AGPL-3.0-only
"""The channels a note plays on: one MIDI channel per entry, with the values it plays with.

Qt-free on purpose, like project.py and settings.py: the project file reads and writes these, the
players receive their playback values, and only the UI draws them. Notes never live here — they
carry the channel number and stay in one flat sequence.

`CHANNEL_COUNT` is 16 and the drum channel is a channel like any other. No name is kept here: a
library name rides on a MIDI track chunk and may repeat, so names live in the project file alone.
`GM_PROGRAMS` is the General MIDI instrument a program number selects, in that number's order, and
`PROGRAM_LABELS` is those names as a picker shows them. A channel may be muted, or soloed to leave
every other one out; `audible` is the single answer to which channels sound. The track clipboard at
the end of the file holds one lifted track in memory - a snapshot of one channel and its notes - so
copying and pasting a whole track needs no temp file and no state of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from namioto.i18n import tr

if TYPE_CHECKING:  # a note carries its channel number, so this module reads notes but never owns them
    from namioto.document import Note

CHANNEL_COUNT = 16
COLOR_HEX = 7

# the General MIDI program list, in the order the program change is meant to select them in
GM_PROGRAMS = (
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Electric Grand Piano",
    "Honky-tonk Piano",
    "Electric Piano 1",
    "Electric Piano 2",
    "Harpsichord",
    "Clavinet",
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)",
    "Electric Guitar (clean)",
    "Electric Guitar (muted)",
    "Overdriven Guitar",
    "Distortion Guitar",
    "Guitar Harmonics",
    "Acoustic Bass",
    "Electric Bass (finger)",
    "Electric Bass (pick)",
    "Fretless Bass",
    "Slap Bass 1",
    "Slap Bass 2",
    "Synth Bass 1",
    "Synth Bass 2",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Tremolo Strings",
    "Pizzicato Strings",
    "Orchestral Harp",
    "Timpani",
    "String Ensemble 1",
    "String Ensemble 2",
    "Synth Strings 1",
    "Synth Strings 2",
    "Choir Aahs",
    "Voice Oohs",
    "Synth Voice",
    "Orchestra Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "French Horn",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Blown Bottle",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Lead 1 (square)",
    "Lead 2 (sawtooth)",
    "Lead 3 (calliope)",
    "Lead 4 (chiff)",
    "Lead 5 (charang)",
    "Lead 6 (voice)",
    "Lead 7 (fifths)",
    "Lead 8 (bass + lead)",
    "Pad 1 (new age)",
    "Pad 2 (warm)",
    "Pad 3 (polysynth)",
    "Pad 4 (choir)",
    "Pad 5 (bowed)",
    "Pad 6 (metallic)",
    "Pad 7 (halo)",
    "Pad 8 (sweep)",
    "FX 1 (rain)",
    "FX 2 (soundtrack)",
    "FX 3 (crystal)",
    "FX 4 (atmosphere)",
    "FX 5 (brightness)",
    "FX 6 (goblins)",
    "FX 7 (echoes)",
    "FX 8 (sci-fi)",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bag pipe",
    "Fiddle",
    "Shanai",
    "Tinkle Bell",
    "Agogo",
    "Steel Drums",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Reverse Cymbal",
    "Guitar Fret Noise",
    "Breath Noise",
    "Seashore",
    "Bird Tweet",
    "Telephone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
)
PROGRAM_LABELS = tuple(f"{index}: {name}" for index, name in enumerate(GM_PROGRAMS))


@dataclass(frozen=True)
class Channel:
    """One MIDI channel of the document. An empty `color` lets the theme assign one."""

    name: str = ""
    color: str = ""
    channel: int = 0
    program: int = 0
    volume: int = 100  # 0-127, 100 being unity
    mute: bool = False
    solo: bool = False
    visible: bool = True
    lock: bool = False

    @property
    def label(self) -> str:
        return self.name or tr("Channel {number}", number=self.channel)


def set_field(channel: Channel, **fields) -> Channel:
    return replace(channel, **fields)


def arranged(channels) -> list[Channel]:
    """The channels in channel order, one entry per number: the first of a repeated number wins."""
    unique: dict[int, Channel] = {}
    for channel in channels:
        unique.setdefault(channel.channel, channel)
    return [unique[number] for number in sorted(unique)]


def free_channel(channels) -> int | None:
    """The lowest channel no entry plays on, or None when all sixteen are taken."""
    used = {channel.channel for channel in channels}
    return next((channel for channel in range(CHANNEL_COUNT) if channel not in used), None)


def audible(channels) -> list[int]:
    """The channel numbers that sound: the one judgment of what plays, shared by players and panel.

    Solo wins outright. As soon as one channel is soloed the mix is exactly the soloed ones, muted or
    not, because pressing solo says "only this" more plainly than an earlier mute said "not that"
    - and pressing mute on the same card then leaves it sounding, never silent by accident.
    With no solo anywhere every channel that is not muted sounds, which is the old behavior.
    """
    soloed = [channel.channel for channel in channels if channel.solo]
    if soloed:
        return soloed
    return [channel.channel for channel in channels if not channel.mute]


def valid_color(value) -> str:
    """A `#rrggbb` string, or "" for anything else."""
    if isinstance(value, str) and len(value) == COLOR_HEX and value[0] == "#":
        try:
            int(value[1:], 16)
        except ValueError:
            return ""
        return value.lower()
    return ""


class TrackClip(NamedTuple):
    """One whole track lifted off the document: a channel's settings and all of its notes.

    A snapshot, not a set of references: the notes are copies, so a copy taken before an edit still
    pastes what was there. The clipboard lives in memory; a temp file was the way to share a track
    between processes, and nothing here has a second process to hand it to.
    """

    channel: Channel
    notes: tuple[Note, ...]


_clipboard: TrackClip | None = None


def copied_track() -> TrackClip | None:
    """The track last copied, or None while nothing has been copied."""
    return _clipboard


def copy_track(channel_index: int, channels, notes) -> TrackClip:
    """Lift one track into the clipboard and return it: the channel and every note it plays.

    The notes keep their pitch, start and length exactly, and are copied, so the document and the
    clip cannot both change under one another. A channel with no notes still copies its settings.
    """
    channel = next((entry for entry in channels if entry.channel == channel_index), None)
    lifted = TrackClip(
        channel if channel is not None else Channel(channel=channel_index),
        tuple(replace(note) for note in notes if note.channel == channel_index),
    )
    global _clipboard
    _clipboard = lifted
    return lifted


def _with_channel(channels, replacement: Channel) -> list[Channel]:
    """The list with `replacement` taking the slot of its number, and added when that number is free."""
    placed = [replacement if channel.channel == replacement.channel else channel for channel in channels]
    if not any(channel.channel == replacement.channel for channel in channels):
        placed.append(replacement)
    return placed


def paste_track(clip: TrackClip, target_index: int, channels, notes) -> tuple[list[Channel], list[Note]]:
    """Put a clip on another channel: its settings land there and its notes join the score.

    Both lists come back new; neither argument is touched, the way the roll's undo snapshots expect.
    The notes keep their time and pitch and take the target channel, and are appended rather than
    replacing what the target already plays. Their ids still name the notes they came from, so the
    document hands the newcomers fresh ones where the two would collide.
    """
    settings = replace(clip.channel, channel=target_index)
    pasted = [replace(note, channel=target_index) for note in clip.notes]
    return _with_channel(channels, settings), [*notes, *pasted]


def delete_track(channel_index: int, channels, notes) -> tuple[list[Channel], list[Note]]:
    """Empty one track: its notes go and its channel returns to the defaults it started with.

    The channel stays in the list - a number is always a place a note can play on, and the sidebar
    keeps a card for it - so only the notes and the settings are cleared. Nothing is touched in
    place, and the other channels come back as they were.
    """
    return (
        _with_channel(channels, Channel(channel=channel_index)),
        [note for note in notes if note.channel != channel_index],
    )
