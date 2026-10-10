# SPDX-License-Identifier: AGPL-3.0-only
"""The channels a note plays on: one MIDI channel per entry, with the values it plays with.

Qt-free on purpose, like project.py and settings.py: the project file reads and writes these, the
players receive their playback values, and only the UI draws them. Notes never live here — they
carry the channel number and stay in one flat sequence.

`CHANNEL_COUNT` is 16 and the drum channel is a channel like any other. No name is kept here: a
library name rides on a MIDI track chunk and may repeat, so names live in the project file alone.
`GM_PROGRAMS` is the General MIDI instrument a program number selects, in that number's order, and
`PROGRAM_LABELS` is those names as a picker shows them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from namioto.i18n import tr

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
    """The channel numbers that sound; the sidebar's mute is the only filter."""
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
