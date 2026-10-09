# SPDX-License-Identifier: AGPL-3.0-only
"""The score a session edits: the notes and the MIDI channels they play on.

Qt-free on purpose, like channels.py and project.py: the roll draws it, the players and the project
file read it, and nothing here knows about a widget or a scene. Notes are timed in beats, the unit
the roll works in; the seconds a file or the audio uses are a conversion at that boundary, not a
second home for the data.

`MIN_DURATION` (a 64th note) is the floor this model keeps, so only a file may bring a note shorter
than that. `OVERLAP_SLACK` is the resolution below which two notes of one voice count as touching.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from namioto.channels import CHANNEL_COUNT, Channel, arranged
from namioto.channels import set_field as channel_set_field

PITCH_MIN = 21
PITCH_MAX = 108
PITCH_COUNT = PITCH_MAX - PITCH_MIN + 1
MIN_DURATION = 0.0625
# A project keeps its times in seconds rounded to a tenth of a millisecond; at the fastest tempo
# (300 bpm) that lets a note drift half a thousandth of a beat off the grid, its start and its end
# alike. An exact overlap test then refuses an edit the grid calls legal, so anything under this is
# touching. It is smaller than the 1/480 beat an exported MIDI tick stands for, so the two are the
# same instant wherever they leave the editor.
OVERLAP_SLACK = 1e-3


@dataclass(eq=False)
class Note:
    """One note, in beats (the roll's scene unit) and semitone rows.

    Identity is the note itself, not its fields: two notes may sound and sit exactly alike, and a
    roll that looks one up by value would drop the wrong one. `id` is the stable number a saved
    mapping names it by; a fresh note carries 0 until the document gives it one.
    """

    pitch: int
    start: float
    duration: float
    channel: int = 0
    id: int = 0

    def __post_init__(self) -> None:
        self.pitch = min(PITCH_MAX, max(PITCH_MIN, int(self.pitch)))
        self.start = max(0.0, float(self.start))
        self.duration = max(MIN_DURATION, float(self.duration))
        self.channel = min(CHANNEL_COUNT - 1, max(0, int(self.channel)))
        self.id = max(0, int(self.id))

    @property
    def end(self) -> float:
        return self.start + self.duration

    def set_duration(self, duration: float) -> None:
        self.duration = max(MIN_DURATION, float(duration))

    def set_range(self, start: float, pitch: int) -> None:
        self.start = max(0.0, float(start))
        self.pitch = min(PITCH_MAX, max(PITCH_MIN, int(pitch)))


class Document:
    """The notes and the channels of one score, with the invariants the roll relies on.

    It holds the data, not the drawing: the roll builds one item per note and reads the note back.
    A channel's number is its identity, so removing one never renumbers the others - the notes that
    stay keep the MIDI channel they play on.
    """

    def __init__(self, channels=None, notes=None):
        self.channels: list[Channel] = arranged(channels or ()) or [Channel()]
        self.notes: list[Note] = list(notes) if notes else []
        self.next_id = 1
        self._assign_ids()
        self._fill_channels()

    def _assign_ids(self) -> None:
        """Give every note a stable id, keeping the ones it already has; the next free id is kept."""
        used: set[int] = set()
        top = max(self.next_id, max((note.id for note in self.notes), default=0) + 1)
        for note in self.notes:
            if note.id <= 0 or note.id in used:
                note.id = top
            used.add(note.id)
            top = max(top, note.id + 1)
        self.next_id = top

    def _take_id(self) -> int:
        taken = self.next_id
        self.next_id += 1
        return taken

    def _fill_channels(self) -> None:
        """Every channel a note names gets an entry, so nothing draws against a missing one."""
        present = {channel.channel for channel in self.channels}
        for number in sorted({note.channel for note in self.notes} - present):
            self.channels.append(Channel(channel=number))
        self.channels.sort(key=lambda channel: channel.channel)

    def add_note(self, note: Note) -> Note:
        # a fresh note takes the next free id; the scan is over the handful a gesture adds at once
        if note.id <= 0 or any(other.id == note.id for other in self.notes):
            note.id = self._take_id()
        else:
            self.next_id = max(self.next_id, note.id + 1)
        self.notes.append(note)
        self._fill_channels()
        return note

    def index_by_voice(self) -> dict[tuple[int, int], list[Note]]:
        """The notes grouped by (channel, pitch), for the overlap tests a drag repeats per moved note.

        A drag mutates only the notes it holds and ignores those same ones, so a caller that rebuilds
        this once per move reads every other note under the voice it still has.
        """
        index: dict[tuple[int, int], list[Note]] = {}
        for note in self.notes:
            index.setdefault((note.channel, note.pitch), []).append(note)
        return index

    def collides(self, pitch: int, start: float, duration: float, channel: int, ignore=(), index=None) -> bool:
        """Whether a note of `pitch` on `channel` would share time with one already there.

        Only the same pitch on the same channel is refused - different pitches may sound together,
        and two notes whose spans overlap by less than `OVERLAP_SLACK` are touching, not clashing.
        `ignore` holds the notes being moved or resized, which are not obstacles to themselves.
        `index` is `index_by_voice` read against the notes as they are now, which saves the caller a
        full scan per moved note.
        """
        end = start + duration
        ignored = ignore if isinstance(ignore, (set, frozenset)) else set(ignore)
        candidates = self.notes if index is None else index.get((channel, pitch), ())
        return any(
            note not in ignored
            and note.pitch == pitch
            and note.channel == channel
            and note.start < end - OVERLAP_SLACK
            and start < note.end - OVERLAP_SLACK
            for note in candidates
        )

    def remove_note(self, note: Note) -> None:
        self.notes.remove(note)

    def clear_notes(self) -> None:
        self.notes.clear()

    def replace_notes(self, notes, next_id: int | None = None) -> None:
        self.notes = list(notes)
        if next_id is not None:
            self.next_id = max(1, int(next_id))
        self._assign_ids()
        self._fill_channels()

    def set_channels(self, channels) -> None:
        """Replace the channel list; a note whose channel is gone gets a plain entry back."""
        self.channels = arranged(channels or ()) or [Channel()]
        self._fill_channels()

    def add_channel(self, channel: Channel) -> None:
        self.channels = arranged([*self.channels, channel])

    def remove_channel(self, number: int) -> list[Note] | None:
        """Drop a channel and the notes on it, leaving the other numbers alone; None when it is the
        last, which stays."""
        if len(self.channels) <= 1:
            return None
        self.channels = [channel for channel in self.channels if channel.channel != number]
        removed = [note for note in self.notes if note.channel == number]
        self.notes = [note for note in self.notes if note.channel != number]
        return removed

    def set_channel_field(self, number: int, **fields) -> None:
        for index, channel in enumerate(self.channels):
            if channel.channel == number:
                self.channels[index] = channel_set_field(channel, **fields)
                return

    def set_channel_number(self, old: int, new: int) -> bool:
        """Move a channel and the notes on it onto another number; the number has to be free."""
        if old == new:
            return True
        if not 0 <= new < CHANNEL_COUNT or any(channel.channel == new for channel in self.channels):
            return False
        for index, channel in enumerate(self.channels):
            if channel.channel == old:
                self.channels[index] = replace(channel, channel=new)
                break
        else:
            return False
        self.channels.sort(key=lambda channel: channel.channel)
        for note in self.notes:
            if note.channel == old:
                note.channel = new
        return True
