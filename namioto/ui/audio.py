# SPDX-License-Identifier: AGPL-3.0-only
"""Note playback through an external MIDI synth, silent when no output is available."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

from PySide6.QtCore import QObject, Signal

PREVIEW_SECONDS = 0.6
NOTE_ON, NOTE_OFF, VELOCITY = 0x90, 0x80, 100
PROGRAM_CHANGE = 0xC0  # program change per channel: which instrument the synth should use
DEFAULT_PROGRAM = 0  # a grand piano
CHANNEL_VOLUME = (0xB0, 0x07)  # control change 7: the volume of a channel, 0 to 127
ALL_NOTES_OFF = 123  # control change 123: releases every note of a channel, the sweep's safety net
DEFAULT_CHANNEL = (0, DEFAULT_PROGRAM, 100)  # what a program without channels still plays on
SYNTH_NAMES = ("timidity", "fluidsynth", "qsynth", "wavetable")


def find_synth_port(port_names: Sequence[str]) -> int | None:
    """Index of the first port that looks like a software synth; Midi Through is a loopback."""
    for index, name in enumerate(port_names):
        lowered = name.lower()
        if "through" not in lowered and any(part in lowered for part in SYNTH_NAMES):
            return index
    return None


def find_port(port_names: Sequence[str], wanted: str = "") -> int | None:
    """The port a name asks for, else the first software synth: either way, None when there is none.

    The name is matched loosely as well, because a synthesiser's port is numbered by its client when
    it starts, and that number changes between sessions.
    """
    if wanted:
        for index, name in enumerate(port_names):
            if name == wanted:
                return index
        for index, name in enumerate(port_names):
            if wanted.lower() in name.lower():
                return index
    return find_synth_port(port_names)


class NotePlayer(QObject):
    """What the window drives: a prepared program, a transport and an audition."""

    finished = Signal()
    silent = False  # the bars disable the MIDI control around a player that cannot sound

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gain = 1.0

    def set_program(self, notes, speed, channels=()) -> None:
        """Prepare the notes to play: `(pitch, start, duration[, channel])` in seconds, the
        playback rate, and each channel's `(channel, program, volume)`."""
        raise NotImplementedError

    def play(self, seconds: float = 0.0) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def seek(self, seconds: float) -> None:
        raise NotImplementedError

    def preview(self, pitch: int, seconds: float = PREVIEW_SECONDS) -> None:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError

    @property
    def position(self) -> float:
        raise NotImplementedError

    @property
    def is_playing(self) -> bool:
        raise NotImplementedError


class SilentPlayer(NotePlayer):
    """Keep the cursor position without producing sound or starting a transport."""

    silent = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self._position = 0.0
        self._duration = 0.0

    def set_program(self, notes, speed, channels=()) -> None:
        self._duration = max((note[1] + note[2] for note in notes), default=0.0)

    def play(self, seconds: float = 0.0) -> None:
        self.seek(seconds)

    def pause(self) -> None:
        pass

    def stop(self) -> None:
        self._position = 0.0

    def seek(self, seconds: float) -> None:
        self._position = max(0.0, seconds)

    def preview(self, pitch: int, seconds: float = PREVIEW_SECONDS) -> None:
        pass

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        return self._position

    @property
    def is_playing(self) -> bool:
        return False


class MidiPortOut(NotePlayer):
    """An external MIDI synth such as TiMidity: the notes are scheduled onto its port."""

    def __init__(self, port, parent=None, velocity: int = VELOCITY, program: int = DEFAULT_PROGRAM):
        self.port = port  # before the base class, whose gain setter sends a control change
        self._programs: dict[int, int] = {0: program}
        self._volumes: dict[int, int] = {0: 100}
        super().__init__(parent)
        self.velocity = velocity
        self._notes: tuple[tuple[int, float, float, int], ...] = ()
        self._speed = 1.0
        self._start = 0.0
        self._started: float | None = None
        self._stopping = False
        self._sounding: dict[tuple[int, int], int] = {}  # (channel, pitch) -> note-ons still owed an off
        self._previews: dict[int, threading.Timer] = {}
        self._thread: threading.Thread | None = None

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        """A synth does its own mixing, so its volume is a control change rather than a scale factor."""
        self._gain = max(0.0, min(1.0, value))
        for channel in self._volumes:
            self._send_volume(channel)

    def _send_volume(self, channel: int) -> None:
        # the channel volume rides on the global one, both on the same control change
        level = min(1.0, min(1.27, self._volumes.get(channel, 100) / 100) * self._gain)
        self.port.send_message([CHANNEL_VOLUME[0] | channel, CHANNEL_VOLUME[1], round(level * 127)])

    def set_program(self, notes, speed, channels=()) -> None:
        self.stop()
        channels = tuple(channels) or (DEFAULT_CHANNEL,)
        self._notes = tuple(
            (note[0], note[1], note[2], note[3] if len(note) > 3 else 0)
            for note in sorted(notes, key=lambda note: note[1])
        )
        self._speed = max(0.01, speed)
        for channel, program, volume in channels:
            self._programs[channel] = program
            self._volumes[channel] = volume
            self.port.send_message([PROGRAM_CHANGE | channel, program])
            self._send_volume(channel)

    @property
    def duration(self) -> float:
        return max((start + duration for _pitch, start, duration, _ch in self._notes), default=0.0)

    @property
    def position(self) -> float:
        if self._started is None:
            return self._start
        return self._start + (time.monotonic() - self._started) * self._speed

    @property
    def is_playing(self) -> bool:
        return self._started is not None

    def play(self, seconds: float = 0.0) -> None:
        self.stop()
        if not self._notes:
            return
        self._start = max(0.0, min(seconds, self.duration))
        self._started = time.monotonic()
        self._stopping = False
        self._thread = threading.Thread(target=self._schedule, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        position = self.position
        self.stop()
        self._start = position

    def seek(self, seconds: float) -> None:
        if self.is_playing:
            self.play(seconds)
        else:
            # the end of the program is not the end of the timeline: play() clamps when the sound starts
            self._start = max(0.0, seconds)

    def stop(self) -> None:
        self._stopping = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        for timer in self._previews.values():
            timer.cancel()
        self._previews.clear()
        for (channel, pitch), count in tuple(self._sounding.items()):  # a synth keeps sounding until it is told to stop
            for _ in range(count):
                self._send(NOTE_OFF, pitch, 0, channel)
        self._sounding.clear()
        for channel in tuple(self._programs):  # catches a voice the bookkeeping lost, which an exact sweep cannot
            self.port.send_message([CHANNEL_VOLUME[0] | channel, ALL_NOTES_OFF, 0])
        self._started = None
        self._start = 0.0

    def preview(self, pitch: int, seconds: float = PREVIEW_SECONDS) -> None:
        """Audition one note; clicking it again restarts it, which needs the old note released first."""
        pending = self._previews.pop(pitch, None)
        if pending is not None:
            pending.cancel()
            self._send(NOTE_OFF, pitch, 0)  # a synth still holding the note only layers a second one
        self._send(NOTE_ON, pitch, self.velocity)

        def release() -> None:
            if self._previews.get(pitch) is timer:  # a newer click owns the note by now
                self._previews.pop(pitch, None)
                self._send(NOTE_OFF, pitch, 0)

        timer = threading.Timer(seconds, release)
        timer.daemon = True
        self._previews[pitch] = timer
        timer.start()

    def _schedule(self) -> None:
        started = self._started
        events: list[tuple[float, int, int, int]] = []
        for pitch, start, duration, channel in self._notes:
            if start + duration <= self._start:
                continue
            begin = max(start, self._start)
            events.append(((begin - self._start) / self._speed, NOTE_ON, pitch, channel))
            events.append(((start + duration - self._start) / self._speed, NOTE_OFF, pitch, channel))
        # at one tick the previous note's off goes before the next note's on; rounding to the
        # microsecond makes a start+duration that lost a bit against the next start count as one tick
        for offset, kind, pitch, channel in sorted(events, key=lambda event: (round(event[0], 6), event[1])):
            if not self._wait_until(started + offset):
                return
            self._send(kind, pitch, self.velocity if kind == NOTE_ON else 0, channel)
        if self._wait_until(started + (self.duration - self._start) / self._speed):
            self._started = None
            self._start = self.duration
            self.finished.emit()

    def _wait_until(self, deadline: float) -> bool:
        while not self._stopping:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.005))
        return False

    def _send(self, status: int, pitch: int, velocity: int, channel: int = 0) -> None:
        self.port.send_message([status | channel, pitch, velocity])
        key = (channel, pitch)
        if status == NOTE_ON:  # a synth voices every note-on of a pitch, so each is owed its own note-off
            self._sounding[key] = self._sounding.get(key, 0) + 1
        elif self._sounding.get(key):
            self._sounding[key] -= 1
            if not self._sounding[key]:
                del self._sounding[key]


def open_player(
    parent=None,
    port_name: str = "",
    velocity: int = VELOCITY,
    program: int = DEFAULT_PROGRAM,
) -> tuple[NotePlayer, str]:
    """An external MIDI player, or a silent player when no MIDI output is available."""
    try:
        import rtmidi

        port = rtmidi.MidiOut()
        index = find_port(port.get_ports(), port_name)
        if index is not None:
            port.open_port(index)
            return MidiPortOut(port, parent, velocity=velocity, program=program), port.get_port_name(index)
    except Exception:
        pass
    return SilentPlayer(parent), "No MIDI output"
