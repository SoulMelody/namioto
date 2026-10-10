# SPDX-License-Identifier: AGPL-3.0-only
"""Note playback outputs: an external MIDI synth, or the built-in synth on the native engine."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from namioto import _audio
from namioto.playback import A4, SAMPLE_RATE, TAIL, render_notes

BUFFER_FRAMES = 1024
FINISHED_POLL_MS = 50
PROGRAM_ID = -1  # the note program's buffer; previews use their pitch as the id
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


class BuiltinSynth(NotePlayer):
    """The built-in synth: the notes rendered by Python and played through the native engine.

    Nothing here stretches anything - a note program is already rendered at the settled speed - so
    the engine only has to mix the rendered buffers. Previews are submitted at the playhead, which
    is what lets them overlap the program the way the old Qt sink mixed them.
    """

    def __init__(self, parent=None, sample_rate: int = SAMPLE_RATE, a4: float = A4):
        self._sample_rate = sample_rate
        self._output = _audio.Output(sample_rate, BUFFER_FRAMES)
        self._program = np.zeros(0, dtype=np.float32)
        self._speed = 1.0
        self._duration = 0.0
        self.a4 = a4
        super().__init__(parent)
        self._finished_poll = QTimer(self)
        self._finished_poll.setInterval(FINISHED_POLL_MS)
        self._finished_poll.timeout.connect(self._check_finished)

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(0.0, value)
        if self._program.size:
            self._output.submit_buffer(self._program, 0.0, self._gain, PROGRAM_ID)

    def set_program(self, notes, speed, channels=()) -> None:
        notes = tuple(notes)
        self._speed = max(0.01, float(speed))
        self._program = render_notes(
            notes, sample_rate=self._sample_rate, speed=self._speed, a4=self.a4, channels=channels
        )
        end = max((start + duration for _pitch, start, duration, *_rest in notes), default=0.0)
        self._duration = end + TAIL if notes else 0.0
        if self._program.size:
            self._output.submit_buffer(self._program, 0.0, self._gain, PROGRAM_ID)

    def play(self, seconds: float = 0.0) -> None:
        if not self._program.size or not self._open():
            return
        # the program is already stretched into output time, so a source second is 1/speed of it
        self._output.play(seconds / self._speed)
        self._finished_poll.start()

    def pause(self) -> None:
        self._finished_poll.stop()
        self._output.pause()

    def stop(self) -> None:
        self._finished_poll.stop()
        self._output.pause()
        self._output.seek(0.0)

    def seek(self, seconds: float) -> None:
        self._output.seek(seconds / self._speed)

    def preview(self, pitch: int, seconds: float = PREVIEW_SECONDS) -> None:
        if not self._open():
            return
        voice = render_notes([(pitch, 0.0, seconds)], sample_rate=self._sample_rate, a4=self.a4)
        self._output.submit_buffer(voice, self._output.position, self._gain, pitch)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        return self._output.position * self._speed

    @property
    def is_playing(self) -> bool:
        return self._output.playing

    def _open(self) -> bool:
        return self._output.open(False) if not self._output.is_open else True

    def _check_finished(self) -> None:
        # the engine holds no song here, so it never ends on its own; the program does
        if not self._output.playing or self.position < self._duration:
            return
        self._finished_poll.stop()
        self._output.pause()
        self.finished.emit()


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
    backend: str = "auto",
    port_name: str = "",
    velocity: int = VELOCITY,
    program: int = DEFAULT_PROGRAM,
    a4: float = A4,
) -> tuple[NotePlayer, str]:
    """A player and a description of where it sends the sound.

    An external synth is preferred because it brings its own patches (TiMidity's piano,
    FluidSynth's SoundFont), which is what the rest of the machine already sounds like; with no MIDI
    service the built-in synth takes over, so the notes are still heard. `backend="builtin"` asks
    for the built-in synth on purpose.
    """
    if backend != "builtin":
        try:
            import rtmidi

            port = rtmidi.MidiOut()
            index = find_port(port.get_ports(), port_name)
        except Exception:  # no MIDI backend on this machine
            index = None
            port = None
        if index is not None:
            port.open_port(index)
            return MidiPortOut(port, parent, velocity=velocity, program=program), port.get_port_name(index)
    return BuiltinSynth(parent, a4=a4), "the built-in synth"
