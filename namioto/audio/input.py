# SPDX-License-Identifier: AGPL-3.0-only
"""Recording: a device input stream that a session can capture while it plays.

Recording is the *input* side of the audio layer and shares nothing with `output.py` beyond
`FrameRing`, the one structure that describes "a producer writes, a consumer drains". The output
`Output` is driven by a playhead and renders; a recorder is driven by the device and stores. Giving
them a common owner would tie two different lifetimes together for no gain.

The PortAudio callback is a real-time thread: it copies frames into the ring and reports the level,
and it never touches a Qt object, allocates a buffer it keeps, or opens a file. Everything that
writes happens in `stop()`, on the thread that called it.

`sounddevice` and `soundfile` are both optional. Missing `sounddevice` means recording cannot start
and says so; missing `soundfile` falls back to the `wave` module, which is in the standard library
and writes the same 16-bit PCM either way.
"""

from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from namioto.audio.output import FrameRing

DEFAULT_RATE = 44100
DEFAULT_CHANNELS = 1
BLOCK_FRAMES = 1024
RING_SECONDS = 8.0


def _open_stream(**kwargs):
    """The real `sounddevice.InputStream`, or None when the package is not installed."""
    try:
        import sounddevice
    except ImportError:  # the optional path: a build without PortAudio still imports this module
        return None
    return sounddevice.InputStream(**kwargs)


def _write_wav(path: Path, samples: np.ndarray, rate: int, channels: int) -> None:
    """Write float samples in `-1..1` as 16-bit PCM, preferring `soundfile` and falling back to `wave`."""
    mono = samples.reshape(len(samples), -1)
    try:
        import soundfile

        soundfile.write(str(path), mono, rate, subtype="PCM_16")
        return
    except ImportError:
        pass
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(mono.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm.tobytes())


@dataclass(frozen=True)
class Recording:
    """What a finished take holds, before it becomes a file."""

    samples: np.ndarray
    rate: int

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def seconds(self) -> float:
        return len(self.samples) / float(self.rate) if self.rate else 0.0


class Recorder(QObject):
    """Capture from an input device into memory, then write one wav when it stops.

    One `FrameRing` per channel: the ring is mono by contract, and mixing channels inside it would
    lose the stereo the device gave us before anyone had a chance to ask for mono.
    """

    level = Signal(float)  # RMS of the last block, for a meter
    failed = Signal(str)
    finished = Signal(str)  # the path that was written
    stopped = Signal()

    def __init__(self, parent=None, *, stream_factory=None, rate: int = DEFAULT_RATE) -> None:
        super().__init__(parent)
        self.rate = int(rate)
        self._factory = stream_factory or _open_stream
        self._stream = None
        self._rings: list[FrameRing] = []
        self._channels = DEFAULT_CHANNELS
        self._lock = threading.Lock()
        self._recording = False
        self._dropped = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def channels(self) -> int:
        return self._channels

    def start(self, *, device=None, rate: int | None = None, channels: int = DEFAULT_CHANNELS) -> bool:
        """Open the device and begin filling the rings; False when the device is not there."""
        if self._recording:
            return True
        self.rate = int(rate or self.rate)
        self._channels = max(1, int(channels))
        capacity = 1 << max(10, int(round(self.rate * RING_SECONDS)).bit_length())
        self._rings = [FrameRing(capacity) for _ in range(self._channels)]
        self._dropped = False
        stream = self._factory(
            samplerate=self.rate,
            channels=self._channels,
            dtype="float32",
            blocksize=BLOCK_FRAMES,
            callback=self._on_block,
            device=device,
        )
        if stream is None:
            self.failed.emit("sounddevice is not installed")
            self._rings = []
            return False
        try:
            stream.start()
        except Exception as error:  # a device that refused to open is a message, not a crash
            self.failed.emit(str(error))
            self._rings = []
            return False
        self._stream = stream
        self._recording = True
        return True

    def _on_block(self, indata, frames, time_info, status) -> None:
        """The device callback: copy, report the level, and nothing else."""
        if not self._recording:
            return
        block = np.asarray(indata, dtype=np.float32)
        columns = block.reshape(len(block), -1)
        for index, ring in enumerate(self._rings[: columns.shape[1]]):
            written = ring.write(columns[:, index])
            if written < len(columns):
                with self._lock:
                    self._dropped = True
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        self.level.emit(rms)

    def drain(self) -> Recording:
        """Everything captured so far, as one array; safe to call while recording."""
        columns = []
        for ring in self._rings:
            chunks = []
            buffer = np.zeros(BLOCK_FRAMES, np.float32)
            while True:
                count = ring.read_into(buffer, len(buffer))
                if not count:
                    break
                chunks.append(buffer[:count].copy())
            columns.append(np.concatenate(chunks) if chunks else np.zeros(0, np.float32))
        if not columns:
            return Recording(np.zeros(0, np.float32), self.rate)
        length = min(len(column) for column in columns)
        if self._channels == 1:
            return Recording(columns[0][:length], self.rate)
        return Recording(np.stack([column[:length] for column in columns], axis=1), self.rate)

    def stop(self, path: str | Path | None = None) -> Recording:
        """Stop the stream, take everything out of the rings, and write `path` when given."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._recording = False
        recording = self.drain()
        self._rings = []
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(target, recording.samples, self.rate, recording.channels)
            self.finished.emit(str(target))
        self.stopped.emit()
        return recording


def available() -> bool:
    """True when a build can record at all, so a UI can hide the control instead of failing."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True
