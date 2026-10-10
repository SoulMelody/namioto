# SPDX-License-Identifier: AGPL-3.0-only
"""Song playback: the decoded file handed to the native realtime engine.

The stretching, the ring and the audio device all live in `namioto._audio`; this wrapper only
carries the window's surface - a `finished` signal and the settings - over to them. The position
stays in the song's own seconds whatever rate it is played at, because the engine counts the source
timeline and re-anchors on every speed change and seek.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from namioto import _audio

ENGINE_RATE = 48000
BLOCK_FRAMES = 1024
FINISHED_POLL_MS = 50


def load_song(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode a file into one mono buffer at the engine rate, ready for the native engine."""
    samples, _sample_rate = librosa.load(path, sr=ENGINE_RATE, mono=True)
    return np.ascontiguousarray(samples, dtype=np.float32), ENGINE_RATE


class SongPlayer(QObject):
    """The loaded audio file, carrying the transport surface the window drives."""

    finished = Signal()
    failed = Signal(str)

    def __init__(self, parent=None, sample_rate: int = ENGINE_RATE):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._output = _audio.Output(sample_rate, BLOCK_FRAMES)
        self._opened = False
        self._finished_poll = QTimer(self)
        self._finished_poll.setInterval(FINISHED_POLL_MS)
        self._finished_poll.timeout.connect(self._check_finished)

    def load(self, samples: np.ndarray, sample_rate: int) -> None:
        if sample_rate != self._sample_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=self._sample_rate)
        self._output.load(np.ascontiguousarray(samples, dtype=np.float32))

    def unload(self) -> None:
        self._output.unload()

    @property
    def is_loaded(self) -> bool:
        return self._output.loaded

    @property
    def duration(self) -> float:
        return self._output.duration

    @property
    def position(self) -> float:
        return self._output.position

    @property
    def is_playing(self) -> bool:
        return self._output.playing

    @property
    def speed(self) -> float:
        return self._output.speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._output.speed = value

    @property
    def gain(self) -> float:
        return self._output.gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._output.gain = value

    def play(self, seconds: float = 0.0) -> None:
        if not self.is_loaded or not self._open():
            return
        self._output.play(seconds)
        self._finished_poll.start()

    def pause(self) -> None:
        self._finished_poll.stop()
        self._output.pause()

    def stop(self) -> None:
        self._finished_poll.stop()
        self._output.pause()
        self._output.seek(0.0)

    def seek(self, seconds: float) -> None:
        self._output.seek(seconds)

    def close(self) -> None:
        self._finished_poll.stop()
        self._output.close()

    def _open(self) -> bool:
        if self._opened:
            return True
        if self._output.open(False):
            self._opened = True
            return True
        self.failed.emit(self._output.error())
        return False

    def _check_finished(self) -> None:
        if not self._output.finished:
            return
        self._finished_poll.stop()
        self.finished.emit()
