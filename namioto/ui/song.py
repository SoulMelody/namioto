# SPDX-License-Identifier: AGPL-3.0-only
"""Song playback: the decoded file handed to the realtime engine in `namioto.audio`.

The stretching, the ring and the audio device all live there; this wrapper only carries the window's
surface - a `finished` signal and the settings - over to them. The position stays in the song's own
seconds whatever rate it is played at, because the engine counts the source timeline and re-anchors
on every speed change and seek.

Resolve librosa's audio decoder at import time so its first import does not race across the
song, spectrum and tempo workers while the GUI restores note items.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from librosa import load as decode_audio
from PySide6.QtCore import QObject, QTimer, Signal

from namioto.audio import Output
from namioto.audio.eq import BAND_COUNT

ENGINE_RATE = 48000
BLOCK_FRAMES = 1024
FINISHED_POLL_MS = 50


def load_song(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode a file into one mono buffer at the engine rate, ready for the native engine."""
    samples, _sample_rate = decode_audio(path, sr=ENGINE_RATE, mono=True)
    return np.ascontiguousarray(samples, dtype=np.float32), ENGINE_RATE


class SongPlayer(QObject):
    """The loaded audio file, carrying the transport surface the window drives."""

    finished = Signal()
    failed = Signal(str)

    def __init__(self, parent=None, sample_rate: int = ENGINE_RATE):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._output = Output(sample_rate, BLOCK_FRAMES)
        self._opened = False
        # the two playback settings the window owns rather than the transport: kept here because
        # the output builds a fresh engine when it opens at another device rate, and a value set
        # before that swap must not be the one the swap forgets
        self._pitch_semitones = 0.0
        self._eq_bands: tuple[float, ...] = (0.0,) * BAND_COUNT
        self._finished_poll = QTimer(self)
        self._finished_poll.setInterval(FINISHED_POLL_MS)
        self._finished_poll.timeout.connect(self._check_finished)

    def load(self, samples: np.ndarray, sample_rate: int) -> None:
        if sample_rate != self._sample_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=self._sample_rate)
        self._output.load(np.ascontiguousarray(samples, dtype=np.float32))
        # loading can re-aim the engine, and a setting the window made before the song was chosen
        # belongs to the song that just arrived, not to the one that was never there
        self._push_effects()

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

    @property
    def pitch_semitones(self) -> float:
        return self._pitch_semitones

    @pitch_semitones.setter
    def pitch_semitones(self, value: float) -> None:
        self._pitch_semitones = float(value)
        self._push_effects()

    @property
    def eq_bands(self) -> tuple[float, ...]:
        return self._eq_bands

    @eq_bands.setter
    def eq_bands(self, bands: tuple[float, ...]) -> None:
        self._eq_bands = tuple(float(band) for band in bands)
        self._push_effects()

    def _push_effects(self) -> None:
        """Hand the two stored settings to the engine the output holds right now.

        The engine can be replaced under us whenever the device rate forces a rebuild, and it is
        reached through the output's own field because the output has no surface for the two stages
        beyond the engine itself. The engine's setters ignore a value that did not change, so
        pushing the same pair again costs nothing and resets nothing.
        """
        engine = self._output._engine
        engine.pitch_semitones = self._pitch_semitones
        engine.eq_bands = self._eq_bands

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
        if not self._output.open(False):
            self.failed.emit(self._output.error())
            return False
        self._opened = True
        # opening at a rate the device insists on rebuilds the engine, and the rebuild knows only
        # the transport: the pitch and the curve are ours to put back
        self._push_effects()
        return True

    def _check_finished(self) -> None:
        if not self._output.finished:
            return
        self._finished_poll.stop()
        self.finished.emit()
