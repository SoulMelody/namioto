# SPDX-License-Identifier: AGPL-3.0-only
"""The song's transport: source samples in, stretched frames out.

This is where the phase vocoder from `stretch` stops being a DSP curiousity and starts behaving
like a player. It owns no device and never blocks: `pull` is driven frame by frame, so the whole
realtime path can be judged deterministically in a test.

The position stays on the source timeline, which is the clock the drawn playhead follows, and it is
counted in output frames pulled at the current rate rather than in source frames read: a seek or a
rate change re-aims the stretcher, so the frames buffered before it do not count against the new
one and the playhead never steps back.
"""

from __future__ import annotations

import numpy as np

from namioto.audio.stretch import MAX_SPEED, MIN_SPEED, Stretcher


class Engine:
    """The loaded song and the transport over it."""

    def __init__(self, sample_rate: int):
        self._sample_rate = int(sample_rate)
        self._source = np.zeros(0, np.float32)
        self._stretcher = Stretcher(self._sample_rate)
        self._position = 0.0  # source seconds heard next
        self._speed = 1.0
        self._gain = 1.0
        self._playing = False
        self._finished = False

    def load(self, samples: np.ndarray) -> None:
        """Take a mono buffer at the engine's own rate, as `song.py` hands it over."""
        self._source = np.ascontiguousarray(samples, dtype=np.float32)
        self._start_at(0.0)
        self._playing = False
        self._finished = False

    def unload(self) -> None:
        self._source = np.zeros(0, np.float32)
        self._start_at(0.0)
        self._playing = False
        self._finished = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def loaded(self) -> bool:
        return self._source.size > 0

    @property
    def duration(self) -> float:
        return self._source.size / self._sample_rate if self._source.size else 0.0

    @property
    def position(self) -> float:
        return self._position

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        # nothing is reset: the stretcher takes the new rate from the next frame on, and dropping
        # what is buffered would skip source the song has already read
        self._speed = min(MAX_SPEED, max(MIN_SPEED, float(value)))

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(0.0, float(value))

    def play(self, seconds: float = 0.0) -> None:
        self._start_at(seconds)
        self._playing = True

    def pause(self) -> None:
        self._playing = False

    def seek(self, seconds: float) -> None:
        was_playing = self._playing
        self._start_at(seconds)
        self._playing = was_playing

    def pull(self, frames: int) -> np.ndarray:
        """The `frames` output samples that come next: silence while paused, the song while playing."""
        if frames <= 0:
            return np.zeros(0, np.float32)
        if not self._playing or not self.loaded:
            return np.zeros(frames, np.float32)

        block = self._stretcher.pull(frames, self._speed)
        if self._gain != 1.0:
            block *= self._gain
        self._position += frames / self._sample_rate * self._speed

        if self._position >= self.duration:
            # the stretcher still holds a frame of tail; it is dropped rather than flushed, which is
            # where the song ends rather than where the source runs out of samples to read
            self._position = self.duration
            self._finished = True
            self._playing = False
        return block

    def _start_at(self, seconds: float) -> None:
        target = min(max(0.0, float(seconds)), self.duration)
        self._stretcher.load(self._source, target)
        self._position = target
        self._finished = False
