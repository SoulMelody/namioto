# SPDX-License-Identifier: AGPL-3.0-only
"""The song's transport: source samples in, stretched, shifted, equalized frames out.

This is where the phase vocoder from `stretch` stops being a DSP curiousity and starts behaving
like a player. It owns no device and never blocks: `pull` is driven frame by frame, so the whole
realtime path can be judged deterministically in a test.

The chain is fixed and flat, because it is only three stages and two of them may be switched out:
the `Stretcher` carries the speed, the `Shifter` carries the pitch, the `Equalizer` carries the
curve. No stage is a general node and none of them is ever reordered, so a chain has nothing to
decide that three lines cannot say outright.

The position stays on the source timeline, which is the clock the drawn playhead follows, and it is
counted in output frames pulled at the current rate rather than in source frames read: a seek or a
rate change re-aims the stretcher, so the frames buffered before it do not count against the new
one and the playhead never steps back. Pitch and gain are not in that count either: they change what
a frame sounds like, never how many frames a second of song takes.
"""

from __future__ import annotations

import numpy as np

from namioto.audio.eq import BAND_COUNT, Equalizer
from namioto.audio.pitch import Shifter
from namioto.audio.stretch import MAX_SPEED, MIN_SPEED, Stretcher


class Engine:
    """The loaded song and the transport over it."""

    def __init__(
        self,
        sample_rate: int,
        *,
        pitch_semitones: float = 0.0,
        eq_bands: tuple[float, ...] = (0.0,) * BAND_COUNT,
    ) -> None:
        self._sample_rate = int(sample_rate)
        self._source = np.zeros(0, np.float32)
        self._stretcher = Stretcher(self._sample_rate)
        self._position = 0.0  # source seconds heard next
        self._speed = 1.0
        self._gain = 1.0
        self._playing = False
        self._finished = False
        # `None` is a stage that is switched out, not one that is not configured yet: a zero shift
        # and a flat curve must cost nothing at all, so they never get an object to run at all.
        # The two setters below own that rule, so the constructor goes through them as well.
        self._pitch_semitones = 0.0
        self._shifter: Shifter | None = None
        self._eq_bands: tuple[float, ...] = ()
        self._equalizer: Equalizer | None = None
        self.pitch_semitones = pitch_semitones
        self.eq_bands = eq_bands

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

    @property
    def pitch_semitones(self) -> float:
        return self._pitch_semitones

    @pitch_semitones.setter
    def pitch_semitones(self, value: float) -> None:
        semitones = float(value)
        if semitones == self._pitch_semitones:
            # the same value is the same shift: rebuilding here would throw away the context the
            # shift is carrying and put a seam in the output for a control that never moved
            return
        self._pitch_semitones = semitones
        # a new shifter rather than a reset one, and a new object rather than a mutated field: the
        # delay line of the old ratio must not be carried into the new one, and the callback can
        # swap a reference whole where it could not safely reset an object it is reading. Writing
        # the fields and reading them once per block is `output.py`'s own arrangement.
        self._shifter = None if semitones == 0.0 else Shifter(semitones=semitones, rate=self._sample_rate)

    @property
    def eq_bands(self) -> tuple[float, ...]:
        return self._eq_bands

    @eq_bands.setter
    def eq_bands(self, bands: tuple[float, ...]) -> None:
        curve = tuple(float(band) for band in bands)
        if curve == self._eq_bands:
            return
        self._eq_bands = curve
        # the band filter carries a tail of the block it last saw, so a new curve starts on a new
        # equalizer; a flat curve starts on none at all, which is how it costs nothing
        self._equalizer = None if all(band == 0.0 for band in curve) else Equalizer(bands=curve, rate=self._sample_rate)

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
        block = self._shifted(block)
        block = self._equalized(block)
        if self._gain != 1.0:
            block *= self._gain
        # the pitch and the curve changed what the frames sound like, not how many of them a second
        # of song takes: a shifted song lasts exactly as long as an unshifted one, so this count
        # stays the one the playhead reads
        self._position += frames / self._sample_rate * self._speed

        if self._position >= self.duration:
            # the stretcher still holds a frame of tail; it is dropped rather than flushed, which is
            # where the song ends rather than where the source runs out of samples to read
            self._position = self.duration
            self._finished = True
            self._playing = False
        return block

    def _shifted(self, block: np.ndarray) -> np.ndarray:
        """The block at its new pitch, or the block itself when there is no shift to make.

        The shift hands back the length and the shape it was given, so nothing is trimmed or padded
        here: the playhead counts output frames, and a shift that changed their number would move it.
        """
        shifter = self._shifter
        if shifter is None:
            return block
        return shifter.shift(block)

    def _equalized(self, block: np.ndarray) -> np.ndarray:
        """The block under the curve, or the block itself when the curve is flat."""
        equalizer = self._equalizer
        if equalizer is None:
            return block
        return equalizer.apply(block)

    def _start_at(self, seconds: float) -> None:
        target = min(max(0.0, float(seconds)), self.duration)
        self._stretcher.load(self._source, target)
        # both stages carry a block of the audio just before where they are, and a seek lands on
        # audio that does not follow it: the carried context would splice the old position into the
        # new one, so it is dropped here rather than left to fade out of the first block
        if self._shifter is not None:
            self._shifter.reset()
        if self._equalizer is not None:
            self._equalizer.reset()
        self._position = target
        self._finished = False
