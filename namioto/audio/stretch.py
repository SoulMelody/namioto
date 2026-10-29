# SPDX-License-Identifier: AGPL-3.0-only
"""Pitch-preserving time-stretch of one mono stream: a phase vocoder over numpy.

This is the one piece of signal processing the program owns, and it replaces the vendored
Signalsmith Stretch the native extension used to compile. The synthesis hop is fixed and the
analysis hop carries the rate, which is the arrangement that keeps the frames exactly a hop apart:
every output sample inside the stream is then covered by the same number of windows, so the
overlap-add needs no per-sample normalising, and no rate can tear a gap between two frames.

The rate is source frames per output frame, so a `speed` above one consumes more source per output
frame and plays the song faster, while the phase advance keeps every partial where it was: the
pitch is what the frames carry, and it never moves.

One frame of source is the latency, and a source shorter than a frame at the end is zero-padded,
which is what the engine hands it at the end of a song anyway.
"""

from __future__ import annotations

import numpy as np

FRAME = 2048  # analysis window, about 43 ms at 48 kHz
OVERLAP = 4  # the hop is FRAME/OVERLAP
MIN_SPEED, MAX_SPEED = 0.01, 100.0  # the engine's own clamp, kept here so the DSP agrees
QUIET = 1e-6  # a window sum under this is a sample the frames have not reached yet
WRAP = 2.0 * np.pi


def _window(frame: int) -> np.ndarray:
    """A periodic Hann window; analysis and synthesis both take it, so a frame is weighted twice."""
    return (0.5 - 0.5 * np.cos(WRAP * np.arange(frame) / frame)).astype(np.float32)


def _weight(frame: int, hop: int) -> float:
    """The sum of the squared window over the frames covering one sample: the overlap-add divisor.

    One number rather than an array, because the frames sit exactly a hop apart: every sample inside
    the stream is covered by the same set of windows. Only the fade-in at the very start falls short
    of it, and only the source the frames have not reached leaves it at zero.
    """
    return float(np.sum(_window(frame) ** 2) / hop)


class Stretcher:
    """A streaming phase vocoder: source samples in, stretched frames out.

    `load` puts a source under it and pins the read cursor, `pull` walks the source by `speed`
    frames per output frame and returns exactly the output it was asked for, and `process` is the
    one-shot form that takes a whole buffer at once.
    """

    def __init__(self, sample_rate: int, frame: int = FRAME, overlap: int = OVERLAP):
        self.sample_rate = int(sample_rate)
        self.frame = max(2, int(frame))
        self.hop = max(1, self.frame // max(1, int(overlap)))
        self._window = _window(self.frame)
        self._omega = WRAP * np.arange(self.frame // 2 + 1) / self.frame
        self._weight = _weight(self.frame, self.hop)
        self._source = np.zeros(0, np.float32)
        self.reset()

    @property
    def latency(self) -> float:
        """Seconds of source the first output frame needs before it can be placed."""
        return self.frame / self.sample_rate

    def reset(self) -> None:
        """Drop the phase, as a seek does before it starts again from a new place."""
        self._in_pos = 0.0  # source position handed to the next analysis frame, fractional
        self._previous = None  # the phase of the frame before, the advance is measured from it
        self._phase = None  # the synthesis phase, carried frame to frame

    def load(self, source: np.ndarray, position: float = 0.0) -> None:
        """Put `source` under the stretcher and aim it at `position` seconds into it."""
        self._source = np.ascontiguousarray(source, dtype=np.float32)
        self.reset()
        self._in_pos = max(0.0, float(position) * self.sample_rate)

    def pull(self, frames: int, speed: float = 1.0) -> np.ndarray:
        """Place as many frames as `frames` output samples need, and return exactly that many."""
        if frames <= 0:
            return np.zeros(0, np.float32)
        rate = min(MAX_SPEED, max(MIN_SPEED, float(speed)))
        # the first frame lands at zero and the rest a hop later, so one buffer holds everything the
        # frames reach: the asked-for output plus the frame of look-ahead they always run into
        acc = np.zeros(frames + self.frame, np.float32)
        placed = 0
        while placed < frames:
            self._place(acc, placed, rate)
            placed += self.hop
            self._in_pos += self.hop * rate
        return acc[:frames] / np.maximum(self._weight, QUIET)

    def process(self, source: np.ndarray, speed: float = 1.0) -> np.ndarray:
        """The one-shot form: stretch a whole buffer, its length the source's over `speed`."""
        source = np.ascontiguousarray(source, dtype=np.float32)
        if not len(source):
            return np.zeros(0, np.float32)
        rate = min(MAX_SPEED, max(MIN_SPEED, float(speed)))
        self.load(source, 0.0)
        return self.pull(max(1, int(round(len(source) / rate))), rate)

    def _place(self, acc: np.ndarray, at: int, speed: float) -> None:
        """One analysis frame in, one synthesis frame out, a hop apart and a hop`*speed` of source."""
        start = int(round(self._in_pos))
        segment = np.zeros(self.frame, np.float32)
        take = max(0, min(self.frame, len(self._source) - start))
        if take:
            segment[:take] = self._source[start : start + take]

        spectrum = np.fft.rfft(segment * self._window)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        if self._previous is None:
            self._phase = phase
        else:
            # a bin's real frequency is what it advanced by over the analysis hop, once that hop's
            # own rotation is taken out; advancing it by the synthesis hop instead is what holds the
            # partial in place while the frames move under it
            deviation = phase - self._previous - self._omega * self.hop * speed
            deviation -= WRAP * np.round(deviation / WRAP)
            self._phase = self._phase + (self._omega + deviation / (self.hop * speed)) * self.hop
        self._previous = phase

        frame = np.fft.irfft(magnitude * np.exp(1j * self._phase), self.frame).astype(np.float32)
        acc[at : at + self.frame] += frame * self._window
