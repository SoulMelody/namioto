# SPDX-License-Identifier: AGPL-3.0-only
"""Ten bands of gain over the playback, in the frequency domain.

WaveTone's mixer carries a ten-band equalizer (`ps_setequalizer`); this is the same ten bands, and
it is deliberately *not* part of `stretch.py`: a stretch changes time, an equalizer changes level,
and neither belongs inside the other's block loop.

The gain curve is interpolated between band centres rather than stepped. A stepped curve is a
rectangular filter in disguise and rings; a straight line between the centres does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EQ_BANDS: tuple[float, ...] = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
BAND_COUNT = len(EQ_BANDS)
MIN_DB = -24.0
MAX_DB = 12.0
CONTEXT = 512


def _curve(spectrum_size: int, bands: tuple[float, ...], rate: int) -> np.ndarray:
    """The linear gain for every `rfft` bin, interpolated between the band centres."""
    frequencies = np.fft.rfftfreq(spectrum_size * 2 - 2, d=1.0 / rate) if spectrum_size > 1 else np.zeros(1)
    centres = np.array(EQ_BANDS, dtype=float)
    gains = np.array(bands, dtype=float)
    db = np.interp(frequencies, centres, gains, left=gains[0], right=gains[-1])
    return 10.0 ** (db / 20.0)


def _apply_block(block: np.ndarray, bands: tuple[float, ...], rate: int) -> np.ndarray:
    if block.size == 0:
        return np.asarray(block)
    spectrum = np.fft.rfft(block)
    return np.fft.irfft(spectrum * _curve(spectrum.size, bands, rate), n=block.size)


def apply_eq(frames: np.ndarray, bands: tuple[float, ...], rate: int = 44100) -> np.ndarray:
    """Apply ten bands of gain to a whole array, mono or `(frames, channels)`."""
    data = np.asarray(frames)
    if data.size == 0 or all(band == 0.0 for band in bands):
        return data
    if data.ndim == 1:
        return _apply_block(data, bands, rate)
    return np.stack([_apply_block(data[:, index], bands, rate) for index in range(data.shape[1])], axis=1)


@dataclass(frozen=True)
class Equalizer:
    """Ten bands of gain in dB, safe to feed one block at a time.

    All-zero bands return the input untouched, sample for sample: an equalizer nobody set must not
    cost a transform, or the engine's output stops matching what it was before.
    """

    bands: tuple[float, ...] = (0.0,) * BAND_COUNT
    rate: int = 44100
    context: int = CONTEXT
    _tail: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # the settings layer already clamps, so this is the second line rather than the message a
        # caller sees; padding and truncating keeps a short or long list from becoming an exception
        clamped = tuple(min(MAX_DB, max(MIN_DB, float(band))) for band in self.bands)
        if len(clamped) < BAND_COUNT:
            clamped += (0.0,) * (BAND_COUNT - len(clamped))
        object.__setattr__(self, "bands", clamped[:BAND_COUNT])

    @property
    def is_flat(self) -> bool:
        return all(band == 0.0 for band in self.bands)

    def reset(self) -> None:
        """Forget the context carried over, as at the start of a new playback."""
        object.__setattr__(self, "_tail", None)

    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Equalize one block and return it, the length and shape it came in."""
        data = np.asarray(frames)
        if self.is_flat or data.size == 0:
            return data
        mono = data.ndim == 1
        columns = data.reshape(-1, 1) if mono else data
        tail = self._tail
        if tail is not None and tail.shape[1] != columns.shape[1]:
            tail = None
        joined = columns if tail is None else np.vstack([tail, columns])
        done = apply_eq(joined, self.bands, self.rate)[-columns.shape[0] :]
        object.__setattr__(self, "_tail", columns[-self.context :].copy() if self.context > 0 else None)
        return done[:, 0] if mono else done
