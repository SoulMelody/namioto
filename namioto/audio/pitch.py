# SPDX-License-Identifier: AGPL-3.0-only
"""A pitch shift that moves the pitch and leaves the length alone.

`stretch.py` moves time and keeps the pitch: its own docstring says the pitch is what the frames
carry and it never moves. This is the other half of that pair, and it must stay the other half -
folding a pitch into the speed argument would make `speed=1.0, pitch=+3` impossible to express and
would quietly break the engine's frame-counted playhead.

The shift is done in the frequency domain rather than by resampling: `rfft` the block, move the
spectrum along the frequency axis, `irfft` it back. A resampler changes the length along with the
pitch, which would need a second time correction here; a spectral move keeps the block the length
it was, which is the whole contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONTEXT = 1024  # raw samples carried into the next block, so a block edge does not ring


def _shift_block(block: np.ndarray, semitones: float) -> np.ndarray:
    """One channel of one block, shifted along the frequency axis."""
    if semitones == 0.0 or block.size == 0:
        return np.asarray(block)
    spectrum = np.fft.rfft(block)
    moved = np.zeros_like(spectrum)
    ratio = 2.0 ** (semitones / 12.0)
    # a bin at index `i` lands at `i * ratio`; splitting the energy between the two bins it falls
    # between is what keeps a shift sounding like a shift instead of a comb
    for index in range(spectrum.size):
        target = index * ratio
        low = int(np.floor(target))
        if low >= spectrum.size:
            break
        weight = target - low
        moved[low] += spectrum[index] * (1.0 - weight)
        if weight > 0.0 and low + 1 < spectrum.size:
            moved[low + 1] += spectrum[index] * weight
    return np.fft.irfft(moved, n=block.size)


def _shift_columns(columns: np.ndarray, semitones: float) -> np.ndarray:
    """A `(frames, channels)` array, each channel shifted on its own and never mixed."""
    return np.stack([_shift_block(columns[:, index], semitones) for index in range(columns.shape[1])], axis=1)


@dataclass(frozen=True)
class Shifter:
    """A pitch shift that keeps the length, safe to feed one block at a time.

    `semitones == 0` returns its input untouched, sample for sample: a shift nobody asked for must
    not cost a single resample, or the engine's output stops matching what it was before.
    """

    semitones: float = 0.0
    rate: int = 44100
    context: int = CONTEXT
    _tail: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def reset(self) -> None:
        """Forget the context carried over, as at the start of a new playback."""
        object.__setattr__(self, "_tail", None)

    def shift(self, frames: np.ndarray) -> np.ndarray:
        """Shift one block and return it, the length and shape it came in."""
        data = np.asarray(frames)
        if self.semitones == 0.0 or data.size == 0:
            return data
        mono = data.ndim == 1
        columns = data.reshape(-1, 1) if mono else data
        tail = self._tail
        if tail is not None and tail.shape[1] != columns.shape[1]:
            tail = None  # the channel count changed, so the context belongs to another stream
        joined = columns if tail is None else np.vstack([tail, columns])
        shifted = _shift_columns(joined, self.semitones)[-columns.shape[0] :]
        object.__setattr__(self, "_tail", columns[-self.context :].copy() if self.context > 0 else None)
        return shifted[:, 0] if mono else shifted


def shift(frames: np.ndarray, semitones: float) -> np.ndarray:
    """Shift a whole array in one call: the shape the tests and the CLI use."""
    data = np.asarray(frames)
    if semitones == 0.0 or data.size == 0:
        return data
    if data.ndim == 1:
        return _shift_block(data, semitones)
    return _shift_columns(data, semitones)
