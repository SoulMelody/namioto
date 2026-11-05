# SPDX-License-Identifier: AGPL-3.0-only
"""The pitch shift: a shift that moves the pitch and leaves the block the length it was."""

from __future__ import annotations

import numpy as np
import pytest

from namioto.audio.pitch import Shifter, shift

RATE = 44100


def sine(frequency: float, seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float64)


def dominant(frames: np.ndarray, rate: int = RATE) -> float:
    """The frequency the most energy sits at, from an `rfft` of the whole array."""
    spectrum = np.abs(np.fft.rfft(frames))
    return float(np.fft.rfftfreq(len(frames), d=1.0 / rate)[int(np.argmax(spectrum))])


def test_no_shift_returns_the_input_sample_for_sample() -> None:
    block = sine(440.0, 0.1)
    assert np.array_equal(Shifter(semitones=0).shift(block), block)
    assert np.array_equal(shift(block, 0.0), block)


def test_twelve_semitones_doubles_the_frequency() -> None:
    block = sine(440.0, 1.0)
    shifted = shift(block, 12.0)
    assert dominant(shifted) == pytest.approx(880.0, rel=0.02)


def test_a_shift_does_not_change_the_number_of_frames() -> None:
    block = sine(440.0, 0.5)
    assert shift(block, 12.0).shape == block.shape
    assert Shifter(semitones=12.0).shift(block).shape == block.shape
    # a stereo block keeps both its channels and its length
    stereo = np.stack([block, block], axis=1)
    assert shift(stereo, 12.0).shape == stereo.shape


def test_a_block_fed_through_the_shifter_keeps_its_length_and_moves_its_pitch() -> None:
    block = sine(440.0, 0.4)
    shifter = Shifter(semitones=12.0)
    pieces = [shifter.shift(block[start : start + 2048]) for start in range(0, len(block), 2048)]
    joined = np.concatenate(pieces)
    # a block is shifted on its own window, so it is not the same arithmetic as one whole array;
    # what it does keep is the contract: the same frames back, at the new pitch
    assert joined.shape == block.shape
    assert dominant(joined) == pytest.approx(880.0, rel=0.02)
