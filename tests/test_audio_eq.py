# SPDX-License-Identifier: AGPL-3.0-only
"""The equalizer: ten bands of gain that leave a flat curve costing nothing."""

from __future__ import annotations

import numpy as np
import pytest

from namioto.audio.eq import BAND_COUNT, EQ_BANDS, Equalizer, apply_eq

RATE = 44100


def sine(frequency: float, seconds: float = 0.5, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float64)


def band_energy(frames: np.ndarray, centre: float, rate: int = RATE, width: float = 0.2) -> float:
    """The energy in an `rfft` of the whole array, inside a window around `centre`."""
    spectrum = np.abs(np.fft.rfft(frames))
    frequencies = np.fft.rfftfreq(len(frames), d=1.0 / rate)
    mask = (frequencies >= centre * (1.0 - width)) & (frequencies <= centre * (1.0 + width))
    return float(np.sum(spectrum[mask] ** 2))


def test_a_flat_curve_returns_the_input_sample_for_sample() -> None:
    block = sine(440.0)
    flat = (0.0,) * BAND_COUNT
    assert np.array_equal(apply_eq(block, flat), block)
    assert np.array_equal(Equalizer().apply(block), block)
    assert Equalizer().is_flat


def test_a_band_cut_takes_more_than_eight_db_out_of_that_band() -> None:
    block = sine(1000.0)
    # the 1 kHz band is index 5 of the ten; -12 dB there must take at least 8 dB off a 1 kHz tone
    bands = [0.0] * BAND_COUNT
    bands[EQ_BANDS.index(1000.0)] = -12.0
    cut = apply_eq(block, tuple(bands))
    before = band_energy(block, 1000.0)
    after = band_energy(cut, 1000.0)
    lost_db = 10.0 * np.log10(before / after)
    assert lost_db >= 8.0
    assert cut.shape == block.shape


def test_a_band_cut_leaves_a_tone_far_from_it_alone() -> None:
    block = sine(1000.0)
    bands = [0.0] * BAND_COUNT
    bands[EQ_BANDS.index(31.5)] = -24.0  # the bottom band, a world away from 1 kHz
    cut = apply_eq(block, tuple(bands))
    assert band_energy(cut, 1000.0) == pytest.approx(band_energy(block, 1000.0), rel=0.01)


def test_a_short_or_long_band_list_is_padded_or_truncated() -> None:
    assert len(Equalizer(bands=(1.0, 2.0)).bands) == BAND_COUNT
    assert Equalizer(bands=(1.0, 2.0)).bands[2:] == (0.0,) * (BAND_COUNT - 2)
    assert len(Equalizer(bands=(1.0,) * (BAND_COUNT + 4)).bands) == BAND_COUNT
    # and the settings layer's range is enforced a second time here
    assert Equalizer(bands=(99.0,) + (0.0,) * (BAND_COUNT - 1)).bands[0] == 12.0


def test_a_block_fed_through_the_equalizer_keeps_its_shape() -> None:
    block = sine(1000.0, 0.4)
    equalizer = Equalizer(bands=(0.0, 0.0, 0.0, 0.0, 0.0, -12.0, 0.0, 0.0, 0.0, 0.0))
    pieces = [equalizer.apply(block[start : start + 2048]) for start in range(0, len(block), 2048)]
    joined = np.concatenate(pieces)
    assert joined.shape == block.shape
    assert band_energy(joined, 1000.0) < band_energy(block, 1000.0)
