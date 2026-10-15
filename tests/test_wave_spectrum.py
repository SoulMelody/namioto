# SPDX-License-Identifier: AGPL-3.0-only
"""Multirate correlation checked against direct quadrature and synthetic audio."""

import math

import numpy as np
import pytest

from namioto.analysis.spectrum import A4_INDEX, freq_table, wave_notes
from namioto.analysis.wave_spectrum import amplitudes, correlation_kernel, decimate


def test_decimation_matches_three_tap_filter():
    samples = np.arange(19, dtype=float) ** 2
    expected = [0] + [(samples[2 * j - 1] + 2 * samples[2 * j] + samples[2 * j + 1]) / 4 for j in range(1, 9)]
    np.testing.assert_array_equal(decimate(samples), expected)


@pytest.mark.parametrize("frequency", [220.0, 440.0, 1760.0])
def test_kernel_matches_scalar_quadrature(frequency):
    rate = 8000
    size = int(26 * rate / frequency + 0.5)
    ramp = int(size / 4 + 0.5)
    step = int(frequency * 2**26 / rate + 0.5)
    expected = []
    for j in range(size):
        weight = (j + 1) / ramp if j < ramp else (size - j - 1) / ramp if j >= size - ramp else 1
        phase = ((j * step) // 65536) % 1024
        angle = 2 * math.pi * phase / 1024
        expected.append(weight * complex(math.cos(angle), math.sin(angle)) * math.pi / (size * 0.75))
    np.testing.assert_allclose(correlation_kernel(frequency, rate), expected, atol=1e-15)


def test_fft_acceleration_matches_direct_local_magnitude_average():
    rate = 8000
    frequency = 440.0
    samples = np.random.default_rng(7).normal(size=800)
    centers = np.array([0, 100, 400, 799])
    kernel = correlation_kernel(frequency, rate)
    count = int(2 * (rate // 40) / int(17 * rate / frequency + 0.5) + 1)
    spacing = int(rate / (count * 40))
    expected = []
    padded = np.pad(samples, (len(kernel) // 2, len(kernel)))
    for center in centers:
        magnitudes = []
        for j in range(count):
            position = center + j * spacing - (count - 1) * spacing // 2
            magnitudes.append(
                abs(np.vdot(kernel, padded[position : position + len(kernel)])) if 0 <= position < len(samples) else 0
            )
        expected.append(np.mean(magnitudes))
    actual = amplitudes([samples], rate, np.array([frequency]), centers, 40)
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-14)


@pytest.mark.parametrize("note", [0, 12, A4_INDEX, 70, 83])
def test_tones_peak_in_their_own_band(note):
    rate = 22050
    audio = np.sin(2 * np.pi * freq_table()[note] * np.arange(rate * 2) / rate)
    result = wave_notes(audio, rate)
    assert result.table[20:60].mean(axis=0).argmax() == note
    assert result.table.dtype == np.float32
    assert result.frames == 80
    assert result.frame_ms == 25


def test_silence_nyquist_progress_and_short_input():
    updates = []
    result = wave_notes(np.zeros(8000), 8000, progress=lambda done, total: updates.append((done, total)))
    assert result.sigma == 0
    assert not result.table.any()
    assert updates[-1] == (84, 84)
    assert all(a[0] < b[0] for a, b in zip(updates, updates[1:], strict=False))
    with pytest.raises(ValueError, match="too short"):
        wave_notes(np.zeros(1), 8000)
    audio = np.random.default_rng(3).normal(size=8000)
    result = wave_notes(audio, 8000)
    assert not result.table[:, freq_table() > 4000].any()


def test_custom_tuning_moves_the_analysis_grid():
    rate = 8000
    audio = np.sin(2 * np.pi * 432 * np.arange(rate) / rate)
    result = wave_notes(audio, rate, a4=432)
    assert result.a4 == 432
    assert result.table[10:30].mean(axis=0).argmax() == A4_INDEX


def test_channels_are_averaged_after_magnitude_not_as_power():
    rate = 8000
    left = np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    right = 0.3 * np.sin(2 * np.pi * 880 * np.arange(rate) / rate)
    centers = np.arange(100, rate, 200)
    frequencies = np.array([110.0, 440.0, 880.0])
    both = amplitudes([left, right], rate, frequencies, centers, 40)
    expected = (
        amplitudes([left], rate, frequencies, centers, 40) + amplitudes([right], rate, frequencies, centers, 40)
    ) / 2
    np.testing.assert_allclose(both, expected, atol=1e-14)


def test_stereo_magnitudes_preserve_antiphase_and_scale_invariance():
    rate = 8000
    audio = np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    single = wave_notes(audio, rate)
    stereo = wave_notes([audio, -audio], rate)
    quiet = wave_notes(audio * 0.01, rate)
    np.testing.assert_allclose(stereo.table, single.table, atol=1e-6)
    np.testing.assert_allclose(quiet.table, single.table, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"t_num": 0}, {"t_num": float("nan")}, {"a4": -440}, {"t_num": 16000}])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        wave_notes(np.zeros(8000), 8000, **kwargs)
