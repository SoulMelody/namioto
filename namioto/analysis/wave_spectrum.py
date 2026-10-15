# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free WaveTone-style multirate, variable-window frequency analysis.

Implements the 26-cycle trapezoidal correlation path in WaveTone 2.74's awlib:
AnalyzeWaveData2 (RVA 0x3980), the decimation pyramid (0x1010), correlation
(0x1810, 0x19e0), and magnitude averaging (0x1d40). No vendor binary or table
is required. This is not a bit-exact port: samples remain floating point,
64 samples per period selects the decimation level, and file edges are padded
rather than shortening and renormalising the window. Namioto's frame centres
and display normalisation are retained by the caller. Overlap-add convolution
avoids song-length FFTs for the short kernels and evaluates the same per-frequency
correlations without materialising every sliding window;
it does not replace the variable kernels with fixed-window STFT bins.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import oaconvolve


def decimate(samples: np.ndarray) -> np.ndarray:
    """Low-pass and halve the rate, keeping the original sample-time origin."""
    result = np.zeros(len(samples) // 2, dtype=np.float64)
    if len(result) > 1:
        result[1:] = (samples[1 : 2 * len(result) - 1 : 2] + samples[3 : 2 * len(result) + 1 : 2]) * 0.25
        result[1:] += samples[2 : 2 * len(result) : 2] * 0.5
    return result


def correlation_kernel(frequency: float, sample_rate: float) -> np.ndarray:
    """Complex quadrature kernel with the 26-cycle, quarter-ramp window."""
    size = max(4, int(26 * sample_rate / frequency + 0.5))
    ramp = max(1, int(size * 0.25 + 0.5))
    window = np.ones(size)
    window[:ramp] = np.arange(1, ramp + 1) / ramp
    window[-ramp:] = np.arange(ramp - 1, -1, -1) / ramp
    step = int(frequency * 0x4000000 / sample_rate + 0.5)
    phase = ((np.arange(size, dtype=np.int64) * step) >> 16) & 1023
    return window * np.exp(2j * np.pi * phase / 1024) * (np.pi / (size * 0.75))


def amplitudes(
    channels: list[np.ndarray],
    sample_rate: int,
    frequencies: np.ndarray,
    centers: np.ndarray,
    t_num: float,
    progress=None,
) -> np.ndarray:
    """Average local magnitudes, then channel magnitudes, on the requested grid."""
    result = np.zeros((len(centers), len(frequencies)), dtype=np.float64)
    total = len(channels) * len(frequencies)
    for channel_index, channel in enumerate(channels):
        pyramid = [np.asarray(channel, dtype=np.float64)]
        for _ in range(7):
            pyramid.append(decimate(pyramid[-1]))
        for note, frequency in enumerate(frequencies):
            if frequency <= sample_rate / 2:
                level = 0
                while level < 7 and (sample_rate >> level) >= 64 * frequency:
                    level += 1
                rate = sample_rate / (1 << level)
                samples = pyramid[level]
                count = min(8, max(1, int(2 * int(sample_rate / t_num) / int(17 * sample_rate / frequency + 0.5) + 1)))
                spacing = int(sample_rate / (count * t_num))
                offsets = np.arange(count) * spacing - (count - 1) * spacing // 2
                positions = (centers[:, None] + offsets) >> level
                kernel = correlation_kernel(float(frequency), rate)
                if len(samples):
                    padded = np.pad(samples, (len(kernel) // 2, len(kernel)))
                    response = oaconvolve(padded, kernel[::-1].conj(), mode="valid")
                    valid = (positions >= 0) & (positions < len(samples))
                    values = np.abs(response[np.clip(positions, 0, len(samples) - 1)])
                    result[:, note] += np.where(valid, values, 0).mean(axis=1)
            if progress is not None:
                progress(channel_index * len(frequencies) + note + 1, total)
    return result / len(channels)
