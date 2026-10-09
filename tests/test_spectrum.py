# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the note-domain spectrum analysis, including a literal port of noteDigger's math."""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile

from namioto.analysis.spectrum import (
    A4_INDEX,
    NOTE_COUNT,
    NoteSpectrum,
    analyse,
    auto_fill,
    build_weights,
    freq_table,
    load_channels,
    note_label,
    stft_notes,
)

SAMPLE_RATE = 44100
FFT_POINTS = 8192


def tone(frequency: float, seconds: float = 1.0, amplitude: float = 0.5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def write_wav(path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    soundfile.write(str(path), audio, sample_rate, subtype="FLOAT")
    return str(path)


def reference_weights(
    freqs: np.ndarray,
    df: float,
    n_bins: int,
    semi_range: float = 0.667,
    leak_range: float = 1.0,
    oversample: int = 32,
) -> np.ndarray:
    """`NoteAnalyser.updateRange` from noteDigger, translated line by line."""
    h_fft = np.zeros((oversample << 1) | 1)
    step = math.pi / (oversample + 1)
    for i in range(oversample):
        h_fft[i] = h_fft[len(h_fft) - 1 - i] = (1 - math.cos((i + 1) * step)) * 0.5
    h_fft[oversample] = 1.0
    over_df = df * leak_range / (oversample + 1)
    omega = math.pi / semi_range

    def pitch_cos_window(x: float, center: float) -> float:
        if x <= 0:
            return 0.0
        distance = 12 * math.log2(x / center)
        return math.cos(distance * omega) + 1 if abs(distance) < semi_range else 0.0

    tuning = 2 ** (semi_range / 12)
    leak_f = over_df * oversample
    weights = np.zeros((len(freqs), n_bins))
    for index, center in enumerate(freqs):
        start = math.ceil((center / tuning - leak_f) / df)
        end = math.floor((center * tuning + leak_f) / df) + 1
        for j in range(start, end):
            if not 0 <= j < n_bins:
                continue
            total = 0.0
            frequency = j * df - oversample * over_df
            for tap in range(len(h_fft)):
                total += pitch_cos_window(frequency, center) * h_fft[tap]
                frequency += over_df
            weights[index, j] = total
    return weights


def reference_table(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, t_num: float = 20.0, a4: float = 440.0):
    """`stftCPU` from noteDigger: same frame geometry, window and normalisation."""
    hop = round(sample_rate / t_num)
    fft_points = FFT_POINTS
    frames = 1 + (len(audio) - hop // 2) // hop
    window = 1 - np.cos(2 * np.pi * np.arange(fft_points) / fft_points)
    weights = reference_weights(freq_table(a4), sample_rate / fft_points, fft_points // 2 + 1)

    energies = np.zeros((frames, NOTE_COUNT))
    for frame in range(frames):
        center = hop // 2 + frame * hop
        start = center - fft_points // 2
        chunk = np.zeros(fft_points)
        for j in range(fft_points):
            index = start + j
            if 0 <= index < len(audio):
                chunk[j] = audio[index]
        power = np.abs(np.fft.rfft(chunk * window)) ** 2
        energies[frame] = weights @ power

    sigma = energies.std()
    return np.sqrt(np.maximum(energies, 0.0) / sigma)


def reference_auto_fill(table: np.ndarray, threshold: float):
    """`NoteAnalyser.autoFill`, translated line by line."""
    spans = []
    last = np.full(table.shape[1], -1)
    time = 0
    for time in range(table.shape[0]):
        for note in range(table.shape[1]):
            below = table[time, note] < threshold
            if last[note] != -1:
                if below:
                    spans.append((note, int(last[note]), time))
                    last[note] = -1
            elif not below:
                last[note] = time
    for note in range(table.shape[1]):
        if last[note] != -1:
            spans.append((note, int(last[note]), time + 1))
    return sorted(spans)


def test_freq_table_covers_c1_to_b7():
    freqs = freq_table()
    assert len(freqs) == NOTE_COUNT
    assert freqs[A4_INDEX] == pytest.approx(440.0)
    assert freqs[0] == pytest.approx(32.703, abs=1e-3)
    assert freqs[-1] == pytest.approx(3951.066, abs=1e-3)
    assert freq_table(442.0)[A4_INDEX] == pytest.approx(442.0)


def test_note_label():
    assert note_label(0) == "C1"
    assert note_label(A4_INDEX) == "A4"
    assert note_label(NOTE_COUNT - 1) == "B7"


def test_build_weights_matches_notedigger():
    df = SAMPLE_RATE / FFT_POINTS
    n_bins = FFT_POINTS // 2 + 1
    ours = build_weights(freq_table(), df, n_bins).toarray()
    expected = reference_weights(freq_table(), df, n_bins)
    assert np.allclose(ours, expected, rtol=1e-10, atol=1e-12)


def test_weights_peak_at_the_band_frequency():
    df = SAMPLE_RATE / FFT_POINTS
    weights = build_weights(freq_table(), df, FFT_POINTS // 2 + 1)
    for index in (0, A4_INDEX, NOTE_COUNT - 1):
        row = weights.getrow(index).toarray()[0]
        peak = int(row.argmax()) * df
        assert peak == pytest.approx(freq_table()[index], rel=0.05)
        assert row.sum() > 0


def test_weights_are_clipped_low_sample_rates():
    df = 8000 / FFT_POINTS
    n_bins = FFT_POINTS // 2 + 1
    weights = build_weights(freq_table(), df, n_bins)
    assert weights.shape == (NOTE_COUNT, n_bins)
    assert np.isfinite(weights.data).all()
    assert (weights.indices < n_bins).all()


def test_weights_ignore_dc():
    df = SAMPLE_RATE / FFT_POINTS
    weights = build_weights(freq_table(), df, FFT_POINTS // 2 + 1).tocsc()
    assert weights[:, 0].nnz == 0


@pytest.mark.parametrize(
    "frequency,label", [(110.0, "A2"), (220.0, "A3"), (440.0, "A4"), (880.0, "A5"), (1046.5, "C6")]
)
def test_single_tone_lands_in_its_band(frequency, label):
    spectrum = stft_notes(tone(frequency), SAMPLE_RATE)
    mean_energy = spectrum.table.mean(axis=0)
    assert note_label(int(mean_energy.argmax())) == label


def test_matches_notedigger_frame_by_frame():
    audio = (tone(220.0, 0.6) + tone(660.0, 0.6, 0.3)).astype(np.float32)
    ours = stft_notes(audio, SAMPLE_RATE, t_num=20.0).table
    expected = reference_table(audio, t_num=20.0)
    assert ours.shape == expected.shape
    assert np.allclose(ours, expected, rtol=1e-4, atol=1e-5)


def test_frame_geometry_matches_notedigger():
    spectrum = stft_notes(tone(440.0, 1.0), SAMPLE_RATE)
    assert spectrum.frames == 40
    assert spectrum.frame_ms == pytest.approx(25.0)
    assert spectrum.hop == round(SAMPLE_RATE / 40)
    assert spectrum.duration == pytest.approx(1.0)


def test_normalisation_makes_amplitude_irrelevant():
    quiet = stft_notes(tone(440.0, 1.0, 0.05), SAMPLE_RATE).table
    loud = stft_notes(tone(440.0, 1.0, 0.8), SAMPLE_RATE).table
    assert np.allclose(quiet, loud, rtol=1e-4, atol=1e-5)


def test_stereo_channels_are_summed_before_normalising():
    left = tone(440.0, 1.0, 0.4)
    right = tone(880.0, 1.0, 0.4)
    both = stft_notes([left, right], SAMPLE_RATE)
    first = stft_notes(left, SAMPLE_RATE)
    second = stft_notes(right, SAMPLE_RATE)

    # noteDigger sums the per-channel energies, so channels are added before the 1/sigma scaling
    energies = both.table**2 * both.sigma
    expected = first.table**2 * first.sigma + second.table**2 * second.sigma
    assert np.allclose(energies, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("channels", [1, 2])
def test_silent_audio_has_a_finite_zero_spectrum(channels):
    audio = np.zeros((channels, SAMPLE_RATE), dtype=np.float32)
    with np.errstate(divide="raise", invalid="raise"):
        spectrum = stft_notes(audio, SAMPLE_RATE)
    assert spectrum.sigma == 0.0
    assert spectrum.table.shape == (40, NOTE_COUNT)
    assert spectrum.table.dtype == np.float32
    assert np.isfinite(spectrum.table).all()
    assert np.count_nonzero(spectrum.table) == 0


@pytest.mark.parametrize("mode,phase", [("side", 1), ("sum", -1)])
def test_cancelling_channels_have_a_finite_zero_spectrum(tmp_path, mode, phase):
    left = tone(440.0)
    path = write_wav(tmp_path / "cancelled.wav", np.stack([left, phase * left], axis=1))
    with np.errstate(divide="raise", invalid="raise"):
        spectrum = analyse(path, channels=mode)
    assert spectrum.sigma == 0.0
    assert spectrum.table.shape == (40, NOTE_COUNT)
    assert np.isfinite(spectrum.table).all()
    assert np.count_nonzero(spectrum.table) == 0


def test_too_short_audio_raises():
    with pytest.raises(ValueError, match="too short"):
        stft_notes(tone(440.0, 0.01), SAMPLE_RATE)


def test_load_channels_modes(tmp_path):
    left = tone(440.0, 1.0, 0.3)
    right = tone(880.0, 1.0, 0.3)
    path = write_wav(tmp_path / "stereo.wav", np.stack([left, right], axis=1))

    assert len(load_channels(path, "mono")[0]) == 1
    assert len(load_channels(path, "both")[0]) == 2
    reference, _ = load_channels(path, "left")
    side, _ = load_channels(path, "side")
    assert np.allclose(reference[0], left, atol=1e-6)
    assert np.allclose(side[0], left - right, atol=1e-6)
    with pytest.raises(ValueError, match="channel mode"):
        load_channels(path, "middle")


def test_analyse_end_to_end(tmp_path):
    path = write_wav(tmp_path / "a440.wav", tone(440.0, 2.0))
    timings: dict = {}
    spectrum = analyse(path, timings=timings)
    assert spectrum.frames == 80  # 2 s at the default 40 frames per second
    assert note_label(int(spectrum.table.mean(axis=0).argmax())) == "A4"
    assert {"decode", "weights", "fft", "reduce", "normalize"} <= set(timings)


def test_auto_fill_matches_notedigger():
    table = np.zeros((6, NOTE_COUNT), dtype=np.float32)
    table[1:4, 10] = 0.5
    table[2:6, 11] = 0.5
    table[0, 12] = 0.5
    table[5, 12] = 0.9
    spans = auto_fill(table, 0.4)

    assert sorted((span.note, span.start_frame, span.end_frame) for span in spans) == reference_auto_fill(table, 0.4)
    assert spans == sorted(spans, key=lambda span: (span.start_frame, span.note))


def test_auto_fill_edge_cases():
    assert auto_fill(np.zeros((4, NOTE_COUNT)), 0.1) == []
    always = np.ones((4, NOTE_COUNT), dtype=np.float32)
    spans = auto_fill(always, 0.5)
    assert len(spans) == NOTE_COUNT
    assert all((span.start_frame, span.end_frame) == (0, 4) for span in spans)
    assert all(span.midi == span.note + 24 for span in spans)


def test_note_spectrum_helpers():
    spectrum = NoteSpectrum(
        table=np.zeros((21, NOTE_COUNT), dtype=np.float32),
        frame_ms=50.0,
        sample_rate=SAMPLE_RATE,
        fft_points=FFT_POINTS,
        hop=2205,
        a4=440.0,
        sigma=1.0,
    )
    assert spectrum.frame_at(0.0) == 0
    assert spectrum.frame_at(0.1) == 2
    assert spectrum.frame_at(99.0) == 20
    assert spectrum.frame_at(-1.0) == 0
    assert spectrum.time_at(4) == pytest.approx(0.2)
