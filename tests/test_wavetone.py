# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the WaveTone onset-flux tempo estimator, which needs no model and no audio."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile

from namioto.analysis.beats import SAMPLE_RATE
from namioto.analysis.wavetone import HOP, WINDOW, estimate_array, main, onset_envelope


def click_track(bpm: float, seconds: float = 90.0, jitter: float = 0.0, missing: float = 0.0, seed: int = 1):
    """A click every beat, optionally loose and with beats dropped, like a real recording."""
    rng = np.random.default_rng(seed)
    audio = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
    burst = np.exp(-np.arange(int(SAMPLE_RATE * 0.05)) / (SAMPLE_RATE * 0.005))
    burst = (burst * np.sin(2 * np.pi * 180 * np.arange(len(burst)) / SAMPLE_RATE)).astype(np.float32)
    for index, start in enumerate(np.arange(0.0, seconds - 0.1, 60.0 / bpm)):
        if index > 2 and rng.random() < missing:
            continue
        at = max(0, int((start + (rng.normal(0.0, jitter) if jitter else 0.0)) * SAMPLE_RATE))
        audio[at : at + len(burst)] += burst
    return audio


def tone_switching(bpm: float = 120.0, seconds: float = 40.0) -> np.ndarray:
    """A steady tone whose pitch steps on every half beat: the loudness never moves, only the spectrum."""
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    frequency = np.where((times // (30.0 / bpm)).astype(int) % 2 == 0, 200.0, 400.0)
    return np.sin(2 * np.pi * frequency * times).astype(np.float32)


@pytest.mark.parametrize("bpm", [93.0, 120.0, 168.0])
def test_a_click_track_is_measured_to_the_beat(bpm: float) -> None:
    assert estimate_array(click_track(bpm), SAMPLE_RATE) == pytest.approx(bpm, abs=0.05)


def test_loose_and_incomplete_beats_are_still_measured() -> None:
    assert estimate_array(click_track(93.0, jitter=0.02, missing=0.1), SAMPLE_RATE) == pytest.approx(93.0, abs=0.5)


def test_the_envelope_is_the_spectrum_and_not_the_loudness() -> None:
    """A pitch step is an onset the amplitude envelope cannot see, so the loudness is the wrong signal."""
    audio = tone_switching()
    peaks = np.abs(audio[: len(audio) // WINDOW * WINDOW].reshape(-1, WINDOW)).max(axis=1)
    assert peaks.max() == peaks.min()  # the loudness is flat all the way through...
    assert onset_envelope(audio, SAMPLE_RATE)[0].max() > 0.0  # ...and every step is an onset all the same
    assert estimate_array(audio, SAMPLE_RATE) == pytest.approx(120.0, abs=0.5)


def test_the_envelope_rate_follows_the_sample_rate() -> None:
    envelope, rate = onset_envelope(np.ones(WINDOW + HOP * 3, dtype=np.float32), SAMPLE_RATE)
    assert rate == SAMPLE_RATE / HOP
    assert envelope.shape == (3,)
    assert not envelope.any()  # a steady spectrum changes by nothing


def test_silence_has_no_tempo() -> None:
    with pytest.raises(ValueError, match="no onset"):
        estimate_array(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE)


def test_too_short_audio_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        estimate_array(np.zeros(WINDOW + HOP - 1, dtype=np.float32), SAMPLE_RATE)


def test_the_cli_prints_the_measured_tempo(tmp_path, capsys) -> None:
    path = tmp_path / "click.wav"
    soundfile.write(path, click_track(168.0, seconds=60.0), SAMPLE_RATE)
    assert main([str(path)]) == 0
    assert "168.0" in capsys.readouterr().out
