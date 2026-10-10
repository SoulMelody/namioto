# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the TempoCNN pipeline and the tempo map helpers."""

from __future__ import annotations

import os

import librosa
import numpy as np
import pytest
import soundfile

from namioto.analysis import model_store
from namioto.analysis.tempo import (
    N_MELS,
    LocalTempo,
    TempoEstimate,
    aggregate,
    default_model_path,
    estimate,
    estimate_array,
    mel_spectrogram,
)

SAMPLE_RATE_IN = 22050


def click_track(bpm: float, seconds: float = 30.0) -> np.ndarray:
    audio = np.zeros(int(seconds * SAMPLE_RATE_IN), dtype=np.float32)
    period = 60.0 / bpm
    length = int(0.12 * SAMPLE_RATE_IN)
    time = np.arange(length) / SAMPLE_RATE_IN
    hit = (0.6 * np.sin(2 * np.pi * 110 * time) + 0.4 * np.sin(2 * np.pi * 2200 * time)) * np.exp(-time * 25)
    for index in range(int(seconds / period)):
        start = int(index * period * SAMPLE_RATE_IN)
        audio[start : start + length] += hit[: len(audio) - start]
    return audio


def write_wav(path, audio: np.ndarray):
    soundfile.write(str(path), audio, SAMPLE_RATE_IN)
    return str(path)


@pytest.fixture(scope="module")
def model_file() -> str:
    """The real graph, which the program downloads on first use rather than shipping.

    A test run has its own data directory, so the model has to be pointed at: either let
    `namioto-cli tempocnn` fetch one, or set the model's environment variable to a local copy.
    """
    env = model_store.MODELS["tempocnn"].env
    if not model_store.installed("tempocnn") and not os.environ.get(env):
        pytest.skip(f"no TempoCNN model installed; set {env} to one, or let namioto-cli tempocnn fetch it")
    return str(default_model_path())


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_estimate_click_track(tmp_path, model_file, bpm):
    audio = write_wav(tmp_path / f"click{bpm}.wav", click_track(bpm))
    assert estimate(audio, model=model_file).bpm == pytest.approx(bpm, abs=2)


def test_estimate_detects_tempo_change(tmp_path, model_file):
    audio = write_wav(tmp_path / "var.wav", np.concatenate([click_track(90, 20.0), click_track(140, 20.0)]))
    result = estimate(audio, model=model_file)

    first, second = result.segments()
    assert (first.bpm, second.bpm) == (90, 140)
    assert first.end == pytest.approx(second.start)
    assert first.start == 0.0
    assert second.start == pytest.approx(20.0, abs=2.0)


def test_too_short_audio_raises(tmp_path, model_file):
    audio = write_wav(tmp_path / "short.wav", click_track(120, 2.0))
    with pytest.raises(ValueError, match="too short"):
        estimate(audio, model=model_file)


def test_segments_merge_runs_of_equal_bpm():
    local = tuple(
        LocalTempo(start=s, end=s + 12.0, bpm=bpm, probability=1.0)
        for s, bpm in ((0.0, 100), (6.0, 100), (12.0, 120), (18.0, 120), (24.0, 100))
    )
    segments = TempoEstimate(bpm=100.0, local=local).segments()

    assert [(s.start, s.end, s.bpm) for s in segments] == [
        (0.0, 15.0, 100),
        (15.0, 27.0, 120),
        (27.0, 36.0, 100),
    ]


def test_empty_estimate_has_no_segments():
    assert TempoEstimate(bpm=0.0, local=()).segments() == ()


def test_aggregate():
    assert aggregate([100, 100, 120], "majority") == 100
    assert aggregate([100, 120], "mean") == 110
    assert aggregate([100, 120, 140], "median") == 120
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate([100], "mode")


def test_mel_spectrogram_has_model_input_shape():
    mel = mel_spectrogram(click_track(120, 5.0))
    assert mel.shape[1] == N_MELS
    assert mel.dtype == np.float32


def test_estimate_array_accepts_other_sample_rates(model_file):
    audio = click_track(120, 20.0)
    resampled = librosa.resample(audio, orig_sr=SAMPLE_RATE_IN, target_sr=48000)
    assert estimate_array(resampled, 48000, model=model_file).bpm == pytest.approx(120, abs=2)
