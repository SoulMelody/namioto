# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for song playback: decoding to the engine rate and the window's wrapper over the engine.

No test opens an audio device: the engine is only driven through its non-playing surface, and the
`finished` wiring is checked with a stand-in. `tests/conftest.py` forbids a real output in the suite.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import soundfile
from PySide6.QtWidgets import QApplication

from namioto import _audio
from namioto.ui import theme
from namioto.ui.song import ENGINE_RATE, SongPlayer, load_song


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    theme.apply(app)
    return app


def test_song_import_resolves_the_audio_decoder_before_workers_start() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import librosa; from namioto.ui.song import decode_audio; "
            "assert vars(librosa).get('load') is decode_audio; "
            "assert 'librosa.core.audio' in sys.modules; "
            "assert 'soundfile' in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_load_song_resamples_to_the_engine_rate(tmp_path) -> None:
    path = tmp_path / "song.wav"
    time = np.arange(4410) / 4410.0
    soundfile.write(str(path), (np.sin(2 * np.pi * 440 * time) * 0.5).astype(np.float32), 4410)

    samples, sample_rate = load_song(path)

    assert sample_rate == ENGINE_RATE
    assert samples.dtype == np.float32
    assert len(samples) == pytest.approx(ENGINE_RATE, rel=0.01)  # the same second, at 48 kHz
    assert np.max(np.abs(samples)) == pytest.approx(0.5, abs=0.05)


def test_the_song_reports_its_length_and_moves_the_playhead() -> None:
    player = SongPlayer()
    assert not player.is_loaded
    assert player.duration == 0.0

    player.load(np.zeros(ENGINE_RATE * 2, dtype=np.float32), ENGINE_RATE)

    assert player.is_loaded
    assert player.duration == pytest.approx(2.0)
    assert not player.is_playing and player.position == 0.0

    player.seek(0.5)
    assert player.position == pytest.approx(0.5)
    player.seek(9.0)  # past the end
    assert player.position == pytest.approx(2.0)
    player.seek(-1.0)
    assert player.position == 0.0

    player.stop()
    assert player.position == 0.0

    player.unload()
    assert not player.is_loaded and player.duration == 0.0


def test_speed_and_gain_reach_the_engine() -> None:
    player = SongPlayer()
    player.load(np.zeros(ENGINE_RATE, dtype=np.float32), ENGINE_RATE)

    player.speed = 1.5
    player.gain = 0.3

    assert player.speed == pytest.approx(1.5)
    assert player.gain == pytest.approx(0.3)


class FakeOutput:
    """The native engine with no device behind it, so the wrapper's plumbing can be checked."""

    def __init__(self, *_args) -> None:
        self.loaded = True
        self.playing = False
        self.finished = False
        self.position = 0.0
        self.duration = 1.0
        self.speed = 1.0
        self.gain = 1.0
        self.opened = False
        self.calls: list[str] = []

    def open(self, null_backend: bool) -> bool:
        self.opened = True
        return True

    def close(self) -> None:
        pass

    def load(self, _samples) -> None:
        self.calls.append("load")

    def unload(self) -> None:
        self.loaded = False

    def play(self, seconds: float = 0.0) -> None:
        self.playing = True
        self.position = seconds
        self.calls.append("play")

    def pause(self) -> None:
        self.playing = False
        self.calls.append("pause")

    def seek(self, seconds: float) -> None:
        self.position = seconds
        self.calls.append("seek")

    def error(self) -> str:
        return ""


def test_the_wrapper_opens_once_and_reports_finished(monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(_audio, "Output", lambda *args: fake)
    player = SongPlayer()
    received: list[bool] = []
    player.finished.connect(lambda: received.append(True))

    player.load(np.zeros(ENGINE_RATE, dtype=np.float32), ENGINE_RATE)
    player.play(0.0)
    assert fake.opened and fake.calls == ["load", "play"]

    fake.finished = True
    player._check_finished()
    assert received == [True]
