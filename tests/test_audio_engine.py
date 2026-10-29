# SPDX-License-Identifier: AGPL-3.0-only
"""The audio backend: the phase vocoder, the transport over it and the output around both.

No audio device is opened here: `Stretcher` is pure DSP and `Engine.pull` is driven frame by frame,
so the whole realtime path is judged deterministically. The device-backed tests ask `Output` for the
null backend, which paces a plain thread at realtime instead of a sound card. `tests/conftest.py`
forbids a display and an audio device in the whole suite.
"""

from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

from namioto.audio import Engine, Output, Stretcher

SAMPLE_RATE = 48000
TONE = 440.0
EDGE = 4096  # latency and window edges carry artefacts, so only the steady middle is judged


def _tone(frequency: float, seconds: float) -> np.ndarray:
    return np.sin(2 * np.pi * frequency * np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE).astype(np.float32)


def _dominant_frequency(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(np.fft.rfftfreq(len(samples), 1 / sample_rate)[np.argmax(spectrum)])


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


@pytest.mark.parametrize("speed", [1.0, 2.0, 0.5])
def test_stretcher_speed_scales_the_length_and_leaves_the_pitch(speed: float):
    source = _tone(TONE, 2.0)

    stretcher = Stretcher(SAMPLE_RATE)
    output = stretcher.process(source, speed)

    assert abs(len(output) - len(source) / speed) <= 2
    assert abs(_dominant_frequency(output[EDGE:-EDGE]) - TONE) < 5.0


@pytest.mark.parametrize("speed", [0.1, 2.0])
def test_stretcher_never_tears_a_gap_at_the_ends_of_the_slider(speed: float):
    """The synthesis hop is fixed, so even the slowest rate the slider offers keeps its amplitude."""
    source = _tone(TONE, 2.0)

    output = Stretcher(SAMPLE_RATE).process(source, speed)

    assert abs(_dominant_frequency(output[EDGE:-EDGE]) - TONE) < 5.0
    assert _rms(output[EDGE:-EDGE]) == pytest.approx(_rms(source[EDGE:-EDGE]), rel=0.1)


@pytest.mark.parametrize("speed", [1.0, 2.0, 0.5])
def test_engine_speed_scales_the_position_and_leaves_the_pitch(speed: float):
    engine = Engine(SAMPLE_RATE)
    engine.load(_tone(TONE, 4.0))
    engine.speed = speed
    engine.play(0.0)

    output = engine.pull(SAMPLE_RATE // 2)  # half a second of output

    assert abs(_dominant_frequency(output[EDGE:-EDGE]) - TONE) < 5.0
    assert engine.position == pytest.approx(0.5 * speed, abs=0.02)


def test_engine_seek_lands_on_the_new_source_position():
    engine = Engine(SAMPLE_RATE)
    engine.load(np.concatenate([_tone(440.0, 1.0), _tone(880.0, 1.0)]))
    engine.play(0.0)
    engine.pull(SAMPLE_RATE // 4)

    engine.seek(1.0)
    output = engine.pull(SAMPLE_RATE // 4)

    assert abs(_dominant_frequency(output[EDGE:-EDGE]) - 880.0) < 10.0
    assert engine.position == pytest.approx(1.25, abs=0.03)


def test_engine_pause_is_silent_and_resumes():
    engine = Engine(SAMPLE_RATE)
    engine.load(_tone(TONE, 2.0))
    engine.play(0.0)
    assert np.any(engine.pull(SAMPLE_RATE // 4) != 0)

    engine.pause()
    assert np.all(engine.pull(SAMPLE_RATE // 4) == 0)

    engine.play(engine.position)
    assert np.any(engine.pull(SAMPLE_RATE // 4) != 0)


def test_engine_gain_scales_the_output():
    loud = Engine(SAMPLE_RATE)
    loud.load(_tone(TONE, 2.0))
    loud.play(0.0)
    quiet = Engine(SAMPLE_RATE)
    quiet.load(_tone(TONE, 2.0))
    quiet.gain = 0.25
    quiet.play(0.0)

    reference = _rms(loud.pull(SAMPLE_RATE // 2)[EDGE:-EDGE])
    scaled = _rms(quiet.pull(SAMPLE_RATE // 2)[EDGE:-EDGE])

    assert scaled == pytest.approx(0.25 * reference, rel=1e-3)


def test_engine_stops_at_the_end():
    engine = Engine(SAMPLE_RATE)
    engine.load(_tone(TONE, 0.1))
    engine.play(0.0)

    engine.pull(SAMPLE_RATE)  # a whole second of output from a tenth of a second of source

    assert engine.finished
    assert not engine.playing
    assert engine.position == pytest.approx(engine.duration)


def test_engine_a_speed_change_keeps_the_source_position():
    engine = Engine(SAMPLE_RATE)
    engine.load(np.zeros(SAMPLE_RATE * 4, dtype=np.float32))
    engine.play(0.0)

    engine.pull(SAMPLE_RATE // 2)  # half a source second at 1x
    engine.speed = 0.5
    engine.pull(SAMPLE_RATE // 2)  # half an output second at 0.5x is a quarter of source

    assert engine.position == pytest.approx(0.75)  # the rate changed where it stood, no reset


def test_engine_a_higher_speed_reaches_a_marker_sooner():
    def arrival(speed: float) -> float:
        engine = Engine(SAMPLE_RATE)
        source = np.zeros(SAMPLE_RATE, dtype=np.float32)
        source[SAMPLE_RATE // 2 :] = 1.0  # a step at half a second
        engine.load(source)
        engine.speed = speed
        engine.play(0.0)
        output = engine.pull(int(SAMPLE_RATE * (0.5 / speed + 0.4)))
        return int(np.argmax(output > 0.5)) / SAMPLE_RATE

    # the step is heard in source time, so a faster rate must reach it sooner
    assert arrival(2.0) < arrival(1.0) < arrival(0.5)


def test_engine_dragging_the_speed_never_sends_the_playhead_back():
    engine = Engine(SAMPLE_RATE)
    engine.load(_tone(TONE, 8.0))
    engine.play(0.0)

    seen = [engine.position]
    for speed in (0.5, 1.0, 2.0, 0.25, 1.0) * 20:  # what dragging the slider settles through
        engine.speed = speed
        engine.pull(1024)
        seen.append(engine.position)

    assert seen == sorted(seen)


def _output(samples: np.ndarray, block_frames: int = 1024) -> Output:
    """An `Output` on its null backend: real time, but no audio device."""
    output = Output(SAMPLE_RATE, block_frames)
    assert output.open(null_backend=True), output.error()
    output.load(samples)
    return output


def test_output_plays_and_reports_the_position():
    output = _output(_tone(TONE, 2.0))
    try:
        output.play(0.0)
        time.sleep(0.2)
        assert output.playing
        assert 0.05 < output.position < 0.6
    finally:
        output.close()


def test_output_pause_freezes_and_seek_moves():
    output = _output(_tone(TONE, 2.0))
    try:
        output.play(0.0)
        time.sleep(0.15)
        output.pause()
        time.sleep(0.05)
        paused = output.position
        time.sleep(0.1)

        assert not output.playing
        assert output.position == pytest.approx(paused, abs=0.02)

        output.seek(1.0)
        assert output.position == pytest.approx(1.0, abs=0.02)
    finally:
        output.close()


def test_output_stops_when_the_song_ends():
    output = _output(_tone(TONE, 0.2))
    try:
        output.play(0.0)
        deadline = time.monotonic() + 3.0
        while not output.finished and time.monotonic() < deadline:
            time.sleep(0.02)

        assert output.finished
        assert not output.playing
        assert output.position == pytest.approx(output.duration, abs=0.05)
    finally:
        output.close()


def test_output_speed_changes_keep_the_playhead_advancing():
    output = _output(_tone(TONE, 4.0))
    try:
        output.play(0.0)
        time.sleep(0.1)
        dragged = output.position
        for speed in (0.5, 1.0, 2.0, 0.5, 1.0):  # what dragging the slider settles through
            output.speed = speed
            time.sleep(0.03)

        assert output.playing
        assert output.position >= dragged  # a rate change never sends the playhead back
        time.sleep(0.2)
        assert output.position > dragged
    finally:
        output.close()


def test_output_reports_its_length_before_any_device_is_opened():
    output = Output(SAMPLE_RATE, 1024)
    output.load(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))

    assert output.loaded
    assert output.duration == pytest.approx(2.0)

    output.seek(9.0)  # past the end
    assert output.position == pytest.approx(2.0)
    output.seek(-1.0)
    assert output.position == 0.0

    output.unload()
    assert not output.loaded and output.duration == 0.0


def _fake_sounddevice(monkeypatch, *, device_rate: int = 44100, accepts: bool = False, fail: bool = False) -> list:
    """A stand-in for sounddevice, so the device paths are judged without a sound card."""
    module = types.ModuleType("sounddevice")
    opened: list = []

    class OutputStream:
        def __init__(self, **kwargs) -> None:
            if fail:
                raise RuntimeError("the device refused the stream")
            opened.append(kwargs)
            self.active = False

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            pass

    def check_output_settings(**_kwargs) -> None:
        if not accepts:
            raise RuntimeError("the device does not take that rate")

    module.OutputStream = OutputStream
    module.check_output_settings = check_output_settings
    module.query_devices = lambda **kwargs: {"default_samplerate": device_rate}
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return opened


def test_a_device_that_refuses_the_engine_rate_moves_the_song_to_its_own(monkeypatch):
    opened = _fake_sounddevice(monkeypatch, device_rate=44100, accepts=False)
    output = Output(SAMPLE_RATE, 1024)
    output.load(_tone(TONE, 1.0))

    assert output.open(False)

    assert opened[0]["samplerate"] == 44100  # the device's rate, not the engine's
    assert output.stream_rate == 44100
    assert output.duration == pytest.approx(1.0)  # resampling keeps the song's length in seconds
    output.close()


def test_a_device_that_takes_the_engine_rate_keeps_it(monkeypatch):
    opened = _fake_sounddevice(monkeypatch, accepts=True)
    output = Output(SAMPLE_RATE, 1024)
    output.load(_tone(TONE, 1.0))

    assert output.open(False)

    assert opened[0]["samplerate"] == SAMPLE_RATE
    assert output.stream_rate == SAMPLE_RATE
    output.close()


def test_without_a_device_the_transport_stays_silent_and_says_why(monkeypatch):
    _fake_sounddevice(monkeypatch, fail=True)
    output = Output(SAMPLE_RATE, 1024)
    output.load(_tone(TONE, 1.0))

    assert not output.open(False)

    assert not output.is_open
    assert output.error()
    assert output.duration == pytest.approx(1.0)  # the song stays loaded, it just cannot be heard
    output.close()
