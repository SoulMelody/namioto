# SPDX-License-Identifier: AGPL-3.0-only
"""The record window: the state machine, the clock, and a wav actually written to disk.

No device is opened. The state tests inject a fake recorder; the one that checks the file drives the
real `Recorder` with a fake audio stream, which is the seam `namioto.audio.input` already exposes.
"""

from __future__ import annotations

import wave

import numpy as np
from PySide6.QtCore import QObject, Signal

from namioto.audio.input import Recorder
from namioto.ui.record_dialog import RecordDialog, clock_text


class FakeRecorder(QObject):
    """A recorder that answers start/stop without a device."""

    level = Signal(float)
    failed = Signal(str)

    def __init__(self, opens: bool = True, error: str = "no device"):
        super().__init__()
        self.opens = opens
        self.error = error
        self.recording = False
        self.started_with: dict = {}
        self.stopped_to: str | None = None

    @property
    def is_recording(self) -> bool:
        return self.recording

    def start(self, *, device=None, rate=None, channels=1) -> bool:
        self.started_with = {"device": device, "rate": rate, "channels": channels}
        if not self.opens:
            self.failed.emit(self.error)
            return False
        self.recording = True
        return True

    def stop(self, path=None) -> None:
        self.stopped_to = None if path is None else str(path)
        self.recording = False


class FakeStream:
    """A sounddevice stream stand-in: it keeps the callback and pushes a block when told to."""

    made: list[FakeStream] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        FakeStream.made.append(self)

    def start(self) -> None:
        self.started = True

    def push(self, level: float = 0.25, frames: int = 256) -> None:
        callback = self.kwargs.get("callback")
        if callback is not None:
            callback(np.full((frames, 1), level, dtype=np.float32), frames, None, None)

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def test_the_clock_reads_minutes_and_tenths() -> None:
    assert clock_text(0.0) == "00:00.0"
    assert clock_text(5.4) == "00:05.4"
    assert clock_text(61.0) == "01:01.0"


def test_start_then_stop_toggles_the_buttons(qt_app) -> None:
    recorder = FakeRecorder()
    dialog = RecordDialog(recorder_factory=lambda: recorder, can_record=lambda: True)
    assert dialog.start() is True
    assert dialog.is_recording is True
    assert dialog.stop_button.isEnabled() is True
    assert dialog.start_button.isEnabled() is False

    dialog._target = dialog._target  # no path chosen: the default take name is used
    dialog.stop()
    assert dialog.is_recording is False
    assert dialog.start_button.isEnabled() is True
    assert dialog.stop_button.isEnabled() is False


def test_the_timer_moves_the_clock(qt_app) -> None:
    recorder = FakeRecorder()
    dialog = RecordDialog(recorder_factory=lambda: recorder, can_record=lambda: True)
    dialog.start()
    dialog._timer.stop()  # drive the ticks by hand, so the test does not wait a second
    for _ in range(5):
        dialog._tick()
    assert dialog.seconds > 0.0
    assert dialog.clock.text() == clock_text(dialog.seconds)


def test_the_level_reaches_the_meter(qt_app) -> None:
    recorder = FakeRecorder()
    dialog = RecordDialog(recorder_factory=lambda: recorder, can_record=lambda: True)
    dialog.start()
    recorder.level.emit(0.5)
    assert dialog.meter.value() == 500


def test_a_missing_device_disables_start_and_says_why(qt_app) -> None:
    dialog = RecordDialog(recorder_factory=FakeRecorder, can_record=lambda: False)
    assert dialog.start_button.isEnabled() is False
    assert "sounddevice" in dialog.status.text()


def test_a_refused_device_leaves_the_dialog_usable(qt_app) -> None:
    recorder = FakeRecorder(opens=False, error="device busy")
    dialog = RecordDialog(recorder_factory=lambda: recorder, can_record=lambda: True)
    assert dialog.start() is False
    assert dialog.is_recording is False
    assert dialog.start_button.isEnabled() is True
    assert "device busy" in dialog.status.text()


def test_stopping_without_a_start_is_harmless(qt_app) -> None:
    dialog = RecordDialog(recorder_factory=FakeRecorder, can_record=lambda: True)
    assert dialog.stop() is None


def test_a_finished_take_is_written_and_reported(qt_app, tmp_path) -> None:
    """The real recorder writes the wav; the fake stream is the only stand-in."""
    target = tmp_path / "take.wav"
    FakeStream.made.clear()
    dialog = RecordDialog(
        recorder_factory=lambda: Recorder(stream_factory=FakeStream),
        can_record=lambda: True,
    )
    dialog._target = target
    assert dialog.start() is True
    # `Recorder` sets `_recording` only after the stream starts, so the block is pushed once the
    # dialog is running rather than from inside `start()`
    FakeStream.made[-1].push()
    recorded: list[str] = []
    dialog.recorded.connect(recorded.append)

    path = dialog.stop()

    assert path == str(target)
    assert recorded == [str(target)]
    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 44100
        assert handle.getnframes() == 256


def test_wire_builds_an_action(qt_app) -> None:
    from namioto.ui.record_dialog import wire

    class FakeSettings:
        device = None
        rate = None

    class FakeWindow(QObject):
        settings = FakeSettings()

        def statusBar(self):
            class Bar:
                def showMessage(self, _text):
                    pass

            return Bar()

    assert wire(FakeWindow()).text()
