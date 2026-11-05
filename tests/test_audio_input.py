# SPDX-License-Identifier: AGPL-3.0-only
"""Recording: the start/stop state machine and the frames a take holds.

No device is opened here: `Recorder` takes its stream from a factory, so a fake one stands in for
PortAudio and feeds the callback by hand. That keeps the suite device-free, which is the same
promise `namioto.audio.output`'s null backend makes for playback.
"""

from __future__ import annotations

import numpy as np

from namioto.audio.input import DEFAULT_RATE, Recorder, Recording


class FakeStream:
    """A stand-in for `sounddevice.InputStream`: it keeps the callback, not a device."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def feed(self, frames: int = 512, channels: int = 1, value: float = 0.25) -> None:
        """Hand the recorder's callback one block, the way PortAudio would."""
        block = np.full((frames, channels), value, np.float32)
        self.callback(block, frames, None, None)


def recorder(**kwargs) -> tuple[Recorder, list[FakeStream]]:
    """A recorder wired to fake streams, plus the list every stream it opened lands in."""
    opened: list[FakeStream] = []

    def factory(**stream_kwargs) -> FakeStream:
        stream = FakeStream(**stream_kwargs)
        opened.append(stream)
        return stream

    return Recorder(stream_factory=factory, **kwargs), opened


def test_starting_opens_one_stream_and_says_it_is_recording() -> None:
    device, opened = recorder()
    assert not device.is_recording
    assert device.start()
    assert device.is_recording
    assert len(opened) == 1
    assert opened[0].started
    assert opened[0].kwargs["samplerate"] == DEFAULT_RATE
    assert opened[0].kwargs["channels"] == device.channels

    # starting again is a no-op, not a second stream
    assert device.start()
    assert len(opened) == 1


def test_stopping_closes_the_stream_and_leaves_the_state_clean() -> None:
    device, opened = recorder()
    device.start()
    opened[0].feed(1024)
    recording = device.stop()
    assert not device.is_recording
    assert opened[0].closed
    assert not opened[0].started
    assert isinstance(recording, Recording)
    assert recording.rate == DEFAULT_RATE


def test_the_take_holds_every_frame_the_callback_delivered() -> None:
    device, opened = recorder(rate=8000)
    device.start(rate=8000)
    for _ in range(4):
        opened[0].feed(512)
    recording = device.stop()
    assert len(recording.samples) == 4 * 512
    assert recording.samples.ndim == 1
    assert recording.channels == 1
    assert recording.seconds == 4 * 512 / 8000


def test_a_stereo_take_keeps_its_two_channels_apart() -> None:
    device, opened = recorder()
    device.start(channels=2)
    assert device.channels == 2
    block = np.stack([np.full(256, 0.5, np.float32), np.full(256, -0.5, np.float32)], axis=1)
    opened[0].callback(block, 256, None, None)
    recording = device.stop()
    assert recording.samples.shape == (256, 2)
    assert recording.channels == 2
    assert np.allclose(recording.samples[:, 0], 0.5)
    assert np.allclose(recording.samples[:, 1], -0.5)


def test_a_missing_device_is_reported_rather_than_raised() -> None:
    device = Recorder(stream_factory=lambda **kwargs: None)
    failures: list[str] = []
    device.failed.connect(failures.append)
    assert not device.start()
    assert not device.is_recording
    assert failures and "sounddevice" in failures[0]


def test_a_device_that_refuses_to_open_is_a_message_not_a_crash() -> None:
    class Refuses(FakeStream):
        def start(self) -> None:
            raise RuntimeError("device busy")

    device = Recorder(stream_factory=lambda **kwargs: Refuses(**kwargs))
    failures: list[str] = []
    device.failed.connect(failures.append)
    assert not device.start()
    assert not device.is_recording
    assert failures == ["device busy"]


def test_a_stopped_recorder_ignores_late_blocks() -> None:
    device, opened = recorder()
    device.start()
    opened[0].feed(128)
    device.stop()
    # a callback that lands after the stop must not resurrect a take
    opened[0].feed(128)
    assert device.stop().samples.size == 0


def test_stopping_with_a_path_writes_a_wav(tmp_path) -> None:
    device, opened = recorder(rate=8000)
    device.start(rate=8000)
    opened[0].feed(256, value=0.0)
    target = tmp_path / "take.wav"
    finished: list[str] = []
    device.finished.connect(finished.append)
    device.stop(target)
    assert target.exists()
    assert finished == [str(target)]
