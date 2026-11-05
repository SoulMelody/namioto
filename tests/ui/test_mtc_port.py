# SPDX-License-Identifier: AGPL-3.0-only
"""The time code port, driven with fake ports: no device is ever opened."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from namioto.mtc import MtcTime, Rate
from namioto.ui.mtc_port import MtcPort, pack_quarter_frame, quarter_frame_interval_ms


class FakeOutput:
    """An output that keeps what it was sent."""

    def __init__(self):
        self.sent: list[list[int]] = []
        self.closed = False

    def send(self, message) -> None:
        self.sent.append(list(message))

    def close(self) -> None:
        self.closed = True


class FakeInput(QObject):
    """An input a test can push messages into, as the real callback thread would."""

    message = Signal(object)

    def __init__(self):
        super().__init__()
        self.on_message = None
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def push(self, data) -> None:
        if self.on_message is not None:
            self.on_message(data)


def make_port(output=None, source=None) -> MtcPort:
    return MtcPort(output_factory=lambda _name: output, input_factory=lambda _name: source)


def test_one_quarter_frame_is_a_quarter_of_a_frame() -> None:
    assert quarter_frame_interval_ms(Rate.FPS_25) == 10.0
    assert quarter_frame_interval_ms(Rate.FPS_30) == 1000.0 / 120.0


def test_sending_eight_quarter_frames_advances_exactly_one_frame(qt_app, monkeypatch) -> None:
    output = FakeOutput()
    port = make_port(output)
    assert port.start_send("fake", Rate.FPS_25) is True
    assert output.sent[0][0] == 0xF0  # a full frame first, so a receiver starts from a position

    port._send_timer.stop()
    before = port._clock.time.to_frames()
    # eight intervals of 10 ms: the clock must land exactly one frame on, not two
    for step in range(1, 9):
        monkeypatch.setattr(port._elapsed, "elapsed", lambda step=step: step * 10)
        port._on_send_tick()

    after = port._clock.time.to_frames()
    quarters = [message for message in output.sent if message[0] == 0xF1]
    assert len(quarters) == 8
    assert [message[1] >> 4 for message in quarters] == list(range(8))
    assert after - before == 1
    port.stop()


def test_a_quarter_frame_goes_out_every_interval(qt_app, monkeypatch) -> None:
    output = FakeOutput()
    port = make_port(output)
    assert port.start_send("fake", Rate.FPS_25) is True
    port._send_timer.stop()

    # 40 ms of a 10 ms interval is four pieces, so the clock moves half a frame
    monkeypatch.setattr(port._elapsed, "elapsed", lambda: 40)
    port._on_send_tick()

    quarters = [message for message in output.sent if message[0] == 0xF1]
    assert len(quarters) == 4
    port.stop()


def test_receiving_eight_pieces_emits_a_position(qt_app) -> None:
    source = FakeInput()
    port = make_port(source=source)
    positions: list[float] = []
    port.position.connect(positions.append)
    assert port.start_receive("fake") is True

    clock = MtcTime(hours=0, minutes=1, seconds=2, frames=3, rate=Rate.FPS_25)
    for index in range(8):
        source.push(pack_quarter_frame(index, clock))

    assert len(positions) == 1
    assert positions[0] == clock.to_seconds()
    port.stop()


def test_a_full_frame_sets_the_position_at_once(qt_app) -> None:
    from namioto.mtc import encode_full_frame

    source = FakeInput()
    port = make_port(source=source)
    positions: list[float] = []
    port.position.connect(positions.append)
    assert port.start_receive("fake") is True

    clock = MtcTime(hours=1, minutes=0, seconds=0, frames=0, rate=Rate.FPS_30)
    source.push(encode_full_frame(clock))

    assert positions == [clock.to_seconds()]
    port.stop()


def test_an_undefined_rate_code_reports_a_failure(qt_app) -> None:
    source = FakeInput()
    port = make_port(source=source)
    failures: list[str] = []
    positions: list[float] = []
    port.failed.connect(failures.append)
    port.position.connect(positions.append)
    assert port.start_receive("fake") is True

    # the eighth nibble is `reserved | rate(2) | hour-high-bit`; setting the reserved bit names no
    # rate the standard defines. Nothing may fall back to 25: a wrong rate is worse than none
    for index in range(8):
        nibble = 0x08 if index == 7 else 0x00
        source.push([0xF1, (index << 4) | nibble])

    assert positions == []
    assert failures and "4" in failures[0]  # (0x08 >> 1) & 0x07 is 4, which MTC does not define
    port.stop()


def test_a_defined_rate_is_not_mistaken_for_an_undefined_one(qt_app) -> None:
    source = FakeInput()
    port = make_port(source=source)
    positions: list[float] = []
    port.position.connect(positions.append)
    assert port.start_receive("fake") is True

    clock = MtcTime(hours=0, minutes=0, seconds=1, frames=0, rate=Rate.FPS_29_97_DROP)
    for index in range(8):
        source.push(pack_quarter_frame(index, clock))

    assert len(positions) == 1
    assert positions[0] == clock.to_seconds()
    port.stop()


def test_stop_without_a_start_is_harmless(qt_app) -> None:
    port = make_port()
    port.stop()
    port.stop()
    assert port._output is None
    assert port._input is None
    assert port._send_timer is None


def test_a_port_that_will_not_open_says_so(qt_app) -> None:
    port = make_port(output=None)
    failures: list[str] = []
    port.failed.connect(failures.append)
    assert port.start_send("fake", Rate.FPS_25) is False
    assert failures and "fake" in failures[0]


def test_stop_closes_both_sides(qt_app) -> None:
    output = FakeOutput()
    source = FakeInput()
    port = make_port(output, source)
    assert port.start_send("fake", Rate.FPS_25) is True
    port.stop()
    assert output.closed is True
    assert port._send_timer is None

    port = make_port(source=source)
    assert port.start_receive("fake") is True
    port.stop()
    assert source.closed is True
