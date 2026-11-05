# SPDX-License-Identifier: AGPL-3.0-only
"""MIDI Time Code over a port: the carriage, never the policy.

`namioto.mtc` already owns the arithmetic - what a quarter frame carries, what eight of them
assemble into, and how 29.97 is kept exact. What it cannot own is a port or a timer, because those
are Qt and RtMidi and neither belongs in a module the CLI imports. This file is that seam and nothing
more: it puts encoded messages on a wire and takes decoded positions off one.

Two rules live here because a port layer is the only place they can be got right:

* **Eight quarter frames advance one frame.** A quarter frame goes out every `1/(4*fps)` seconds, so
  the eighth one lands exactly one frame after the first. The reference implementation this project
  was ported from advanced its frame counter by *two* after each eight, which hid the fact that its
  output ran at half rate and made any third-party device read 2x. Advancing one quarter per piece
  keeps the rate standard.
* **The RtMidi callback thread never touches Qt state.** supriya-midi calls back on its own thread. Emitting a
  Qt signal there is safe, because Qt queues a cross-thread connection onto the receiver's thread;
  calling a slot or a `seek` directly is not. So the callback decodes, emits, and returns.

Where a position *goes* is not this file's business: `position` is a number, and app.py decides
whether that number is a seek, a status line, or nothing.
"""

from __future__ import annotations

from contextlib import suppress

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal

from namioto import midi_port
from namioto.i18n import tr
from namioto.mtc import (
    MtcClock,
    MtcTime,
    Rate,
    decode_full_frame,
    decode_quarter_frame,
    encode_full_frame,
    encode_quarter_frame,
)

# how many pieces one timer tick may catch up on: a late tick is caught up rather than skipped, but a
# machine that was suspended for a minute must not fire a minute of messages at once
MAX_CATCH_UP = 4
QUARTER_FRAMES_PER_FRAME = 8
QUARTER_FRAME = 0xF1
SYSEX_START = 0xF0


def quarter_frame_interval_ms(rate: Rate) -> float:
    """How long one quarter frame lasts, in milliseconds, at `rate`."""
    return 1000.0 / float(4 * rate.fps)


def message_bytes(message) -> list[int]:
    """A message's bytes, however a backend handed it over: `bytes` or a plain sequence."""
    for name in ("bytes", "data"):
        attribute = getattr(message, name, None)
        if callable(attribute):
            return [int(value) for value in attribute()]
        if attribute is not None:
            return [int(value) for value in attribute]
    return [int(value) for value in message]


def pack_quarter_frame(index: int, clock: MtcTime) -> list[int]:
    """One quarter frame as `[0xF1, (index << 4) | nibble]`, the shape MTC expects.

    `namioto.mtc.encode_quarter_frame` returns `[0xF1, nibble]` with the index left out, so its own
    `decode_quarter_frame` reads index 0 for every piece and a stream it encodes cannot be decoded
    back into a position. That module is the analysis side and is not this file's to change, so the
    index is put where the standard says it goes here: the nibble's arithmetic still comes from
    `namioto.mtc`, only the packaging is redone.
    """
    message = encode_quarter_frame(index, clock)
    return [message[0], ((int(index) & 0x07) << 4) | (int(message[1]) & 0x0F)]


class MtcPort(QObject):
    """One MIDI port carrying time code, sending or receiving.

    A port is either sending or receiving, never both: each `start_*` stops whatever was running
    first, so a caller never has to remember to.

    Signals:

    * `position(seconds)` - a received time code, in the song's own seconds. Emitted from the RtMidi
      thread, so a receiver must connect it with the default (queued) connection.
    * `synced(bool)` - whether the receive side has assembled a frame, or the send side is running.
    * `failed(text)` - the port would not open, or carried a rate code MTC does not define.

    `output_factory` / `input_factory` are the seam the tests replace; they default to
    `namioto.midi_port`, and they are handed a port name rather than a callback, so a port that is
    opened here is also where its callback is set.
    """

    position = Signal(float)
    synced = Signal(bool)
    failed = Signal(str)

    def __init__(self, parent=None, *, output_factory=None, input_factory=None):
        super().__init__(parent)
        self._output_factory = output_factory or _open_output
        self._input_factory = input_factory or _open_input
        self._output = None
        self._input = None
        self._send_timer: QTimer | None = None
        self._interval_ms = 0.0
        self._clock: MtcClock | None = None
        self._elapsed = QElapsedTimer()
        self._advanced = 0  # quarter frames the clock has moved, which is also the sequence index
        self._accounted = 0  # quarter frames the timer has been credited for, so catch-up is bounded
        self._base_frames = 0  # the whole frame the clock started on, so eight quarters is one frame
        self._pieces: list[int | None] = [None] * QUARTER_FRAMES_PER_FRAME
        self._next_piece = 0

    # -- sending -----------------------------------------------------------------------------

    def start_send(self, port_name: str, rate: Rate) -> bool:
        """Send time code on `port_name` from 00:00:00:00, advancing at `rate`.

        The clock starts at zero and moves with the timer. A caller that wants to send from the
        playhead stops, calls `seek_to`, and starts again - this file has no opinion about when.
        """
        self.stop()
        output = self._output_factory(port_name)
        if output is None:
            self.failed.emit(tr("MIDI port {port} could not be opened", port=port_name))
            return False
        self._output = output
        self._clock = MtcClock(MtcTime(rate=rate))
        self._base_frames = 0
        self._advanced = 0
        self._accounted = 0
        self._output_send(encode_full_frame(self._clock.time))  # a receiver starts on a full frame
        # the timer can only count whole milliseconds, but 1000/(4*29.97) is 8.342: the tick uses
        # the exact interval and the timer only decides how often it is asked, so the rate stays
        # right instead of running 4% fast
        self._interval_ms = quarter_frame_interval_ms(rate)
        self._send_timer = QTimer(self)
        self._send_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._send_timer.setInterval(max(1, round(self._interval_ms)))
        self._send_timer.timeout.connect(self._on_send_tick)
        self._elapsed.start()
        self._send_timer.start()
        self.synced.emit(True)
        return True

    def seek_to(self, seconds: float) -> None:
        """Move the send clock to `seconds`, so the next piece carries that position."""
        if self._clock is None:
            return
        self._clock = MtcClock.from_seconds(seconds, self._clock.rate)
        self._base_frames = self._clock.time.to_frames()
        self._advanced = 0
        self._accounted = 0
        self._output_send(encode_full_frame(self._clock.time))

    def _on_send_tick(self) -> None:
        """Account for every quarter frame whose time has come, sending at most `MAX_CATCH_UP`."""
        if self._clock is None or self._send_timer is None:
            return
        interval = float(self._interval_ms)
        target = int(self._elapsed.elapsed() // interval)
        due = target - self._accounted
        if due <= 0:
            return
        sent = min(due, MAX_CATCH_UP)
        # the pieces that were skipped still happened: the clock moves on regardless, so a machine
        # that was too busy to send every piece still reports the right position, just coarser
        self._advance(due - sent)
        for _ in range(sent):
            # the index is the clock's own quarter position, not a counter of what was sent: a
            # dropped piece then leaves a gap in the sequence a receiver can see, instead of
            # relabelling the pieces that did arrive and assembling them into a wrong clock
            self._output_send(pack_quarter_frame(self._advanced % QUARTER_FRAMES_PER_FRAME, self._clock.time))
            self._advance(1)
        self._accounted = target

    def _advance(self, count: int) -> None:
        """Move the send clock on by `count` quarter frames; eight of them make exactly one frame.

        The clock is rebuilt from whole frames plus the quarters not yet a frame, rather than using
        `MtcClock.advance_quarters`: that helper divides by four, so eight quarters would move it two
        frames, which is the very half-rate error this port exists to avoid.
        """
        if self._clock is None or count <= 0:
            return
        self._advanced += int(count)
        frames = self._base_frames + self._advanced // QUARTER_FRAMES_PER_FRAME
        self._clock = MtcClock(MtcTime.from_frames(frames, self._clock.rate))

    def _output_send(self, message: list[int]) -> None:
        if self._output is None:
            return
        try:
            self._output.send(message)
        except Exception as error:  # a port unplugged mid-song is a message, not a crash
            self.failed.emit(str(error))

    # -- receiving ---------------------------------------------------------------------------

    def start_receive(self, port_name: str) -> bool:
        """Listen on `port_name` for time code; every complete frame emits `position`."""
        self.stop()
        source = self._input_factory(port_name)
        if source is None:
            self.failed.emit(tr("MIDI port {port} could not be opened", port=port_name))
            return False
        self._input = source
        try:
            source.on_message = self._on_message
            source.start()
        except Exception as error:  # a device that refused to open is a message, not a crash
            self._input = None
            self.failed.emit(str(error))
            return False
        return True

    def _on_message(self, message) -> None:
        """The port callback: decode and emit only. Never touch a Qt object here."""
        data = message_bytes(message)
        if not data:
            return
        if data[0] == QUARTER_FRAME:
            self._on_quarter_frame(data)
        elif data[0] == SYSEX_START:
            self._on_full_frame(data)

    def _on_quarter_frame(self, data: list[int]) -> None:
        decoded = decode_quarter_frame(data)
        if decoded is None:
            return
        index, nibble = decoded
        if index != self._next_piece:  # a lost piece leaves the rest unusable, so start over
            self._reset_pieces()
            if index != 0:
                return
        self._pieces[index] = nibble
        self._next_piece = index + 1
        if self._next_piece < QUARTER_FRAMES_PER_FRAME:
            return
        self._assemble()

    def _assemble(self) -> None:
        """Eight pieces in order become one position; a rate code MTC does not define is reported."""
        pieces = self._pieces
        self._reset_pieces()
        if any(piece is None for piece in pieces):
            return
        # the nibble's bit 0 is the hour's high bit, bits 1-2 are the rate, and bit 3 is reserved.
        # Reading three bits rather than masking to two is what makes an undefined rate *reportable*:
        # a stream that sets the reserved bit names no rate the standard defines, and the reference
        # implementation quietly fell back to 25 for it, turning a 30 fps stream into a clock that
        # ran 20% slow. A wrong rate is worse than no position, so this reports instead of guessing.
        code = (pieces[7] >> 1) & 0x07
        rate = Rate.from_code(code)
        if rate is None:
            self.failed.emit(tr("Time code uses an undefined frame rate ({code})", code=code))
            return
        clock = MtcTime(
            # the eighth nibble carries the rate in its middle bits and the hour's high bit in its
            # lowest one, which is why the rate is read before the hour is put back together
            hours=pieces[6] | ((pieces[7] & 0x01) << 4),
            minutes=pieces[4] | (pieces[5] << 4),
            seconds=pieces[2] | (pieces[3] << 4),
            frames=pieces[0] | (pieces[1] << 4),
            rate=rate,
        )
        self.synced.emit(True)
        self.position.emit(clock.to_seconds())

    def _on_full_frame(self, data: list[int]) -> None:
        clock = decode_full_frame(data)
        if clock is None:  # a SysEx that is not MTC is not ours to judge
            return
        self._reset_pieces()
        self.synced.emit(True)
        self.position.emit(clock.to_seconds())

    def _reset_pieces(self) -> None:
        self._pieces = [None] * QUARTER_FRAMES_PER_FRAME
        self._next_piece = 0

    # -- lifetime ----------------------------------------------------------------------------

    def stop(self) -> None:
        """Close whatever is open. Safe on a port that was never started."""
        if self._send_timer is not None:
            self._send_timer.stop()
            self._send_timer.deleteLater()
            self._send_timer = None
        if self._output is not None:
            with suppress(Exception):
                self._output.close()
            self._output = None
        if self._input is not None:
            with suppress(Exception):
                self._input.close()  # a receiving port drops its callback as it closes
            self._input = None
        self._clock = None
        self._advanced = 0
        self._accounted = 0
        self._base_frames = 0
        self._reset_pieces()
        self.synced.emit(False)


def _open_output(name: str) -> midi_port.Port | None:
    """The real output, or None when the backend cannot reach the port."""
    return midi_port.open_output(name)


def _open_input(name: str) -> midi_port.Port | None:
    """The real input, or None when the backend cannot reach the port.

    No callback yet: the factory is handed a name, and `start_receive` sets the callback once the
    port is known to have opened, which is also when this one starts listening.
    """
    return midi_port.open_input(name)
