# SPDX-License-Identifier: AGPL-3.0-only
"""MIDI Time Code: the frame clock a transport can hand to, or take from, another machine.

MTC is eight quarter frames per frame, four per second per frame rate, and a full frame message
that says where a clock jumped to. Nothing here talks to a port: it encodes and decodes `list[int]`
messages, so the port layer above it owns the timing and this owns the arithmetic.

Two things a naive implementation gets wrong, both of them corrected here:

* **Eight quarter frames are one frame, not two.** A quarter frame goes out every `1/(4*fps)`
  seconds, so the eighth one lands exactly one frame after the first. The reference implementation
  this was ported from advanced its frame counter by two after each eight, which made its clock run
  at half speed and read as 2x to any third-party device.
* **29.97 is `30000/1001`, not 29.97 and certainly not 29.** Drop-frame is a real rate, and a float
  approximation accumulates: about 3.6 seconds of drift over an hour. `Fraction` keeps it exact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

QUARTER_FRAME = 0xF1
SYSEX_START = 0xF0
SYSEX_END = 0xF7
MTC_SYSEX = (0x7F, 0x7F, 0x01, 0x01)  # universal real time, sub-id 1, time code full frame
RATE_CODES = {"FPS_24": 0, "FPS_25": 1, "FPS_29_97_DROP": 2, "FPS_30": 3}


class Rate(Enum):
    """The four frame rates MTC can carry, 29.97 drop kept exact."""

    FPS_24 = Fraction(24, 1)
    FPS_25 = Fraction(25, 1)
    FPS_29_97_DROP = Fraction(30000, 1001)
    FPS_30 = Fraction(30, 1)

    @property
    def fps(self) -> Fraction:
        return self.value

    @property
    def code(self) -> int:
        return RATE_CODES[self.name]

    @classmethod
    def from_code(cls, code: int) -> Rate | None:
        return next((rate for rate in cls if rate.code == code), None)

    @classmethod
    def from_name(cls, name: str) -> Rate | None:
        return next((rate for rate in cls if rate.name == name), None)


def _whole_frames(rate: Rate) -> int:
    """The frame count a whole second holds: 24, 25, 30, and 30 for drop-frame too."""
    return int(rate.fps) if rate is not Rate.FPS_29_97_DROP else 30


@dataclass(frozen=True)
class MtcTime:
    """A position in hours, minutes, seconds and frames at one rate."""

    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    frames: int = 0
    rate: Rate = Rate.FPS_25

    def __post_init__(self) -> None:
        nominal = _whole_frames(self.rate)
        # normalise past the end of a second rather than reject: a caller adding frames should not
        # have to carry by hand, and a clock that overflows is a clock that wrapped, not an error
        frames_total = (self.hours * 3600 + self.minutes * 60 + self.seconds) * nominal + self.frames
        object.__setattr__(self, "frames", frames_total % nominal)
        remainder = frames_total // nominal
        object.__setattr__(self, "seconds", remainder % 60)
        object.__setattr__(self, "minutes", (remainder // 60) % 60)
        object.__setattr__(self, "hours", (remainder // 3600) % 24)

    def to_seconds(self) -> float:
        """The exact position in seconds, drop-frame included."""
        nominal = _whole_frames(self.rate)
        frames = (self.hours * 3600 + self.minutes * 60 + self.seconds) * nominal + self.frames
        return float(Fraction(frames) / self.rate.fps)

    def to_frames(self) -> int:
        """The position as a count of whole frames at this rate."""
        return (self.hours * 3600 + self.minutes * 60 + self.seconds) * _whole_frames(self.rate) + self.frames

    @classmethod
    def from_seconds(cls, seconds: float, rate: Rate = Rate.FPS_25) -> MtcTime:
        """The position `seconds` names, rounded to the nearest frame."""
        total = int(round(Fraction(float(seconds)) * rate.fps))
        return cls.from_frames(total, rate)

    @classmethod
    def from_frames(cls, frames: int, rate: Rate = Rate.FPS_25) -> MtcTime:
        nominal = _whole_frames(rate)
        frames = int(frames)
        return cls(
            hours=(frames // (nominal * 3600)) % 24,
            minutes=(frames // (nominal * 60)) % 60,
            seconds=(frames // nominal) % 60,
            frames=frames % nominal,
            rate=rate,
        )


def _nibble(clock: MtcTime, index: int) -> int:
    """The nibble one quarter frame carries, in the order the standard lays them out."""
    if index == 0:
        return clock.frames & 0x0F
    if index == 1:
        return (clock.frames >> 4) & 0x0F
    if index == 2:
        return clock.seconds & 0x0F
    if index == 3:
        return (clock.seconds >> 4) & 0x0F
    if index == 4:
        return clock.minutes & 0x0F
    if index == 5:
        return (clock.minutes >> 4) & 0x0F
    if index == 6:
        return clock.hours & 0x0F
    if index == 7:
        return ((clock.rate.code << 1) | ((clock.hours >> 4) & 0x01)) & 0x0F
    raise ValueError(f"quarter frame index must be 0..7, got {index!r}")


def encode_quarter_frame(index: int, clock: MtcTime) -> list[int]:
    """One quarter frame as a MIDI message: `[0xF1, nnn_dddd]`.

    The high nibble of the data byte is the piece's own index and the low nibble the value it
    carries; leaving the index out would make every piece decode as piece 0, which is exactly the
    half-speed clock a decoder assembled from such a stream would keep.
    """
    if not 0 <= index <= 7:
        raise ValueError(f"quarter frame index must be 0..7, got {index!r}")
    return [QUARTER_FRAME, ((index & 0x07) << 4) | (_nibble(clock, index) & 0x0F)]


def decode_quarter_frame(data) -> tuple[int, int] | None:
    """`(index, nibble)` from a quarter frame message, or None for anything else."""
    if not data or len(data) < 2 or int(data[0]) != QUARTER_FRAME:
        return None
    value = int(data[1])
    return (value >> 4) & 0x07, value & 0x0F


def encode_full_frame(clock: MtcTime) -> list[int]:
    """A full frame SysEx: `F0 7F 7F 01 01 hr mn sc fr F7`."""
    hour_byte = ((clock.rate.code << 5) | (clock.hours & 0x1F)) & 0xFF
    return [
        SYSEX_START,
        *MTC_SYSEX,
        hour_byte,
        clock.minutes & 0x7F,
        clock.seconds & 0x7F,
        clock.frames & 0x1F,
        SYSEX_END,
    ]


def decode_full_frame(data) -> MtcTime | None:
    """The clock a full frame SysEx carries, or None when it is not one."""
    if not data or len(data) < 10:
        return None
    payload = [int(value) for value in data]
    if payload[0] != SYSEX_START or payload[-1] != SYSEX_END:
        return None
    if tuple(payload[1:5]) != MTC_SYSEX:
        return None
    hour_byte = payload[5]
    rate = Rate.from_code((hour_byte >> 5) & 0x03)
    if rate is None:
        return None
    return MtcTime(hour_byte & 0x1F, payload[6] & 0x7F, payload[7] & 0x7F, payload[8] & 0x1F, rate)


class QuarterFrameAssembler:
    """Eight quarter frames in, one clock out.

    A set only counts when its indices arrive in order from 0: a frame lost in the middle of a
    stream leaves seven good pieces that would otherwise be assembled into a clock that is wrong by
    however much the missing piece carried.
    """

    def __init__(self) -> None:
        self._pieces: list[int | None] = [None] * 8
        self._next = 0

    def reset(self) -> None:
        self._pieces = [None] * 8
        self._next = 0

    def feed(self, data) -> MtcTime | None:
        """Add one quarter frame; returns a clock once eight of them have arrived in order."""
        decoded = decode_quarter_frame(data)
        if decoded is None:
            return None
        index, nibble = decoded
        if index != self._next:
            self.reset()
            if index != 0:
                return None
            self._next = 0
        self._pieces[index] = nibble
        self._next = index + 1
        if self._next < 8:
            return None
        return self._assemble()

    def _assemble(self) -> MtcTime | None:
        pieces = self._pieces
        if any(piece is None for piece in pieces):
            self.reset()
            return None
        frames = pieces[0] | (pieces[1] << 4)
        seconds = pieces[2] | (pieces[3] << 4)
        minutes = pieces[4] | (pieces[5] << 4)
        hours = pieces[6] | ((pieces[7] & 0x01) << 4)
        rate = Rate.from_code((pieces[7] >> 1) & 0x03)
        self.reset()
        if rate is None:
            return None
        return MtcTime(hours, minutes, seconds, frames, rate)


def qf_interval_seconds(rate: Rate) -> Fraction:
    """How long one quarter frame lasts: a quarter of a frame."""
    return Fraction(1, 1) / (rate.fps * 4)


@dataclass(frozen=True)
class MtcClock:
    """A running MTC position, advanced a quarter frame at a time.

    Advancing by quarters rather than by frames is what keeps the eight-piece sequence and the
    clock the same thing: four quarters move the clock a second's quarter, and eight move a frame.
    """

    time: MtcTime = MtcTime()

    @property
    def rate(self) -> Rate:
        return self.time.rate

    def to_seconds(self) -> float:
        return self.time.to_seconds()

    @classmethod
    def from_seconds(cls, seconds: float, rate: Rate = Rate.FPS_25) -> MtcClock:
        return cls(MtcTime.from_seconds(seconds, rate))

    def advance_quarters(self, count: int = 1) -> MtcClock:
        """Move the clock on by `count` quarter frames; eight of them make a frame."""
        frames = self.time.to_frames() * 4 + int(count)
        return MtcClock(MtcTime.from_frames(frames // 4, self.rate))

    def advance_frames(self, count: int = 1) -> MtcClock:
        return MtcClock(MtcTime.from_frames(self.time.to_frames() + int(count), self.rate))

    def quarter_frame(self, index: int) -> list[int]:
        return encode_quarter_frame(index, self.time)

    def quarter_frames(self) -> list[list[int]]:
        """The eight pieces that describe this clock, in the order they go out."""
        return [encode_quarter_frame(index, self.time) for index in range(8)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namioto.mtc", description="encode an MTC position")
    parser.add_argument("--rate", choices=[rate.name for rate in Rate], default=Rate.FPS_25.name)
    parser.add_argument("--seconds", type=float, default=3661.5)
    args = parser.parse_args(argv)
    rate = Rate.from_name(args.rate) or Rate.FPS_25
    clock = MtcClock.from_seconds(args.seconds, rate)
    print(f"{clock.time} at {rate.name}, {clock.to_seconds():.6f} s")
    for index, message in enumerate(clock.quarter_frames()):
        print(f"  QF {index}: {message[0]:02X} {message[1]:02X}")
    print("  full frame: " + " ".join(f"{value:02X}" for value in encode_full_frame(clock.time)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
