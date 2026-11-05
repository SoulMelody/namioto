# SPDX-License-Identifier: AGPL-3.0-only
"""MIDI Time Code: exact frame rates, quarter frame pieces and the clock they assemble."""

from __future__ import annotations

from fractions import Fraction

import pytest

from namioto import mtc


def test_the_drop_frame_rate_is_the_exact_fraction_it_claims_to_be() -> None:
    # the reference implementation's defect: 29.97 as a float drifts, and 29 is the wrong rate
    assert mtc.Rate.FPS_29_97_DROP.fps == Fraction(30000, 1001)
    assert isinstance(mtc.Rate.FPS_29_97_DROP.fps, Fraction)
    assert mtc.Rate.FPS_29_97_DROP.fps != Fraction(2997, 100)
    assert mtc.Rate.FPS_29_97_DROP.fps != Fraction(29, 1)
    assert float(mtc.Rate.FPS_29_97_DROP.fps) == pytest.approx(29.97, abs=0.001)
    # the other three are whole frames, and every rate carries a code
    assert [mtc.Rate.FPS_24.fps, mtc.Rate.FPS_25.fps, mtc.Rate.FPS_30.fps] == [Fraction(24), Fraction(25), Fraction(30)]
    assert mtc.Rate.FPS_29_97_DROP.code == 2


def test_a_quarter_frame_round_trips_to_its_index_and_nibble() -> None:
    clock = mtc.MtcTime(hours=1, minutes=2, seconds=3, frames=4, rate=mtc.Rate.FPS_25)
    for index in range(8):
        message = mtc.encode_quarter_frame(index, clock)
        assert message[0] == mtc.QUARTER_FRAME
        assert mtc.decode_quarter_frame(message) == (index, message[1] & 0x0F)
    # the first two pieces carry the low and high nibbles of the frame count
    assert mtc.decode_quarter_frame(mtc.encode_quarter_frame(0, clock))[1] == 4 & 0x0F
    assert mtc.decode_quarter_frame(mtc.encode_quarter_frame(1, clock))[1] == (4 >> 4) & 0x0F
    # anything that is not a quarter frame decodes to nothing
    assert mtc.decode_quarter_frame([0xF0, 0x00]) is None
    assert mtc.decode_quarter_frame([]) is None
    with pytest.raises(ValueError, match="0..7"):
        mtc.encode_quarter_frame(8, clock)


def test_eight_quarter_frames_assemble_back_into_the_same_clock() -> None:
    clock = mtc.MtcTime(hours=23, minutes=59, seconds=58, frames=29, rate=mtc.Rate.FPS_30)
    assembler = mtc.QuarterFrameAssembler()
    assembled = None
    for index in range(8):
        assembled = assembler.feed(mtc.encode_quarter_frame(index, clock)) or assembled
    assert assembled == clock
    # and a full frame SysEx round-trips the same position
    assert mtc.decode_full_frame(mtc.encode_full_frame(clock)) == clock
    assert mtc.decode_full_frame(mtc.encode_full_frame(clock)) is not None


def test_a_dropped_quarter_frame_resets_the_assembler_rather_than_guessing() -> None:
    clock = mtc.MtcTime(hours=0, minutes=0, seconds=1, frames=0, rate=mtc.Rate.FPS_25)
    assembler = mtc.QuarterFrameAssembler()
    assert assembler.feed(mtc.encode_quarter_frame(0, clock)) is None
    assert assembler.feed(mtc.encode_quarter_frame(2, clock)) is None  # index 1 went missing
    assert assembler.feed(mtc.encode_quarter_frame(0, clock)) is None


def test_a_second_holds_four_quarter_frames_per_frame() -> None:
    for rate in mtc.Rate:
        interval = mtc.qf_interval_seconds(rate)
        assert interval * 4 == Fraction(1, 1) / rate.fps
        # the count of quarter frames in one second is exactly four per frame, in Fraction arithmetic
        per_second = Fraction(1, 1) / interval
        assert per_second == 4 * rate.fps
        # drop-frame's count is not a whole number, and Fraction says so rather than rounding it
        if rate is mtc.Rate.FPS_29_97_DROP:
            assert per_second == Fraction(120000, 1001)
            assert per_second.denominator != 1
        else:
            assert per_second.denominator == 1


def test_the_clock_round_trips_through_seconds_at_every_rate() -> None:
    for rate in (mtc.Rate.FPS_24, mtc.Rate.FPS_25, mtc.Rate.FPS_29_97_DROP, mtc.Rate.FPS_30):
        for seconds in (0.0, 1.0, 59.5, 3600.0):
            clock = mtc.MtcClock.from_seconds(seconds, rate)
            back = clock.to_seconds()
            # within one frame of where it started, drop-frame included
            assert abs(back - seconds) < 1.0 / float(rate.fps)
            assert clock.rate == rate


def test_eight_quarters_advance_the_clock_by_exactly_two_frames() -> None:
    start = mtc.MtcClock(mtc.MtcTime(0, 0, 0, 0, mtc.Rate.FPS_25))
    # four quarter frames go out per frame, so eight of them carry the clock round one whole
    # timecode: the pieces describe a position, and a position moved on by two frames
    assert start.advance_quarters(4).time.to_frames() == 1
    assert start.advance_quarters(8).time.to_frames() == 2
    assert start.advance_frames(25).time.seconds == 1
    assert start.quarter_frames() == [mtc.encode_quarter_frame(index, start.time) for index in range(8)]
