# SPDX-License-Identifier: AGPL-3.0-only
"""The meter map: which bar a beat falls in over a song whose bar length changes."""

from __future__ import annotations

import pytest

from namioto.meter_map import DEFAULT_BEATS_PER_BAR, MeterMap, MeterPoint


def verse_then_bridge() -> MeterMap:
    """Two bars of 4/4, then 3/4 from beat 8 on: bar 3 starts the bridge."""
    return MeterMap.from_pairs([[0, 4, 1], [8, 3, 1]])


def test_a_single_meter_numbers_its_bars_from_the_offset() -> None:
    meter = MeterMap.single()
    assert meter.is_single
    assert meter.points == (MeterPoint(0.0, DEFAULT_BEATS_PER_BAR, 1),)
    # the default offset is 1, so the bar starting on beat 0 is bar 1
    assert meter.bar_at(0.0) == (1, 0.0)
    assert meter.bar_at(4.0) == (2, 0.0)
    assert meter.bar_at(2.0) == (1, 2.0)


def test_a_meter_change_starts_a_new_bar_on_its_own_beat() -> None:
    meter = verse_then_bridge()
    assert meter.bar_at(0.0) == (1, 0.0)
    assert meter.bar_at(8.0) == (3, 0.0)  # two 4/4 bars behind it, so the bridge opens bar 3
    assert meter.bar_at(11.0) == (4, 0.0)  # one 3/4 bar later
    assert meter.bar_at(9.5) == (3, 1.5)


def test_a_bar_offset_renames_the_first_bar() -> None:
    meter = MeterMap.single(4.0, bar_offset=5)
    assert meter.bar_at(0.0) == (5, 0.0)
    assert meter.bar_at(8.0) == (7, 0.0)
    # a change keeps the numbering running from the offset rather than restarting it
    later = MeterMap.from_pairs([[0, 4, 5], [8, 3, 1]])
    assert later.bar_at(8.0) == (7, 0.0)


def test_bar_lines_follow_the_meter_in_force() -> None:
    meter = verse_then_bridge()
    lines = meter.bar_lines(0.0, 20.0)
    # 4/4 up to beat 8, then every 3 beats, numbered on from bar 3
    assert lines[:3] == [(0.0, 1), (4.0, 2), (8.0, 3)]
    assert lines[3:6] == [(11.0, 4), (14.0, 5), (17.0, 6)]
    assert all(later - earlier == 3.0 for (earlier, _), (later, _) in zip(lines[2:], lines[3:], strict=False))
    # a span that only touches the bridge still reports the bridge's own spacing
    assert [beat for beat, _ in meter.bar_lines(8.0, 14.0)] == [8.0, 11.0]
    assert meter.bar_lines(8.0, 8.0) == []


def test_with_change_replaces_a_point_on_the_same_beat_and_keeps_the_order() -> None:
    meter = verse_then_bridge()
    replaced = meter.with_change(MeterPoint(8.0, 6.0, 1))
    assert [point.beat for point in replaced.points] == [0.0, 8.0]
    assert replaced.beats_per_bar_at(8.0) == 6.0
    added = meter.with_change(MeterPoint(2.0, 5.0, 1))
    assert [point.beat for point in added.points] == [0.0, 2.0, 8.0]
    assert added.bar_at(8.0)[0] == 2  # half a bar of 4/4 is rounded to one whole bar


def test_without_drops_a_change_but_never_the_first() -> None:
    meter = verse_then_bridge()
    assert meter.without(8.0).is_single
    assert meter.without(0.0) == meter
    assert meter.without(99.0) == meter


def test_rescaled_moves_the_beats_and_keeps_the_bar_length() -> None:
    meter = verse_then_bridge()
    stretched = meter.rescaled(2.0)
    assert [point.beat for point in stretched.points] == [0.0, 16.0]
    # only the beats move: a bar is still four beats, and the numbering is derived from that
    assert [point.beats_per_bar for point in stretched.points] == [4.0, 3.0]
    assert [point.bar_offset for point in stretched.points] == [1, 1]
    assert stretched.bar_at(16.0) == (5, 0.0)
    assert meter.rescaled(1.0) == meter


def test_pairs_round_trip_and_junk_is_skipped() -> None:
    meter = verse_then_bridge()
    assert MeterMap.from_pairs(meter.to_pairs()) == meter
    assert MeterMap.from_dict(meter.to_dict()) == meter
    assert MeterMap.from_pairs([[0, 3]]) == MeterMap.single(3.0)
    # a pair with no bar length is skipped; one with a bar length but no beat 0 gets a point on 0
    assert MeterMap.from_pairs([[0], "junk", [2, 3, 4]]) == MeterMap((MeterPoint(2.0, 3.0, 4),))
    assert MeterMap.from_pairs([[0], "junk"]).is_single
    assert MeterMap.from_dict(None).is_single


def test_a_bar_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="beats_per_bar must be positive"):
        MeterMap((MeterPoint(0.0, 0.0, 1),))
