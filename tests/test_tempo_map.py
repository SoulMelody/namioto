# SPDX-License-Identifier: AGPL-3.0-only
"""The tempo map: beats to seconds over a song whose tempo changes."""

from __future__ import annotations

import pytest

from namioto.tempo_map import DEFAULT_BPM, MAX_BPM, MIN_BPM, TempoMap, TempoPoint


def two_speeds() -> TempoMap:
    """Four beats at 120 BPM (2 s), then 60 BPM from there on."""
    return TempoMap((TempoPoint(0.0, 120.0), TempoPoint(4.0, 60.0)))


def test_a_single_tempo_converts_both_ways() -> None:
    tempo = TempoMap.single(120.0)
    assert tempo.is_single
    assert tempo.bpm == 120.0
    assert tempo.seconds_at(0.0) == 0.0
    assert tempo.seconds_at(4.0) == pytest.approx(2.0)
    assert tempo.beats_at(2.0) == pytest.approx(4.0)
    assert tempo.beats_at(0.0) == 0.0


def test_a_change_takes_effect_on_its_own_beat() -> None:
    tempo = two_speeds()
    assert tempo.bpm_at(3.999) == 120.0
    assert tempo.bpm_at(4.0) == 60.0
    assert tempo.seconds_at(4.0) == pytest.approx(2.0)  # the boundary itself keeps its time
    assert tempo.seconds_at(8.0) == pytest.approx(6.0)  # four more beats at half the speed
    assert tempo.seconds_per_beat(4.0) == pytest.approx(1.0)
    assert tempo.seconds_per_beat(3.0) == pytest.approx(0.5)


def test_the_two_conversions_are_inverses_across_a_change() -> None:
    tempo = two_speeds()
    for beat in (0.0, 0.5, 2.0, 4.0, 4.25, 7.0, 12.5):
        assert tempo.beats_at(tempo.seconds_at(beat)) == pytest.approx(beat, abs=1e-9)


def test_negative_times_extend_the_first_tempo_backwards() -> None:
    tempo = two_speeds()
    assert tempo.seconds_at(-3.0) == pytest.approx(-1.5)
    assert tempo.beats_at(-3.0) == pytest.approx(-6.0)


def test_the_map_is_normalised_as_it_is_built() -> None:
    unsorted = TempoMap((TempoPoint(4.0, 60.0), TempoPoint(0.0, 120.0)))
    assert [point.beat for point in unsorted.points] == [0.0, 4.0]
    # a map that never names beat 0 starts from the tempo it does name
    late = TempoMap((TempoPoint(3.0, 90.0),))
    assert late.points == (TempoPoint(0.0, 90.0), TempoPoint(3.0, 90.0))
    # two changes on one beat leave the later one
    doubled = TempoMap((TempoPoint(0.0, 120.0), TempoPoint(0.0, 60.0)))
    assert doubled.points == (TempoPoint(0.0, 60.0),)
    # an empty map is the default tempo, and an absurd one is pulled back into range
    assert TempoMap(()).points == (TempoPoint(0.0, DEFAULT_BPM),)
    assert TempoMap((TempoPoint(0.0, 10_000.0),)).bpm == MAX_BPM
    assert TempoMap((TempoPoint(0.0, 0.0),)).bpm == MIN_BPM


def test_with_change_replaces_a_point_on_the_same_beat() -> None:
    tempo = two_speeds()
    assert tempo.with_change(4.0, 90.0).points == (TempoPoint(0.0, 120.0), TempoPoint(4.0, 90.0))
    added = tempo.with_change(8.0, 150.0)
    assert [point.beat for point in added.points] == [0.0, 4.0, 8.0]
    assert added.bpm_at(8.0) == 150.0
    assert added.seconds_at(4.0) == pytest.approx(2.0)  # adding later does not move earlier beats


def test_without_drops_a_change_but_never_the_first() -> None:
    tempo = two_speeds()
    assert tempo.without(4.0).is_single
    assert tempo.without(0.0) == tempo
    assert tempo.without(99.0) == tempo


def test_segments_walk_the_changes_and_leave_the_last_open() -> None:
    assert list(two_speeds().segments()) == [(0.0, 4.0, 120.0), (4.0, None, 60.0)]
    assert list(TempoMap.single(100.0).segments()) == [(0.0, None, 100.0)]


def test_rescaled_moves_the_beats_and_the_tempos_together() -> None:
    tempo = two_speeds().rescaled(180.0)
    assert [point.bpm for point in tempo.points] == [180.0, 90.0]
    assert [point.beat for point in tempo.points] == [0.0, 6.0]  # the change keeps its time
    assert tempo.seconds_at(6.0) == pytest.approx(two_speeds().seconds_at(4.0))
    assert two_speeds().rescaled(120.0) == two_speeds()


def test_pairs_round_trip_and_junk_is_skipped() -> None:
    tempo = two_speeds()
    assert TempoMap.from_pairs(tempo.to_pairs()) == tempo
    assert TempoMap.from_pairs([[0, 100]]) == TempoMap.single(100.0)
    assert TempoMap.from_pairs([[0], "junk", [2, 3, 4]]) == TempoMap.single(DEFAULT_BPM)
    assert TempoMap.from_pairs(None).is_single
