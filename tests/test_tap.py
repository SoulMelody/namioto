# SPDX-License-Identifier: AGPL-3.0-only
"""Tapping: the tempo a run of human taps implies, and the histogram it is checked against."""

from __future__ import annotations

import pytest

from namioto.analysis import tap


def test_four_even_taps_at_half_a_second_are_120_bpm() -> None:
    assert tap.tap_tempo([0.0, 0.5, 1.0, 1.5]) == pytest.approx(120.0)


def test_one_tap_implies_no_tempo_at_all() -> None:
    assert tap.tap_tempo([]) is None
    assert tap.tap_tempo([0.25]) is None
    # two taps on the same instant are one tap as far as a gap goes
    assert tap.tap_tempo([1.0, 1.0]) is None


def test_a_gap_longer_than_a_missed_beat_is_not_a_slow_tempo() -> None:
    # the third tap arrives late enough to be a skipped beat, so it cannot drag the estimate
    assert tap.tap_tempo([0.0, 0.5, 4.0]) == pytest.approx(120.0)


def test_a_shaky_run_still_lands_on_a_sane_tempo() -> None:
    # a stray double-click among even taps is one gap in five, so the median stays near the even taps
    assert tap.tap_tempo([0.0, 0.5, 0.55, 1.0, 1.5]) == pytest.approx(126.0, abs=8.0)
    # the same tempo tapped with human jitter: the median gap is still a second, so 60 BPM
    assert tap.tap_tempo([0.0, 1.0, 1.95, 3.05, 4.0]) == pytest.approx(60.0, abs=3.0)


def test_the_tempo_is_clamped_to_the_range_a_song_can_have() -> None:
    assert tap.tap_tempo([0.0, 0.01]) == tap.MAX_BPM
    # two seconds is the longest gap still read as a beat, and it is exactly the slowest tempo
    assert tap.tap_tempo([0.0, 2.0]) == tap.MIN_BPM


def test_only_the_last_taps_count_when_a_window_is_named() -> None:
    # four slow taps, then three fast ones; the window keeps the recent tempo
    assert tap.tap_tempo([0.0, 1.0, 2.0, 3.0, 3.5, 4.0], window=3) == pytest.approx(120.0)
    assert tap.tap_tempo([0.0, 1.0, 2.0, 3.0], window=0) == pytest.approx(60.0)


def test_the_histogram_counts_gaps_inside_its_range_only() -> None:
    centres, weights = tap.tempo_histogram([0.0, 0.5, 1.0, 1.5])
    assert centres[0] == tap.HISTOGRAM_LOW
    assert len(centres) == len(weights)
    assert sum(weights) > 0.0
    # a gap whose tempo falls outside the range is skipped rather than piled onto an edge bucket
    _, empty_weights = tap.tempo_histogram([0.0, 100.0])
    assert sum(empty_weights) == 0.0
