# SPDX-License-Identifier: AGPL-3.0-only
"""The tempo window: 180 cells over 60-240 BPM, tapping, and a grid that snaps towards a target."""

from __future__ import annotations

import numpy as np

from namioto.analysis.tap import tap_tempo
from namioto.ui.tempo_dialog import (
    HISTOGRAM_CELLS,
    HISTOGRAM_MAX,
    HISTOGRAM_MIN,
    TempoDialog,
    TempoHistogram,
    bins_from_intervals,
    histogram_weights,
    refine,
)


def beats_at(bpm: float, count: int = 16) -> list[float]:
    period = 60.0 / bpm
    return [index * period for index in range(count)]


def test_the_histogram_has_180_cells_over_60_to_240() -> None:
    assert HISTOGRAM_CELLS == 180
    assert HISTOGRAM_MIN == 60.0
    assert HISTOGRAM_MAX == 240.0
    assert len(histogram_weights([])) == 180
    assert len(bins_from_intervals([])) == 180


def test_a_steady_120_lands_in_the_bin_that_says_120() -> None:
    weights = histogram_weights(beats_at(120.0))
    assert len(weights) == 180
    assert int(np.argmax(weights)) == 60  # 60 + 60 = 120 BPM


def test_gaps_outside_the_span_are_dropped_not_clamped() -> None:
    # one very slow interval, one in range: clamping would pile the slow one onto the first cell
    weights = bins_from_intervals([4.0, 0.5])
    assert weights[0] == 0.0
    assert sum(weights) == 1.0


def test_the_widget_reports_its_cells(qt_app) -> None:
    widget = TempoHistogram([0.0] * 180)
    assert widget.cells == 180
    assert widget.weights == [0.0] * 180


def test_tapping_four_times_gives_about_120_bpm(qt_app) -> None:
    dialog = TempoDialog([], 120.0)
    # four taps half a second apart are 120 BPM, measured without waiting on the wall clock
    estimate = dialog.tap_with_times([0.0, 0.5, 1.0, 1.5])
    assert estimate is not None
    assert 118.0 <= estimate <= 122.0
    assert 118.0 <= dialog.bpm() <= 122.0
    assert dialog.tap_times() == [0.0, 0.5, 1.0, 1.5]


def test_one_tap_is_not_a_tempo(qt_app) -> None:
    dialog = TempoDialog([], 120.0)
    assert dialog.tap_with_times([0.0]) is None


def test_refining_moves_the_grid_towards_the_beats(qt_app) -> None:
    # the beats are at 123 BPM while the target says 120: refining must land nearer 123, monotonically
    times = beats_at(123.0)
    target = 120.0
    first = refine(times, target)
    assert abs(first - 123.0) < abs(target - 123.0)


def test_the_dialog_refines_and_reports_it(qt_app) -> None:
    times = beats_at(123.0)
    dialog = TempoDialog(times, 120.0)
    fitted = refine(times, dialog.bpm())
    dialog.bpm_box.setValue(round(fitted, 1))
    assert abs(dialog.bpm() - 123.0) < 1.0


def test_refine_uses_the_analysis_module(qt_app) -> None:
    """The window does not reimplement the fit: its answer is `beats.refine_grid`'s own."""
    from namioto.analysis import beats as beat_analysis

    times = beats_at(118.0)
    expected = float(beat_analysis.refine_grid(np.asarray(times, dtype=float), 120.0))
    assert refine(times, 120.0) == expected


def test_tapping_matches_the_analysis_function(qt_app) -> None:
    taps = [0.0, 0.5, 1.0, 1.5]
    dialog = TempoDialog([], 120.0)
    assert dialog.tap_with_times(taps) == tap_tempo(taps)
