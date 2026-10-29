# SPDX-License-Identifier: AGPL-3.0-only
"""The beat grid's tempo map in the editor: notes keep their seconds, and a change is one step."""

from __future__ import annotations

import pytest

from namioto.tempo_map import TempoMap, TempoPoint

from .helpers import make_spectrum


def note_seconds(view) -> list[float]:
    """Every note's start and duration in seconds, flattened so one comparison covers them all."""
    return [
        value
        for note in view.notes()
        for value in (view.to_seconds(note.start), view.length_to_seconds(note.duration, note.start))
    ]


def two_speeds() -> TempoMap:
    return TempoMap((TempoPoint(0.0, 120.0), TempoPoint(4.0, 60.0)))


def test_a_tempo_change_leaves_the_notes_where_they_sound(own_window) -> None:
    view = own_window.view
    view.clear_notes()
    for pitch, start, duration in ((60, 0.0, 2.0), (64, 4.0, 2.0), (67, 8.0, 1.0)):
        view.add_note(pitch, start, duration)
    before = note_seconds(view)

    view.set_tempo_map(two_speeds(), "Set tempo change")

    assert note_seconds(view) == pytest.approx(before)
    assert view.bpm == 120.0  # the one-tempo control names the map's first point
    assert view.tempo.bpm_at(4.0) == 60.0


def test_a_tempo_change_is_one_undo_step(own_window) -> None:
    view = own_window.view
    view.clear_notes()
    view.add_note(60, 4.0, 2.0)
    view.undo_stack.clear()
    original = view.tempo

    view.set_tempo_map(original.with_change(4.0, 60.0), "Set tempo change")
    assert view.undo_stack.count() == 1
    assert view.tempo.bpm_at(4.0) == 60.0

    view.undo_stack.undo()
    assert view.tempo == original
    view.undo_stack.redo()
    assert view.tempo.bpm_at(4.0) == 60.0


def test_typing_a_tempo_rescales_every_change(own_window) -> None:
    view = own_window.view
    view.clear_notes()
    view.add_note(60, 4.0, 2.0)
    view.set_tempo_map(two_speeds(), "Set tempo change")
    before = note_seconds(view)

    view.bpm = 240.0  # the toolbar control: twice as fast, so every point moves with it

    assert [point.bpm for point in view.tempo.points] == [240.0, 120.0]
    assert note_seconds(view) == pytest.approx(before)


def test_the_spectrum_is_drawn_one_stretch_per_tempo(own_window) -> None:
    view = own_window.view
    view.set_spectrum(make_spectrum(frames=400))
    view.set_tempo_map(two_speeds(), "Set tempo change")

    segments = view._spectrum_segments()
    assert len(segments) == 2
    assert segments[0][0] == 0.0
    assert segments[0][1] == pytest.approx(2.0)  # four beats at 120 BPM
    assert segments[1][0] == pytest.approx(2.0)
    # the change sits exactly on its beat, so the two stretches meet without a gap
    assert view.to_beats(2.0) == pytest.approx(4.0)


def test_a_single_tempo_still_draws_one_stretch(own_window) -> None:
    view = own_window.view
    view.set_spectrum(make_spectrum(frames=400))
    segments = view._spectrum_segments()
    assert len(segments) == 1
    assert segments[0][0] == 0.0
    assert segments[0][1] == pytest.approx(view.to_seconds(view.to_beats(segments[0][1])))


def test_the_ruler_snaps_to_the_bar_line_under_the_pointer(own_window) -> None:
    ruler = own_window.ruler
    left = ruler.origin().x()
    first = ruler.bar_at(left + 2)
    second = ruler.bar_at(left + 2 + 4 * own_window.view.pixels_per_beat())
    assert first % 4 == pytest.approx(0.0, abs=1e-9)
    assert second - first == pytest.approx(4.0)
    assert ruler.bar_at(left + 2) == first  # a few pixels later is still the same bar
