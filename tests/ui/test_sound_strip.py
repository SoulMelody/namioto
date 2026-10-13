# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from namioto import project
from namioto.channels import Channel
from namioto.karaoke.operations import Drop, Match, Merge, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyrics import text_key
from namioto.ui import theme

from .helpers import sound_mouse, wait_for_lyric_mapping


def test_a_narrow_sound_still_names_itself(window) -> None:
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    window.sound_strip._update_hover((0, 1))
    assert window.sound_strip.toolTip() == "い"
    window.sound_strip._toggle_raw(True)
    window.sound_strip._update_hover((0, 1))
    assert window.sound_strip.toolTip() == "い\nReference duration: 1.000 s"
    window.sound_strip._toggle_raw(False)
    window.view.load_lyrics((), ())
    window.sound_strip.setVisible(False)


def test_the_sound_strip_shows_every_line(window) -> None:
    window.view.load_lyrics(natural_sounds("あい\nうえ"), [[(0.0, 1.0), (1.0, 2.0)], [(3.0, 4.0), (4.0, 5.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    # the second line's blocks show too, not just the first line's
    assert window.sound_strip._sound_at(window.sound_strip._x(3.5)) == (1, 0)
    assert not window.sound_strip.grab().isNull()
    window.view.load_lyrics((), ())
    window.sound_strip.setVisible(False)


def test_the_sound_strip_renders_the_current_line(window) -> None:
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    image = window.sound_strip.grab()
    assert not image.isNull() and image.width() > 0
    window.view.load_lyrics((), ())
    window.sound_strip.setVisible(False)


def test_alignment_overlay_does_not_edit_reference_onsets(window) -> None:
    window.sound_strip._toggle_raw(True)
    times = [[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]]
    window.view.load_lyrics(natural_sounds("あいう"), times)
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    before = window.view.undo_stack.count()

    sound_mouse(window, QEvent.Type.MouseButtonPress, window.sound_strip._x(1.0))
    sound_mouse(window, QEvent.Type.MouseMove, window.sound_strip._x(1.5))
    sound_mouse(window, QEvent.Type.MouseButtonRelease, window.sound_strip._x(1.5))

    assert window.view.lyric_raw[0] == ((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    assert window.view.undo_stack.count() == before
    wait_for_lyric_mapping(window)

    window.sound_strip._toggle_raw(False)


def test_alignment_overlay_does_not_move_the_final_onset(window) -> None:
    window.sound_strip._toggle_raw(True)
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()

    sound_mouse(window, QEvent.Type.MouseButtonPress, window.sound_strip._x(1.0))
    sound_mouse(window, QEvent.Type.MouseMove, window.sound_strip._x(5.0))
    sound_mouse(window, QEvent.Type.MouseButtonRelease, window.sound_strip._x(5.0))

    assert window.view.lyric_raw[0] == ((0.0, 1.0), (1.0, 1.0))
    window.sound_strip._toggle_raw(False)


def test_zero_duration_alignment_evidence_is_read_only(own_window) -> None:
    own_window.sound_strip._toggle_raw(True)
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.load_lyrics(natural_sounds("ーす"), [[(0.0, 1.0), (1.0, 2.0)]], raw=[[(0.0, 1.0), (1.0, 0.0)]])
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()
    strip = own_window.sound_strip
    before = view.undo_stack.count()

    assert strip._boundary_x(0, 1) == pytest.approx(strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.5))
    wait_for_lyric_mapping(own_window)

    assert view.lyric_raw == (((0.0, 1.0), (1.0, 0.0)),)
    assert view.undo_stack.count() == before
    view.undo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_raw == (((0.0, 1.0), (1.0, 0.0)),)
    view.redo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_raw == (((0.0, 1.0), (1.0, 0.0)),)


def test_sound_onsets_stop_at_global_neighbours_across_empty_lines(own_window) -> None:
    view = own_window.view
    raw = [[(1.0, 0.2)], [], [(2.0, 0.0)], [(3.0, 0.1)]]
    view.load_lyrics(natural_sounds("あ\n！？\nい\nう"), (), raw=raw)

    assert view.set_sound_onset(2, 0, 5.0)
    assert view.lyric_raw[2] == ((3.0, 0.0),)
    assert view.set_sound_onset(2, 0, 0.0)
    assert view.lyric_raw[2] == ((1.0, 0.0),)
    assert view.set_sound_onset(0, 0, -1.0)
    assert view.lyric_raw[0] == ((0.0, 0.2),)
    assert view.set_sound_onset(3, 0, 10.0)
    assert view.lyric_raw[3] == ((10.0, 0.1),)


def test_an_onset_drag_cannot_cross_the_next_onset_in_its_line(own_window) -> None:
    view = own_window.view
    view.load_lyrics(natural_sounds("あいう"), (), raw=[[(0.0, 9.0), (1.0, 0.1), (2.0, 0.2)]])
    assert view.set_sound_onset(0, 1, 5.0)
    assert view.lyric_raw == (((0.0, 9.0), (2.0, 0.1), (2.0, 0.2)),)
    assert view.set_sound_onset(0, 1, -5.0)
    assert view.lyric_raw == (((0.0, 9.0), (0.0, 0.1), (2.0, 0.2)),)


def test_only_the_operation_of_the_moved_onset_loses_confirmation(own_window) -> None:
    view = own_window.view
    previous = Match(SoundRef(0, 0), (1,), confirmed=True)
    moved = Match(SoundRef(0, 1), (2,), confirmed=True)
    following = Drop(SoundRef(0, 2), confirmed=True)
    view.load_lyrics(
        natural_sounds("あいう"),
        (),
        raw=[[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]],
        operations=[previous, moved, following],
    )
    assert not view.set_sound_onset(0, 1, 1.0)
    assert view.lyric_operations == (previous, moved, following)
    assert view.set_sound_onset(0, 1, 1.5)
    assert view.lyric_operations == (previous, following)
    assert view.lyric_raw == (((0.0, 1.0), (1.5, 1.0), (2.0, 1.0)),)


def test_the_sound_strip_has_no_line_end_handle(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.load_lyrics(natural_sounds("あ"), [[(0.0, 2.0)]], raw=[[(0.0, 1.0)]])
    strip = own_window.sound_strip
    assert len(strip._layout(0)) == 1
    assert strip._boundary_at(strip._x(1.0)) is None
    assert strip._boundary_at(strip._x(2.0)) is None
    assert not view.set_sound_onset(0, 1, 3.0)


def test_a_lyric_drag_is_smooth_but_magnets_to_the_drawn_grid(window) -> None:
    window.view.set_zoom(48.0, 16.0)
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    strip = window.sound_strip
    step = window.view.grid_step() * window.view.seconds_per_beat
    line = 1.0

    # near a drawn line the time sticks to it; halfway between two lines it stays where it is put
    assert strip._magnet_seconds(line + step * 0.1, strip._x(line)) == pytest.approx(line)
    between = line + step * 0.5
    assert strip._magnet_seconds(between, strip._x(between)) == pytest.approx(between)


def test_hovering_a_shared_note_lights_the_whole_group(window) -> None:
    times = [[(0.0, 1.0), (1.0, 2.0)]]
    window.view.load_lyrics(natural_sounds("あい"), times, group=[[0, 0]])
    assert window.sound_strip._group_run((0, 0)) == (0, 0, 1)
    assert window.sound_strip._group_run((0, 1)) == (0, 0, 1)
    assert window.sound_strip._group_run(None) is None


def test_the_lyric_note_lights_with_the_sounds_that_map_to_it(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so the spans read in seconds
    view = own_window.view
    view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]], mapped=[[(1,), (2,)]])

    view.set_playhead(0.5)  # the highlight follows the playhead, playing or paused
    assert view.highlight_note == 1
    assert view.highlight_sounds == ((0, 0),)

    view.set_playhead(1.5)
    assert view.highlight_note == 2
    assert view.highlight_sounds == ((0, 1),)

    view.set_playhead(2.5)  # past the last note: nothing is sung there
    assert view.highlight_note is None
    assert view.highlight_sounds == ()


def test_a_merge_lights_the_whole_run(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.set_notes(((60, 0.0, 1.0, 0),))
    view.load_lyrics(
        natural_sounds("あい"), [[(0.0, 0.5), (0.5, 1.0)]], raw=[[(0.0, 0.5), (0.5, 0.5)]], mapped=[[(1,), (1,)]]
    )

    view.set_playhead(0.25)
    assert view.highlight_note == 1
    assert view.highlight_sounds == ((0, 0), (0, 1))

    view.playing = False  # a pause keeps the highlight where the playhead rests
    assert view.highlight_note == 1
    assert view.highlight_sounds == ((0, 0), (0, 1))


def test_the_strip_paints_the_lyric_highlight(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.set_notes(((60, 0.0, 1.0, 0),))
    view.load_lyrics(natural_sounds("あ"), [[(0.0, 1.0)]], mapped=[[(1,)]])
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()

    view.set_playhead(0.5)
    QApplication.processEvents()
    image = own_window.sound_strip.grab().toImage()
    pixels = [image.pixelColor(x, y).rgb() for x in range(image.width()) for y in range(image.height())]
    assert QColor(theme.LYRIC_HIGHLIGHT).rgb() in pixels  # the NOTE block takes the blue
    assert QColor(theme.LYRIC_SELECT).rgb() in pixels  # its text takes the group colour, not blue

    view.set_playhead(None)
    own_window.sound_strip.setVisible(False)


def test_the_sound_strip_lays_a_row_out_once_per_refresh(window) -> None:
    window.view.load_lyrics(natural_sounds("あい\nうえ"), [[(0.0, 1.0), (1.0, 2.0)], [(3.0, 4.0), (4.0, 5.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    strip = window.sound_strip
    strip._refresh()
    x = strip._x(3.5)
    strip._boundary_at(x)  # fills the cache

    calls: list[float] = []
    original = strip._x
    strip._x = lambda seconds, origin=None: (calls.append(seconds), original(seconds, origin))[1]
    try:
        strip._boundary_at(x)
        assert not calls  # a second hit test reuses the rows already laid out
        strip._refresh()
        strip._boundary_at(x)
        assert calls  # a refresh lays them out again
    finally:
        strip._x = original
        window.view.load_lyrics((), ())
        window.sound_strip.setVisible(False)


def test_sound_strip_only_measures_visible_labels(own_window, monkeypatch) -> None:
    own_window.show()
    view, strip = own_window.view, own_window.sound_strip
    view.load_lyrics(natural_sounds("あいう"), [[(1.0, 2.0), (3.0, 4.0), (1000.0, 1001.0)]])
    view.center_on(2.0, 60.0)
    strip.show()
    QApplication.processEvents()
    calls = []
    original = strip._fit_label
    monkeypatch.setattr(
        strip, "_fit_label", lambda sound, metrics, room: (calls.append(sound), original(sound, metrics, room))[1]
    )
    strip.grab()
    assert calls == list(view.lyric_lines[0].sounds[:2])
    calls.clear()
    block = strip._blocks()[(0, 0)]
    strip.update(block.toAlignedRect())
    QApplication.processEvents()
    assert calls == [view.lyric_lines[0].sounds[0]]


@pytest.mark.parametrize("modifiers", [Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier])
def test_sound_strip_hover_queries_each_hit_once(own_window, monkeypatch, modifiers) -> None:
    view, strip = own_window.view, own_window.sound_strip
    view.load_lyrics(natural_sounds("あい"), [[(1.0, 2.0), (3.0, 4.0)]])
    strip.show()
    QApplication.processEvents()
    calls = []
    original = strip._sound_at
    monkeypatch.setattr(strip, "_sound_at", lambda x, y=None: (calls.append((x, y)), original(x, y))[1])
    rect = strip._blocks()[(0, 0)]
    sound_mouse(own_window, QEvent.Type.MouseMove, rect.center().x(), rect.center().y(), modifiers)
    assert len(calls) == 1
    assert strip._hover == (0, 0)
    expected = Qt.CursorShape.SizeHorCursor if modifiers else Qt.CursorShape.OpenHandCursor
    assert strip.cursor().shape() == expected


def test_sound_strip_scroll_reuses_layout_and_runs(own_window, monkeypatch) -> None:
    view, strip = own_window.view, own_window.sound_strip
    view.load_lyrics(natural_sounds("あいう"), [[(1.0, 2.0), (None, None), (3.0, 4.0)]])
    strip.show()
    QApplication.processEvents()
    blocks = strip._blocks()
    runs = strip._note_runs()
    lanes = strip._lanes()
    original_x = strip._x(0.0)
    calls = []
    original = strip._x
    monkeypatch.setattr(strip, "_x", lambda seconds, origin=None: (calls.append(seconds), original(seconds, origin))[1])
    bar = view.horizontalScrollBar()
    bar.setValue(bar.value() + 20)
    shifted = strip._blocks()
    assert calls == [0.0]
    delta = original(0.0) - original_x
    assert delta != 0
    assert shifted == {key: rect.translated(delta, 0) for key, rect in blocks.items()}
    assert strip._note_runs() is runs
    assert strip._lanes() is lanes
    calls.clear()
    bar = view.verticalScrollBar()
    bar.setValue(bar.value() + 10)
    assert calls == [0.0]
    assert strip._blocks() is shifted
    view.set_zoom(view.zoom[0] * 2, view.zoom[1])
    assert strip._block_cache is None
    rebuilt = strip._blocks()
    assert rebuilt[(0, 0)].width() > shifted[(0, 0)].width()
    assert strip._note_runs() is runs
    assert strip._lanes() is lanes


def test_drop_labels_use_nearest_mapped_neighbours_across_lines(own_window) -> None:
    view, strip = own_window.view, own_window.sound_strip
    view.load_lyrics(
        natural_sounds("あい\nうえ\nおか"),
        [[(None, None), (2.0, 3.0)], [(None, None), (5.0, 6.0)], [(None, None), (None, None)]],
    )
    blocks = strip._blocks()
    assert blocks[(0, 0)].left() == strip._x(2.0)
    assert blocks[(1, 0)].left() == max(strip._x(3.0), blocks[(0, 0)].right() + 3)
    assert blocks[(2, 0)].left() == max(strip._x(6.0), blocks[(1, 0)].right() + 3)
    assert blocks[(2, 1)].left() >= blocks[(2, 0)].right() + 3


def test_drop_labels_never_land_on_each_other(window) -> None:
    window.view.load_lyrics(natural_sounds("あいう"), [[(0.0, 1.0), (None, None), (None, None)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()

    first = window.sound_strip._boundary_x(0, 1)
    second = window.sound_strip._boundary_x(0, 2)
    assert first is not None and second is not None
    blocks = window.sound_strip._blocks()
    assert blocks[(0, 1)].top() > blocks[(0, 0)].bottom()
    assert blocks[(0, 2)].left() > blocks[(0, 1)].right()


def test_mapped_labels_ignore_raw_onsets(window) -> None:
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]], raw=[[(0.2, 0.1), (0.8, 0.1)]])
    strip = window.sound_strip
    assert strip._boundary_x(0, 1) == pytest.approx(strip._x(1.0))
    assert strip._raw_boundary_at(strip._x(0.8)) == (0, 1)


class _Metrics:
    """A stand-in for QFontMetrics: ten pixels a character, whatever the character."""

    def horizontalAdvance(self, text: str) -> int:
        return len(text) * 10


def test_a_label_is_dropped_when_it_does_not_fit(window) -> None:
    strip = window.sound_strip
    rubied = natural_sounds("世界[せ,かい]")[0].sounds[0]  # label せ(世), reading せ
    plain = natural_sounds("あい")[0].sounds[0]  # label あ

    assert strip._fit_label(rubied, _Metrics(), 50) == "せ(世)"
    assert strip._fit_label(rubied, _Metrics(), 20) == "せ"  # the base in brackets gives way first
    assert strip._fit_label(rubied, _Metrics(), 5) == ""
    assert strip._fit_label(plain, _Metrics(), 5) == ""


def test_without_alignment_lyrics_use_faithful_mapping(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", mode="read")
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 2.0, 1.0, 0)))
    wait_for_lyric_mapping(own_window)

    assert own_window._lyric_error == ""
    assert own_window.view.lyric_times == (((0.0, 1.0), (2.0, 3.0)),)
    assert own_window.view.lyric_editable is True


def test_an_alignment_puts_each_sound_on_the_note_its_time_covers(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", mode="read")
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))  # one note under both sounds
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)  # the merge tiles the one note
    assert own_window.view.lyric_red == ((False, False),)


def test_moving_a_lyric_line_preserves_its_mapping_as_krc(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes(((60, 0.0, 2.0, 0), (62, 4.0, 2.0, 1)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="(あい).1", key=text_key("(あい).1"), raw=(((0.0, 1.0), (1.0, 1.0)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)
    own_window.view.set_active_channel(1)
    assert own_window.view.lyric_lines[0].track == 0
    before = own_window.view.undo_stack.count()
    own_window._set_lyric_channel(0, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "@1{(あい).1}"
    assert own_window.view.lyric_lines[0].track == 1
    assert own_window.view.lyric_times == (((4.0, 5.0), (5.0, 6.0)),)
    assert own_window.view.lyric_operations == (Merge((SoundRef(0, 0), SoundRef(0, 1)), 2, confirmed=True),)
    assert own_window.view.undo_stack.count() == before + 1
    assert own_window.save_project(own_window.project_path)
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "(あい).1"
    assert own_window.view.lyric_lines[0].track == 0
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)
    own_window.view.redo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "@1{(あい).1}"
    assert own_window.load_project(own_window.project_path)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_lines[0].track == 1
    assert own_window.view.lyric_times == (((4.0, 5.0), (5.0, 6.0)),)


def test_loading_lyrics_batches_filtering_and_skips_unchanged_refreshes(own_window, monkeypatch):
    view = own_window.view
    note = view.add_note(60, 0.0, 1.0)
    lines = natural_sounds("あ")
    calls, changes = [], []
    original = view._sync_channel_visuals

    def sync():
        calls.append(True)
        original()

    monkeypatch.setattr(view, "_sync_channel_visuals", sync)
    view.lyrics_changed.connect(lambda: changes.append(True))
    request = {"lines": lines, "times": [[(0.0, 1.0)]], "mapped": [[(note.id,)]], "filtered": (note.id,)}
    view.load_lyrics(**request)
    assert calls == [True]
    assert changes == [True]
    assert note.filtered
    assert "Overlapping note" in note.toolTip()
    view.load_lyrics(**request)
    view.set_filtered_notes((note.id,))
    assert calls == [True]
    assert changes == [True]
    view.load_lyrics(**dict(request, filtered=()))
    assert calls == [True, True]
    assert changes == [True]
    assert not note.filtered
    assert "Overlapping note" not in note.toolTip()

