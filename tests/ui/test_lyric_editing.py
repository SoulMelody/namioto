# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from namioto import project
from namioto.channels import Channel
from namioto.karaoke.operations import Drop, Match, Merge, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyrics import text_key
from namioto.ui import theme

from .helpers import _mapped_pair, drag_sound, sound_mouse, wait_for_lyric_mapping


def test_note_lyrics_merge_same_ruby_base(own_window) -> None:
    view = own_window.view
    view.set_notes(((60, 0, 1, 0),))
    note = view.notes()[0]
    view.load_lyrics(natural_sounds("今日[きょう]"), [[(0.0, 0.5), (0.5, 1.0)]], mapped=[[(note.id,), (note.id,)]])
    assert note.lyric_text == "きょう(今日)"
    assert note.lyric_runs == (("きょう(今日)", False),)
    assert len(view.lyric_lines[0].sounds) == 2


def test_note_lyrics_append_zero_to_previous_note(own_window) -> None:
    view = own_window.view
    view.set_notes(((60, 0, 1, 0), (62, 1, 1, 0)))
    first, second = view.notes()
    view.load_lyrics(
        natural_sounds("あいうえ"),
        [[(0.0, 1.0), (None, None), (None, None), (1.0, 2.0)]],
        zero=[[False, True, True, False]],
        mapped=[[(first.id,), (), (), (second.id,)]],
    )
    assert first.lyric_text == "あ い う"
    assert first.lyric_runs == (("あ", False), (" い", True), (" う", True))
    assert second.lyric_text == "え"
    assert view.lyric_mapped[0][1:3] == ((), ())


def test_note_lyrics_leading_zero_uses_first_mapped_note(own_window) -> None:
    view = own_window.view
    view.set_notes(((60, 0, 1, 0),))
    note = view.notes()[0]
    view.load_lyrics(
        natural_sounds("あい"),
        [[(None, None), (0.0, 1.0)]],
        zero=[[True, False]],
        mapped=[[(), (note.id,)]],
    )
    assert note.lyric_runs == (("あ", True), (" い", False))


def test_note_lyrics_toggle_preserves_mapping(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view = own_window.view
    labels = [note.lyric_text for note in view.notes()]
    operations = view.lyric_operations
    before = view.undo_stack.count()
    assert own_window.transport.note_lyrics.isChecked()
    assert any(labels)
    own_window.transport.note_lyrics.click()
    assert all(not note.lyric_text for note in view.notes())
    assert [note.toolTip() for note in view.notes()] == labels
    view._sync_channel_visuals()
    assert all(not note.lyric_text for note in view.notes())
    own_window.transport.note_lyrics.click()
    assert [note.lyric_text for note in view.notes()] == labels
    assert view.lyric_operations == operations
    assert view.undo_stack.count() == before


def test_note_lyric_font_is_bold_and_tracks_vertical_zoom(own_window, monkeypatch) -> None:
    own_window.view.set_notes(((60, 0, 1, 0),))
    note = own_window.view.notes()[0]
    note.lyric_text = "あ"
    fonts = []
    original = QPainter.setFont

    def capture_font(painter, font):
        fonts.append(QFont(font))
        original(painter, font)

    monkeypatch.setattr(QPainter, "setFont", capture_font)
    for zoom_x, zoom_y in ((100, 16), (100, 60), (100, 120), (200, 120)):
        image = QImage(400, 200, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.scale(zoom_x, zoom_y)
        note.paint(painter, None)
        painter.end()

    assert all(font.bold() for font in fonts)
    assert [font.pixelSize() for font in fonts] == [9, 31, 62, 62]
    assert fonts[0].pixelSize() < fonts[1].pixelSize() < fonts[2].pixelSize()
    assert fonts[2].pixelSize() > 14
    assert fonts[3].pixelSize() == fonts[2].pixelSize()


def test_boundary_drag_changes_mapping_without_editing_raw(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    raw = view.lyric_raw
    ids = tuple(note.id for note in view.notes())
    before = view.undo_stack.count()
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.8))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.8))
    wait_for_lyric_mapping(own_window)
    assert view.lyric_operations == (Match(SoundRef(0, 0), ids, True), Drop(SoundRef(0, 1), True))
    assert view.lyric_raw == raw
    assert view.undo_stack.count() == before + 1
    assert [note.lyric_text for note in view.notes()] == ["あ", "↳ あ い"]
    assert view.notes()[1].lyric_runs == (("↳ あ", False), (" い", True))
    view.undo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_mapped == (((ids[0],), (ids[1],)),)
    view.redo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_zero == ((False, True),)
    operations = view.lyric_operations
    assert own_window.save_project(own_window.project_path)
    saved = project.load(own_window.project_path).lyrics
    assert saved.raw == raw
    assert saved.operations == ()
    assert saved.text == own_window.lyrics_text
    assert own_window.load_project(own_window.project_path)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == operations


def test_lyric_mapping_preview_is_cancelled_by_escape(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    baseline = view.lyric_operations
    before = view.undo_stack.count()
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.8))
    QTest.keyClick(strip, Qt.Key.Key_Escape)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_operations == baseline
    assert view.undo_stack.count() == before
    assert not view.gesture_active


def test_lyric_drop_layer_drag_joins_the_target_without_displacing_it(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    view.set_notes(((60, 0.0, 2.0, 0),))
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)
    raw = view.lyric_raw
    second = strip._blocks()[(0, 1)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, second.center().x(), second.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, second.center().x(), 55)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, second.center().x(), 55)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_zero == ((False, True),)
    dropped = strip._blocks()[(0, 1)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, dropped.center().x(), dropped.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.0), 15)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.0), 15)
    wait_for_lyric_mapping(own_window)
    assert isinstance(view.lyric_operations[0], Merge)
    assert view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)
    assert view.lyric_raw == raw
    assert view.notes()[0].lyric_text == "あ い"
    view.undo()
    wait_for_lyric_mapping(own_window)
    dropped = strip._blocks()[(0, 1)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, dropped.center().x(), dropped.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(0.3), 15)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(0.3), 15)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "(あい).1"
    assert view.lyric_zero == ((False, False),)
    assert view.lyric_raw == raw


def test_dropping_a_sound_preserves_other_krc_counts_not_note_identity(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    before = view.undo_stack.count()
    first = strip._blocks()[(0, 0)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, first.center().x(), first.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, first.center().x(), 35)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, first.center().x(), 35)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あ.0い"
    assert view.lyric_mapped == (((), (1,)),)
    assert view.undo_stack.count() == before + 1
    assert not view.gesture_active


def test_middle_group_member_cannot_be_dropped_from_the_menu(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60)
    own_window.view.set_notes(((60, 0, 3, 0),))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="(あいう).1", key=text_key("(あいう).1"), raw=(((0.0, 1.0), (1.0, 1.0), (2.0, 1.0)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    baseline = own_window.view.lyric_operations
    assert isinstance(baseline[0], Merge)
    before = own_window.view.undo_stack.count()
    own_window._on_lyric_action("drop", 0, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == baseline
    assert own_window.view.undo_stack.count() == before


def test_advanced_raw_drag_escape_restores_evidence(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    raw, operations = view.lyric_raw, view.lyric_operations
    before = view.undo_stack.count()
    strip._toggle_raw(True)
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.3))
    QTest.keyClick(strip, Qt.Key.Key_Escape)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_raw == raw
    assert view.lyric_operations == operations
    assert view.undo_stack.count() == before


def test_dropping_a_sound_marks_it_zero_and_keeps_the_rest(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    own_window.sound_strip.lyric_action_requested.emit("drop", 0, 1)
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_zero == ((False, True),)
    assert any(
        isinstance(operation, Drop) and operation.confirmed and operation.sound.index == 1
        for operation in own_window.view.lyric_operations
    )
    assert own_window.project_dirty is True


def test_a_lyric_edit_is_one_undo_step(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    own_window.sound_strip.lyric_action_requested.emit("drop", 0, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_zero == ((False, True),)

    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_zero == ((False, False),)
    assert own_window.lyrics_text == "あい"
    assert own_window.view.lyric_mapped == (((1,), (2,)),)


def test_a_cross_container_merge_is_green_and_exports_a_continuation(own_window, tmp_path) -> None:
    from namioto import project

    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="泣[な]い.+", key=text_key("泣[な]い.+"), raw=(((0.0, 0.5), (0.5, 0.5)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == (Merge((SoundRef(0, 0), SoundRef(0, 1)), 1, confirmed=True),)
    assert own_window.view.lyric_red == ((False, False),)
    assert own_window.export_krc(tmp_path / "exported.krc")
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "泣[な]い.+"


def test_undo_alignment_restores_literal_krc_continuations(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="泣[な]い.+")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms", use_constraints=False)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_mapped == (((1,), (2,)),)
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "泣[な]い.+"
    assert own_window.view.lyric_times == (((0.0, 0.5), (0.5, 1.0)),)
    assert own_window.view.lyric_mapped == (((1,), (1,)),)
    assert own_window.view.lyric_raw == (((None, None), (None, None)),)


def test_merging_two_sounds_confirms_the_pair(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.0, 0.5), (0.5, 0.5)))
    drag_sound(own_window, (0, 1), own_window.sound_strip._blocks()[(0, 0)].center().x())

    assert any(
        isinstance(operation, Merge) and operation.confirmed and len(operation.sounds) == 2
        for operation in own_window.view.lyric_operations
    )
    assert own_window.view.lyric_times == (((0.0, 0.5), (0.5, 1.0)),)


def test_a_group_edge_member_can_be_dragged_out_again(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.0, 0.5), (0.5, 0.5)))
    drag_sound(own_window, (0, 1), own_window.sound_strip._blocks()[(0, 0)].center().x())
    assert own_window.lyrics_text == "(あい).1"
    drag_sound(own_window, (0, 1), own_window.sound_strip._x(2.2))

    assert all(isinstance(operation, Match) and operation.slots == 1 for operation in own_window.view.lyric_operations)
    assert own_window.lyrics_text == "あい"


def test_a_grouped_run_is_marked_on_the_strip(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="あい\n", key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),)
    )
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))
    own_window._watch_lyrics()
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_group == ((0, 0),)


def test_shared_sound_blocks_have_a_thin_internal_separator(window) -> None:
    text = "あい"
    window.transport.bpm.setValue(60.0)
    window.view.load_lyrics(
        natural_sounds(text),
        [[(0.0, 2.0), (1.0, 2.0)]],
        raw=[[(0.0, 1.5), (1.5, 0.5)]],
        zero=[[False, False]],
        group=[[1, 1]],
    )
    window.view.set_zoom(300.0, 16.0)
    window.view.horizontalScrollBar().setValue(0)
    window.view.verticalScrollBar().setValue(0)
    window.sound_strip.setVisible(True)
    QApplication.processEvents()

    strip = window.sound_strip
    strip._hover = None  # a live pointer may have tinted the group; the plain body is what it tested
    body = theme.lyric_shades(True)[0]
    seam = round(strip._x(1.0))  # い's left edge, inside あ's block
    image = strip.grab().toImage()
    mid = strip.height() // 2
    colors = [image.pixelColor(seam + dx, mid).name() for dx in (-1, 0, 1)]
    assert colors == [body.name(), QColor(theme.LYRIC_ZERO).name(), body.name()]


def test_a_sound_that_covers_no_note_is_marked_grey(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あか\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="あか\n", key=text_key("あか\n"), model="mms", raw=(((0.0, 1.0), (3.0, 0.5)),)
    )
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_zero == ((False, True),)


@pytest.mark.parametrize("model", ["mms", "yohane"])
def test_pattern_weights_apply_to_both_aligners_without_changing_evidence(own_window, tmp_path, model) -> None:
    text = "泣[な]い"
    raw = (((0.0, 0.74), (0.74, 0.26)),)
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), model=model, raw=raw)
    own_window._watch_lyrics()
    own_window._adopt_alignment(raw, model)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == (Merge((SoundRef(0, 0), SoundRef(0, 1)), 1, confirmed=True),)
    assert own_window.view.lyric_raw == raw
    assert own_window.export_krc(tmp_path / "exported.krc")
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "泣[な]い.+"


@pytest.mark.parametrize("raw_count", [1, 3])
def test_loading_lyrics_discards_evidence_with_a_different_sound_count(own_window, tmp_path, raw_count) -> None:
    text = "S.2S.2"
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(tuple((60, float(at), 1.0, 0) for at in range(4)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text=text,
        key=text_key(text),
        model="mms",
        raw=(tuple((float(at), 1.0) for at in range(raw_count)),),
        scores=(tuple(0.9 for _ in range(raw_count)),),
        flagged=(True,),
    )

    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window.lyrics_text == text
    assert own_window.view.lyric_raw == (((None, None), (None, None)),)
    assert own_window.view.lyric_times == (((0.0, 2.0), (2.0, 4.0)),)
    assert own_window._lyric_model == ""
    assert own_window._lyric_scores is None
    assert own_window._lyric_flags == ()
    assert own_window._lyric_error == ""


def test_short_sounds_on_one_note_fall_to_zero(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("きにく\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="きにく\n", key=text_key("きにく\n"), model="mms", raw=(((0.0, 0.9), (0.9, 0.05), (0.95, 0.05)),)
    )
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))  # one note under three sounds: the first keeps it
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_raw == (((0.0, 0.9), (0.9, 0.05), (0.95, 0.05)),)
    assert own_window.view.lyric_zero == ((False, True, True),)
    assert own_window.view.lyric_group == ((-1, -1, -1),)


def test_a_block_is_the_note_the_sound_maps_to(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("タに\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="タに\n", key=text_key("タに\n"), model="mms", raw=(((0.0, 0.8), (0.8, 0.5)),)
    )
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 0.4, 0)))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    # タ sits on note 0 and に on note 1: a block is the whole note a sound maps to, not its raw span
    assert own_window.view.lyric_zero == ((False, False),)
    assert own_window.view.lyric_times[0][0] == (0.0, 1.0)
    # に's raw span (0.8, 1.3) crosses the boundary, but its block is note 1 whole
    assert own_window.view.lyric_times[0][1] == (1.0, 1.4)
    assert own_window.view.lyric_raw[0][1][0] == 0.8


def test_a_lyric_drag_is_kept_in_the_project(own_window, tmp_path) -> None:
    project_path = tmp_path / "song.nto"
    own_window.project_path = project_path
    own_window._stored_lyrics = project.Lyrics(
        text="あい\n", key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),)
    )
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))
    own_window._watch_lyrics()
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()

    drag_sound(own_window, (0, 1), own_window.sound_strip._x(1.5), 55)

    assert own_window.project_dirty is True
    assert own_window.save_project(project_path) is True
    assert project.load(project_path).lyrics.text == "あい.0"
    assert project.load(project_path).lyrics.raw == (((0.0, 1.0), (1.0, 1.0)),)


def test_notes_on_another_channel_are_not_mapped(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", mode="read")
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=1),))
    own_window.view.set_notes(((60, 0.0, 1.0, 1), (62, 1.0, 1.0, 1)))

    assert own_window.view.lyric_times == (((None, None), (None, None)),)
