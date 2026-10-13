# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QProcess, QRectF
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu

from namioto import lyrics, project
from namioto.channels import Channel
from namioto.karaoke.operations import Merge, SoundRef
from namioto.lyrics import text_key
from namioto.ui.align_dialog import AlignDialog
from namioto.ui.lyrics_dialog import LyricsDialog

from .helpers import _mapped_pair, sound_mouse, wait_for_lyric_mapping


def test_the_grid_offset_only_slides_the_drawn_grid(own_window) -> None:
    view = own_window.view
    view.set_channels(())
    view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.transport.bpm.setValue(120.0)
    rect = QRectF(0.0, 0.0, 3.0, 1.0)
    assert [x for x, _level in view.division_lines(rect, 0.0)] == [0.0, 1.0, 2.0, 3.0]

    own_window.transport.grid_offset.setValue(125)  # +0.125 s = a quarter beat at 120 BPM
    assert [x for x, _level in view.division_lines(rect, 0.0)] == [-0.75, 0.25, 1.25, 2.25]
    assert view.notes()[0].start == 0.0  # the note keeps its own time
    assert view._snap_beats(0.6) == pytest.approx(0.75)  # snapping follows the drawn lines
    assert view._snap_floor_beats(0.6) == pytest.approx(0.25)
    assert view._snap_ceil_beats(0.6) == pytest.approx(0.75)


def test_the_grid_offset_leaves_the_time_clock_absolute(own_window) -> None:
    view = own_window.view
    own_window.transport.bpm.setValue(120.0)
    rect = QRectF(0.0, 0.0, 3.0, 1.0)
    assert [x for x, _seconds, _major in view.seconds_lines(rect, 40.0)] == [0.0, 1.0, 2.0, 3.0]

    own_window.transport.grid_offset.setValue(125)  # +0.125 s = a quarter beat at 120 BPM
    assert [x for x, _seconds, _major in view.seconds_lines(rect, 40.0)] == [0.0, 1.0, 2.0, 3.0]
    view.division = "seconds"
    assert [x for x, _level in view.division_lines(rect, 40.0)] == [0.0, 1.0, 2.0, 3.0]


def test_the_grid_offset_leaves_the_playhead_at_its_timestamp(own_window, monkeypatch) -> None:
    own_window.transport.grid_offset.setValue(-300)
    monkeypatch.setattr(own_window, "_position", lambda: 1.0)
    own_window._show_position()
    assert own_window.view.playhead == 1.0


def test_unmapped_lyrics_remain_visible_until_notes_and_alignment_arrive(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window.lyrics_text = "あい\n"
    own_window._load_sounds()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_times == (((None, None), (None, None)),)
    assert not own_window.sound_strip.isHidden()

    own_window.transport.bpm.setValue(60)
    own_window.view.set_notes([(60, 0, 1), (62, 1, 1)])
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)
    assert not own_window.sound_strip.isHidden()


def test_project_lyrics_ignore_existing_sidecars(own_window, tmp_path) -> None:
    path = tmp_path / "song.nto"
    sidecar = path.with_suffix(".krc")
    sidecar.write_text("あう\n", encoding="utf-8")
    project.save(project.Project(lyrics=project.Lyrics(text="あい\n", mode="read")), path)
    assert own_window.load_project(path)
    assert own_window.lyrics_text == "あい\n"
    assert own_window.lyrics_path().parent != tmp_path
    assert own_window.save_project(path)
    assert project.load(path).lyrics.text == "あい\n"
    assert sidecar.read_text(encoding="utf-8") == "あう\n"
    sidecar.unlink()
    assert own_window.load_project(path)
    assert own_window.lyrics_text == "あい\n"
    assert not sidecar.exists()


def test_alignment_writes_krc_without_saving_or_entering_a_mode(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.0, 0.5), (0.5, 0.5)))
    own_window.view.set_notes([(60, 0.0, 1.0, 0, 1)])
    wait_for_lyric_mapping(own_window)
    before = own_window.view.undo_stack.count()
    own_window._adopt_alignment([[(0.0, 0.5), (0.5, 0.5)]], "mms")
    wait_for_lyric_mapping(own_window)
    expected = own_window.view.lyric_mapped
    assert own_window.lyrics_text == "(あい).1"
    assert not own_window.project_path.exists()
    assert own_window.project_dirty
    assert own_window.view.undo_stack.count() == before + 1
    assert own_window.view.lyric_editable
    assert own_window.edit.lyrics.isEnabled()
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あい"
    own_window.view.redo()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_mapped == expected
    assert own_window.save_project(own_window.project_path)
    saved = project.load(own_window.project_path)
    assert saved.lyrics.text == "(あい).1"
    assert saved.lyrics.mode == "read"
    assert saved.lyrics.operations == ()
    assert saved.lyrics.raw == (((0.0, 0.5), (0.5, 0.5)),)
    assert saved.lyrics.key == text_key(saved.lyrics.text)
    assert saved.lyrics.model == "mms"
    assert own_window.load_project(own_window.project_path)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_mapped == expected


@pytest.mark.parametrize("use_constraints,slots", [(True, (2, 1)), (False, (1, 2))])
def test_alignment_mapping_can_use_or_ignore_krc_constraints(own_window, tmp_path, use_constraints, slots):
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0), Channel(channel=5)))
    own_window.view.set_active_channel(5)
    own_window.view.set_notes([(60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0), (64, 2.0, 1.0, 0)])
    assert own_window._on_lyrics_saved("あ.2い")
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 2.0)]], "mms", use_constraints=use_constraints)
    wait_for_lyric_mapping(own_window)
    assert tuple(operation.slots for operation in own_window.view.lyric_operations) == slots
    assert own_window.save_project(own_window.project_path)
    assert own_window.load_project(own_window.project_path)
    wait_for_lyric_mapping(own_window)
    assert not hasattr(own_window, "_lyric_mode")
    assert tuple(operation.slots for operation in own_window.view.lyric_operations) == slots
    expected = own_window.view.lyric_mapped
    assert own_window._prepare_lyrics_copy() is not None
    assert lyrics.load(own_window.lyrics_path()) == own_window.lyrics_text
    assert own_window.load_project(own_window.project_path)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_lines[0].track == 0
    assert own_window.view.lyric_mapped == expected


def test_extra_lyric_sounds_remain_editable_without_mapping_errors(own_window, tmp_path):
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window.view.set_notes([(60, 0.0, 1.0)])
    assert own_window._on_lyrics_saved("あいう")
    wait_for_lyric_mapping(own_window)
    assert not own_window._lyric_error
    assert own_window.view.lyric_editable
    assert own_window.view.lyric_times == (((0.0, own_window.view.to_seconds(1.0)), (None, None), (None, None)),)
    own_window._on_lyric_action("drop", 0, 0)
    assert own_window.lyrics_text == "あ.0いう"
    assert not own_window._lyric_error
    assert own_window.view.lyric_times == (((None, None), (0.0, own_window.view.to_seconds(1.0)), (None, None)),)
    own_window.view.undo_stack.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あいう"
    assert not own_window._lyric_error


def test_incompatible_krc_constraints_leave_the_original_text_until_rebuilt(own_window, tmp_path):
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window.view.set_notes([(60, 0.0, 1.0)])
    assert own_window._on_lyrics_saved("あ.3")
    own_window._adopt_alignment([[(0.0, 1.0)]], "mms", use_constraints=True)
    wait_for_lyric_mapping(own_window)
    assert not own_window._lyric_error
    assert own_window.view.lyric_times == (((0.0, own_window.view.to_seconds(1.0)),),)
    assert own_window.view.lyric_editable
    assert not own_window.sound_strip.isHidden()
    assert own_window.lyrics_text == "あ.3"
    own_window._adopt_alignment([[(0.0, 1.0)]], "mms", use_constraints=False)
    wait_for_lyric_mapping(own_window)
    assert not own_window._lyric_error
    assert own_window.lyrics_text == "あ"


def test_unreadable_temporary_lyrics_do_not_replace_project_text(lyrics_window):
    assert lyrics_window._on_lyrics_saved("あい")
    lyrics_window.lyrics_path().write_bytes(b"\xff")
    lyrics_window._on_lyrics_file_changed()
    assert lyrics_window.lyrics_text == "あい"
    assert "could not be read" in lyrics_window.statusBar().currentMessage().lower()


def test_failed_project_save_keeps_the_current_krc_and_evidence(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    before = own_window.lyrics_text, own_window.view.lyric_raw, own_window.view.lyric_operations

    def fail(*args):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(project, "save", fail)
    assert not own_window.save_project(own_window.project_path)
    assert (own_window.lyrics_text, own_window.view.lyric_raw, own_window.view.lyric_operations) == before
    assert own_window.view.lyric_editable
    assert "read-only filesystem" in own_window.statusBar().currentMessage()


def test_legacy_edit_project_is_migrated_without_overwriting_disk(own_window, tmp_path):
    path = tmp_path / "legacy.nto"
    project.save(
        project.Project(
            notes=(project.Note(0, 1, 60, id=10),),
            lyrics=project.Lyrics(
                text="あい",
                mode="edit",
                key=text_key("あい"),
                raw=(((0, 0.5), (0.5, 0.5)),),
                operations=(Merge((SoundRef(0, 0), SoundRef(0, 1)), 10),),
            ),
        ),
        path,
    )
    before = path.read_bytes()
    assert own_window.load_project(path)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "(あい).1"
    assert own_window.project_dirty
    assert path.read_bytes() == before
    assert own_window.save_project(path)
    saved = project.load(path).lyrics
    assert saved.mode == "read"
    assert saved.operations == ()
    assert saved.text == "(あい).1"


def test_mapping_does_not_block_lyrics_text_entries(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    opened = []
    monkeypatch.setattr(LyricsDialog, "exec", lambda self: opened.append(self))
    monkeypatch.setattr(QProcess, "startDetached", lambda *args: (True, 1))
    own_window._open_lyrics()
    own_window._open_lyrics_editor()
    assert len(opened) == 1
    assert own_window._on_lyrics_saved("あう")
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あう"
    lyrics.save(own_window.lyrics_path(), "あえ")
    own_window._on_lyrics_file_changed()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あえ"
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あう"


def test_temporary_lyrics_copy_is_released_when_switching_documents(lyrics_window, tmp_path):
    old = lyrics_window.lyrics_path()
    lyrics.save(old, "あい")
    next_project = tmp_path / "next.nto"
    project.save(project.Project(lyrics=project.Lyrics(text="あう", mode="read")), next_project)
    assert lyrics_window.load_project(next_project)
    assert not old.parent.exists()
    assert lyrics_window.lyrics_path().parent != old.parent
    assert not next_project.with_suffix(".krc").exists()
    current = lyrics_window.lyrics_path()
    lyrics_window.project_dirty = False
    lyrics_window.close()
    assert not current.parent.exists()


def test_timeline_blank_menu_has_no_mapping_session_actions(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    menus = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *args: menus.append(self))
    point = QPoint(own_window.sound_strip.width() - 1, 1)
    event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, point)
    own_window.sound_strip.contextMenuEvent(event)
    actions = {action.text(): action for action in menus[-1].actions()}
    assert "Save lyrics map" not in actions
    assert "Discard lyrics map" not in actions
    assert "Confirm its operation" not in actions
    own_window.view.lyric_mapping_pending = True
    own_window.sound_strip.contextMenuEvent(event)
    actions = {action.text(): action for action in menus[-1].actions()}
    assert "Save lyrics map" not in actions
    assert "Discard lyrics map" not in actions


@pytest.mark.parametrize("text", ["あ.2", "あ.2い.0"])
def test_faithful_ass_export_uses_krc_spans_without_edit_operations(own_window, tmp_path, text):
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_notes([(60, 0.0, 1.0), (62, 1.0, 1.0)])
    own_window.lyrics_text = text
    own_window._load_sounds()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations[0].slots == 2
    assert own_window.view.lyric_times[0][0] == (0.0, 2.0)
    assert own_window.export_ass(tmp_path / "lyrics.ass")
    assert (tmp_path / "lyrics.ass").exists()


def test_faithful_krc_export_does_not_require_edit_operations(own_window, monkeypatch, tmp_path) -> None:
    own_window.lyrics_text = "あ.2い.0\n"
    own_window._load_sounds()
    wait_for_lyric_mapping(own_window)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "Lyrics file (*.krc)")
    )

    assert own_window._on_export_krc() is True
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "あ.2い.0\n"
    assert own_window.project_path is None


def test_exporting_krc_writes_the_mapping_back(own_window, monkeypatch, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))  # one note under both sounds
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="(あい).1", key=text_key("(あい).1"), raw=(((0.0, 0.5), (0.5, 0.5)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "Lyrics file (*.krc)")
    )

    assert own_window._on_export_krc() is True
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "(あい).1"


def test_exporting_krc_without_lyrics_says_so(own_window, monkeypatch) -> None:
    own_window.lyrics_text = ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: pytest.fail("must not ask"))

    assert own_window._on_export_krc() is False
    assert "no lyrics" in own_window.statusBar().currentMessage().lower()


def test_faithful_mapping_lays_the_krcs_own_dot_n_onto_the_notes(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あ.2\n", key=text_key("あ.2\n"), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert not hasattr(own_window, "_lyric_mode")
    assert own_window.view.lyric_times == (((0.0, 2.0),),)  # one sound held over both notes
    assert own_window.view.lyric_mapped == (((1, 2),),)  # the ids the playback highlight reads
    assert own_window.view.lyric_editable is True
    assert own_window.edit.align.isEnabled() is True


def test_read_mode_draws_a_dot_n_sound_where_it_holds_off(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい.0う\n", key=text_key("あい.0う\n"), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()

    assert own_window.view.lyric_zero == ((False, True, False),)
    blocks = own_window.sound_strip._blocks()
    assert blocks[(0, 1)].top() > blocks[(0, 0)].bottom()
    assert own_window.sound_strip._boundary_x(0, 1) == pytest.approx(own_window.sound_strip._x(1.0))


def test_read_mode_reads_a_ruby_part_by_part(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(tuple((60 + i, float(i), 1.0, 0) for i in range(5)))
    own_window.project_path = tmp_path / "song.nto"
    text = "確信犯[かく,(しん).1,はん]\n"
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    # the `.1` group's two sounds share the third note; か, く, は and ん take one each
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0)),)


def test_a_broken_outside_change_keeps_the_read_only_lyrics(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    text = "あ.0\n"
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), mode="read")
    own_window._watch_lyrics()
    before = own_window.view.lyric_lines
    assert before

    lyrics.save(own_window.lyrics_path(), "あ.0)\n")  # a syntax error in the file beside us
    own_window._on_lyrics_file_changed()

    assert own_window.lyrics_text == text
    assert own_window.view.lyric_lines == before
    assert "could not be read" in own_window.statusBar().currentMessage().lower()


def test_a_broken_read_only_lyrics_file_opens_empty(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    text = "あ.0)\n"  # a syntax error
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), mode="read")
    own_window._watch_lyrics()  # must not take the window down

    assert not own_window.view.lyric_lines


def test_faithful_lyrics_can_start_alignment(own_window, tmp_path, monkeypatch) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window.audio_path = str(tmp_path / "song.wav")
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), mode="read")
    own_window._watch_lyrics()
    opened = []
    monkeypatch.setattr(AlignDialog, "exec", lambda self: opened.append(self) or 0)
    own_window._open_align()
    assert len(opened) == 1
    own_window.view.set_notes([(60, 0, 1), (62, 2, 1)])
    opened[0].aligned.emit([[(0.0, 1.0), (1.0, 1.0)]], "mms", ())
    assert not hasattr(own_window, "_lyric_mode")
    assert own_window.edit.lyrics.isEnabled()


def test_the_lyrics_mode_button_is_removed(lyrics_window) -> None:
    assert not hasattr(lyrics_window.edit, "lyric_lock")
    assert not hasattr(lyrics_window, "_lyric_mode")
    assert lyrics_window.edit.lyrics.isEnabled()
    assert lyrics_window.edit.align.isEnabled()


def test_faithful_project_allows_text_and_mapping_edits_and_alignment(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), mode="read")
    own_window._watch_lyrics()
    assert own_window.edit.lyrics.isEnabled()
    assert own_window.edit.align.isEnabled()
    assert own_window.view.lyric_editable


def test_faithful_strip_dragging_does_not_change_alignment_evidence(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()
    before = own_window.view.lyric_raw

    sound_mouse(own_window, QEvent.Type.MouseButtonPress, own_window.sound_strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, own_window.sound_strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, own_window.sound_strip._x(1.5))

    assert own_window.view.lyric_raw == before


def test_realigning_keeps_the_faithful_editor_available(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.1, 0.2), (0.8, 0.4)))
    own_window._adopt_alignment([[(0.1, 0.2), (0.8, 0.4)]], "mms", use_constraints=True)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_editable
    assert own_window.view.lyric_raw == (((0.1, 0.2), (0.8, 0.4)),)
    assert own_window.edit.lyrics.isEnabled()
