# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QMenu

from namioto import project
from namioto import settings as store
from namioto.channels import Channel
from namioto.karaoke.operations import Drop, Match, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyrics import text_key
from namioto.ui.align_dialog import AlignDialog
from namioto.ui.settings_dialog import SettingsDialog

from .helpers import _manual_mapping, _mapped_pair, drag_sound, wait_for_lyric_mapping


def _mixed_lyrics(own_window, tmp_path, *, confirmed=True):
    text = "あ\n@1{い}\nう"
    own_window.project_path = tmp_path / "mixed.nto"
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1), Channel(channel=2)))
    own_window.view.set_notes(
        ((60, 0.0, 1.0, 0, 10), (62, 2.0, 1.0, 0, 20), (64, 4.0, 1.0, 1, 30), (65, 6.0, 1.0, 1, 40))
    )
    own_window._stored_lyrics = project.Lyrics(
        text=text,
        key=text_key(text),
        raw=(((0.0, 1.0),), ((4.0, 1.0),), ((2.0, 1.0),)),
        operations=(Match(SoundRef(2, 0), (20,), confirmed=True),) if confirmed else (),
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)


def test_mixed_channels_all_use_faithful_mapping(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    assert own_window.view.lyric_times == (((0.0, 1.0),), ((4.0, 5.0),), ((2.0, 3.0),))
    assert {ref.line for op in own_window.view.lyric_operations for ref in op.sounds} == {0, 1, 2}
    assert Match(SoundRef(2, 0), (20,), confirmed=True) in own_window.view.lyric_operations
    assert not own_window._lyric_error
    assert own_window.export_krc(tmp_path / "mixed.krc")
    assert natural_sounds((tmp_path / "mixed.krc").read_text())[1].track == 1
    assert own_window.export_ass(tmp_path / "mixed.ass")


def test_alignment_adopts_all_channel_rows(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    own_window._adopt_alignment([[(0.1, 0.9)], [(4.1, 0.9)], [(2.1, 0.9)]], "mms", (True, False, False))
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == (((0.1, 0.9),), ((4.1, 0.9),), ((2.1, 0.9),))
    assert own_window._lyric_flags == (True, False, False)
    assert own_window.view.lyric_mapped[1] == ((30, 40),)


def test_align_dialog_receives_all_channel_text(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    own_window.audio_path = Path("/tmp/song.wav")
    seen = []
    monkeypatch.setattr(AlignDialog, "exec", lambda dialog: seen.append(dialog.text))
    own_window._open_align()
    assert seen == ["あ\n@1{い}\nう"]


def test_moving_one_line_preserves_krc_counts_and_restarts_channel_cursors(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    own_window._set_lyric_channel(0, 1)
    wait_for_lyric_mapping(own_window)
    assert [line.track for line in own_window.view.lyric_lines] == [1, 1, 0]
    assert Match(SoundRef(2, 0), (10,), confirmed=True) in own_window.view.lyric_operations
    assert own_window.view.lyric_mapped[:2] == (((30,),), ((40,),))
    assert not own_window._lyric_error
    assert own_window.view.lyric_times[2] == ((0.0, 1.0),)


def test_empty_channel_does_not_block_other_channel_edits(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path, confirmed=False)
    own_window._set_lyric_channel(1, 2)
    wait_for_lyric_mapping(own_window)
    assert "Channel 2" in own_window._lyric_error
    assert own_window.view.lyric_times[1] == ((None, None),)
    own_window._on_lyric_action("drop", 0, 0)
    wait_for_lyric_mapping(own_window)
    assert any(
        isinstance(op, Drop) and op.sound == SoundRef(0, 0) and op.confirmed for op in own_window.view.lyric_operations
    )
    own_window.view.add_note(67, 4.0, 1.0, channel=2)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_mapped[1]
    assert not own_window._lyric_error


def test_nonzero_channel_edits_write_krc_directly(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    count = own_window.view.undo_stack.count()
    own_window._on_lyric_action("drop", 1, 0)
    wait_for_lyric_mapping(own_window)
    assert Drop(SoundRef(1, 0), confirmed=True) in own_window.view.lyric_operations
    assert own_window.lyrics_text == "あ\n@1{い.0}\nう"
    assert own_window.view.undo_stack.count() == count + 1


def test_nonzero_only_lyrics_can_be_aligned(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path, confirmed=False)
    own_window._set_lyric_channel(0, 1)
    wait_for_lyric_mapping(own_window)
    own_window._set_lyric_channel(2, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.edit.align.isEnabled()
    own_window.view.set_active_channel(1)
    own_window.audio_path = Path("/tmp/song.wav")
    seen = []
    monkeypatch.setattr(AlignDialog, "exec", lambda dialog: seen.append(dialog.text))
    own_window._open_align()
    assert seen == [own_window.lyrics_text]


def test_lyric_context_menu_selects_a_channel(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    menus = []

    def choose(menu, *_args):
        submenu = next(action.menu() for action in menu.actions() if action.text() == "Lyric channel")
        menus.append(submenu)
        assert [action.isChecked() for action in submenu.actions()] == [False, True, False]
        assert any(action.text() == "Drop this sound" for action in menu.actions())
        submenu.actions()[2].trigger()

    monkeypatch.setattr(QMenu, "exec", choose)
    strip = own_window.sound_strip
    point = QPoint(round(strip._x(4.5)), round(strip._blocks()[(1, 0)].center().y()))
    strip.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, strip.mapToGlobal(point)))
    wait_for_lyric_mapping(own_window)
    assert menus
    assert own_window.view.lyric_lines[1].track == 2


def test_assigning_one_sound_changes_only_ownership_and_is_one_undo_step(own_window, tmp_path):
    text = "あい"
    own_window.project_path = tmp_path / "voices.nto"
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes(((60, 0.0, 1.0, 0, 10), (62, 1.0, 1.0, 0, 20), (64, 1.0, 1.0, 1, 30)))
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    before = own_window.view.undo_stack.count()
    raw = own_window.view.lyric_raw
    own_window._set_lyric_channel(0, 1, index=1)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あ@1{い}"
    assert [sound.channel for sound in own_window.view.lyric_lines[0].sounds] == [0, 1]
    assert own_window.view.lyric_raw == raw
    assert own_window.view.lyric_mapped[0][1] == (30,)
    assert own_window.view.undo_stack.count() == before + 1
    blocks = own_window.sound_strip._blocks()
    assert blocks[(0, 0)].bottom() < blocks[(0, 1)].top()
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == text
    assert [sound.channel for sound in own_window.view.lyric_lines[0].sounds] == [0, 0]
    assert own_window.view.lyric_mapped == (((10,), (20,)),)


def test_raw_onsets_are_limited_only_by_their_own_channel(own_window):
    view = own_window.view
    view.load_lyrics(natural_sounds("@1{あ}@2{い}@1{う}"), (), raw=[[(0.0, 1.0), (0.5, 1.0), (2.0, 1.0)]])
    assert view.set_sound_onset(0, 0, 1.5)
    assert view.lyric_raw == (((1.5, 1.0), (0.5, 1.0), (2.0, 1.0)),)


def test_channel_alignment_preserves_other_channels_evidence_and_anchors(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    anchor = Match(SoundRef(2, 0), (20,), confirmed=True)
    own_window._adopt_alignment([[(4.1, 0.9)]], "mms", channel=1)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == (((0.0, 1.0),), ((4.1, 0.9),), ((2.0, 1.0),))
    assert anchor in own_window.view.lyric_operations


def test_align_dialog_selects_the_active_channel_and_separates_its_cache(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    own_window.view.set_active_channel(1)
    own_window.audio_path = Path("/tmp/song.wav")
    seen = []
    monkeypatch.setattr(AlignDialog, "exec", lambda dialog: seen.append((dialog.channel, dialog.cache_text)))
    own_window._open_align()
    assert seen == [(1, "# Align channel 1\n" + own_window.lyrics_text)]


def test_project_colour_pickers_validate_and_leave_preferences_untouched(own_window):
    from PySide6.QtWidgets import QDialogButtonBox, QPushButton

    original = store.to_dict(own_window.settings)
    original_midi = own_window.view.channel_color(1)
    dialog = SettingsDialog(own_window.settings, lyric_colors=own_window.view.lyric_colors)
    dialog.lyric_colors_applied.connect(own_window._apply_lyric_colors)
    dialog.color_editors[1].setText("#FF0000")
    dialog.color_editors[2].setText("#00FF00")
    picker = dialog.color_editors[1].parentWidget().findChild(QPushButton)
    assert not picker.icon().isNull()
    assert picker.icon().pixmap(20, 20).toImage().pixelColor(10, 10).name() == "#ff0000"
    dialog.apply()
    assert own_window.view.lyric_color(1).name() == "#ff0000"
    assert own_window.view.lyric_color(2).name() == "#00ff00"
    assert own_window.view.channel_color(1) == original_midi
    assert own_window._ass_settings().channel_colours()[1] == "FF0000"
    dialog.color_editors[1].setText("invalid")
    assert picker.icon().isNull()
    assert not dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).isEnabled()
    dialog._restore_colors()
    assert dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).isEnabled()
    assert store.to_dict(own_window.settings) == original


def test_hand_edits_and_note_changes_never_call_the_solver(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    monkeypatch.setattr("namioto.ui.lyric_map.solve", lambda *args, **kwargs: pytest.fail("must not solve"))
    drag_sound(own_window, (0, 1), own_window.sound_strip._blocks()[(0, 0)].center().x())
    assert own_window.lyrics_text == "(あい).1"
    drag_sound(own_window, (0, 1), own_window.sound_strip._x(2.2))
    assert own_window.lyrics_text == "あい"
    own_window._on_lyric_action("drop", 0, 0)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あ.0い"
    own_window._on_lyric_action("keep", 0, 0)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あい"
    own_window.view.set_notes(((60, 0.0, 0.4, 0, 1), (62, 1.0, 1.0, 0, 2)))
    own_window.view.add_note(64, 0.5, 0.25)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あい"
    assert own_window.view.lyric_times == (((0.0, 0.4), (0.5, 0.75)),)


def test_alignment_flags_never_mark_notes_red_and_are_undoable(own_window, tmp_path):
    _mapped_pair(own_window, tmp_path)
    raw = own_window.view.lyric_raw
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms", (True,))
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_red == ((False, False),)
    assert own_window.lyrics_text == "あい"
    assert own_window.export_krc(tmp_path / "low.krc")
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == raw
    assert own_window._lyric_flags == ()
    assert own_window.view.lyric_red == ((False, False),)
    own_window.view.redo()
    wait_for_lyric_mapping(own_window)
    assert own_window._lyric_flags == (True,)
    assert own_window.view.lyric_red == ((False, False),)
    assert own_window.save_project(own_window.project_path)
    assert own_window.load_project(own_window.project_path)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_red == ((False, False),)


def test_failed_legacy_migration_preserves_the_project_and_blocks_overwriting(own_window, tmp_path):
    path = tmp_path / "broken.nto"
    project.save(
        project.Project(
            notes=(project.Note(0, 1, 60, id=10),),
            lyrics=project.Lyrics(
                text="あ",
                mode="edit",
                operations=(Match(SoundRef(0, 0), (999,), confirmed=True),),
            ),
        ),
        path,
    )
    before = path.read_bytes()
    assert own_window.load_project(path)
    assert own_window._lyric_migration_error
    assert not own_window.save_project(path)
    assert path.read_bytes() == before
    assert own_window._stored_lyrics.operations == (Match(SoundRef(0, 0), (999,), confirmed=True),)


def test_stale_automatic_mapping_does_not_replace_a_text_edit(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    assert workers[0].request["mode"] == "edit"
    assert own_window._on_lyrics_saved("あう")
    workers[0].complete()
    assert own_window.lyrics_text == "あう"
    workers[1].complete()
    assert own_window.lyrics_text == "あう"
    assert own_window.view.lyric_raw == (((None, None), (None, None)),)


def test_channel_alignment_preserves_an_unmapped_other_channel(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    own_window._set_lyric_channel(1, 2)
    wait_for_lyric_mapping(own_window)
    own_window._adopt_alignment([[(0.0, 1.0)], [(2.0, 1.0)]], "mms", channel=0)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あ\n@2{い}\nう"
    assert own_window.view.lyric_mapped == (((10,),), ((),), ((20,),))
    assert "Channel 2" in own_window._lyric_error


def test_align_dialog_exposes_auto_map_and_rebuild_as_distinct_actions(own_window, monkeypatch):
    dialog = AlignDialog("vocal.wav", "あい", 120.0, own_window)
    calls = []
    monkeypatch.setattr(dialog, "_start", calls.append)
    dialog.run.click()
    dialog.rebuild.click()
    assert calls == [True, False]
    assert all(field.name != "use_constraints" for field, _read, _write in dialog._fields)


def test_saving_while_auto_mapping_runs_does_not_cancel_the_result(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms", (True,))
    assert own_window.save_project(own_window.project_path)
    assert own_window._lyric_alignment is not None
    assert not own_window.project_dirty
    workers[0].complete()
    assert own_window.project_dirty
    assert own_window._lyric_flags == (True,)
    assert len(workers) == 2
    workers[1].complete()
    assert own_window.view.lyric_red == ((False, False),)


def test_automatic_solver_runs_outside_the_gui_thread(own_window, tmp_path, monkeypatch):
    from threading import get_ident

    from namioto.ui import lyric_map

    _mapped_pair(own_window, tmp_path)
    main = get_ident()
    called = []
    original = lyric_map.solve

    def record(*args, **kwargs):
        called.append(get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(lyric_map, "solve", record)
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)
    assert called and all(thread != main for thread in called)
    assert own_window.view.lyric_mapped == (((1,), (2,)),)


def test_alignment_of_one_channel_in_a_shared_line_preserves_the_other_krc_slots(own_window, tmp_path):
    text = "@1{あ}@2{い.0}"
    own_window.project_path = tmp_path / "voices.nto"
    own_window.transport.bpm.setValue(60)
    own_window.view.set_channels((Channel(channel=1), Channel(channel=2)))
    own_window.view.set_notes(((60, 0, 1, 1, 10), (62, 1, 1, 1, 20)))
    own_window._stored_lyrics = project.Lyrics(text=text)
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window._adopt_alignment([[(0.0, 2.0)]], "mms", channel=1)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "@1{あ.2}@2{い.0}"
    assert own_window.view.lyric_mapped == (((10, 20), ()),)
    assert own_window.view.lyric_raw == (((0.0, 2.0), (None, None)),)


def test_text_edits_to_note_counts_keep_valid_alignment_evidence(own_window, tmp_path):
    _mapped_pair(own_window, tmp_path)
    raw = own_window.view.lyric_raw
    assert own_window._on_lyrics_saved("あ.2い.0")
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == raw
    assert own_window.view.lyric_mapped == (((1, 2), ()),)
    assert own_window._on_lyrics_saved("あ.2う.0")
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == (((None, None), (None, None)),)
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == raw
