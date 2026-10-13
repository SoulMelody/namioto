# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent

from namioto import project
from namioto.karaoke.operations import Match, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyrics import text_key

from .helpers import _manual_mapping, _mapped_pair, sound_mouse, wait_for_lyric_mapping


def test_faithful_reading_runs_in_the_background_and_hand_edits_are_local(own_window, tmp_path, monkeypatch):
    from threading import get_ident

    from namioto.ui import lyric_map

    main = get_ident()
    called = []
    original = lyric_map.map_lyrics

    def record(*args, **request):
        called.append(get_ident())
        return original(*args, **request)

    monkeypatch.setattr(lyric_map, "map_lyrics", record)
    _mapped_pair(own_window, tmp_path)
    assert called
    assert all(thread != main for thread in called)
    own_window._on_lyric_action("drop", 0, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_zero == ((False, True),)
    assert own_window.lyrics_text == "あい.0"


def test_same_lyric_channel_is_a_noop(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    before = own_window.view.undo_stack.count()
    own_window._set_lyric_channel(0, 0)
    assert workers == []
    assert own_window.view.undo_stack.count() == before
    assert not hasattr(own_window.edit, "map_channel")


def test_saving_during_faithful_reading_does_not_request_a_solve(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window.view.begin_gesture("Move lyrics")
    own_window.view.set_sound_onset(0, 1, 1.1)
    own_window.view.commit_gesture()
    revision = own_window._lyric_map_revision
    assert own_window.save_project(tmp_path / "saved.nto")
    assert own_window._lyric_map_revision == revision
    assert own_window._lyric_map_pending is None
    assert workers[0].request["mode"] == "read"
    assert "previous" not in workers[0].request
    assert project.load(tmp_path / "saved.nto").lyrics.raw == (((0.0, 1.0), (1.1, 1.0)),)
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending


def test_loading_same_lyrics_replaces_evidence_and_only_starts_one_full_mapping(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    raw = (((0.1, 1.0), (1.2, 1.0)),)
    path = tmp_path / "other.nto"
    project.save(
        project.Project(
            notes=(project.Note(0.0, 1.0, 60, id=41), project.Note(1.0, 1.0, 62, id=99)),
            lyrics=project.Lyrics(text="あい", key=text_key("あい"), raw=raw),
        ),
        path,
    )
    workers = _manual_mapping(monkeypatch)
    assert own_window.load_project(path)
    assert len(workers) == 1
    assert workers[0].request["mode"] == "read"
    assert "previous" not in workers[0].request
    assert own_window.view.lyric_raw == raw
    assert [note.id for note in own_window.view.notes()] == [41, 99]
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending
    assert not own_window._lyric_error


def test_a_mapping_result_cannot_overwrite_a_drag_in_progress(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    view = own_window.view
    original = view.lyric_operations
    before = view.undo_stack.count()
    own_window._remap_lyrics_async()
    view.begin_gesture("Move lyrics")
    view.set_sound_onset(0, 1, 1.1)
    workers[0].complete()
    assert view.lyric_raw == (((0.0, 1.0), (1.1, 1.0)),)
    assert view.lyric_operations == original[:1]
    assert view.lyric_mapping_pending
    assert len(workers) == 1
    view.commit_gesture()
    assert len(workers) == 2
    assert workers[1].request["raw"] == [[(0.0, 1.0), (1.1, 1.0)]]
    workers[1].complete()
    assert not view.lyric_mapping_pending
    assert view.undo_stack.count() == before + 1


def test_continuous_edits_only_queue_the_latest_mapping(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._remap_lyrics_async()
    for onset in (1.1, 1.2, 1.3):
        own_window.view.begin_gesture("Move lyrics")
        own_window.view.set_sound_onset(0, 1, onset)
        own_window.view.commit_gesture()
    assert len(workers) == 1
    assert own_window._lyric_map_pending[1]["raw"] == [[(0.0, 1.0), (1.3, 1.0)]]
    workers[0].failed.emit("obsolete failure")
    assert own_window.view.lyric_mapping_pending
    assert not own_window._lyric_error
    workers[0].complete()
    assert len(workers) == 2
    assert own_window.view.lyric_mapping_pending
    assert workers[1].request["raw"] == [[(0.0, 1.0), (1.3, 1.0)]]
    workers[1].complete()
    assert own_window.view.lyric_raw == (((0.0, 1.0), (1.3, 1.0)),)
    assert not own_window.view.lyric_mapping_pending


def test_pending_mapping_blocks_lyric_actions_and_exports_but_not_saving_or_midi(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    before = own_window.view.undo_stack.count()
    operations = own_window.view.lyric_operations
    own_window._remap_lyrics()
    assert own_window.view.lyric_mapping_pending
    assert not own_window.transport.export_krc_action.isEnabled()
    assert not own_window.transport.export_ass_action.isEnabled()
    assert own_window.transport.export_midi_action.isEnabled()
    assert not own_window.export_krc(tmp_path / "pending.krc")
    assert not own_window.export_ass(tmp_path / "pending.ass")
    assert not (tmp_path / "pending.krc").exists()
    assert not (tmp_path / "pending.ass").exists()
    assert own_window._verification().counts() == {"mapping_pending": 1}
    own_window._on_lyric_action("drop", 0, 1)
    assert own_window.view.lyric_operations == operations
    assert own_window.view.undo_stack.count() == before
    assert own_window.save_project(tmp_path / "pending.nto")
    assert own_window.export_midi(tmp_path / "pending.mid")
    workers[0].complete()
    assert own_window.transport.export_krc_action.isEnabled()
    assert own_window.transport.export_ass_action.isEnabled()
    assert own_window.export_krc(tmp_path / "ready.krc")
    assert own_window.view.undo_stack.count() == before


@pytest.mark.parametrize("finish", ["commit_gesture", "cancel_gesture"])
def test_an_unchanged_gesture_does_not_leave_mapping_pending(own_window, tmp_path, monkeypatch, finish):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._remap_lyrics_async()
    own_window.view.begin_gesture("Move lyrics")
    workers[0].complete()
    getattr(own_window.view, finish)()
    assert len(workers) == 2
    workers[1].complete()
    assert not own_window.view.lyric_mapping_pending


def test_pending_faithful_mapping_disables_gestures(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._remap_lyrics()
    assert own_window.view.lyric_mapping_pending
    assert own_window.view.lyric_editable
    before = own_window.view.lyric_raw
    strip = own_window.sound_strip
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.5))
    assert own_window.view.lyric_raw == before
    assert not own_window.view.gesture_active
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending


def test_changed_conflict_filtering_invalidates_the_cached_mapping(monkeypatch):
    from namioto.ui import lyric_map

    text = "\n".join(["あ"] * 5)
    request = {
        "text": text,
        "lines": natural_sounds(text),
        "raw": [[(row * 4.0, 1.0)] for row in range(5)],
        "notes": [(row * 4.0, row * 4.0 + 1.0, 60, row + 1) for row in range(5)],
        "mode": "edit",
    }
    previous = lyric_map.map_lyrics(**request)
    assert previous.state is not None
    cached = []
    original = lyric_map.solve

    def record(lines, raw, notes, anchors, state, constraints):
        cached.append(state)
        return original(lines, raw, notes, anchors, state, constraints)

    monkeypatch.setattr(lyric_map, "solve", record)
    request["notes"].append((0.0, 1.0, 62, 99))
    result = lyric_map.map_lyrics(**request, previous=previous)
    assert cached == [None]
    assert [note.id for note in result.filtered] == [99]
    assert result.operations == previous.operations
    assert not result.error


def test_time_edit_undo_and_redo_preserve_sparse_note_ids_and_anchors(own_window, tmp_path):
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_notes(((60, 0.0, 1.0, 0, 41), (62, 1.0, 1.0, 0, 99)))
    own_window.project_path = tmp_path / "song.nto"
    anchor = Match(SoundRef(0, 0), (41,), confirmed=True)
    own_window._stored_lyrics = project.Lyrics(
        text="あい", key=text_key("あい"), raw=(((0.0, 1.0), (1.0, 1.0)),), operations=(anchor,)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    view = own_window.view
    view.begin_gesture("Move lyrics")
    view.set_sound_onset(0, 1, 1.1)
    view.commit_gesture()
    wait_for_lyric_mapping(own_window)
    for restore, onset in ((view.undo, 1.0), (view.redo, 1.1)):
        restore()
        wait_for_lyric_mapping(own_window)
        assert [note.id for note in view.notes()] == [41, 99]
        assert view.lyric_raw[0][1][0] == onset
        assert anchor in view.lyric_operations
        assert not own_window._lyric_error
