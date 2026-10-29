# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from namioto import project
from namioto.tempo_map import TempoMap, TempoPoint
from namioto.ui.app import MainWindow

from .helpers import fake_loaders


def test_saving_a_project_takes_the_notes_and_the_values_with_it(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(120.0)
    own_window.view.set_notes([(64, 2.0, 1.0), (67, 3.0, 0.5)])  # beats, as the roll holds them
    own_window.mix.gain.set_value(300.0)
    path = tmp_path / "song.nto"
    assert own_window.save_project(path) is True
    assert own_window.project_path == path
    assert own_window.project_dirty is False
    assert own_window._document_name() == "song"
    saved = project.load(path)  # beat 2 at 120 BPM is one second in, and it is seconds that are kept
    assert saved.notes == (project.Note(1.0, 0.5, 64, 0, 1), project.Note(1.5, 0.25, 67, 0, 2))
    assert saved.settings.tempo.bpm == 120.0
    assert saved.settings.spectrum.gain == 300.0
    assert saved.settings.editor.snap == own_window.view.snap
    assert "Saved song.nto" in own_window.statusBar().currentMessage()


def test_loading_a_project_brings_the_notes_and_the_values_back(own_window, tmp_path, monkeypatch) -> None:
    other = project.default_settings()
    other.tempo.bpm = 120.0  # not 60: at 120 BPM a beat and a second differ, so the conversion shows
    other.analysis.a4 = 432.0
    other.spectrum.gain = 300.0
    other.editor.snap = 0.25
    path = tmp_path / "song.nto"
    opened = project.Project(
        settings=other,
        audio="vocal.wav",
        notes=(project.Note(2.0, 0.5, 64), project.Note(3.5, 0.5, 67)),
    )
    (tmp_path / "vocal.wav").write_bytes(b"")
    project.save(opened, path)
    played: list[str] = []  # the analysis and the audio device are not what this checks
    monkeypatch.setattr(type(own_window), "load_audio", lambda _self, path: played.append(path))

    assert own_window.load_project(path) is True
    assert sorted(note.pitch for note in own_window.view.notes()) == [64, 67]
    beats = sorted((note.start, note.duration) for note in own_window.view.notes())
    assert beats[0] == pytest.approx((4.0, 1.0))  # 2 s at 120 BPM is beat 4, and half a second is a beat
    assert beats[1] == pytest.approx((7.0, 1.0))
    assert own_window.view.bpm == 120.0
    assert own_window.project_settings.analysis.a4 == 432.0
    assert own_window.view.gain == 300.0
    assert own_window.view.snap == 0.25
    assert played == [str(tmp_path / "vocal.wav")]
    assert own_window.project_dirty is False
    assert own_window.project_path == path
    assert own_window._document_name() == "song"
    assert "Opened song.nto — 2 notes" in own_window.statusBar().currentMessage()

    own_window.view.set_notes([(60, 0.0, 1.0)])
    assert own_window._document_name() == "song*"


def test_a_project_without_its_audio_still_opens(own_window, tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(
        project.Project(
            settings=project.default_settings(),
            audio="gone.wav",
            notes=(project.Note(1.0, 0.5, 60),),
        ),
        path,
    )
    assert own_window.load_project(path) is True
    assert [note.pitch for note in own_window.view.notes()] == [60]
    assert own_window.audio_path is None
    assert own_window.song.is_loaded is False
    assert "audio not found" in own_window.statusBar().currentMessage()


def test_a_file_that_is_not_a_project_says_so(own_window, tmp_path) -> None:
    path = tmp_path / "not.nto"
    path.write_text("{}")
    assert own_window.load_project(path) is False
    assert "could not be opened" in own_window.statusBar().currentMessage()
    assert own_window.project_path is None


def test_opening_a_project_keeps_its_values_out_of_the_preferences(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.spectrum.gain = 300.0
    other.analysis.a4 = 432.0
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)

    assert own_window.load_project(path) is True
    assert own_window.project_settings.spectrum.gain == 300.0
    assert own_window.mix.gain.value() == 300.0
    assert not (tmp_path / "settings.json").exists()  # opening is not saving

    own_window.settings_store.flush()
    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert "spectrum" not in written  # a document's values never become the program's
    assert "analysis" not in written
    assert "playback" not in written


def test_a_change_made_with_a_project_open_stays_in_the_document(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.spectrum.gain = 300.0
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)
    assert own_window.load_project(path) is True

    own_window.mix.gain.set_value(340.0)  # the document's gain, not the program's
    own_window.settings_store.flush()
    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert own_window.project_settings.spectrum.gain == 340.0
    assert "spectrum" not in written


def test_drawing_marks_the_document_that_has_a_name(own_window) -> None:
    assert own_window.project_dirty is False
    own_window.transport.bpm.setValue(90.0)
    assert own_window.project_dirty is True
    assert own_window._document_name() == "Untitled*"  # nothing to ask about until it has a file


def test_auto_save_is_off_until_it_is_turned_on(own_window, tmp_path) -> None:
    path = tmp_path / "song.nto"
    own_window.project_path = path
    own_window.project_dirty = False

    own_window.view.set_notes([(64, 0.0, 1.0)])

    assert own_window.settings.general.auto_save is False
    assert own_window.project_dirty is True
    assert own_window.autosave_timer.isActive() is False
    assert path.exists() is False


def test_auto_save_writes_the_project_once_editing_stops(own_window, tmp_path) -> None:
    own_window.settings.general.auto_save = True
    path = tmp_path / "song.nto"
    assert own_window.save_project(path) is True

    own_window.view.set_notes([(64, 0.0, 1.0)])
    assert own_window.project_dirty is True
    assert own_window.autosave_timer.interval() == 3000
    assert own_window.autosave_timer.isActive() is True

    own_window.autosave_timer.stop()
    own_window._autosave()

    assert own_window.project_dirty is False
    assert project.load(path).notes == (project.Note(0.0, 0.5, 64, 0, 1),)


def test_losing_focus_writes_the_project_too(own_window, tmp_path, monkeypatch) -> None:
    own_window.settings.general.auto_save = True
    path = tmp_path / "song.nto"
    own_window.save_project(path)
    own_window.view.set_notes([(60, 0.0, 1.0)])
    own_window.autosave_timer.stop()

    monkeypatch.setattr(own_window, "isActiveWindow", lambda: False)
    own_window.changeEvent(QEvent(QEvent.Type.ActivationChange))

    assert own_window.project_dirty is False
    assert project.load(path).notes == (project.Note(0.0, 0.5, 60, 0, 1),)


def test_auto_save_leaves_a_document_without_a_file_alone(own_window) -> None:
    own_window.settings.general.auto_save = True
    own_window.view.set_notes([(60, 0.0, 1.0)])
    assert own_window.project_path is None

    own_window.autosave_timer.stop()
    own_window._autosave()

    assert own_window.project_dirty is True  # a sketch waits for Save As


def test_a_modal_dialog_holds_auto_save_back(own_window, tmp_path, monkeypatch) -> None:
    own_window.settings.general.auto_save = True
    path = tmp_path / "song.nto"
    own_window.save_project(path)
    own_window.view.set_notes([(60, 0.0, 1.0)])
    own_window.autosave_timer.stop()

    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: object()))
    own_window._autosave()

    assert own_window.project_dirty is True
    assert project.load(path).notes == ()  # the prompt, not the timer, is what decides


def test_moving_a_note_counts_as_a_change(own_window) -> None:
    note = own_window.view.add_note(69, 2.0, 2.0)
    own_window.project_dirty = False

    own_window.view.begin_gesture("Move notes")
    note.set_range(3.0, 69)
    own_window.view.commit_gesture()

    assert own_window.project_dirty is True


def test_unsaved_notes_are_asked_about_once(own_window, monkeypatch) -> None:
    asked: list[tuple] = []

    def warning(*args, **_kwargs):
        asked.append(args)
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "warning", warning)
    assert own_window._confirm_discard() is True  # a sketch is not worth interrupting anyone over
    assert asked == []

    own_window.view.set_notes([(64, 0.0, 1.0)])
    own_window.project_path = Path("song.nto")
    assert own_window._confirm_discard() is True
    assert len(asked) == 1
    assert "song.nto" in asked[0][2]

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    assert own_window._confirm_discard() is False
    own_window.project_dirty = False


def test_the_chosen_file_is_left_alone_when_the_notes_are_kept(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_notes([(64, 0.0, 1.0)])
    own_window.project_path = Path("song.nto")
    other = tmp_path / "other.nto"
    project.save(project.Project(settings=project.default_settings()), other)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), ""))

    own_window._on_open()

    assert own_window.project_path == Path("song.nto")  # the prompt was answered no, so nothing opened
    assert [note.pitch for note in own_window.view.notes()] == [64]
    own_window.project_dirty = False


def test_a_chosen_audio_file_is_left_alone_when_the_notes_are_kept(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_notes([(64, 0.0, 1.0)])
    own_window.project_path = Path("song.nto")
    audio = tmp_path / "other.wav"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(audio), ""))

    own_window._on_open()

    assert own_window.project_path == Path("song.nto")  # the prompt was answered no, so nothing opened
    assert own_window.audio_path is None  # and the sound was not analysed behind its back
    own_window.project_dirty = False


def test_saving_as_adds_the_suffix_when_it_is_missing(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_notes([(64, 0.0, 1.0)])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "mysong"), ""))
    assert own_window._on_save() is True
    assert own_window.project_path == tmp_path / "mysong.nto"
    assert (tmp_path / "mysong.nto").exists()


def test_saving_as_suggests_the_name_of_the_audio_file(own_window, monkeypatch, tmp_path) -> None:
    own_window.audio_path = str(tmp_path / "vocal.wav")
    own_window.state.last_audio_dir = str(tmp_path)
    asked: list[str] = []

    def choose(_parent, _caption, suggested, *_filters):
        asked.append(suggested)
        return (str(tmp_path / "vocal.nto"), "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose)
    assert own_window._on_save_as() is True
    assert Path(asked[0]) == tmp_path / "vocal.nto"


def test_a_project_beside_the_audio_is_opened_instead(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    project.save(
        project.Project(
            settings=project.default_settings(),
            audio="song.wav",
            notes=(project.Note(1.0, 0.5, 62),),
        ),
        tmp_path / "song.nto",
    )
    fake_loaders(monkeypatch)  # the project's own audio is analysed, not the file handed in

    assert own_window.open_audio(str(audio)) is True

    assert own_window.project_path == tmp_path / "song.nto"
    assert [note.pitch for note in own_window.view.notes()] == [62]
    assert own_window.audio_path == str(audio)


def test_a_broken_project_beside_the_audio_stops_the_open(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    (tmp_path / "song.nto").write_text("{}")
    captured = fake_loaders(monkeypatch)

    assert own_window.open_audio(str(audio)) is False

    assert own_window.project_path is None
    assert own_window.audio_path is None  # the project is the document, so a broken one opens nothing
    assert not captured  # and the sound is not analysed behind its back


def test_a_project_is_picked_up_from_the_command_line(qt_app, tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(
        project.Project(settings=project.default_settings(), notes=(project.Note(1.0, 0.5, 62),)),
        path,
    )
    window = MainWindow()
    window.open_file(str(path))
    try:
        assert window.project_path == path
        assert [note.pitch for note in window.view.notes()] == [62]
    finally:
        window.project_dirty = False
        window.close()


def test_a_new_project_needs_no_audio(own_window, tmp_path) -> None:
    own_window.view.set_notes([(60, 0.0, 1.0), (64, 2.0, 1.0)])
    own_window.view.set_tempo(
        TempoMap((TempoPoint(0.0, 120.0), TempoPoint(4.0, 60.0))), keep_notes=False
    )
    own_window.project_dirty = False

    assert own_window.new_project() is True

    assert own_window.project_path is None
    assert own_window.audio_path is None
    assert own_window.view.notes() == []
    assert own_window.view.tempo.is_single
    assert own_window.view.spectrum is None
    assert own_window._document_name() == "Untitled"

    # notes can be drawn and the project saved with no recording behind it at all
    own_window.view.add_note(60, 0.0, 1.0)
    path = tmp_path / "empty.nto"
    assert own_window.save_project(path) is True
    saved = project.load(path)
    assert [note.pitch for note in saved.notes] == [60]
    assert saved.audio == ""
