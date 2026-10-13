# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QFileDialog

from namioto import lyrics, project
from namioto.analysis import align, devices
from namioto.channels import Channel
from namioto.karaoke.sounds import natural_sounds
from namioto.lyrics import text_key
from namioto.ui.align_dialog import AlignDialog, Aligner

from .helpers import parameter_writer, wait_for_lyric_mapping


def test_the_aligned_times_are_kept_in_the_project(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window._on_lyrics_saved("あい\n")
    own_window.transport.bpm.setValue(60)
    own_window.view.set_notes([(60, 0, 1), (62, 1, 1)])
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)

    assert own_window.save_project(tmp_path / "song.nto") is True
    saved = project.load(tmp_path / "song.nto")
    assert saved.lyrics == project.Lyrics(
        text="あい", key=text_key("あい"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),), flagged=(False,)
    )
    assert own_window.project_dirty is False


def test_the_align_dialog_hands_the_times_over(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append((times, model)))
    dialog._done(([[(0.0, 1.0), (1.0, 2.0)]], "mms", [], []))
    assert got == [([[(0.0, 1.0), (1.0, 1.0)]], "mms")]
    assert dialog.run.isEnabled()


def test_a_missing_runtime_stops_the_alignment(own_window, monkeypatch) -> None:
    monkeypatch.setattr(devices, "validate", lambda: devices.MISSING_RUNTIME)
    monkeypatch.setattr(Aligner, "start", lambda self: pytest.fail("must not run"))
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)

    dialog._start()

    assert devices.MISSING_RUNTIME in dialog.log.toPlainText()
    assert dialog.run.isEnabled()


def test_the_align_dialog_snaps_to_the_beat_grid_when_asked(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "quantize")(1)  # 1/4 notes, one beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.0, 0.5), (0.5, pytest.approx(0.5))]]]


def test_the_align_dialog_can_quantize_to_eighth_notes(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "quantize")(2)  # 1/8 notes, half a beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.0, 0.5), (0.5, pytest.approx(0.5))]]]


def test_the_align_dialog_quantizes_onto_the_offset_grid(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window, offset=0.25)
    parameter_writer(dialog, "quantize")(1)  # 1/4 notes, one beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.25, 0.5), (0.75, pytest.approx(0.5))]]]


def test_a_cached_alignment_is_reused_and_resnapped(own_window, monkeypatch) -> None:
    align.save_alignment("/tmp/vocal.wav", "mms", "cpu", "あん\n", [[(0.1, 0.6), (0.6, 1.1)]], [], [])
    monkeypatch.setattr(align, "align", lambda *args, **kwargs: pytest.fail("must not run the model"))
    dialog = AlignDialog("/tmp/vocal.wav", "あん\n", 120.0, parent=own_window, offset=0.25)
    parameter_writer(dialog, "quantize")(1)  # 1/4 notes, one beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))

    dialog._start()

    assert dialog._thread is None  # the model never ran
    assert got == [[[(0.25, 0.5), (0.75, pytest.approx(0.5))]]]
    assert "saved alignment reused" in dialog.log.toPlainText()


def test_a_cached_whole_song_alignment_is_reused_when_chunking_is_off(own_window, monkeypatch) -> None:
    align.save_alignment("/tmp/vocal.wav", "mms", "cpu", "あん\n", [[(0.1, 0.6), (0.6, 1.1)]], [], [], chunk="off")
    monkeypatch.setattr(align, "align", lambda *args, **kwargs: pytest.fail("must not run the model"))
    dialog = AlignDialog("/tmp/vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "chunk")("off")
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))

    dialog._start()

    assert dialog._thread is None
    assert got == [[[(0.1, 0.5), (0.6, pytest.approx(0.5))]]]
    assert "saved alignment reused" in dialog.log.toPlainText()


def test_the_align_dialog_remembers_what_was_chosen(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "model")("yohane")
    parameter_writer(dialog, "quantize")(4)
    parameter_writer(dialog, "chunk")("off")
    dialog.reject()

    again = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    assert again.parameters() == {
        "model": "yohane",
        "device": "cpu",
        "quantize": 4,
        "chunk": "off",
        "use_constraints": True,
    }


def test_the_align_dialog_runs_chunked_unless_told_otherwise(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    assert dialog.parameters()["chunk"] == align.DEFAULT_MODE


def test_the_align_dialog_leaves_the_times_alone_by_default(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.1, 0.5), (0.6, pytest.approx(0.5))]]]


def test_the_align_dialog_shows_a_failure(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)
    dialog._fail("FileNotFoundError: no mms aligner")
    assert "FileNotFoundError" in dialog.log.toPlainText()
    assert dialog.run.isEnabled()


def test_the_align_dialog_keeps_its_progress_beside_the_problems(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)
    dialog.log.setPlainText("Aligning over 204.0s of audio…")
    dialog._done(([[(0.0, 1.0), (1.0, 2.0)]], "mms", ["あい: empty"], [True]))
    assert "Aligning over 204.0s of audio…" in dialog.log.toPlainText()
    assert "あい: empty" in dialog.log.toPlainText()


def test_the_aligner_logs_a_download_a_tenth_at_a_time(qt_app) -> None:
    aligner = Aligner("vocal.wav", "あい\n", "yohane")
    logged: list[str] = []
    aligner.message.connect(logged.append)
    aligner._downloading(0, 1000)
    aligner._downloading(50, 1000)
    aligner._downloading(100, 1000)
    assert logged == ["Downloading the yohane model… 0%", "Downloading the yohane model… 10%"]


def test_the_aligner_reports_the_lines_it_doubts() -> None:
    lines = natural_sounds("あい\n")
    found = align.AlignedSegment(0.0, 2.0, (align.Token("a"), align.Token("i")))
    flagged, reported = Aligner._problems(found, lines)
    assert flagged == [True]
    assert reported == ["あい: empty"]


def test_the_align_button_opens_the_dialog(own_window, monkeypatch, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あん\n", mode="read")
    own_window._watch_lyrics()
    own_window.audio_path = str(tmp_path / "vocal.wav")

    opened: list = []
    monkeypatch.setattr(AlignDialog, "exec", lambda self: opened.append(self) or 0)
    own_window.transport.grid_offset.setValue(125)
    own_window._open_align()
    assert len(opened) == 1
    assert opened[0].offset == 0.125  # the dialog quantises on the grid that is drawn


def test_saving_lyrics_lets_align_see_them_at_once(own_window, monkeypatch, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window.audio_path = str(tmp_path / "vocal.wav")
    own_window._watch_lyrics()  # no .krc yet, so nothing to align
    assert own_window.view.lyric_lines == ()

    own_window._on_lyrics_saved("あん\n")  # the dialog wrote the file and told the window
    assert [sound.reading for line in own_window.view.lyric_lines for sound in line.sounds] == ["あ", "ん"]

    opened: list = []
    monkeypatch.setattr(AlignDialog, "exec", lambda self: opened.append(self) or 0)
    own_window._open_align()
    assert len(opened) == 1


def test_the_align_button_says_why_it_cannot_open(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window.audio_path = str(tmp_path / "vocal.wav")
    own_window._open_align()
    assert "no lyrics" in own_window.statusBar().currentMessage()

    own_window._stored_lyrics = project.Lyrics(text="世界\n", mode="read")
    own_window._watch_lyrics()
    own_window._open_align()
    assert "could not be read" in own_window.statusBar().currentMessage()


def test_text_changes_do_not_start_alignment(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(Aligner, "start", lambda self: pytest.fail("must not align automatically"))
    lyrics_window._on_lyrics_saved("あい")
    wait_for_lyric_mapping(lyrics_window)
    assert not hasattr(lyrics_window, "_lyric_mode")
    assert lyrics_window.view.lyric_raw == (((None, None), (None, None)),)
    lyrics.save(lyrics_window.lyrics_path(), "あう")
    lyrics_window._on_lyrics_file_changed()
    assert lyrics_window.lyrics_text == "あう"
    assert not hasattr(lyrics_window, "_lyric_mode")


def test_exporting_is_not_gated_on_the_lyrics(own_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あん\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))  # one note where two sounds need two
    own_window.view.set_lyrics(natural_sounds("あん\n"), [[(0.0, 1.0), (1.0, 2.0)]])
    own_window._lyric_key = text_key("あん\n")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "out"), "MIDI"))

    assert own_window._on_export_midi() is True


def test_exporting_leaves_the_krc_alone(own_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あん\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0), (64, 2.0, 1.0, 0)))
    own_window.view.set_lyrics(natural_sounds("あん\n"), [[(0.0, 1.0), (1.0, 3.0)]])
    own_window._lyric_key = text_key("あん\n")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "out"), "MIDI"))

    assert own_window._on_export_midi() is True
    assert (tmp_path / "song.krc").read_text(encoding="utf-8") == "あん\n"
