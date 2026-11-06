# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pathlib
import queue

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QToolButton

from namioto.analysis import devices, separation, transcription
from namioto.channels import Channel
from namioto.ui.transcription_dialog import TranscriptionDialog
from namioto.utils import file_stamp, write_json

from .helpers import parameter_writer


class FakeProcess:
    """A child that has already stopped, so the dialog sees what a real one leaves behind."""

    def __init__(self, exitcode: int = 0):
        self.exitcode = exitcode
        self.terminated = False

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout=None) -> None:
        pass


def fake_job(messages, exitcode: int = 0):
    """The spawn seam replaced by a queue that is already full and a child that already stopped."""
    channel: queue.Queue = queue.Queue()
    for message in messages:
        channel.put(message)
    process = FakeProcess(exitcode)

    def start_job(_audio, _parameters):
        return process, channel

    return start_job


def test_the_transcribe_button_needs_audio(own_window) -> None:
    assert own_window.edit.transcribe.isEnabled() is False


def test_the_transcribe_button_opens_the_dialog(own_window, monkeypatch) -> None:
    own_window.audio_path = "/tmp/song.wav"
    own_window.edit.transcribe.setEnabled(True)
    opened: list[TranscriptionDialog] = []
    monkeypatch.setattr(TranscriptionDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    own_window.edit.transcribe.click()

    assert len(opened) == 1
    assert opened[0].parent() is own_window
    assert opened[0].audio == "/tmp/song.wav"


def test_the_transcription_dialog_carries_the_grid_offset(own_window, monkeypatch) -> None:
    own_window.audio_path = "/tmp/song.wav"
    own_window.transport.grid_offset.setValue(125)
    opened: list[TranscriptionDialog] = []
    monkeypatch.setattr(TranscriptionDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    own_window._open_transcription()

    assert opened[0].offset == 0.125


def test_the_transcription_dialog_snaps_the_raw_notes_on_its_grid(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.51, 0.98, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, offset=0.25)
    parameter_writer(dialog, "quantize")(4)  # 1/16 notes: a 0.125 s cell at 120 BPM

    dialog._start()
    dialog._poll()

    assert dialog.notes() == [(0.5, 1.0, 60.0)]  # the 0.25 s offset grid, not the absolute one
    assert "offset 250.0 ms" in dialog.log.toPlainText()


def test_transcription_phase_estimation_is_optional(own_window) -> None:
    raw = [(0.15, 0.65, 60.0), (0.65, 1.15, 62.0)]
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    dialog._parameters["quantize"] = 2
    assert dialog._snapped(raw) == [(0.25, 0.75, 60.0), (0.75, 1.25, 62.0)]
    assert dialog.offset == 0.0
    dialog._parameters["estimate_phase"] = True
    assert dialog._snapped(raw) == [pytest.approx(values) for values in raw]
    assert dialog.offset == pytest.approx(0.15)
    assert transcription.run_key("/tmp/song.wav", dialog._parameters) == transcription.run_key(
        "/tmp/song.wav", dict(dialog._parameters, estimate_phase=False)
    )
    dialog.close()


def test_the_transcription_dialog_offers_every_parameter(own_window) -> None:
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    assert [field.name for field, _read, _write in dialog._fields] == [item.name for item in transcription.PARAMETERS]
    dialog.close()


def test_the_transcription_dialog_remembers_what_was_typed(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    parameter_writer(dialog, "size")("large")
    parameter_writer(dialog, "language")("zh")

    dialog._start()
    dialog._poll()

    saved = transcription.load_parameters()
    assert saved["size"] == "large"
    assert saved["language"] == "zh"
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_the_transcription_run_shows_its_log_and_progress(own_window, monkeypatch) -> None:
    messages = [
        ("log", "GAME model small"),
        ("progress", "parts", 1, 2),
        ("progress", "parts", 2, 2),
        ("done", [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)]),
    ]
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job(messages))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert "GAME model small" in dialog.log.toPlainText()
    assert dialog.progress_label.text() == "2 notes"
    assert dialog.notes() == [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)]
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_a_cached_run_is_offered_instead_of_running(own_window, monkeypatch) -> None:
    transcription.save_run("/tmp/song.wav", transcription.default_parameters(), [(0.0, 0.5, 60.0)])
    asked: list[bool] = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: asked.append(True) or QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", lambda *args: pytest.fail("must not run"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()

    assert asked == [True]
    assert dialog.notes() == [(0.0, 0.5, 60.0)]
    assert "saved run reused" in dialog.log.toPlainText()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_a_declined_cache_still_runs(own_window, monkeypatch) -> None:
    transcription.save_run("/tmp/song.wav", transcription.default_parameters(), [(0.0, 0.5, 60.0)])
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(1.0, 1.5, 64.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert dialog.notes() == [(1.0, 1.5, 64.0)]


def test_a_crashed_transcription_leaves_the_window_alone(own_window, monkeypatch) -> None:
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_job", fake_job([("log", "GAME model small")], exitcode=1)
    )
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert "exit code 1" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_a_failed_transcription_shows_the_traceback(own_window, monkeypatch) -> None:
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_job",
        fake_job([("error", "Traceback\nRuntimeError: no model")]),
    )
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert "RuntimeError: no model" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()


def test_a_child_that_cannot_start_is_reported(own_window, monkeypatch) -> None:
    def refuse(*args):
        raise OSError("no processes left")

    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", refuse)
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()

    assert "no processes left" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()


def test_a_missing_runtime_stops_the_transcription(own_window, monkeypatch) -> None:
    monkeypatch.setattr(devices, "validate", lambda: devices.MISSING_RUNTIME)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", lambda *args: pytest.fail("must not run"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()

    assert devices.MISSING_RUNTIME in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()


def test_transcription_has_no_target_choice(own_window, monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: pytest.fail("must not ask"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, channel_has_notes=True)
    assert "target" not in dialog.parameters()
    assert all(field.name != "target" for field, _read, _write in dialog._fields)
    dialog.close()


def test_declining_channel_zero_overwrite_does_not_start_transcription(own_window, monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", lambda *args: pytest.fail("must not run"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, channel_has_notes=True)
    dialog._start()
    assert dialog.notes() == []
    assert dialog._process is None
    assert dialog.run_button.isEnabled()


def test_an_empty_channel_zero_takes_the_notes_without_asking(own_window, monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: pytest.fail("must not ask"))
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    monkeypatch.setattr(devices, "validate", lambda: None)

    dialog._start()
    dialog._poll()

    assert dialog.notes() == [(0.0, 0.5, 60.0)]


def test_accepting_channel_zero_overwrite_runs_transcription(own_window, monkeypatch) -> None:
    monkeypatch.setattr(devices, "validate", lambda: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, channel_has_notes=True)
    dialog._start()
    dialog._poll()
    assert dialog.notes() == [(0.0, 0.5, 60.0)]


def test_the_button_closes_the_window_when_idle(own_window) -> None:
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    assert dialog.close_button.text() == "Close"

    dialog._close_or_cancel()

    assert dialog.result() == QDialog.DialogCode.Rejected


def test_cancelling_a_run_keeps_the_window_and_offers_another(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("log", "GAME model small")]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    assert dialog.close_button.text() == "Cancel"

    dialog._close_or_cancel()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.run_button.isEnabled()
    assert dialog.close_button.text() == "Close"
    assert "Cancelled" in dialog.log.toPlainText()


def test_the_log_stays_hidden_until_there_is_something_to_say(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("log", "GAME model small")]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    assert dialog.log.isHidden()

    dialog._start()

    assert not dialog.log.isHidden()


def test_the_window_is_pinned_to_its_content(own_window) -> None:
    """Hyprland resizes a native toplevel only while min == max, so the dialog follows its content
    by staying rigid."""
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    collapsed = dialog.size().height()
    assert dialog.minimumSize() == dialog.maximumSize()

    dialog.findChild(QToolButton).click()
    QApplication.processEvents()
    QApplication.processEvents()

    assert dialog.size().height() > collapsed
    assert dialog.minimumSize() == dialog.maximumSize() == dialog.size()


def test_transcription_creates_channel_zero_when_missing(own_window) -> None:
    own_window.view.set_channels((Channel(channel=5),))
    own_window.view.set_active_channel(5)
    own_window._adopt_transcription([(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)])
    assert {channel.channel for channel in own_window.view.channels} == {0, 5}
    assert [note.pitch for note in own_window.view.notes()] == [60, 62]
    assert {note.channel for note in own_window.view.notes()} == {0}
    own_window.view.undo()
    assert own_window.view.notes() == []
    assert [channel.channel for channel in own_window.view.channels] == [5]


def test_transcription_replaces_only_channel_zero(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes([(60, 0.0, 1.0, 0), (62, 0.0, 1.0, 1)])
    own_window.view.set_active_channel(1)

    own_window._adopt_transcription([(1.0, 1.5, 64.0)])

    assert len(own_window.view.channels) == 2
    assert sorted((note.pitch, note.channel) for note in own_window.view.notes()) == [(62, 1), (64, 0)]


def fake_stage(messages, seen: list | None = None, exitcode: int = 0):
    """`fake_job`, but it also records the audio it was handed, so a test can see which one a stage read."""
    channel: queue.Queue = queue.Queue()
    for message in messages:
        channel.put(message)
    process = FakeProcess(exitcode)

    def start(audio, parameters):
        if seen is not None:
            seen.append((audio, parameters))
        return process, channel

    return start


def source_writer(dialog, name: str):
    """The write half of one row of the Source section, to change it the way a widget would."""
    for field, _read, write in dialog._separation_fields:
        if field.name == name:
            return write
    raise AssertionError(f"this dialog has no {name} source field")


def cached_stems(audio: str, model: str) -> dict[str, pathlib.Path]:
    """A finished separation's directory, written by hand so a check needs no model."""
    folder = separation.stems_dir(audio, model)
    folder.mkdir(parents=True)
    written = {}
    for name in ("vocals", "instrumental"):
        path = folder / f"{name}.wav"
        path.write_bytes(b"wav")
        written[name] = path
    write_json(
        {
            "version": separation.STORE_VERSION,
            "audio": str(pathlib.Path(audio).resolve()),
            "stamp": list(file_stamp(audio)),
            "model": model,
            "rate": 44100,
            "stems": {name: path.name for name, path in written.items()},
        },
        folder / separation.MANIFEST,
    )
    return written


def test_the_switch_separates_before_transcribing(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    stems = {"vocals": str(tmp_path / "vocals.wav"), "instrumental": str(tmp_path / "instrumental.wav")}
    separated: list = []
    transcribed: list = []
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_separation",
        fake_stage([("log", "separation model x"), ("progress", "separating", 1, 1), ("done", stems)], separated),
    )
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_job", fake_stage([("done", [(0.0, 0.5, 60.0)])], transcribed)
    )
    dialog = TranscriptionDialog(str(audio), 120.0, parent=own_window)
    source_writer(dialog, "enabled")(True)

    dialog._start()
    dialog._poll()  # the stems land, the transcription stage begins
    dialog._poll()  # the notes land

    assert len(separated) == 1
    assert separated[0][0] == str(audio)
    assert [entry[0] for entry in transcribed] == [str(stems["vocals"])]
    assert dialog.notes() == [(0.0, 0.5, 60.0)]
    assert "Transcribing the vocals stem" in dialog.log.toPlainText()


def test_cached_stems_skip_the_separation_stage(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    model = separation.MODELS[0]
    stems = cached_stems(str(audio), model)
    transcribed: list = []
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_separation", lambda *args: pytest.fail("must not split"))
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_job", fake_stage([("done", [(0.0, 0.5, 60.0)])], transcribed)
    )
    dialog = TranscriptionDialog(str(audio), 120.0, parent=own_window)
    source_writer(dialog, "enabled")(True)
    source_writer(dialog, "model")(model)

    dialog._start()
    dialog._poll()

    assert [entry[0] for entry in transcribed] == [str(stems["vocals"])]
    assert "stems reused" in dialog.log.toPlainText()
    assert dialog.notes() == [(0.0, 0.5, 60.0)]


def test_a_separation_without_the_vocals_stem_fails_in_the_log(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_separation",
        fake_stage([("done", {"drums": str(tmp_path / "drums.wav")})]),
    )
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", lambda *args: pytest.fail("must not transcribe"))
    dialog = TranscriptionDialog(str(audio), 120.0, parent=own_window)
    source_writer(dialog, "enabled")(True)

    dialog._start()
    dialog._poll()

    assert "vocals" in dialog.log.toPlainText()
    assert dialog.progress_label.text() == "Failed"
    assert dialog.notes() == []


def test_an_unreported_download_shows_an_indeterminate_bar(own_window) -> None:
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._show_progress("download", 0, 0)

    assert dialog.progress.minimum() == 0
    assert dialog.progress.maximum() == 0
    assert dialog.progress_label.text() == "Downloading the model …"

    dialog._show_progress("separating", 2, 4)

    assert dialog.progress_label.text() == "Separating … 2/4"
    assert dialog.progress.value() == 2
    dialog.close()
