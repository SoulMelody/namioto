# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the alignment window: its form, its two stages, and what it hands back."""

from __future__ import annotations

import queue

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from namioto.analysis import alignment, separation
from namioto.ui import align_dialog
from namioto.ui.align_dialog import AlignDialog

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


def fake_job(messages, exitcode: int = 0, calls: list | None = None):
    """The spawn seam replaced by a queue that is already full and a child that already stopped."""
    channel: queue.Queue = queue.Queue()
    for message in messages:
        channel.put(message)
    process = FakeProcess(exitcode)

    def start(*args):
        if calls is not None:
            calls.append(args)
        return process, channel

    return start


@pytest.fixture
def agreed(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)


def make_dialog(window) -> AlignDialog:
    dialog = AlignDialog("/tmp/song.wav", ["き", "み"], [0.0, 0.5], 120.0, parent=window)
    return dialog


def test_the_align_button_needs_audio(own_window) -> None:
    assert own_window.edit.align.isEnabled() is False


def test_the_align_button_opens_the_dialog(own_window, monkeypatch) -> None:
    own_window.audio_path = "/tmp/song.wav"
    own_window.edit.align.setEnabled(True)
    item = own_window.view.add_note(60, 0.0, 1.0)
    item.note.lyric = "き"
    opened: list[AlignDialog] = []
    monkeypatch.setattr(AlignDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    own_window.edit.align.click()

    assert len(opened) == 1
    assert opened[0].parent() is own_window
    assert opened[0].words == ["き"]


def test_the_window_offers_every_parameter(own_window) -> None:
    dialog = make_dialog(own_window)
    names = [field.name for field, _read, _write in dialog._fields]
    assert names == [item.name for item in alignment.PARAMETERS]
    assert [field.name for field, _read, _write in dialog._separation_fields] == [
        item.name for item in separation.PARAMETERS
    ]
    dialog.close()


def test_a_declined_confirmation_does_not_start_the_run(own_window, monkeypatch, agreed) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr(align_dialog, "start_job", lambda *args: pytest.fail("must not run"))
    dialog = make_dialog(own_window)

    dialog._start()

    assert dialog.spans() == []
    assert dialog._process is None
    dialog.close()


def test_a_finished_run_snaps_its_spans_and_accepts(own_window, monkeypatch, agreed) -> None:
    monkeypatch.setattr(align_dialog, "has_runtime", lambda: True)
    monkeypatch.setattr(
        align_dialog,
        "start_job",
        fake_job([("done", [(0.51, 1.0), None])]),
    )
    dialog = make_dialog(own_window)
    parameter_writer(dialog, "quantize")(4)  # 1/16 notes: a 0.125 s cell at 120 BPM

    dialog._start()
    dialog._poll()

    assert dialog.spans() == [(0.5, 1.0), None]
    dialog.close()


def test_a_missing_runtime_stops_the_run(own_window, monkeypatch, agreed) -> None:
    monkeypatch.setattr(align_dialog, "has_runtime", lambda: False)
    monkeypatch.setattr(align_dialog, "start_job", lambda *args: pytest.fail("must not run"))
    dialog = make_dialog(own_window)

    dialog._start()

    assert "qwen-asr" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()
    dialog.close()


def test_the_separation_stage_hands_the_vocal_stem_to_the_aligner(own_window, monkeypatch, agreed) -> None:
    monkeypatch.setattr(align_dialog, "has_runtime", lambda: True)
    monkeypatch.setattr(
        align_dialog,
        "start_separation",
        fake_job([("done", {"vocals": "/tmp/vocals.wav", "instrumental": "/tmp/inst.wav"})]),
    )
    calls: list = []
    monkeypatch.setattr(align_dialog, "start_job", fake_job([("done", [(0.0, 0.4), (0.4, 0.8)])], calls=calls))
    dialog = make_dialog(own_window)
    for field, _read, write in dialog._separation_fields:
        if field.name == "enabled":
            write(True)

    dialog._start()
    dialog._poll()
    dialog._poll()

    assert calls and calls[0][0] == "/tmp/vocals.wav"
    assert dialog.spans() == [(0.0, 0.4), (0.4, 0.8)]
    dialog.close()


def test_a_child_that_cannot_start_is_reported(own_window, monkeypatch, agreed) -> None:
    monkeypatch.setattr(align_dialog, "has_runtime", lambda: True)

    def refuse(*args):
        raise OSError("no processes left")

    monkeypatch.setattr(align_dialog, "start_job", refuse)
    dialog = make_dialog(own_window)

    dialog._start()

    assert "no processes left" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()
    dialog.close()
