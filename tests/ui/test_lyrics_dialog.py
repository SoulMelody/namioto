# SPDX-License-Identifier: AGPL-3.0-only
"""The lyrics window: one box of lyric text, handed back on `Save` and never written anywhere here."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from namioto.ui.lyrics_dialog import LyricsDialog


def test_the_box_opens_on_the_lyrics_of_the_project(lyrics_window) -> None:
    lyrics_window.view.set_notes(((60, 0.0, 1.0, 0, 0, "歌"),))
    dialog = LyricsDialog("歌", parent=lyrics_window)

    assert dialog.editor.toPlainText() == "歌"
    assert dialog.parent() is lyrics_window
    dialog.close()


def test_saving_hands_the_edited_text_back() -> None:
    dialog = LyricsDialog("歌")
    saved: list[str] = []
    finished: list[int] = []
    dialog.saved.connect(saved.append)
    dialog.finished.connect(finished.append)

    dialog.editor.setPlainText("歌[うた]")
    dialog.save_button.click()

    assert saved == ["歌[うた]"]
    assert finished == [QDialog.DialogCode.Accepted]
    dialog.close()


def test_accept_emits_the_text_as_it_stands() -> None:
    dialog = LyricsDialog("あい")
    saved: list[str] = []
    dialog.saved.connect(saved.append)

    dialog.editor.setPlainText("あい\nうえ")
    dialog.accept()

    assert saved == ["あい\nうえ"]
    dialog.close()


def test_a_close_without_edits_does_not_ask(monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: pytest.fail("must not ask"))
    dialog = LyricsDialog("あい")
    finished: list[int] = []
    dialog.finished.connect(finished.append)
    dialog.reject()
    assert finished == [QDialog.DialogCode.Rejected]


def test_a_modified_text_asks_before_the_close_is_thrown_away(monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)
    dialog = LyricsDialog("あい")
    saved: list[str] = []
    finished: list[int] = []
    dialog.saved.connect(saved.append)
    dialog.finished.connect(finished.append)

    dialog.editor.setPlainText("あう")
    dialog.reject()

    assert saved == []  # Discard hands nothing back
    assert finished == [QDialog.DialogCode.Rejected]


def test_answering_save_at_the_close_saves_and_closes(monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)
    dialog = LyricsDialog("あい")
    saved: list[str] = []
    finished: list[int] = []
    dialog.saved.connect(saved.append)
    dialog.finished.connect(finished.append)

    dialog.editor.setPlainText("あう")
    dialog.reject()

    assert saved == ["あう"]
    assert finished == [QDialog.DialogCode.Accepted]


def test_cancelling_that_question_keeps_the_window_open(monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    dialog = LyricsDialog("あい")
    finished: list[int] = []
    dialog.finished.connect(finished.append)

    dialog.editor.setPlainText("あう")
    dialog.reject()

    assert finished == []  # the close never happened
    dialog.close()


def test_the_window_opens_it_on_the_lyrics_it_already_sings(lyrics_window, monkeypatch) -> None:
    lyrics_window.view.set_notes(((60, 0.0, 1.0, 0, 0, "あん"),))
    opened: list[LyricsDialog] = []
    monkeypatch.setattr(LyricsDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    lyrics_window._open_lyrics()

    assert len(opened) == 1
    assert opened[0].editor.toPlainText() == "あん"
