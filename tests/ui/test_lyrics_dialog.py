# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QFileDialog, QLineEdit, QMessageBox

from namioto import lyrics
from namioto import settings as store
from namioto.ui.form import field_editor
from namioto.ui.lyrics_dialog import LyricsConverter, LyricsDialog


def lyrics_dialog(lyrics_window, **settings):
    config = store.clone(lyrics_window.settings).lyrics
    for name, value in settings.items():
        setattr(config, name, value)
    return LyricsDialog(lyrics_window.lyrics_path(), config, parent=lyrics_window)


def test_the_lyrics_button_waits_for_a_project(own_window) -> None:
    assert own_window.lyrics_path() is None
    assert own_window.edit.lyrics.isEnabled() is False


def test_the_lyrics_button_opens_the_window(lyrics_window, monkeypatch) -> None:
    opened: list[LyricsDialog] = []
    monkeypatch.setattr(LyricsDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    lyrics_window.edit.lyrics.click()

    assert len(opened) == 1
    assert opened[0].path == lyrics_window.lyrics_path()
    assert opened[0].parent() is lyrics_window


def test_the_window_opens_with_the_lyrics_already_there(lyrics_window) -> None:
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.result.toPlainText() == "歌[うた]"
    dialog.close()


def test_the_plain_text_panel_opens_when_there_are_no_lyrics(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.plain_open is True
    assert dialog.plain_body.isVisibleTo(dialog) is True
    dialog.close()


def test_the_plain_text_panel_stays_folded_over_existing_lyrics(lyrics_window) -> None:
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.plain_open is False
    assert dialog.plain_body.isVisibleTo(dialog) is False
    dialog.close()


def test_an_unset_api_points_at_the_prompt(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.convert_button.isEnabled() is False
    assert dialog.copy_button.isEnabled() is True
    assert dialog.hint.isVisibleTo(dialog) is True
    dialog.close()


def test_a_set_api_hides_the_prompt_hint(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    assert dialog.convert_button.isEnabled() is True
    assert dialog.hint.isVisibleTo(dialog) is False
    dialog.close()


def test_closing_with_unsaved_lyrics_asks(lyrics_window, monkeypatch) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("歌[うた]")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)

    dialog.reject()

    assert lyrics.load(lyrics_window.lyrics_path()) == ""  # Discard wrote nothing


def test_closing_saves_the_lyrics_when_asked(lyrics_window, monkeypatch) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("歌[うた]")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)

    dialog.reject()

    assert lyrics.load(lyrics_window.lyrics_path()) == "歌[うた]"


def test_cancelling_the_close_keeps_the_unsaved_lyrics(lyrics_window, monkeypatch) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("歌[うた]")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    finished: list[int] = []
    dialog.finished.connect(finished.append)

    dialog.reject()

    assert finished == []
    assert lyrics.load(lyrics_window.lyrics_path()) == ""


def test_importing_a_krc_fills_the_lyrics_box(lyrics_window, tmp_path, monkeypatch) -> None:
    other = tmp_path / "other.krc"
    other.write_text("季節[き,せつ]", encoding="utf-8")
    dialog = lyrics_dialog(lyrics_window)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), "Lyrics file (*.krc)"))

    dialog.load_krc_button.click()

    assert dialog.result.toPlainText() == "季節[き,せつ]"
    assert "Imported" in dialog.status_label.text()
    dialog.close()


def test_loading_plain_text_fills_the_source_box(lyrics_window, tmp_path, monkeypatch) -> None:
    other = tmp_path / "words.txt"
    other.write_text("季節は移ろい", encoding="utf-8")
    dialog = lyrics_dialog(lyrics_window)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), "Text file (*.txt)"))

    dialog.load_text_button.click()

    assert dialog.source.toPlainText() == "季節は移ろい"
    assert dialog.result.toPlainText() == ""  # plain text is the source, not the lyrics
    dialog.close()


def test_copying_the_prompt_puts_it_on_the_clipboard(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.source.setPlainText("君の名は")
    dialog.copy_button.click()
    assert QGuiApplication.clipboard().text() == lyrics.build_prompt("君の名は")
    assert "web model" in dialog.status_label.text()
    dialog.close()


def test_saving_lyrics_writes_the_temporary_copy(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("季節[き,せつ]は移[うつ]ろい")
    dialog.save_button.click()
    assert lyrics.load(lyrics_window.lyrics_path()) == "季節[き,せつ]は移[うつ]ろい"
    assert "Saved to lyrics.krc" in dialog.status_label.text()
    assert not lyrics_window.project_path.with_suffix(".krc").exists()
    dialog.close()


def test_the_api_button_waits_for_the_settings(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.convert_button.isEnabled() is False
    dialog.close()

    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    assert dialog.convert_button.isEnabled() is True
    dialog.close()


def test_a_missing_source_is_not_sent(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(lyrics, "convert", lambda *args, **kwargs: pytest.fail("must not call"))
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.convert_button.click()
    assert "Paste the lyrics" in dialog.status_label.text()
    dialog.close()


def test_the_converter_asks_the_model_and_reports_it_back(lyrics_window, monkeypatch) -> None:
    asked: dict = {}

    def fake(source, **fields):
        asked.update(fields, source=source)
        return "歌[うた]"

    monkeypatch.setattr(lyrics, "convert", fake)
    dialog = lyrics_dialog(
        lyrics_window,
        api_base="https://api.example.com/v1",
        api_key="k",
        model="m",
        temperature=0.5,
        timeout=30.0,
    )
    converter = LyricsConverter("歌", dialog.config, dialog.path)
    dialog.converter = converter
    converter.converted.connect(dialog._converted)
    converter.load()

    assert asked["source"] == "歌"
    assert asked["base_url"] == "https://api.example.com/v1"
    assert asked["api_key"] == "k"
    assert asked["model"] == "m"
    assert asked["temperature"] == 0.5
    assert asked["timeout"] == 30.0
    assert asked["stream"] is True
    assert callable(asked["on_delta"])
    assert dialog.result.toPlainText() == "歌[うた]"
    assert "Converted" in dialog.status_label.text()
    dialog.close()


def test_a_wrong_answer_is_sent_back_for_another_try(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(LyricsConverter, "start", lambda self: None)
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.source.setPlainText("歌")
    dialog.result.setPlainText("元の歌詞")
    converter = LyricsConverter("歌", dialog.config, dialog.path)
    dialog.converter = converter
    converter.converted.connect(dialog._converted)

    converter.converted.emit("別[べつ]の歌詞[か,し]")  # the model rewrote the lyrics

    assert dialog.result.toPlainText() == "元の歌詞"  # the box keeps the last good lyrics
    assert dialog.converter is not converter  # a correction request replaced the first one
    assert "別[べつ]" in dialog.converter.retry
    assert "changed the lyrics" in dialog.converter.retry
    assert "asking the model" in dialog.status_label.text()
    dialog.close()


def test_the_last_bad_answer_warns_and_keeps_the_correction(lyrics_window, monkeypatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warned.append(message))
    monkeypatch.setattr(LyricsConverter, "start", lambda self: None)
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.source.setPlainText("歌")
    converter = LyricsConverter("歌", dialog.config, dialog.path)
    dialog.converter = converter
    converter.converted.connect(dialog._converted)

    for _ in range(3):
        dialog.converter.converted.emit("別[べつ]の歌詞[か,し]")

    assert dialog.result.toPlainText() == ""
    assert warned
    assert "copy the prompt" in dialog.status_label.text()
    dialog.close()


def test_copying_after_a_refusal_carries_the_correction(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.source.setPlainText("歌")
    dialog._rejected = ("別[べつ]の歌詞[か,し]", "the answer changed the lyrics instead of annotating them")

    dialog.copy_button.click()

    text = QGuiApplication.clipboard().text()
    assert "別[べつ]" in text
    assert "changed the lyrics" in text
    assert "web model" in dialog.status_label.text()
    dialog.close()


def test_a_manual_answer_without_readings_is_refused_at_save(lyrics_window, monkeypatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warned.append(message))
    dialog = lyrics_dialog(lyrics_window)
    dialog.source.setPlainText("歌")
    dialog.result.setPlainText("歌")  # every kanji still wants a reading

    dialog.save_button.click()

    assert lyrics.load(lyrics_window.lyrics_path()) == ""
    assert warned
    assert dialog._rejected is not None
    dialog.copy_button.click()
    assert "no reading" in QGuiApplication.clipboard().text()
    dialog.close()


def test_a_failed_conversion_shows_why(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog._failed("HTTP 401 Unauthorized")
    assert "HTTP 401" in dialog.status_label.text()
    assert dialog.convert_button.isEnabled() is True
    dialog.close()


def test_a_broken_krc_is_not_saved(lyrics_window, monkeypatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warned.append(message))
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("見[み")

    dialog.save_button.click()

    assert lyrics.load(lyrics_window.lyrics_path()) == ""
    assert "Not saved" in dialog.status_label.text()
    assert warned
    dialog.close()


def test_the_output_box_is_only_shown_once_the_api_is_asked(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(LyricsConverter, "start", lambda self: None)
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    assert dialog.log.isHidden() is True

    dialog.source.setPlainText("歌")
    dialog.convert_button.click()

    assert dialog.log.isHidden() is False
    assert dialog.log.toPlainText() == ""  # it starts empty, waiting for the model
    dialog.close()


def test_the_streamed_output_is_logged_as_it_comes(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.log.show()

    dialog._on_delta("reasoning", "考え")
    dialog._on_delta("reasoning", "て")
    dialog._on_delta("content", "歌[うた]を")

    text = dialog.log.toPlainText()
    assert "考えて" in text
    assert text.index("考えて") < text.index("歌[うた]を")  # the reasoning runs before the answer
    assert text.count("歌[うた]を") == 1
    dialog.close()


def test_the_api_key_is_typed_back_hidden() -> None:
    spec = store.FIELD_SPECS[("lyrics", "api_key")]
    widget, read, write = field_editor("hunter2", spec)
    assert isinstance(widget, QLineEdit)
    assert widget.echoMode() == QLineEdit.EchoMode.Password
    write("secret")
    assert read() == "secret"
    widget.deleteLater()


def test_the_lyrics_open_in_an_external_editor(lyrics_window, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(QProcess, "startDetached", lambda program, args: calls.append((program, args)) or (True, 1))
    lyrics_window.settings.lyrics.editor = "code --wait"

    lyrics_window._open_lyrics_editor()

    assert calls == [("code", ["--wait", str(lyrics_window.lyrics_path())])]


def test_an_outside_change_is_reloaded(lyrics_window) -> None:
    lyrics_window.statusBar().clearMessage()
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")

    lyrics_window._on_lyrics_file_changed()

    assert lyrics_window.lyrics_text == "歌[うた]"
    assert "reloaded" in lyrics_window.statusBar().currentMessage().lower()


def test_a_save_of_our_own_is_not_read_back(lyrics_window) -> None:
    lyrics_window._on_lyrics_saved("歌[うた]")
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")
    lyrics_window.statusBar().clearMessage()

    lyrics_window._on_lyrics_file_changed()

    assert lyrics_window.statusBar().currentMessage() == ""


def test_the_watcher_moves_with_the_project(lyrics_window, tmp_path) -> None:
    other = tmp_path / "other.nto"
    other.write_text("{}")

    lyrics_window.project_path = other
    lyrics_window._watch_lyrics()

    assert lyrics_window.lyrics_path().parent != tmp_path
    assert not (tmp_path / "other.krc").exists()
    assert lyrics_window.lyrics_text == ""
