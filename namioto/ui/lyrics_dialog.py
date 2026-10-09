# SPDX-License-Identifier: AGPL-3.0-only
"""The lyrics window: the source text, the `.krc` it becomes, and the model call in between.

The project's lyrics are the box at the top, and the plain text below is the optional upstream that
feeds it - the model, through the endpoint in the settings or through the clipboard, turns it into
the `.krc` above. A request runs off the GUI thread (`LyricsConverter`) and a refused answer is fed
back for another try, up to `MAX_ATTEMPTS`; a refusal that outlives them is kept so `Copy prompt` can
carry it to a web model. `LyricsWatcher` follows the temporary working copy so an external save
can be validated and adopted by the project.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QFileSystemWatcher, QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from namioto import lyrics
from namioto.i18n import tr
from namioto.ui import icons
from namioto.ui.controls import ICON_SIZE
from namioto.ui.loading import LoadingThread

LOAD_HEIGHT = 180
LOG_HEIGHT = 120
MAX_ATTEMPTS = 3


class LyricsConverter(LoadingThread):
    """One request to the model, off the GUI thread; `LoadingThread` reports whatever it throws.

    The endpoint is asked to stream, so `delta` carries the model's own output while it comes - the
    reasoning first, then the answer - and `converted` still carries the answer alone at the end.
    `retry` is what an earlier refusal said, sent with the source for another try.
    """

    converted = pyqtSignal(str)
    delta = pyqtSignal(str, str)

    def __init__(self, source: str, config, path, retry: str = "", parent=None):
        super().__init__(path, parent)
        self.source = source
        self.config = config
        self.retry = retry

    def load(self) -> None:
        self.converted.emit(
            lyrics.convert(
                self.source,
                base_url=self.config.api_base,
                api_key=self.config.api_key,
                model=self.config.model,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
                retry=self.retry,
                stream=True,
                on_delta=lambda kind, text: self.delta.emit(kind, text),
            )
        )


class LyricsWatcher(QObject):
    """Follow a `.krc` and its folder, so a change made outside the editor is seen.

    A save replaces the file rather than writing it in place, which drops the watch Qt holds on the
    path, so every event puts the watches back; the text is not compared here but by the caller,
    which keeps our own write from reading back as somebody else's.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._notify)
        self._watcher.directoryChanged.connect(self._notify)
        self._paths: tuple[str, ...] = ()

    def watch(self, path: str | Path | None) -> None:
        self._release()
        if path is None:
            return
        target = Path(path)
        self._paths = (str(target), str(target.parent))
        self._readd()

    def _release(self) -> None:
        if self._paths:
            self._watcher.removePaths(list(self._paths))
        self._paths = ()

    def _readd(self) -> None:
        self._watcher.addPaths([watch for watch in self._paths if Path(watch).exists()])

    def _notify(self, _path: str) -> None:
        self._readd()
        self.changed.emit()


def _fold_button(title: str, tooltip: str) -> QToolButton:
    """A heading that folds its body away.

    Not checkable: the widget styles paint a checked tool button as a pressed toggle, which reads as
    a switch rather than as the heading over the panel.
    """
    button = QToolButton()
    button.setText(title)
    button.setToolTip(tooltip)
    button.setAutoRaise(True)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setArrowType(Qt.ArrowType.RightArrow)
    return button


class LyricsDialog(QDialog):
    """The `.krc` beside the project, and the one pipeline that fills it.

    `Save` writes the box at the top to the temporary KRC working copy; the plain text below is the only
    reason a file that is not a `.krc` is read at all, and it folds away once there are lyrics to
    work on.
    """

    saved = pyqtSignal(str)
    open_requested = pyqtSignal()

    def __init__(self, path: str | Path, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Lyrics"))
        self.resize(680, 640)
        self.path = Path(path)
        self.config = config
        self.converter: LyricsConverter | None = None
        self._rejected: tuple[str, str] | None = None
        self._attempts = 0
        self._log_kind: str | None = None

        self.path_label = QLabel(str(self.path))
        self.path_label.setToolTip(tr("The lyrics file beside the project, written as .krc"))

        self.result = QPlainTextEdit()
        self.result.setPlaceholderText(tr("The lyrics the project sings: import a .krc, or annotate plain text below"))
        self.result.setPlainText(lyrics.load(self.path))
        self._saved_text = self.result.toPlainText()

        self.load_krc_button = QPushButton(tr("Import .krc…"))
        self.load_krc_button.setIcon(icons.icon("import"))
        self.load_krc_button.setToolTip(tr("Replace these lyrics with another .krc"))
        self.load_krc_button.clicked.connect(self._load_krc)
        self.save_button = QPushButton(tr("Save"))
        self.save_button.setIcon(icons.icon("save"))
        self.save_button.setToolTip(tr("Write the lyrics to the project's .krc"))
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._save)
        self.open_button = QPushButton(tr("Open in external editor"))
        self.open_button.setIcon(icons.icon("external"))
        self.open_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.open_button.setToolTip(tr("Open the lyrics file with the editor named in the settings"))
        self.open_button.clicked.connect(self.open_requested)

        krc = QGroupBox(tr("Project lyrics (.krc)"))
        krc.setToolTip(tr("The lyrics the project sings, kept beside it as a .krc"))
        krc_buttons = QHBoxLayout()
        krc_buttons.addWidget(self.load_krc_button)
        krc_buttons.addStretch(1)
        krc_buttons.addWidget(self.open_button)
        krc_buttons.addWidget(self.save_button)
        krc_body = QVBoxLayout(krc)
        krc_body.addWidget(self.result, 1)
        krc_body.addLayout(krc_buttons)

        self.source = QPlainTextEdit()
        self.source.setPlaceholderText(tr("Paste plain-text lyrics to annotate"))
        self.source.setFixedHeight(LOAD_HEIGHT)
        self.load_text_button = QPushButton(tr("Load text…"))
        self.load_text_button.setIcon(icons.icon("loadtext"))
        self.load_text_button.setToolTip(tr("Load a plain-text lyrics file to annotate"))
        self.load_text_button.clicked.connect(self._load_source)
        self.configured = bool(config.api_base.strip() and config.api_key.strip() and config.model.strip())
        self.convert_button = QPushButton(tr("Convert with the API"))
        self.convert_button.setIcon(icons.icon("translate"))
        self.convert_button.setEnabled(self.configured)
        self.convert_button.setToolTip(
            tr("Ask the endpoint in the settings to annotate the text above")
            if self.configured
            else tr("Set the API base, key and model in the settings first")
        )
        self.convert_button.clicked.connect(self._convert)
        self.copy_button = QPushButton(tr("Copy prompt"))
        self.copy_button.setIcon(icons.icon("copy"))
        self.copy_button.setToolTip(tr("Put the prompt and the lyrics on the clipboard, for a web model"))
        self.copy_button.clicked.connect(self._copy_prompt)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(LOG_HEIGHT)
        self.log.setPlaceholderText(tr("The model's own output, streamed as it arrives"))
        self.log.hide()
        self.hint = QLabel(
            tr("No API is set up: copy the prompt into a web model, then paste its answer into the box above.")
        )
        self.hint.setWordWrap(True)
        self.hint.setVisible(not self.configured)

        text_buttons = QHBoxLayout()
        text_buttons.addWidget(self.load_text_button)
        text_buttons.addStretch(1)
        text_buttons.addWidget(self.copy_button)
        text_buttons.addWidget(self.convert_button)

        plain = QWidget()
        plain_body = QVBoxLayout(plain)
        plain_body.setContentsMargins(0, 0, 0, 0)
        plain_body.addWidget(self.source)
        plain_body.addWidget(self.log)
        plain_body.addWidget(self.hint)
        plain_body.addLayout(text_buttons)

        self.plain_toggle = _fold_button(
            tr("Annotate plain text (optional)"),
            tr("Turns plain lyrics into the .krc above: the model adds the rubies"),
        )
        self.plain_body = plain
        # an empty box is the one case where the plain-text route is the thing to do
        self.plain_open = not self._saved_text.strip()
        self.plain_toggle.clicked.connect(self._toggle_plain)
        self._render_plain()

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.close_button = QPushButton(tr("Close"))
        self.close_button.clicked.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addWidget(krc, 3)
        layout.addWidget(self.plain_toggle)
        layout.addWidget(plain, 2)
        layout.addLayout(bottom)

    def _toggle_plain(self) -> None:
        self.plain_open = not self.plain_open
        self._render_plain()

    def _render_plain(self) -> None:
        self.plain_toggle.setArrowType(Qt.ArrowType.DownArrow if self.plain_open else Qt.ArrowType.RightArrow)
        self.plain_body.setVisible(self.plain_open)

    def _load_krc(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            tr("Import a .krc"),
            str(self.path.parent),
            tr("Lyrics file (*.krc)") + ";;" + tr("All files (*)"),
        )
        if not chosen:
            return
        text = lyrics.load(chosen)
        if not text:
            self.status_label.setText(tr("That file could not be read"))
            return
        self.result.setPlainText(text)
        self.status_label.setText(tr("Imported {name}; Save writes it into the project", name=Path(chosen).name))

    def _load_source(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            tr("Load plain-text lyrics"),
            str(self.path.parent),
            tr("Text file ({patterns})", patterns="*.txt *.md *.lrc") + ";;" + tr("All files (*)"),
        )
        if not chosen:
            return
        text = lyrics.load(chosen)
        if not text:
            self.status_label.setText(tr("That file could not be read"))
            return
        self.source.setPlainText(text)
        self.status_label.setText(tr("Loaded {name} — convert it, or copy the prompt", name=Path(chosen).name))

    def _copy_prompt(self) -> None:
        source = self.source.toPlainText()
        prompt = lyrics.build_prompt(source)
        if self._rejected is not None:
            prompt = lyrics.build_prompt(source, retry=lyrics.retry_prompt(*self._rejected))
        QGuiApplication.clipboard().setText(prompt)
        self.status_label.setText(
            tr("Prompt copied: paste it into a web model, then paste its answer into the box above and save")
        )

    def _convert(self) -> None:
        if not self.source.toPlainText().strip():
            self.status_label.setText(tr("Paste the lyrics to annotate first"))
            return
        retry = lyrics.retry_prompt(*self._rejected) if self._rejected is not None else ""
        self._attempts = 1 if retry else 0
        self._request(retry)

    def _request(self, retry: str) -> None:
        self._set_running(True)
        if retry:
            self.log.appendPlainText("")
        else:
            self.log.clear()
        self.log.show()
        self._log_kind = None
        converter = LyricsConverter(self.source.toPlainText().strip(), self.config, self.path, retry=retry, parent=self)
        self.converter = converter
        converter.converted.connect(self._converted)
        converter.delta.connect(self._on_delta)
        converter.failed.connect(self._failed)
        converter.start()

    def _on_delta(self, kind: str, text: str) -> None:
        """Show the model's own stream, starting the answer on a line of its own."""
        if self._log_kind is not None and kind != self._log_kind:
            self.log.appendPlainText("")
        self._log_kind = kind
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)

    def _converted(self, text: str) -> None:
        self._set_running(False)
        source = self.converter.source if self.converter is not None else ""
        problem = lyrics.conversion_error(text, source)
        if not problem:
            self._rejected = None
            self.result.setPlainText(text)
            self.status_label.setText(tr("Converted: check it over, then save"))
            return
        self._rejected = (text, problem)
        self._attempts += 1
        if self._attempts < MAX_ATTEMPTS:
            self.status_label.setText(tr("Not annotated: {error} — asking the model to correct it", error=problem))
            self._request(lyrics.retry_prompt(text, problem))
            return
        QMessageBox.warning(
            self,
            tr("Lyrics"),
            tr("The model did not annotate the lyrics: {error}", error=problem),
        )
        self.status_label.setText(
            tr("The answer was not applied — copy the prompt to try again: {error}", error=problem)
        )

    def _failed(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText(tr("Conversion failed: {error}", error=message))

    def _set_running(self, running: bool) -> None:
        self.convert_button.setEnabled(self.configured and not running)
        self.convert_button.setText(tr("Converting …") if running else tr("Convert with the API"))

    def _save(self) -> None:
        text = self.result.toPlainText()
        source = self.source.toPlainText().strip()
        problem = lyrics.conversion_error(text, source) if source else lyrics.syntax_error(text)
        if problem:
            if source:
                self._rejected = (text, problem)
            QMessageBox.warning(
                self,
                tr("Lyrics"),
                tr("The lyrics are not the plain text annotated: {error}", error=problem)
                if source
                else tr("The lyrics are not a readable .krc: {error}", error=problem),
            )
            self.status_label.setText(tr("Not saved: {error}", error=problem))
            return
        try:
            lyrics.save(self.path, text)
        except OSError as error:
            self.status_label.setText(tr("Could not save: {error}", error=error))
            return
        self._rejected = None
        self._saved_text = text
        self.status_label.setText(tr("Saved to {name}", name=self.path.name))
        self.saved.emit(text)

    def _confirm_close(self) -> bool:
        """A close that would drop unsaved lyrics asks first; a failed save keeps the window open."""
        if self.result.toPlainText() == self._saved_text:
            return True
        answer = QMessageBox.question(
            self,
            tr("Lyrics"),
            tr("Save the lyrics to {name} before closing?", name=self.path.name),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            self._save()
            return self.result.toPlainText() == self._saved_text
        return True

    def _detach(self) -> None:
        """A request already on its way cannot be recalled, so it is left to finish unobserved."""
        converter, self.converter = self.converter, None
        if converter is None:
            return
        converter.blockSignals(True)
        if converter.isRunning():
            converter.setParent(QApplication.instance())

    def reject(self) -> None:
        if not self._confirm_close():
            return
        self._detach()
        super().reject()
