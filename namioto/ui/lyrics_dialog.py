# SPDX-License-Identifier: AGPL-3.0-only
"""The lyrics window: the open project's lyric text, edited in one box.

The box holds the words the notes of one channel sing, one word per note, separated by whitespace;
nothing is read from or written to a file here. `Save` hands the text back through `saved`, and what
adopting it does to the notes is the caller's to decide. A close that would drop unsaved edits asks
first.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from namioto.i18n import tr
from namioto.ui import icons


class LyricsDialog(QDialog):
    """The lyric text of the open project, in a box that hands it back on `Save`."""

    saved = Signal(str)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Lyrics"))
        self.resize(680, 600)

        self.editor = QPlainTextEdit()
        self.editor.setPlainText(text)
        self._saved_text = text

        self.save_button = QPushButton(tr("Save"))
        self.save_button.setIcon(icons.icon("save"))
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor, 1)
        layout.addLayout(buttons)

    def accept(self) -> None:
        """Hand the text back and close; the caller reads it, checks it and adopts it."""
        self._saved_text = self.editor.toPlainText()
        self.saved.emit(self._saved_text)
        super().accept()

    def reject(self) -> None:
        if self._confirm_close():
            super().reject()

    def _confirm_close(self) -> bool:
        """A close over unsaved edits asks first; `Save` takes the answer as the save it asked for."""
        if self.editor.toPlainText() == self._saved_text:
            return True
        answer = QMessageBox.question(
            self,
            tr("Lyrics"),
            tr("Save the lyrics before closing?"),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            self.accept()
            return False
        return True
