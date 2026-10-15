# SPDX-License-Identifier: AGPL-3.0-only
"""Edit the project's lyric palette without changing application preferences."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout

from namioto.channels import valid_color
from namioto.i18n import tr
from namioto.lyric_colors import DEFAULT_COLORS
from namioto.ui.colors import colour_row


class PaletteDialog(QDialog):
    applied = Signal(object)

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Project lyric colours"))
        self.color_editors = []
        form = QFormLayout()
        for number, color in enumerate(colors):
            editor = QLineEdit(color)
            self.color_editors.append(editor)
            form.addRow(tr("Channel {number}", number=number), colour_row(editor))
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        restore = self._buttons.addButton(tr("Restore default colours"), QDialogButtonBox.ButtonRole.ResetRole)
        restore.clicked.connect(self._restore_colors)
        self._buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel(tr("Colours are saved with the project.")))
        layout.addWidget(self._buttons)
        for editor in self.color_editors:
            editor.textChanged.connect(self._validate_colors)
        self._validate_colors()

    def _restore_colors(self):
        for editor, color in zip(self.color_editors, DEFAULT_COLORS, strict=True):
            editor.setText(color)

    def _validate_colors(self):
        valid = all(valid_color(editor.text().strip()) for editor in self.color_editors)
        for role in (QDialogButtonBox.StandardButton.Save, QDialogButtonBox.StandardButton.Apply):
            self._buttons.button(role).setEnabled(valid)

    def apply(self):
        colors = tuple(valid_color(editor.text().strip()) for editor in self.color_editors)
        if not all(colors):
            return False
        self.applied.emit(colors)
        return True

    def _save(self):
        if self.apply():
            self.accept()
