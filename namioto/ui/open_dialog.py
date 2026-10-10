# SPDX-License-Identifier: AGPL-3.0-only
"""The window that opens a sound: where its project is saved, and how the spectrum is made.

The values below the path are the analysis section of `namioto.project.ProjectSettings`, rendered
from the same table the project file reads and writes, so a field's default, range and caption live
in one place. The window edits a copy, so a cancelled dialog leaves the document untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from namioto import params, project
from namioto.i18n import tr
from namioto.ui.form import add_row, field_editor


class OpenAudioDialog(QDialog):
    """Where a new project goes and how its audio is analysed."""

    def __init__(self, audio: str, document: project.ProjectSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Open audio"))
        self._document = params.clone(document)
        self._rows: list[tuple[str, QWidget, Callable[[], Any]]] = []

        self.path = QLineEdit(str(Path(audio).with_suffix(project.SUFFIX)))
        browse = QPushButton(tr("Browse…"))
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(browse)

        form = QFormLayout()
        form.addRow(QLabel(tr("Project file")), row)
        section = next(item for item in project.PROJECT_SECTIONS if item.name == "analysis")
        form.addRow(QLabel(tr(section.title)))
        for field in section.fields:
            widget, read, _write = field_editor(params.get_value(self._document, section.name, field.name), field)
            add_row(form, widget, field)
            self._rows.append((field.name, widget, read))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(bool(self.path.text()))
        self.path.textChanged.connect(lambda text: ok.setEnabled(bool(text)))

        column = QVBoxLayout(self)
        column.addLayout(form)
        column.addWidget(buttons)

    def _browse(self) -> None:
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            tr("Save project"),
            self.path.text(),
            tr("Namioto project (*{suffix})", suffix=project.SUFFIX),
        )
        if chosen:
            self.path.setText(chosen)

    def document(self) -> project.ProjectSettings:
        """The values the form holds, as the document a new project is saved with."""
        for name, _widget, read in self._rows:
            params.set_value(self._document, "analysis", name, read())
        return self._document

    def target(self) -> Path:
        """Where the project goes, carrying the suffix whatever the path field holds."""
        target = Path(self.path.text()).expanduser()
        return target if project.looks_like_project(target) else target.with_name(target.name + project.SUFFIX)
