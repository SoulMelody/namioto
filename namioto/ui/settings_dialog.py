# SPDX-License-Identifier: AGPL-3.0-only
"""The settings window, and the store that keeps the file in step with what is running.

The window is built from `namioto.settings`: one page per section named below, one row per field of
it. `editor` holds the switches the program remembers by itself; `ass` is edited in the subtitle
export form instead.

`field_editor`, `add_row` and `advanced_section` come from `namioto.ui.form`, which the align and
transcription windows share.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from namioto import settings as store
from namioto.channels import valid_color
from namioto.i18n import tr
from namioto.lyric_colors import DEFAULT_COLORS
from namioto.params import Field
from namioto.ui.colors import colour_row
from namioto.ui.form import add_row, advanced_section, field_editor

SAVE_DELAY_MS = 1000

PAGES = ("general", "devices", "tempo", "lyrics", "network", "midi", "remote")
# the rows a page folds away under its Advanced heading
ADVANCED = {
    "tempo": ("window_seconds", "window_hop_seconds"),
    "lyrics": ("temperature", "timeout"),
}


class SettingsStore(QObject):
    """The settings the program runs with, written out once the changes stop coming."""

    changed = Signal(object)
    failed = Signal(str)

    def __init__(self, settings, path=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.path = path or store.default_path()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SAVE_DELAY_MS)
        self._timer.timeout.connect(self.flush)

    def touch(self) -> None:
        """Something changed, and more may follow: write it out when they settle."""
        self._timer.start()

    def apply(self, settings) -> None:
        """Take whole settings over, at once, from the settings window."""
        self.settings = settings
        self._timer.stop()  # anything still pending is older than what is being applied
        self.changed.emit(settings)
        self.flush()

    def flush(self) -> None:
        self._timer.stop()
        try:
            store.save(self.settings, self.path)
        except OSError as error:  # a read-only home must not take the editor down
            self.failed.emit(tr("Settings could not be saved: {error}", error=error))


class SettingsDialog(QDialog):
    """Every preference, on a page per section, over a copy that is only handed over when applied."""

    applied = Signal(object)
    lyric_colors_applied = Signal(object)

    def __init__(
        self,
        settings,
        *,
        lyric_colors=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("Settings"))
        self.resize(560, 460)
        self._settings = store.clone(settings)
        self._rows: list[tuple[str, Field, Callable[[], Any], Callable[[Any], None]]] = []
        self._sections = {section.name: section for section in store.SECTIONS}

        pages = QTabWidget()
        for name in PAGES:
            section = self._sections[name]
            pages.addTab(self._page(section), tr(section.title))

        self.color_editors = []
        if lyric_colors is not None:
            page = QWidget()
            form = QFormLayout(page)
            for number, color in enumerate(lyric_colors):
                editor = QLineEdit(color)
                self.color_editors.append(editor)
                form.addRow(tr("Channel {number}", number=number), colour_row(editor))
            restore_colors = QPushButton(tr("Restore default colours"))
            restore_colors.clicked.connect(self._restore_colors)
            form.addRow(restore_colors)
            pages.addTab(page, tr("Project lyric colours"))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        restore = buttons.addButton(tr("Restore defaults"), QDialogButtonBox.ButtonRole.ResetRole)
        restore.clicked.connect(self.restore_defaults)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(pages)
        layout.addWidget(self._path_hint())
        layout.addWidget(buttons)
        self._buttons = buttons
        for editor in self.color_editors:
            editor.textChanged.connect(self._validate_colors)
        self._validate_colors()

    def _restore_colors(self):
        for editor, color in zip(self.color_editors, DEFAULT_COLORS, strict=True):
            editor.setText(color)

    def _validate_colors(self):
        valid = all(valid_color(editor.text().strip()) for editor in self.color_editors)
        for role in (QDialogButtonBox.StandardButton.Ok, QDialogButtonBox.StandardButton.Apply):
            self._buttons.button(role).setEnabled(valid)

    def values(self):
        """The copy, with everything the widgets hold read back into it."""
        for section, field, read, _write_value in self._rows:
            store.set_value(self._settings, section, field.name, read())
        return self._settings

    def apply(self) -> None:
        if not all(valid_color(editor.text().strip()) for editor in self.color_editors):
            return
        self.applied.emit(self.values())
        if self.color_editors:
            self.lyric_colors_applied.emit(tuple(editor.text().strip() for editor in self.color_editors))

    def restore_defaults(self) -> None:
        ass = self._settings.ass
        self._settings = store.Settings()
        self._settings.ass = ass
        for _section, field, _read_value, write in self._rows:
            write(field.default)

    def _accept(self) -> None:
        self.apply()
        self.accept()

    def _page(self, section) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        advanced_names = ADVANCED.get(section.name, ())
        for advanced in (False, True):
            rows = [field for field in section.fields if (field.name in advanced_names) is advanced]
            if not rows:
                continue
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            for field in rows:
                editor, read, write = self._editor(section.name, field)
                self._rows.append((section.name, field, read, write))
                add_row(form, editor, field)
            if advanced:
                layout.addWidget(advanced_section(form))
            layout.addLayout(form)
        layout.addStretch(1)
        return widget

    def _path_hint(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(tr("Configuration file")))
        path = QLineEdit(str(store.default_path()))
        path.setReadOnly(True)
        layout.addWidget(path, 1)
        folder = QPushButton(tr("Show folder"))
        folder.setToolTip(tr("Open the directory holding the settings file"))
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(store.default_path().parent))))
        layout.addWidget(folder)
        return widget

    def _editor(self, section: str, field: Field) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        return field_editor(store.get_value(self._settings, section, field.name), field)
