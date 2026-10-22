# SPDX-License-Identifier: AGPL-3.0-only
"""The settings window, and the store that keeps the file in step with what is running.

The window is built from `namioto.settings`: one page per section named below, one row per field of
it. `editor` holds the switches the program remembers by itself; `ass` is edited in the subtitle
export form instead.

`field_editor`, `add_row` and `advanced_section` come from `namioto.ui.form`, which the align and
transcription windows share.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QLibraryInfo, QObject, Qt, QTimer, QUrl, Signal
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

from namioto import qtplugins
from namioto import settings as store
from namioto.i18n import tr
from namioto.params import Field
from namioto.ui.form import add_row, advanced_section, field_editor

SAVE_DELAY_MS = 1000

PAGES = ("general", "devices", "tempo", "lyrics", "network", "midi", "remote")
# the rows a page folds away under its Advanced heading
ADVANCED = {
    "tempo": ("window_seconds", "window_hop_seconds"),
    "lyrics": ("temperature", "timeout"),
}


def system_plugins_can_help() -> bool:
    """Whether this program's Qt could be missing the plugins the machine's own Qt has.

    A PySide6 the distribution built for its own Qt searches the system directory already, and only
    Linux has a Qt to find in the first place; either way the switch would change nothing.
    """
    if sys.platform != "linux":
        return False
    own = own_plugins_dir()
    return not any(own.is_relative_to(path.resolve()) for path in qtplugins.system_dirs())


def own_plugins_dir() -> Path:
    """The plugin directory of the Qt this build of PySide6 draws with."""
    return Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)).resolve()


# the rows that are not built at all on a machine where the switch could change nothing


def _shown(field: Field) -> bool:
    """A row whose switch would change nothing on this machine is left out of the window."""
    return field.name != "system_qt_plugins" or system_plugins_can_help()


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

    def __init__(
        self,
        settings,
        *,
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

    def values(self):
        """The copy, with everything the widgets hold read back into it."""
        for section, field, read, _write_value in self._rows:
            store.set_value(self._settings, section, field.name, read())
        return self._settings

    def apply(self) -> None:
        self.applied.emit(self.values())

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
            rows = [field for field in section.fields if (field.name in advanced_names) is advanced and _shown(field)]
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
