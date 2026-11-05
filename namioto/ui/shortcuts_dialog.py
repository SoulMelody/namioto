# SPDX-License-Identifier: AGPL-3.0-only
"""The shortcuts help: the table from `namioto.shortcuts`, drawn. No key is named here.

The window used to put a sentence about the keys in its status bar at startup. That sentence could
not stay true by construction - every new gesture had to be added to it by hand - it overflowed a
narrow window, and, because the status bar was permanently occupied, it hid every message the
window said afterwards. A dialog is read when it is asked for, has room for more than one line,
and leaves the status bar free; the table behind it is checked by a test that needs no widgets.

The page per group is the same shape `namioto.ui.settings_dialog` uses: `QTabWidget` plus
`addTab(page, tr(title))`, so a group is a tab and a shortcut is a row. The keys column is
monospace because the entries are read as key caps, and a proportional font would make `Ctrl+Wheel`
and `Ctrl+Shift+Wheel` line up differently on every row.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from namioto import shortcuts
from namioto.i18n import tr


class ShortcutsDialog(QDialog):
    """One tab per group, one row per shortcut, with a single Close button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Keyboard shortcuts"))
        # a readable sheet rather than one long row: the table is wide, and a dialog sized by its
        # content would stretch past the screen on the longest action
        self.resize(640, 480)

        self.pages = QTabWidget()
        for group in shortcuts.GROUPS:
            self.pages.addTab(self._page(group), tr(group.title))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.pages)
        layout.addWidget(buttons)

    def _page(self, group: shortcuts.Group) -> QWidget:
        """The keys on the left and the action on the right, one group's worth."""
        page = QWidget()
        form = QFormLayout(page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        keys_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        for shortcut in group.shortcuts:
            keys = QLabel(shortcut.keys)
            keys.setFont(keys_font)
            keys.setMargin(4)
            action = QLabel(tr(shortcut.action))
            action.setWordWrap(True)  # a long action wraps instead of widening the dialog
            form.addRow(keys, action)
        return page


def wire(window) -> QAction:
    """An action that opens the help over `window`.

    Like `properties_dialog.wire`, the dialog is built when it opens: the table is fixed, but a
    dialog kept alive would outlive the window it was parented to.
    """

    def show() -> None:
        ShortcutsDialog(window).exec()

    action = QAction(tr("Keyboard shortcuts…"), window)
    action.triggered.connect(show)
    return action
