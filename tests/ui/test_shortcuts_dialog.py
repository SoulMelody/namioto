# SPDX-License-Identifier: AGPL-3.0-only
"""The help dialog draws the table and adds nothing: a page per group, a row per shortcut."""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel

from namioto import shortcuts
from namioto.i18n import tr
from namioto.ui.shortcuts_dialog import ShortcutsDialog, wire


def _form(dialog: ShortcutsDialog, index: int) -> QFormLayout:
    return dialog.pages.widget(index).layout()


def _row(form: QFormLayout, row: int) -> tuple[QLabel, QLabel]:
    label = form.itemAt(row, QFormLayout.ItemRole.LabelRole).widget()
    field = form.itemAt(row, QFormLayout.ItemRole.FieldRole).widget()
    return label, field


def test_a_page_per_group_named_after_it() -> None:
    dialog = ShortcutsDialog()
    assert dialog.pages.count() == len(shortcuts.GROUPS)
    assert [dialog.pages.tabText(index) for index in range(dialog.pages.count())] == [
        tr(group.title) for group in shortcuts.GROUPS
    ]


def test_each_page_has_one_row_per_shortcut() -> None:
    dialog = ShortcutsDialog()
    for index, group in enumerate(shortcuts.GROUPS):
        form = _form(dialog, index)
        assert form.rowCount() == len(group.shortcuts)
        for row, shortcut in enumerate(group.shortcuts):
            keys, action = _row(form, row)
            # key caps stay English: `shortcuts.py` does not hand them to `tr()`
            assert keys.text() == shortcut.keys
            assert action.text() == tr(shortcut.action)


def test_nothing_is_drawn_blank_so_no_translation_is_missing() -> None:
    dialog = ShortcutsDialog()
    for index in range(len(shortcuts.GROUPS)):
        form = _form(dialog, index)
        for row in range(form.rowCount()):
            keys, action = _row(form, row)
            assert keys.text().strip()
            assert action.text().strip()


def test_the_menu_action_opens_the_same_sheet() -> None:
    action = wire(None)
    assert action.text() == tr("Keyboard shortcuts…")
    assert action.isEnabled()
