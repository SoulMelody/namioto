# SPDX-License-Identifier: AGPL-3.0-only
"""RGB editors with a live swatch and a native colour chooser."""

from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QColorDialog, QHBoxLayout, QPushButton, QWidget

from namioto.i18n import tr


def bind_swatch(editor, picker):
    def refresh():
        color = QColor("#" + editor.text().strip().lstrip("#"))
        if color.isValid():
            pixmap = QPixmap(20, 20)
            pixmap.fill(color)
            picker.setIcon(QIcon(pixmap))
        else:
            picker.setIcon(QIcon())

    editor.textChanged.connect(refresh)
    refresh()
    return refresh


def colour_row(editor, parent=None):
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(editor, 1)
    picker = QPushButton(tr("Choose colour"))
    layout.addWidget(picker)
    bind_swatch(editor, picker)

    def choose():
        color = QColorDialog.getColor(QColor("#" + editor.text().strip().lstrip("#")), row)
        if color.isValid():
            editor.setText(color.name().upper())

    picker.clicked.connect(choose)
    return row
