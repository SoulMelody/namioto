# SPDX-License-Identifier: AGPL-3.0-only
"""Single-track subtitle export options, edited on a copy until the file is written."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
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
from namioto.i18n import tr
from namioto.karaoke import AssSettings
from namioto.ui.form import add_row, field_editor

GROUPS = (
    ("Timing", ("offset_ms", "lead_time_ms", "fade_in_ms", "fade_out_ms", "guide_dot_duration_ms")),
    ("Font and layout", ("font", "font_size", "margin_h", "margin_v", "ruby_offset")),
    (
        "Colours and effects",
        (
            "overlay_color",
            "base_outline_color",
            "overlay_outline_color",
            "overlay_blur_color",
            "base_blur_color",
            "border",
            "border_furi",
            "blur",
            "blur_scale",
            "clip_size",
        ),
    ),
)


class AssDialog(QDialog):
    """The export form; accepting it does not change the program's saved preferences."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Export ASS subtitle"))
        self._settings = store.clone(settings)
        self.editors = {}
        self._readers = {}
        self._writers = {}
        self.automatic = {}
        fields = {field.name: field for section in store.SECTIONS if section.name == "ass" for field in section.fields}
        pages = QTabWidget()
        for title, names in GROUPS:
            page = QWidget()
            form = QFormLayout(page)
            for name in names:
                field = fields[name]
                editor, read, write = field_editor(store.get_value(settings, "ass", name), field)
                self.editors[name] = editor
                self._readers[name] = read
                self._writers[name] = write
                row = self._colour_row(name, editor) if name.endswith("_color") else editor
                add_row(form, row, field)
                if isinstance(editor, QLineEdit):
                    editor.textChanged.connect(self._validate)
                else:
                    editor.valueChanged.connect(self._validate)
            pages.addTab(page, tr(title))

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.export_button = buttons.addButton(tr("Export"), QDialogButtonBox.ButtonRole.AcceptRole)
        restore = buttons.addButton(tr("Restore defaults"), QDialogButtonBox.ButtonRole.ResetRole)
        restore.clicked.connect(self.restore_defaults)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(pages)
        layout.addWidget(self.hint)
        layout.addWidget(buttons)
        self._validate()

    def _colour_row(self, name, editor):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(editor, 1)
        picker = QPushButton(tr("Choose colour"))
        picker.clicked.connect(lambda: self._choose_colour(editor))
        layout.addWidget(picker)
        if name in ("overlay_blur_color", "base_blur_color"):
            automatic = QCheckBox(tr("Automatic"))
            automatic.setChecked(not editor.text())
            editor.setEnabled(not automatic.isChecked())
            picker.setEnabled(not automatic.isChecked())
            automatic.toggled.connect(lambda checked: editor.setEnabled(not checked))
            automatic.toggled.connect(lambda checked: picker.setEnabled(not checked))
            automatic.toggled.connect(self._validate)
            self.automatic[name] = automatic
            layout.addWidget(automatic)
        return row

    def _choose_colour(self, editor):
        colour = QColorDialog.getColor(QColor("#" + editor.text().lstrip("#")), self)
        if colour.isValid():
            editor.setText(colour.name()[1:].upper())

    def values(self):
        for name, read in self._readers.items():
            value = read()
            if name in self.automatic and self.automatic[name].isChecked():
                value = ""
            store.set_value(self._settings, "ass", name, value)
        return self._settings

    def _validate(self, *_):
        settings = AssSettings(**vars(self.values().ass))
        error = ""
        if not settings.font or any(char in settings.font for char in ",\n\r"):
            error = tr("Enter a font name without commas or line breaks")
        elif settings.fade_in_ms > settings.lead_time_ms:
            error = tr("Fade in must not exceed the lead time")
        else:
            for name in self.editors:
                if not name.endswith("_color"):
                    continue
                value = getattr(settings, name)
                if name in self.automatic and self.automatic[name].isChecked():
                    continue
                value = value.lstrip("#")
                if len(value) != 6 or any(char not in "0123456789abcdefABCDEF" for char in value):
                    error = tr("Enter colours as six-digit RGB hex values")
                    break
        if not error:
            for name, colour in zip(("overlay_blur_color", "base_blur_color"), settings.blur_colours(), strict=True):
                if self.automatic[name].isChecked():
                    editor = self.editors[name]
                    editor.blockSignals(True)
                    editor.setText(colour)
                    editor.blockSignals(False)
        self.hint.setText(error)
        self.export_button.setEnabled(not error)

    def restore_defaults(self):
        defaults = store.Settings()
        for name, write in self._writers.items():
            write(getattr(defaults.ass, name))
        for automatic in self.automatic.values():
            automatic.setChecked(True)
        self._validate()

    def _accept(self):
        self._validate()
        if self.export_button.isEnabled():
            self.accept()
