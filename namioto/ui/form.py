# SPDX-License-Identifier: AGPL-3.0-only
"""The pieces a form of settings is made of: one row per field, and the widgets that edit one.

The settings window and the align and transcription windows are all a `QFormLayout` of `Field`s, so
the row, its editors and the fold-away Advanced heading live here once. `namioto.params.Field` is the
only thing the code below reads, so a new kind of value is a branch in `field_editor` and nothing
else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStyleFactory,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from namioto.analysis import devices
from namioto.i18n import tr
from namioto.params import Field
from namioto.ui import theme

FIELD_WIDTH = 300  # a form of numbers that stretch across the page is hard to read


def add_row(form: QFormLayout, editor: QWidget, field: Field) -> None:
    """One row of a form: the spec's caption and tooltip, then the widget that edits its value.

    The field column grows to the rows' width whatever the style says: Breeze's style hint would
    keep a combo at its size hint, leaving the controls of one form unaligned.
    """
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    editor.setMaximumWidth(FIELD_WIDTH)
    label = QLabel(tr(field.caption))
    if field.tooltip:
        label.setToolTip(tr(field.tooltip))
        editor.setToolTip(tr(field.tooltip))
    if field.kind == "device":
        form.addRow(label, device_row(editor))
        return
    form.addRow(label, editor)


class WrappedLabel(QLabel):
    """A wrapped line that keeps itself as tall as the lines it actually has.

    A word-wrapped `QLabel` inside a form row is given only the one-line height its `sizeHint`
    carries, so the rest of the text is clipped. The height the text needs at the label's real width
    is forced as a minimum whenever either changes, which makes the row grow to fit.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWordWrap(True)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def setText(self, text: str) -> None:
        super().setText(text)
        self._fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self.width() <= 0:
            return
        needed = self.heightForWidth(self.width())
        if needed > 0 and needed != self.minimumHeight():
            self.setMinimumHeight(needed)


def device_row(editor: QComboBox) -> QWidget:
    """A device combo with a line under it saying whether the runtime it needs is installed."""
    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(2)
    box.addWidget(editor)
    status = WrappedLabel()
    box.addWidget(status)
    editor.currentIndexChanged.connect(lambda *_: _show_device_status(editor, status))
    _show_device_status(editor, status)
    return holder


def _show_device_status(editor: QComboBox, status: QLabel) -> None:
    """What the device the row names needs, and whether this machine has it."""
    chosen = editor.currentData()
    if chosen in devices.GPU_KEYS:
        key = chosen
    else:
        found = [name for name in devices.GPU_KEYS if devices.available(name)]
        if not found:
            status.setText(tr("No GPU backend is available; a GPU run falls back to the CPU"))
            status.setToolTip("")
            _warn(status)
            return
        key = found[0]
    device = devices.get(key)
    name = device.runtime or device.label
    if devices.available(key):
        status.setText(tr("{name} is available", name=name))
        status.setToolTip("")
        status.setStyleSheet("")
        return
    missing = ", ".join(devices.missing(key)) or device.providers[0]
    status.setText(tr("{name} is not available: {missing} is missing", name=name, missing=missing))
    status.setToolTip(tr(device.hint))
    _warn(status)


def _warn(status: QLabel) -> None:
    status.setStyleSheet(f"color: {theme.canvas().note_selected_edge.name()};")


def _read(widget: QWidget) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QDoubleSpinBox):  # before QSpinBox's sibling check, they do not nest
        return widget.value()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QComboBox):
        return widget.currentData()
    raise TypeError(f"no way to read a {type(widget).__name__}")


def _write(widget: QWidget, value: Any) -> None:
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, QComboBox):
        widget.setCurrentIndex(max(0, widget.findData(value)))
    else:
        raise TypeError(f"no way to fill a {type(widget).__name__}")


def field_editor(value: Any, field: Field) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
    """The widget one setting is edited with, plus how to read it back and how to fill it in."""
    if field.kind == "bool":
        widget = QCheckBox()
    elif field.kind in ("choice", "device"):
        labels = field.labels or tuple(str(choice) for choice in field.choices)
        return combo_editor(list(zip((tr(label) for label in labels), field.choices, strict=True)), value)
    elif field.kind == "style":
        return style_editor(value)
    elif field.kind == "int":
        widget = QSpinBox()
        widget.setRange(int(field.low), int(field.high))
        widget.setSingleStep(max(1, int(field.step) or 1))
        widget.setSuffix(field.suffix)
    elif field.kind == "float":
        widget = QDoubleSpinBox()
        widget.setRange(field.low, field.high)
        widget.setDecimals(field.decimals)
        widget.setSingleStep(field.step or 0.1)
        widget.setSuffix(field.suffix)
        widget.setKeyboardTracking(False)
    elif field.kind == "text":
        widget = QLineEdit()
    elif field.kind == "secret":
        widget = QLineEdit()
        widget.setEchoMode(QLineEdit.EchoMode.Password)
    else:
        raise TypeError(f"no editor for a {field.kind} field")
    _write(widget, value)
    return widget, lambda widget=widget: _read(widget), lambda new, widget=widget: _write(widget, new)


def combo_editor(entries: Sequence[tuple[str, Any]], value: Any):
    widget = QComboBox()
    for caption, data in entries:
        widget.addItem(caption, data)
    _write(widget, value)
    return widget, lambda widget=widget: _read(widget), lambda new, widget=widget: _write(widget, new)


def style_editor(value: Any):
    """Every widget style this build can draw with, the one the desktop hands out named first."""
    available = QStyleFactory.keys()
    entries = [(tr("System default ({name})", name=theme.platform_style()), "")]
    entries += [(name, name) for name in available]
    return combo_editor(entries, value)


def advanced_section(form: QFormLayout) -> QToolButton:
    """The heading over a form of advanced rows, folded away until it is clicked."""
    button = QToolButton()
    button.setText(tr("Advanced"))
    button.setCheckable(True)
    button.setAutoRaise(True)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setArrowType(Qt.ArrowType.RightArrow)

    def reveal(open: bool) -> None:
        button.setArrowType(Qt.ArrowType.DownArrow if open else Qt.ArrowType.RightArrow)
        for row in range(form.rowCount()):
            form.setRowVisible(row, open)

    button.toggled.connect(reveal)
    reveal(False)
    return button
