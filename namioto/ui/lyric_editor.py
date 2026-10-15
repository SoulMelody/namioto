"""Visible ruby, direct text selection and in-place editing of the authoritative KRC."""

from copy import deepcopy

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QScrollArea,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from namioto.i18n import tr
from namioto.karaoke.model import Group, Ruby, Unit, Word, unit_words
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds, sound_lines
from namioto.karaoke.writer import dumps
from namioto.lyricmap.faithful import structure


def reading(unit):
    return ",".join("".join(inner.text for inner in part) for part in unit.ruby.parts) if unit.ruby else ""


def unit_rects(units, font):
    metrics = QFontMetrics(font)
    x = 8
    result = []
    for unit in units:
        width = max(metrics.horizontalAdvance(unit.text), metrics.horizontalAdvance(reading(unit)), 20) + 14
        result.append(QRect(x, 0, width, 58))
        x += width
    return result


def draw_units(painter, units, rects, view, palette, selected=()):
    for index, (unit, rect) in enumerate(zip(units, rects, strict=True)):
        if index in selected:
            painter.fillRect(rect, palette.highlight())
        painter.setPen(palette.highlightedText().color() if index in selected else palette.text().color())
        painter.drawText(rect.adjusted(2, 0, -2, -32), Qt.AlignmentFlag.AlignCenter, reading(unit))
        words = unit.base.words if isinstance(unit.base, Group) else [unit.base]
        metrics = painter.fontMetrics()
        x = rect.center().x() - metrics.horizontalAdvance(unit.text) // 2
        for word in words:
            painter.setPen(view.lyric_color(word.color_channel if word.color_channel is not None else word.channel))
            width = metrics.horizontalAdvance(word.char)
            painter.drawText(QRect(x, 26, width, 28), Qt.AlignmentFlag.AlignCenter, word.char)
            if word.color_channel is not None:
                painter.fillRect(QRect(x, 54, max(width, 2), 2), view.lyric_color(word.color_channel))
            x += width


class LyricDelegate(QStyledItemDelegate):
    def __init__(self, panel):
        super().__init__(panel.lines)
        self.panel = panel

    def sizeHint(self, option, index):
        return QSize(200, 62)

    def paint(self, painter, option, index):
        painter.save()
        painter.setClipRect(option.rect)
        painter.fillRect(
            option.rect,
            option.palette.highlight() if self.panel.lines.item(index.row()).isSelected() else option.palette.base(),
        )
        painter.translate(option.rect.topLeft())
        painter.setFont(option.font)
        units = self.panel.rows[index.row()].units
        selected = range(len(units)) if self.panel.lines.item(index.row()).isSelected() else ()
        draw_units(painter, units, unit_rects(units, option.font), self.panel.window.view, option.palette, selected)
        painter.restore()


class LyricList(QListWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        item = self.itemAt(event.position().toPoint())
        if item is None or event.button() != Qt.MouseButton.LeftButton:
            return
        row_rect = self.visualItemRect(item)
        point = event.position().toPoint() - row_rect.topLeft()
        units = self.panel.rows[self.row(item)].units
        for index, rect in enumerate(unit_rects(units, self.font())):
            if rect.contains(point):
                self.panel.canvas.begin_edit(
                    index, point.y() < 26, self.viewport(), rect.translated(row_rect.topLeft())
                )
                return


class LyricCanvas(QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.selected = set()
        self.anchor = None
        self.editor = QLineEdit(self)
        self.editor.hide()
        self.editor.installEventFilter(self)
        self.editor.editingFinished.connect(self.finish_edit)
        self.edit_target = None
        self.setFixedHeight(64)
        self.setToolTip(tr("Drag to select. Right-click for channels and colors. Double-click text or ruby to edit."))

    def units(self):
        selected = self.panel.lines.selectedIndexes()
        return self.panel.rows[selected[0].row()].units if len(selected) == 1 else []

    def refresh(self):
        self.cancel_edit()
        self.selected.clear()
        self.anchor = None
        rects = unit_rects(self.units(), self.font())
        self.setMinimumWidth(rects[-1].right() + 8 if rects else 0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        draw_units(
            painter,
            self.units(),
            unit_rects(self.units(), self.font()),
            self.panel.window.view,
            self.palette(),
            self.selected,
        )

    def hit(self, position):
        return next(
            (index for index, rect in enumerate(unit_rects(self.units(), self.font())) if rect.contains(position)), None
        )

    def mousePressEvent(self, event):
        index = self.hit(event.position().toPoint())
        if index is not None and event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and self.anchor is not None:
                self.selected = set(range(min(self.anchor, index), max(self.anchor, index) + 1))
            else:
                self.anchor = index
                self.selected = {index}
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        index = self.hit(event.position().toPoint())
        if event.buttons() & Qt.MouseButton.LeftButton and self.anchor is not None and index is not None:
            self.selected = set(range(min(self.anchor, index), max(self.anchor, index) + 1))
            self.update()

    def contextMenuEvent(self, event):
        index = self.hit(event.pos())
        if index is not None and index not in self.selected:
            self.selected = {index}
            self.update()
        self.panel.context_menu(event.globalPos())

    def mouseDoubleClickEvent(self, event):
        index = self.hit(event.position().toPoint())
        if index is not None and event.button() == Qt.MouseButton.LeftButton:
            self.begin_edit(index, event.position().y() < 26)

    def begin_edit(self, index, ruby, parent=None, rect=None):
        row = self.panel.lines.selectedIndexes()[0].row()
        unit = self.units()[index]
        self.edit_target = (self.panel.source, deepcopy(self.panel.model), row, index, ruby)
        rect = rect if rect is not None else unit_rects(self.units(), self.font())[index]
        self.editor.setParent(parent if parent is not None else self)
        self.editor.setGeometry(rect.x(), rect.y() + (0 if ruby else 26), max(rect.width(), 100), 28)
        self.editor.setText(reading(unit) if ruby else unit.text)
        self.editor.setPlaceholderText(tr("Ruby (comma-separated parts)"))
        self.editor.show()
        self.editor.setFocus()
        self.editor.selectAll()

    def cancel_edit(self):
        self.edit_target = None
        self.editor.hide()

    def eventFilter(self, watched, event):
        if watched is self.editor and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_edit()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish_edit()
                return True
        return super().eventFilter(watched, event)

    def finish_edit(self):
        if self.edit_target is None:
            return
        target = self.edit_target
        value = self.editor.text()
        self.cancel_edit()
        self.panel.edit_unit(*target, value)


class LyricEditor(QDockWidget):
    def __init__(self, window):
        super().__init__(tr("Edit lyrics"), window)
        self.window = window
        self.source = ""
        self.rows = []
        body = QWidget()
        layout = QVBoxLayout(body)
        self.lines = LyricList(self)
        self.lines.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lines.setItemDelegate(LyricDelegate(self))
        self.lines.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lines.customContextMenuRequested.connect(self.line_menu)
        layout.addWidget(self.lines, 1)
        self.canvas = LyricCanvas(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        scroll.setFixedHeight(88)
        layout.addWidget(scroll)
        hint = QLabel(tr("Drag to select · Right-click to move · Double-click text/ruby to edit"))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.setWidget(body)
        self.lines.itemSelectionChanged.connect(self.canvas.refresh)
        window.view.lyrics_changed.connect(self.refresh)
        window.view.view_changed.connect(self.lines.viewport().update)
        self.hide()

    def open_at(self, row=None):
        self.refresh(force=True)
        if row is not None and row < self.lines.count():
            self.lines.setCurrentRow(row)
        self.show()
        self.raise_()

    def refresh(self, force=False):
        text = self.window.lyrics_text
        if text == self.source and not force:
            return
        selected = [index.row() for index in self.lines.selectedIndexes()]
        self.source = text
        self.model = parse(text)
        self.rows = [line for chapter in self.model.chapters for line in chapter.lines]
        self.lines.blockSignals(True)
        self.lines.clear()
        for row in self.rows:
            self.lines.addItem(QListWidgetItem("".join(unit.text for unit in row.units)))
        for row in selected:
            if row < self.lines.count():
                self.lines.item(row).setSelected(True)
        self.lines.blockSignals(False)
        self.canvas.refresh()

    def targets(self):
        selected = [index.row() for index in self.lines.selectedIndexes()]
        return [
            (row, index)
            for row in selected
            for index in range(len(self.rows[row].units))
            if len(selected) != 1 or not self.canvas.selected or index in self.canvas.selected
        ]

    def refs(self):
        marked = deepcopy(self.model)
        rows = [line for chapter in marked.chapters for line in chapter.lines]
        for row, index in self.targets():
            for word in unit_words(rows[row].units[index]):
                word.channel = 127
        return {
            (row, sound.index)
            for row, line in enumerate(sound_lines(marked))
            for sound in line.sounds
            if sound.channel == 127
        }

    def note_ids(self):
        mapped = self.window.view.lyric_mapped
        return {
            note
            for row, index in self.refs()
            if row < len(mapped) and index < len(mapped[row])
            for note in mapped[row][index]
        }

    def line_menu(self, position):
        self.canvas.selected.clear()
        self.canvas.update()
        self.context_menu(self.lines.viewport().mapToGlobal(position))

    def context_menu(self, position):
        menu = QMenu(self)
        move = menu.addMenu(tr("Move to channel"))
        sync = QCheckBox(tr("Move linked notes too"), move)
        sync_action = QWidgetAction(move)
        sync_action.setDefaultWidget(sync)
        move.addAction(sync_action)
        move.addSeparator()
        color = menu.addMenu(tr("Set lyric color"))
        color.addAction(tr("Follow lyric channel"), lambda: self.apply("color", None))
        select = menu.addMenu(tr("Select all lines in channel"))
        enabled = bool(self.targets()) and not self.window.view.lyric_mapping_pending
        move.setEnabled(enabled)
        color.setEnabled(enabled)
        for channel in self.window.view.channels:
            number = channel.channel
            label = tr("Channel {number}", number=number) + (" — " + channel.name if channel.name else "")
            move.addAction(label, lambda checked=False, n=number: self.apply("channel", n, sync.isChecked()))
            color.addAction(label, lambda checked=False, n=number: self.apply("color", n))
            select.addAction(label, lambda checked=False, n=number: self.select_channel(n))
        menu.exec(position)

    def select_channel(self, number):
        self.lines.clearSelection()
        for index, row in enumerate(self.rows):
            if any(word.channel == number for unit in row.units for word in unit_words(unit)):
                self.lines.item(index).setSelected(True)

    def apply(self, kind, value, sync=False):
        if not self.targets():
            return
        model = deepcopy(self.model)
        rows = [line for chapter in model.chapters for line in chapter.lines]
        for row, index in self.targets():
            for word in unit_words(rows[row].units[index]):
                setattr(word, "channel" if kind == "channel" else "color_channel", value)
        ids = self.note_ids() if kind == "channel" and sync else set()
        refs = self.refs()
        if ids and any(
            ids.intersection(notes)
            for row, line in enumerate(self.window.view.lyric_mapped)
            for index, notes in enumerate(line)
            if (row, index) not in refs
        ):
            QMessageBox.warning(
                self,
                tr("Edit lyrics"),
                tr("A linked note is shared with unselected lyrics. Expand the selection or disable moving notes."),
            )
            return
        self.window._apply_lyric_editor(self.source, dumps(model, dotted=True), ids, value)

    def edit_unit(self, source, model, row, index, ruby, value):
        unit = [line for chapter in model.chapters for line in chapter.lines][row].units[index]
        if value == (reading(unit) if ruby else unit.text):
            return
        words = unit_words(unit)
        channel, color = words[0].channel, words[0].color_channel
        if ruby:
            unit.ruby = (
                Ruby([[Unit(Word(char, channel, color)) for char in part] for part in value.split(",")])
                if value
                else None
            )
        else:
            if not value:
                QMessageBox.warning(self, tr("Edit lyrics"), tr("Text must not be empty."))
                return
            new = [Word(char, channel, color) for char in value]
            unit.base = new[0] if len(new) == 1 else Group(new, explicit=True)
        try:
            text = dumps(model, dotted=True)
            natural_sounds(text)
            changed = structure(text).groups != structure(source).groups
        except ValueError as error:
            QMessageBox.warning(self, tr("Edit lyrics"), str(error))
            return
        if (
            changed
            and QMessageBox.question(
                self,
                tr("Edit lyrics"),
                tr("NOTE consumption changed. Following lyrics may map to different notes. Apply?"),
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.window._apply_lyric_editor(source, text, set(), None)
