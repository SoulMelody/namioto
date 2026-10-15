"""Visible ruby, direct text selection and in-place editing of the authoritative KRC."""

from copy import deepcopy

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QScrollArea,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from namioto.i18n import tr
from namioto.karaoke.model import Chapter, Group, Line, Lyrics, Ruby, Unit, Word, unit_words
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
        return QSize(200, 86 if index.row() in self.panel.chapter_starts else 62)

    def paint(self, painter, option, index):
        painter.save()
        painter.setClipRect(option.rect)
        painter.fillRect(
            option.rect,
            option.palette.highlight() if self.panel.lines.item(index.row()).isSelected() else option.palette.base(),
        )
        painter.translate(option.rect.topLeft())
        painter.setFont(option.font)
        if index.row() in self.panel.chapter_starts:
            painter.setPen(option.palette.text().color())
            painter.drawText(
                QRect(8, 0, option.rect.width() - 16, 24),
                tr("Chapter {number}", number=self.panel.chapter_starts[index.row()] + 1),
            )
            painter.translate(0, 24)
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
        if self.row(item) in self.panel.chapter_starts:
            row_rect.adjust(0, 24, 0, 0)
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
        self.cut = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self.cut = None
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

        if self.cut is not None:
            rects = unit_rects(self.units(), self.font())
            painter.setPen(self.palette().highlight().color())
            x = rects[self.cut].left()
            painter.drawLine(x, 2, x, 60)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.cut is not None:
            self.panel.split_line()
            return
        super().keyPressEvent(event)

    def hit(self, position):
        return next(
            (index for index, rect in enumerate(unit_rects(self.units(), self.font())) if rect.contains(position)), None
        )

    def mousePressEvent(self, event):
        index = self.hit(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.cut = None
            for boundary, rect in enumerate(unit_rects(self.units(), self.font())):
                if boundary and abs(event.position().x() - rect.left()) <= 5:
                    self.cut = boundary
                    self.selected.clear()
                    self.anchor = None
                    self.update()
                    return
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
        self.chapter_starts = {}
        self.row_selections = {}
        body = QWidget()
        layout = QVBoxLayout(body)
        toolbar = QHBoxLayout()
        insert = QToolButton()
        insert.setText(tr("Insert line"))
        insert.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        insert.clicked.connect(self.insert_line)
        insert_menu = QMenu(insert)
        insert_menu.addAction(tr("Insert above"), lambda: self.insert_line(above=True))
        insert_menu.addAction(tr("Insert below"), self.insert_line)
        insert.setMenu(insert_menu)
        toolbar.addWidget(insert)
        delete = QToolButton()
        delete.setText(tr("Delete lines"))
        delete.setToolTip(tr("Delete lyrics, not MIDI notes. Following lyrics may map to different notes."))
        delete.clicked.connect(self.delete_lines)
        toolbar.addWidget(delete)
        chapter = QToolButton()
        chapter.setText(tr("Chapter"))
        chapter.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        chapter_menu = QMenu(chapter)
        start_chapter = chapter_menu.addAction(tr("Start chapter here"), self.start_chapter)
        merge_chapter = chapter_menu.addAction(tr("Merge with previous chapter"), self.merge_chapter)
        chapter_menu.aboutToShow.connect(
            lambda: start_chapter.setEnabled(
                self.single_row() is not None and self.single_row() not in self.chapter_starts
            )
        )
        chapter_menu.aboutToShow.connect(
            lambda: merge_chapter.setEnabled(self.single_row() in self.chapter_starts and self.single_row() != 0)
        )
        chapter.setMenu(chapter_menu)
        toolbar.addWidget(chapter)
        toolbar.addStretch()
        layout.addLayout(toolbar)
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
        hint = QLabel(tr("Drag to select · Double-click to edit · Click between units, then Enter to split"))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.setWidget(body)
        self.lines.itemSelectionChanged.connect(self.canvas.refresh)
        self.lines.itemSelectionChanged.connect(lambda: delete.setEnabled(bool(self.lines.selectedIndexes())))
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
        self.row_selections[self.source] = selected
        selected = self.row_selections.get(text, selected)
        self.source = text
        self.model = parse(text) if text.strip() else Lyrics([])
        self.rows = [line for chapter in self.model.chapters for line in chapter.lines]
        self.chapter_starts = {}
        offset = 0
        for chapter, part in enumerate(self.model.chapters):
            if part.lines:
                self.chapter_starts[offset] = chapter
            offset += len(part.lines)
        self.lines.blockSignals(True)
        self.lines.clear()
        for row in self.rows:
            self.lines.addItem(QListWidgetItem("".join(unit.text for unit in row.units)))
        for row in selected:
            if row < self.lines.count():
                self.lines.item(row).setSelected(True)
        self.lines.blockSignals(False)
        self.canvas.refresh()

    def single_row(self):
        selected = self.lines.selectedIndexes()
        return selected[0].row() if len(selected) == 1 else None

    def location(self, model, row):
        for chapter, part in enumerate(model.chapters):
            if row < len(part.lines):
                return chapter, row
            row -= len(part.lines)
        raise IndexError(row)

    def commit_structure(self, model, row=0, *, preserve_mapping=False):
        model.chapters = [part for part in model.chapters if part.lines]
        try:
            text = dumps(model, dotted=True)
            layout = structure(text)
            if preserve_mapping:

                def signature(layout):
                    return [
                        (slots, tuple(line.sounds[index].reading for index in indices))
                        for line, groups in zip(layout.lines, layout.groups, strict=True)
                        for slots, indices in groups
                    ]

                if signature(layout) != signature(structure(self.source)):
                    raise ValueError(tr("This boundary would change note mapping. Choose another boundary."))
        except ValueError as error:
            QMessageBox.warning(self, tr("Edit lyrics"), str(error))
            return
        if text != self.source and self.window._apply_lyric_editor(self.source, text, set(), None):
            self.refresh(force=True)
            if self.lines.count():
                self.lines.setCurrentRow(min(row, self.lines.count() - 1))
            self.row_selections[text] = [index.row() for index in self.lines.selectedIndexes()]

    def insert_line(self, checked=False, *, above=False):
        row = self.single_row()
        if self.rows and row is None:
            return
        source = self.source
        value, accepted = QInputDialog.getText(
            self, tr("Insert line"), tr("Lyrics (.krc, one line; following lyrics may map to different notes)")
        )
        if not accepted or not value.strip():
            return
        if source != self.source:
            QMessageBox.warning(
                self, tr("Edit lyrics"), tr("Lyrics changed or are busy. Reopen the editor and try again.")
            )
            return
        try:
            inserted = parse(value)
            if len(inserted.chapters) != 1 or len(inserted.chapters[0].lines) != 1:
                raise ValueError(tr("Enter exactly one lyric line."))
            line = inserted.chapters[0].lines[0]
        except ValueError as error:
            QMessageBox.warning(self, tr("Edit lyrics"), str(error))
            return
        channel = self.rows[row].track if row is not None else self.window.view.active_channel
        line.track = max(channel, 0)
        model = deepcopy(self.model)
        if row is None:
            model.chapters.append(Chapter([line]))
            target = 0
        else:
            chapter, index = self.location(model, row)
            model.chapters[chapter].lines.insert(index + (not above), line)
            target = row + (not above)
        self.commit_structure(model, target)

    def delete_lines(self):
        selected = sorted((index.row() for index in self.lines.selectedIndexes()), reverse=True)
        if not selected:
            return
        model = deepcopy(self.model)
        for row in selected:
            chapter, index = self.location(model, row)
            del model.chapters[chapter].lines[index]
        self.commit_structure(model, selected[-1])

    def split_line(self):
        row, cut = self.single_row(), self.canvas.cut
        if row is None or cut is None or not 0 < cut < len(self.rows[row].units):
            return
        model = deepcopy(self.model)
        chapter, index = self.location(model, row)
        lines = model.chapters[chapter].lines
        if lines[index].units[cut].join_previous:
            QMessageBox.warning(self, tr("Edit lyrics"), tr("Cannot split units sharing a note."))
            return
        lines.insert(index + 1, Line(lines[index].units[cut:]))
        lines[index].units = lines[index].units[:cut]
        self.commit_structure(model, row + 1, preserve_mapping=True)
        self.canvas.setFocus()

    def merge_line(self):
        row = self.single_row()
        if row is None or row == 0 or row in self.chapter_starts:
            return
        model = deepcopy(self.model)
        chapter, index = self.location(model, row)
        lines = model.chapters[chapter].lines
        lines[index - 1].units.extend(lines.pop(index).units)
        self.commit_structure(model, row - 1, preserve_mapping=True)

    def start_chapter(self):
        row = self.single_row()
        if row is None or row in self.chapter_starts:
            return
        model = deepcopy(self.model)
        chapter, index = self.location(model, row)
        lines = model.chapters[chapter].lines
        model.chapters.insert(chapter + 1, Chapter(lines[index:]))
        del lines[index:]
        self.commit_structure(model, row, preserve_mapping=True)

    def merge_chapter(self):
        row = self.single_row()
        if row is None or row == 0 or row not in self.chapter_starts:
            return
        model = deepcopy(self.model)
        chapter, _ = self.location(model, row)
        model.chapters[chapter - 1].lines.extend(model.chapters.pop(chapter).lines)
        self.commit_structure(model, row, preserve_mapping=True)

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
        menu.addSeparator()
        menu.addAction(tr("Insert above"), lambda: self.insert_line(above=True))
        menu.addAction(tr("Insert below"), self.insert_line)
        menu.addAction(tr("Delete lines"), self.delete_lines).setEnabled(bool(self.lines.selectedIndexes()))
        menu.addAction(tr("Split here"), self.split_line).setEnabled(self.canvas.cut is not None)
        row = self.single_row()
        menu.addAction(tr("Merge with previous line"), self.merge_line).setEnabled(
            row is not None and row > 0 and row not in self.chapter_starts
        )
        menu.addAction(tr("Start chapter here"), self.start_chapter).setEnabled(
            row is not None and row not in self.chapter_starts
        )
        menu.addAction(tr("Merge with previous chapter"), self.merge_chapter).setEnabled(
            row in self.chapter_starts and row != 0
        )
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
