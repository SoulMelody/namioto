# SPDX-License-Identifier: AGPL-3.0-only
"""Mapped Sounds on the roll's time axis, with a separate lane for Sounds consuming no NOTE.

Normal gestures request discrete mapping anchors; raw timings are untouched. The advanced
raw overlay exposes onset editing separately. The window solves previews off the GUI thread,
and the roll owns the single undo step committed when a gesture ends.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QMenu

from namioto.i18n import tr
from namioto.karaoke.operations import SoundRef
from namioto.lyricmap.editing import drop_sound, insert_sound, move_boundary, operation_at
from namioto.lyricmap.notes import TimedNote
from namioto.ui import theme
from namioto.ui.viewport import ViewportStrip

SOUND_HEIGHT = 68
SOUND_GRAB_PX = 6
SOUND_PAD = 3
SOUND_MARGIN = 5
SOUND_MAGNET_PX = 4
MAIN_BOTTOM = 36


class SoundStrip(ViewportStrip):
    """Independent Sound blocks, grouped by NOTE, and position-only grey Drop labels."""

    lyric_action_requested = Signal(str, int, int)
    channel_requested = Signal(int, int)
    save_map_requested = Signal()
    discard_map_requested = Signal()
    mapping_drag_started = Signal()
    mapping_preview_requested = Signal(object)
    mapping_drag_finished = Signal(bool)

    def __init__(self, view):
        super().__init__(view)
        self.setFixedHeight(SOUND_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._hover = None
        self._drag = None
        self._base = ()
        self._base_blocks = {}
        self._base_notes = ()
        self._press_x = 0.0
        self._press_seconds = 0.0
        self._candidate = None
        self._candidate_x = 0.0
        self._candidate_y = 0.0
        self._layouts = {}
        self._block_cache = None
        self.show_raw = False
        view.viewport_changed.connect(self._refresh)
        view.lyrics_changed.connect(self._refresh)
        view.notes_changed.connect(self._refresh)
        view.lyric_highlight_changed.connect(self.update)

    def _x(self, seconds, origin=None):
        base = self.origin().x() if origin is None else origin
        return base + self.view.mapFromScene(QPointF(self.view.to_beats(seconds), 0)).x()

    def _blocks(self):
        if self._block_cache is not None:
            return self._block_cache
        font = QFont()
        font.setPixelSize(14)
        metrics = QFontMetrics(font)
        blocks = {}
        mapped = []
        for row, spans in enumerate(self.view.lyric_times):
            for column, (start, end) in enumerate(spans):
                if start is not None and end is not None:
                    mapped.append((SoundRef(row, column), start, end))
                    blocks[(row, column)] = QRectF(
                        self._x(start), SOUND_MARGIN, max(1, self._x(end) - self._x(start)), MAIN_BOTTOM - SOUND_MARGIN
                    )
        label_end = float("-inf")
        for row, line in enumerate(self.view.lyric_lines):
            for column, sound in enumerate(line.sounds):
                ref = SoundRef(row, column)
                if (row, column) in blocks:
                    continue
                if not mapped:
                    continue
                previous = [entry for entry in mapped if (entry[0].line, entry[0].index) < (row, column)]
                following = [entry for entry in mapped if (entry[0].line, entry[0].index) > (row, column)]
                seconds = previous[-1][2] if previous else following[0][1]
                x = max(self._x(seconds), label_end + SOUND_PAD)
                width = metrics.horizontalAdvance(sound.label) + 2 * SOUND_PAD
                label_end = x + width
                blocks[(ref.line, ref.index)] = QRectF(
                    x, MAIN_BOTTOM + SOUND_MARGIN, width, self.height() - MAIN_BOTTOM - 2 * SOUND_MARGIN
                )
        self._block_cache = blocks
        return blocks

    def paintEvent(self, event):
        painter = QPainter(self)
        colors = theme.canvas()
        painter.fillRect(self.rect(), colors.panel)
        origin = self.origin().x()
        painter.setClipRect(QRectF(origin, 0, self.view.viewport().width(), self.height()))
        font = QFont()
        font.setPixelSize(14)
        font.setBold(True)
        painter.setFont(font)
        blocks = self._blocks()
        for (row, column), rect in blocks.items():
            zero = rect.top() > MAIN_BOTTOM
            flags = self.view.lyric_red[row] if row < len(self.view.lyric_red) else ()
            good = not (column < len(flags) and flags[column])
            groups = self.view.lyric_group[row] if row < len(self.view.lyric_group) else ()
            group = groups[column] if column < len(groups) else -1
            left_join = group >= 0 and column > 0 and groups[column - 1] == group
            right_join = group >= 0 and column + 1 < len(groups) and groups[column + 1] == group
            if zero:
                painter.fillRect(rect, QColor(theme.LYRIC_ZERO).darker(250))
                painter.setPen(QColor(theme.LYRIC_ZERO))
                painter.drawRect(rect)
            else:
                self._paint_body(painter, rect, good, left_join, right_join)
                if left_join:
                    painter.setPen(QPen(QColor(theme.LYRIC_ZERO), 1))
                    painter.drawLine(rect.topLeft(), rect.bottomLeft())
            if (row, column) in self.view.highlight_sounds or self._hover == (row, column):
                tint = QColor(theme.LYRIC_HIGHLIGHT)
                tint.setAlpha(70)
                painter.fillRect(rect, tint)
                painter.setPen(QPen(QColor(theme.LYRIC_HIGHLIGHT), 2))
                painter.drawRect(rect)
            text_color = theme.LYRIC_ZERO if zero else theme.LYRIC_TEXT
            if (row, column) in self.view.highlight_sounds:
                text_color = theme.LYRIC_SELECT
            painter.setPen(QColor(text_color))
            sound = self.view.lyric_lines[row].sounds[column]
            room = max(0, int(rect.width()) - 2 * SOUND_PAD)
            text = self._fit_label(sound, painter.fontMetrics(), room)
            if not text and room:
                text = painter.fontMetrics().elidedText(sound.label, Qt.TextElideMode.ElideRight, room)
            painter.drawText(
                rect.adjusted(SOUND_PAD, 0, -SOUND_PAD, 0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                text,
            )
        if self.show_raw:
            painter.setPen(QPen(QColor(theme.LYRIC_ZERO), 1, Qt.PenStyle.DashLine))
            for spans in self.view.lyric_raw:
                for start, _length in spans:
                    if start is not None:
                        x = self._x(start)
                        painter.drawLine(QPointF(x, 0), QPointF(x, MAIN_BOTTOM))
        painter.setPen(colors.ruler_line)
        painter.drawLine(QPointF(origin, MAIN_BOTTOM), QPointF(origin + self.view.viewport().width(), MAIN_BOTTOM))

    def _paint_body(self, painter, rect, good, left_join=False, right_join=False):
        body, light, dark = theme.lyric_shades(good)
        painter.fillRect(rect, body)
        painter.setPen(QPen(light, 0))
        painter.drawLine(rect.topLeft(), rect.topRight())
        if not left_join:
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
        painter.setPen(QPen(dark, 0))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        if not right_join:
            painter.drawLine(rect.topRight(), rect.bottomRight())

    def mousePressEvent(self, event):
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not self.view.lyric_editable
            or self.view.lyric_mapping_pending
        ):
            return
        x, y = event.position().x(), event.position().y()
        self.setFocus()
        self._press_x = x
        if self.show_raw:
            found = self._raw_boundary_at(x)
            if found is None or self.view.lyric_lines[found[0]].track != 0:
                return
            row, column = found
            self._drag = ("raw", row, column)
            self._base = self.view.lyric_raw[row]
            self._press_seconds = self._base[column][0]
            self.view.begin_gesture(tr("Move lyrics"))
            return
        self._base_blocks = self._blocks()
        found = self._boundary_at(x) if y <= MAIN_BOTTOM else None
        kind = "boundary"
        if found is None:
            found = self._sound_at(x, y)
            kind = "body"
        if found is None or self.view.lyric_lines[found[0]].track != 0 or not self.view.lyric_operations:
            return
        self._base = self.view.lyric_operations
        ids = {identifier for spans in self.view.lyric_mapped for chunk in spans for identifier in chunk}
        self._base_notes = tuple(
            sorted(
                (
                    TimedNote(self.view.to_seconds(note.start), self.view.to_seconds(note.end), note.pitch, note.id)
                    for note in self.view.notes()
                    if note.channel == 0 and note.id in ids
                ),
                key=lambda note: (note.start, note.end, note.pitch, note.id),
            )
        )
        self._drag = (kind, *found)
        self._candidate = None
        self._candidate_x = x
        self._candidate_y = y
        self.mapping_drag_started.emit()

    def mouseMoveEvent(self, event):
        x, y = event.position().x(), event.position().y()
        if self._drag is None:
            self._update_hover(self._sound_at(x, y))
            self._set_cursor(self._raw_boundary_at(x) if self.show_raw else self._boundary_at(x))
            return
        kind, row, column = self._drag
        if not self.view.gesture_active:
            self._drag = None
            return
        if kind == "raw":
            origin = self.origin().x()
            travelled = self.view.seconds_at_viewport_x(x - origin) - self.view.seconds_at_viewport_x(
                self._press_x - origin
            )
            self.view.set_sound_onset(
                row, column, self._magnet_seconds(self._press_seconds + travelled, x), base=self._base
            )
            return
        if abs(x - self._press_x) < SOUND_GRAB_PX and kind == "boundary":
            candidate = None
        else:
            ref = SoundRef(row, column)
            if kind == "boundary":
                notes = self._base_notes
                candidate = move_boundary(
                    self._base,
                    SoundRef(row, column - 1),
                    ref,
                    notes,
                    self.view.seconds_at_viewport_x(x - self.origin().x()),
                )
            elif y > MAIN_BOTTOM + SOUND_MARGIN:
                candidate = drop_sound(self._base, ref)
            elif y < MAIN_BOTTOM - SOUND_MARGIN:
                target = next(
                    (
                        key
                        for key, rect in self._base_blocks.items()
                        if rect.top() < MAIN_BOTTOM and rect.left() <= x <= rect.right()
                    ),
                    None,
                )
                candidate = None
                if target is not None:
                    operation = operation_at(self._base, SoundRef(*target))
                    if operation is not None:
                        rects = [self._base_blocks[(sound.line, sound.index)] for sound in operation.sounds]
                        left, right = min(rect.left() for rect in rects), max(rect.right() for rect in rects)
                        candidate = insert_sound(self._base, ref, operation, (x - left) / max(1, right - left))
            else:
                return
        if candidate == self._candidate:
            return
        if (
            candidate is not None
            and self._candidate is not None
            and abs(x - self._candidate_x) < SOUND_MAGNET_PX
            and abs(y - self._candidate_y) < SOUND_MAGNET_PX
        ):
            return
        self._candidate, self._candidate_x, self._candidate_y = candidate, x, y
        self.mapping_preview_requested.emit(candidate)

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        kind = self._drag[0]
        self._drag = None
        if kind == "raw":
            self.view.commit_gesture()
        else:
            self.mapping_drag_finished.emit(True)
        self._refresh()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._drag is not None:
            kind = self._drag[0]
            self._drag = None
            if kind == "raw":
                self.view.cancel_gesture()
            else:
                self.mapping_drag_finished.emit(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        if not self.view.lyric_editable:
            return
        menu = QMenu(self)
        save = menu.addAction(tr("Save lyrics map"))
        save.setEnabled(not self.view.lyric_mapping_pending and not self.view.gesture_active)
        save.triggered.connect(self.save_map_requested)
        discard = menu.addAction(tr("Discard lyrics map"))
        discard.setEnabled(not self.view.gesture_active)
        discard.triggered.connect(self.discard_map_requested)
        menu.addSeparator()
        raw = QAction(tr("Show raw timing (advanced)"), menu)
        raw.setCheckable(True)
        raw.setChecked(self.show_raw)
        raw.toggled.connect(self._toggle_raw)
        menu.addAction(raw)
        found = self._sound_at(event.pos().x(), event.pos().y())
        if found is not None:
            channels = menu.addMenu(tr("Lyric channel"))
            channels.setEnabled(not self.view.lyric_mapping_pending and not self.view.gesture_active)
            for channel in self.view.channels:
                action = channels.addAction(
                    tr("Channel {number}", number=channel.channel) + (f" — {channel.name}" if channel.name else "")
                )
                action.setCheckable(True)
                action.setChecked(channel.channel == self.view.lyric_lines[found[0]].track)
                action.triggered.connect(
                    lambda _checked=False, number=channel.channel: self.channel_requested.emit(found[0], number)
                )
        if found is not None and not self.view.lyric_mapping_pending and self.view.lyric_lines[found[0]].track == 0:
            for kind, label in (
                ("drop", "Drop this sound"),
                ("keep", "Keep this sound"),
                ("merge", "Merge with the sound before it"),
                ("dissolve", "Dissolve its merge"),
            ):
                action = QAction(tr(label), menu)
                action.triggered.connect(
                    lambda _checked=False, name=kind: self.lyric_action_requested.emit(name, *found)
                )
                menu.addAction(action)
        menu.exec(event.globalPos())

    def _toggle_raw(self, checked):
        self.show_raw = checked
        self._refresh()

    def wheelEvent(self, event):
        bar = self.view.horizontalScrollBar()
        bar.setValue(bar.value() - event.angleDelta().y())
        event.accept()

    def leaveEvent(self, event):
        self._update_hover(None)
        self.unsetCursor()
        super().leaveEvent(event)

    def _layout(self, row, origin=None):
        if row not in self._layouts:
            blocks = self._blocks()
            self._layouts[row] = [
                blocks[(row, column)].left() if (row, column) in blocks else None
                for column in range(len(self.view.lyric_lines[row].sounds))
            ]
        return self._layouts[row]

    def _boundary_x(self, row, index):
        layout = self._layout(row)
        return layout[index] if index < len(layout) else None

    def _boundary_at(self, x):
        blocks = self._blocks()
        candidates = []
        for (row, column), rect in blocks.items():
            if column == 0:
                continue
            edge = rect.left()
            if rect.top() > MAIN_BOTTOM:
                previous = blocks.get((row, column - 1))
                if previous is None or previous.top() > MAIN_BOTTOM:
                    continue
                edge = previous.right()
            if abs(x - edge) <= SOUND_GRAB_PX:
                candidates.append((abs(x - edge), (row, column)))
        return min(candidates)[1] if candidates else None

    def _raw_boundary_at(self, x):
        candidates = [
            (abs(x - self._x(start)), (row, column))
            for row, spans in enumerate(self.view.lyric_raw)
            for column, (start, _length) in enumerate(spans)
            if start is not None and abs(x - self._x(start)) <= SOUND_GRAB_PX
        ]
        return min(candidates)[1] if candidates else None

    def _sound_at(self, x, y=None):
        for key, rect in self._blocks().items():
            if rect.left() <= x <= rect.right() and (y is None or rect.top() <= y <= rect.bottom()):
                return key
        return None

    def _fit_label(self, sound, metrics, room):
        if room <= 0:
            return ""
        if metrics.horizontalAdvance(sound.label) <= room:
            return sound.label
        if sound.rubied and metrics.horizontalAdvance(sound.reading) <= room:
            return sound.reading
        return ""

    def _magnet_seconds(self, seconds, x):
        step = self.view.to_seconds(self.view.grid_step())
        nearest = self.view.offset + round((seconds - self.view.offset) / step) * step
        return nearest if abs(self._x(nearest) - x) <= SOUND_MAGNET_PX else seconds

    def _group_run(self, found):
        if found is None:
            return None
        row, column = found
        if row >= len(self.view.lyric_group):
            return None
        groups = self.view.lyric_group[row]
        if column >= len(groups) or groups[column] < 0:
            return None
        first = last = column
        while first > 0 and groups[first - 1] == groups[column]:
            first -= 1
        while last + 1 < len(groups) and groups[last + 1] == groups[column]:
            last += 1
        return (row, first, last) if last > first else None

    def _set_cursor(self, found):
        if found is not None and self.view.lyric_editable and self.view.lyric_lines[found[0]].track == 0:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()

    def _update_hover(self, found):
        label = self.view.lyric_lines[found[0]].sounds[found[1]].label if found else ""
        if found and self.show_raw and self.view.lyric_editable:
            length = self.view.lyric_raw[found[0]][found[1]][1]
            if length is not None:
                label += "\n" + tr("Reference duration: {seconds:.3f} s", seconds=length)
        self.setToolTip(label)
        if found != self._hover:
            self._hover = found
            self.update()

    def _refresh(self):
        self._block_cache = None
        self._layouts.clear()
        self.update()
