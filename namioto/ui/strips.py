# SPDX-License-Identifier: AGPL-3.0-only
"""Mapped Sounds on the roll's time axis, with a separate lane for Sounds consuming no NOTE.

Normal gestures change discrete KRC NOTE counts; alignment evidence is read-only.
The roll owns the single undo step committed when a gesture ends.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QMenu

from namioto.i18n import tr
from namioto.karaoke.operations import SoundRef
from namioto.lyricmap.editing import detach, insert_sound, move_boundary, note_runs, operation_at, resize
from namioto.lyricmap.notes import TimedNote, resolve
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
    channel_requested = Signal(int, int, int)
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
        self._runs = None
        self._run_lookup = {}
        self._press_x = 0.0
        self._press_y = 0.0
        self._press_note_position = 0
        self._candidate = None
        self._candidate_x = 0.0
        self._candidate_y = 0.0
        self._layouts = {}
        self._block_cache = None
        self._block_origin = 0.0
        self._block_scale = 0.0
        self._channel_lanes = None
        self.show_raw = False
        view.viewport_changed.connect(self._refresh_viewport)
        view.lyrics_changed.connect(self._refresh)
        view.notes_changed.connect(self._refresh)
        view.lyric_highlight_changed.connect(self.update)

    def _x(self, seconds, origin=None):
        base = self.origin().x() if origin is None else origin
        return base + self.view.mapFromScene(QPointF(self.view.to_beats(seconds), 0)).x()

    def _lanes(self):
        if self._channel_lanes is None:
            self._channel_lanes = {
                channel: at
                for at, channel in enumerate(
                    sorted({sound.channel for line in self.view.lyric_lines for sound in line.sounds})
                )
            }
        return self._channel_lanes

    def _lane(self, row, column):
        return self._lanes().get(self.view.lyric_lines[row].sounds[column].channel, 0) * SOUND_HEIGHT

    def _blocks(self):
        if self._block_cache is not None:
            return self._block_cache
        font = QFont()
        font.setPixelSize(14)
        metrics = QFontMetrics(font)
        blocks = {}
        first_start = {}
        for row, spans in enumerate(self.view.lyric_times):
            for column, (start, end) in enumerate(spans):
                if start is not None and end is not None:
                    channel = self.view.lyric_lines[row].sounds[column].channel
                    first_start.setdefault(channel, start)
                    blocks[(row, column)] = QRectF(
                        self._x(start),
                        self._lane(row, column) + SOUND_MARGIN,
                        max(1, self._x(end) - self._x(start)),
                        MAIN_BOTTOM - SOUND_MARGIN,
                    )
        label_ends = {}
        previous_end = {}
        for row, line in enumerate(self.view.lyric_lines):
            for column, sound in enumerate(line.sounds):
                ref = SoundRef(row, column)
                if (row, column) in blocks:
                    previous_end[sound.channel] = self.view.lyric_times[row][column][1]
                    continue
                seconds = previous_end.get(sound.channel, first_start.get(sound.channel, 0.0))
                x = max(self._x(seconds), label_ends.get(sound.channel, float("-inf")) + SOUND_PAD)
                width = metrics.horizontalAdvance(sound.label) + 2 * SOUND_PAD
                label_ends[sound.channel] = x + width
                blocks[(ref.line, ref.index)] = QRectF(
                    x,
                    self._lane(row, column) + MAIN_BOTTOM + SOUND_MARGIN,
                    width,
                    SOUND_HEIGHT - MAIN_BOTTOM - 2 * SOUND_MARGIN,
                )
        self._block_cache = blocks
        self._block_origin = self._x(0.0)
        self._block_scale = self.view.pixels_per_second()
        return blocks

    def paintEvent(self, event):
        painter = QPainter(self)
        colors = theme.canvas()
        painter.fillRect(self.rect(), colors.panel)
        origin = self.origin().x()
        visible = QRectF(origin, 0, self.view.viewport().width(), self.height())
        painter.setClipRect(visible)
        dirty = visible.intersected(QRectF(event.rect())).adjusted(-2, -2, 2, 2)
        font = QFont()
        font.setPixelSize(14)
        font.setBold(True)
        painter.setFont(font)
        blocks = self._blocks()
        hovered_group = self._group_run(self._hover)
        highlighted = set(self.view.highlight_sounds)
        metrics = painter.fontMetrics()
        for (row, column), rect in blocks.items():
            if not rect.intersects(dirty):
                continue
            zero = rect.top() - self._lane(row, column) > MAIN_BOTTOM
            flags = self.view.lyric_red[row] if row < len(self.view.lyric_red) else ()
            good = not (column < len(flags) and flags[column])
            groups = self.view.lyric_group[row] if row < len(self.view.lyric_group) else ()
            group = groups[column] if column < len(groups) else -1
            left_join = group >= 0 and column > 0 and groups[column - 1] == group
            right_join = group >= 0 and column + 1 < len(groups) and groups[column + 1] == group
            if zero:
                painter.fillRect(rect, QColor(theme.LYRIC_ZERO).darker(250))
                self._note_runs()
                run = self._run_lookup.get(SoundRef(row, column))
                unmatched = run is not None and run.slots > 0
                painter.setPen(
                    QPen(QColor(theme.LYRIC_ZERO), 1, Qt.PenStyle.DashLine if unmatched else Qt.PenStyle.SolidLine)
                )
                painter.drawRect(rect)
            else:
                self._paint_body(painter, rect, good, left_join, right_join)
                if left_join:
                    painter.setPen(QPen(QColor(theme.LYRIC_ZERO), 1))
                    painter.drawLine(rect.topLeft(), rect.bottomLeft())
            if (
                (row, column) in highlighted
                or self._hover == (row, column)
                or (
                    hovered_group is not None
                    and row == hovered_group[0]
                    and hovered_group[1] <= column <= hovered_group[2]
                )
            ):
                tint = QColor(theme.LYRIC_HIGHLIGHT)
                tint.setAlpha(70)
                painter.fillRect(rect, tint)
                painter.setPen(QPen(QColor(theme.LYRIC_HIGHLIGHT), 2))
                painter.drawRect(rect)
            text_color = theme.LYRIC_ZERO if zero else theme.LYRIC_TEXT
            if (row, column) in highlighted:
                text_color = theme.LYRIC_SELECT
            painter.setPen(QColor(text_color))
            sound = self.view.lyric_lines[row].sounds[column]
            if not zero:
                painter.fillRect(
                    QRectF(rect.left(), rect.top(), rect.width(), 2), self.view.channel_color(sound.channel)
                )
            room = max(0, int(rect.width()) - 2 * SOUND_PAD)
            text = self._fit_label(sound, metrics, room)
            if not text and room:
                text = metrics.elidedText(sound.label, Qt.TextElideMode.ElideRight, room)
            painter.drawText(
                rect.adjusted(SOUND_PAD, 0, -SOUND_PAD, 0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                text,
            )
        if self.show_raw:
            painter.setPen(QPen(QColor(theme.LYRIC_ZERO), 1, Qt.PenStyle.DashLine))
            for row, spans in enumerate(self.view.lyric_raw):
                for column, (start, _length) in enumerate(spans):
                    if start is not None:
                        x = self._x(start)
                        lane = self._lane(row, column)
                        painter.drawLine(QPointF(x, lane), QPointF(x, lane + MAIN_BOTTOM))
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
        self._press_x, self._press_y = x, y
        self._base_blocks = self._blocks()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        found = None if shift else self._boundary_at(x, y) if y % SOUND_HEIGHT <= MAIN_BOTTOM else None
        kind = "boundary"
        if found is None:
            found = self._sound_at(x, y)
            kind = "body"
        if found is None:
            return
        self._base = self._note_runs()
        if not self._base:
            return
        run = operation_at(self._base, SoundRef(*found))
        if run is None or any(not self._same_channel(SoundRef(*found), ref) for ref in run.sounds):
            return
        if shift:
            rect = self._run_rect(run)
            kind = "resize_left" if x < rect.center().x() else "resize_right"
        self._base_notes = tuple(
            resolve(
                [
                    TimedNote(self.view.to_seconds(note.start), self.view.to_seconds(note.end), note.pitch, note.id)
                    for note in self.view.notes()
                    if note.channel == self.view.lyric_lines[found[0]].sounds[found[1]].channel
                ]
            ).stream
        )
        self._press_note_position = self._note_position(x)
        self._drag = (kind, *found)
        self._candidate = None
        self._candidate_x = x
        self._candidate_y = y
        self.mapping_drag_started.emit()

    def mouseMoveEvent(self, event):
        x, y = event.position().x(), event.position().y()
        if self._drag is None:
            sound = self._sound_at(x, y)
            self._update_hover(sound)
            found = self._boundary_at(x, y)
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if shift and sound is not None:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif found is not None:
                self._set_cursor(found)
            elif sound is not None:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()
            return
        kind, row, column = self._drag
        main_bottom = self._lane(row, column) + MAIN_BOTTOM
        if not self.view.gesture_active:
            self._drag = None
            return
        if abs(x - self._press_x) < SOUND_GRAB_PX and abs(y - self._press_y) < SOUND_GRAB_PX:
            candidate = None
        else:
            ref = SoundRef(row, column)
            run = operation_at(self._base, ref)
            delta = self._note_position(x) - self._press_note_position
            if kind == "boundary":
                left = SoundRef(row, column - 1)
                first = operation_at(self._base, left)
                total = first.slots + run.slots
                candidate = move_boundary(self._base, left, ref, max(0, min(total, first.slots + delta)))
            elif kind.startswith("resize"):
                slots = max(0, run.slots + (-delta if kind == "resize_left" else delta))
                previous = operation_at(self._base, SoundRef(row, run.sounds[0].index - 1))
                if kind == "resize_left" and previous is not None:
                    slots = min(slots, previous.slots + run.slots)
                candidate = resize(self._base, ref, slots, left=kind == "resize_left")
                if kind == "resize_left" and previous is not None and not self._same_channel(ref, previous.sounds[-1]):
                    candidate = None
            elif not self._lane(row, column) <= y < self._lane(row, column) + SOUND_HEIGHT:
                candidate = None
            elif y > main_bottom + SOUND_MARGIN:
                candidate = detach(self._base, ref, 0)
            elif y < main_bottom - SOUND_MARGIN:
                target = next(
                    (
                        key
                        for key, rect in self._base_blocks.items()
                        if self._lane(*key) == self._lane(row, column)
                        and rect.top() < main_bottom
                        and rect.top() <= y <= rect.bottom()
                        and rect.left() <= x <= rect.right()
                    ),
                    None,
                )
                rect = self._run_rect(run)
                edge = (ref == run.sounds[0] and x < rect.left()) or (ref == run.sounds[-1] and x > rect.right())
                grouped = len(run.sounds) > 1
                candidate = detach(self._base, ref, 1) if grouped and edge else None
                if target is not None:
                    operation = operation_at(self._base, SoundRef(*target))
                    if operation != run and self._same_channel(ref, operation.sounds[0]):
                        receiving = self._run_rect(operation)
                        inset = receiving.width() / 4 if grouped else 0
                        if receiving.left() + inset <= x <= receiving.right() - inset:
                            candidate = insert_sound(self._base, ref, operation) or candidate
                elif not grouped:
                    candidate = detach(self._base, ref, 1 if run.slots == 0 else run.slots)
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
        self._drag = None
        self.mapping_drag_finished.emit(True)
        self._refresh()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._drag is not None:
            self._drag = None
            self.mapping_drag_finished.emit(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        if not self.view.lyric_editable:
            return
        menu = QMenu(self)
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
                action.setChecked(channel.channel == self.view.lyric_lines[found[0]].sounds[found[1]].channel)
                action.triggered.connect(
                    lambda _checked=False, number=channel.channel: self.channel_requested.emit(*found, number)
                )
        if found is not None and not self.view.lyric_mapping_pending:
            for kind, label in (
                ("drop", "Drop this sound"),
                ("keep", "Keep this sound"),
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

    def _note_runs(self):
        if self._runs is None:
            self._runs = note_runs(self.view.lyric_text)
            self._run_lookup = {ref: run for run in self._runs for ref in run.sounds}
        return self._runs

    def _same_channel(self, left, right):
        return (
            self.view.lyric_lines[left.line].sounds[left.index].channel
            == self.view.lyric_lines[right.line].sounds[right.index].channel
        )

    def _run_rect(self, run):
        rects = [self._base_blocks[(ref.line, ref.index)] for ref in run.sounds]
        rect = QRectF(rects[0])
        for other in rects[1:]:
            rect = rect.united(other)
        return rect

    def _note_position(self, x):
        seconds = self.view.seconds_at_viewport_x(x - self.origin().x())
        return sum(seconds >= (note.start + note.end) / 2 for note in self._base_notes)

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

    def _boundary_at(self, x, y=None):
        blocks = self._blocks()
        self._note_runs()
        candidates = []
        for (row, column), rect in blocks.items():
            if (
                column == 0
                or self.view.lyric_lines[row].sounds[column - 1].channel
                != self.view.lyric_lines[row].sounds[column].channel
            ):
                continue
            ref = SoundRef(row, column)
            if self._run_lookup.get(ref) == self._run_lookup.get(SoundRef(row, column - 1)):
                continue
            lane = self._lane(row, column)
            if y is not None and not lane <= y < lane + SOUND_HEIGHT:
                continue
            edge = rect.left()
            if rect.top() > lane + MAIN_BOTTOM:
                previous = blocks.get((row, column - 1))
                if previous is None or previous.top() > lane + MAIN_BOTTOM:
                    continue
                edge = previous.right()
            previous = blocks.get((row, column - 1))
            band = min(SOUND_GRAB_PX, rect.width() / 4, previous.width() / 4) if previous is not None else 0
            if abs(x - edge) <= band:
                candidates.append((abs(x - edge), (row, column)))
        return min(candidates)[1] if candidates else None

    def _raw_boundary_at(self, x, y=None):
        candidates = [
            (abs(x - self._x(start)), (row, column))
            for row, spans in enumerate(self.view.lyric_raw)
            for column, (start, _length) in enumerate(spans)
            if start is not None
            and abs(x - self._x(start)) <= SOUND_GRAB_PX
            and (y is None or self._lane(row, column) <= y < self._lane(row, column) + SOUND_HEIGHT)
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
        if found is not None and self.view.lyric_editable:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()

    def _update_hover(self, found):
        label = self.view.lyric_lines[found[0]].sounds[found[1]].label if found else ""
        if found:
            self._note_runs()
            run = self._run_lookup.get(SoundRef(*found))
            if run is not None:
                label += "\n" + tr("NOTE count: {count}", count=run.slots)
        if found and self.show_raw and self.view.lyric_editable:
            length = self.view.lyric_raw[found[0]][found[1]][1]
            if length is not None:
                label += "\n" + tr("Reference duration: {seconds:.3f} s", seconds=length)
        self.setToolTip(label)
        if found != self._hover:
            self._hover = found
            self.update()

    def _refresh_viewport(self):
        if self._block_cache is not None:
            origin = self._x(0.0)
            if self._block_scale != self.view.pixels_per_second():
                self._block_cache = None
                self._layouts.clear()
            elif origin != self._block_origin:
                delta = origin - self._block_origin
                self._block_cache = {key: rect.translated(delta, 0) for key, rect in self._block_cache.items()}
                self._layouts.clear()
            self._block_origin = origin
        self.update()

    def _refresh(self):
        self._channel_lanes = None
        self.setFixedHeight(max(1, len(self._lanes())) * SOUND_HEIGHT)
        self._block_cache = None
        self._runs = None
        self._layouts.clear()
        self.update()
