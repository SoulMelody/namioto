# SPDX-License-Identifier: AGPL-3.0-only
"""Piano-roll widgets on one shared time axis: the notes and the grid, the view that edits them, and
the ruler and keyboard round it. The lyrics strip lives in `namioto.ui.strips`.

`PianoRollView` is the editor: it owns the document, the channels, the sounds and their times, and
the undo stack of whole-document snapshots, and it is where a gesture is begun and committed. The
ruler and the keyboard are the strips that stay here; all three share `viewport.ViewportStrip`, and
read the view's scroll position rather than keeping one of their own.

Undo lives here, never in the document: one `QUndoStack` of whole-document `_RollState` snapshots
stored in seconds (a tempo change rescales beats, so seconds are what a note still sounds at), one
gesture one step. A discrete edit goes through `_edit`, whose depth guard makes nested edits one
step, and opening a project clears the stack. `Ctrl+C`/`Ctrl+V` copy and paste a selection snapped to
`snap`; `Ctrl+D` and `Delete`/`Backspace` remove it; clearing every note is `Ctrl+A` then `Delete` in
edit mode, with no button for it. `Escape` drops the gesture in flight - a move or a trim goes back,
a draw takes back the note it opened - with no undo step. A drag snaps the movement, not the place
on the grid
(`round(movement / snap) * snap`), so an off-grid note keeps the offset it had; trimming an edge is
the exception and lands on the drawn grid.

`editor.grid_offset_ms` slides the drawn beat grid through `set_offset`, negative left and positive
right; the snap helpers, the ruler's beat divisions and the align dialog's Quantize follow it, while
the notes, the spectrum, the playhead and the ruler's clock keep their exact timestamps.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QTransform, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

from namioto.analysis.spectrum import MIDI_OFFSET, NOTE_COUNT, NoteSpectrum
from namioto.channels import Channel, free_channel
from namioto.document import MIN_DURATION, PITCH_COUNT, PITCH_MAX, PITCH_MIN, Document, Note
from namioto.i18n import tr
from namioto.interaction import Interaction, Tool
from namioto.karaoke.sounds import SoundLine
from namioto.lyricmap.raw import from_spans
from namioto.ui import theme
from namioto.ui.blocks import CLICK_SLOP_PX, Press, block_part
from namioto.ui.spectrogram import IMAGE_BLOCK_FRAMES, SpectrumImage
from namioto.ui.text import format_time, note_name
from namioto.ui.viewport import ViewportStrip

LENGTH_BEATS = 64
CONTENT_MARGIN = 4.0
NOTE_INSET = 0.06
NOTE_VISUAL_INSET = 0.14
BAR_BEATS = 4.0
PAGE_TURN_MARGIN = 0.15  # how close to the right edge the playhead may get before the page turns
PAGE_TURN_LEAD = 0.2  # where on the fresh page the playhead then sits

SNAP_CHOICES = (("1/1", 4.0), ("1/2", 2.0), ("1/4", 1.0), ("1/8", 0.5), ("1/16", 0.25), ("1/32", 0.125))

# The colours are read while painting rather than once at import: a window outlives a theme switch,
# and a name read early would keep painting the theme that was in force when it was read.
SPECTRUM_TOP = PITCH_MAX - (MIDI_OFFSET + NOTE_COUNT - 1)
MIN_LINE_SPACING = 16.0
OVERTONES = (2, 3, 4)  # f, 2f, 3f and 4f: the four partials WaveTone marks over the row under the mouse
EDIT_DIM = 0.45
MIN_GRID_SPACING = 9.0
HISTORY_LIMIT = 50  # snapshots of the whole document, so the depth trades memory for how far back undo goes
NEW_CHANNEL = -1  # the channel menu's "make one" entry, since no channel number is negative
RULER_TIME_ROW = 24
RULER_HEIGHT = 46
TIME_LABEL_SPACING = 84.0
MEASURE_LABEL_SPACING = 30.0
TIME_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0)


def is_black_key(pitch: int) -> bool:
    return pitch % 12 in (1, 3, 6, 8, 10)


def _is_multiple(value: float, step: float) -> bool:
    return abs(value / step - round(value / step)) < 1e-6


def time_step(pixels_per_second: float, minimum: float) -> float:
    """Smallest 1-2-5 step that keeps the time grid at least `minimum` pixels apart."""
    for step in TIME_STEPS:
        if step * pixels_per_second >= minimum:
            return step
    return TIME_STEPS[-1]


@dataclass(frozen=True)
class _RollState:
    """A document snapshot as one undo step: channels, notes and what was selected.

    Note times are seconds, not the beats the roll draws in: a tempo change rescales every beat
    coordinate, so a snapshot in beats would restore a note at the beat it used to sit on rather
    than the time it still sounds at.
    """

    channels: tuple[Channel, ...]
    notes: tuple[tuple[int, float, float, int, int], ...]
    selected: frozenset[int]
    active_channel: int
    lyrics: tuple = ()
    lyric_raw: tuple = ()
    lyric_operations: tuple = ()
    lyric_text: str = ""
    lyric_evidence: tuple = ("", (), (), ())


def _state_data(state: _RollState) -> tuple:
    return (state.channels, state.notes, state.lyric_raw, state.lyric_text, state.lyric_evidence)


class _RollEdit(QUndoCommand):
    """One undo step, holding the state before and after it and restoring either by rebuilding.

    `push` runs `redo` once with the edit already applied, so the first call stays a no-op and only
    a following undo/redo pair moves the document.
    """

    def __init__(self, view: PianoRollView, before: _RollState, after: _RollState, text: str):
        super().__init__(text)
        self._view = view
        self._before = before
        self._after = after
        self._applied = False

    def undo(self) -> None:
        self._view._restore_state(self._before)

    def redo(self) -> None:
        if self._applied:
            self._view._restore_state(self._after)
        self._applied = True


class NoteItem(QGraphicsRectItem):
    """The drawn body of one `Note`; scene units are beats (x) and semitone rows (y).

    It is a view of the note, not a copy: the data lives in `namioto.document`, and the item reads
    and writes it. The note's channel is also the stacking order: a higher channel draws over, and
    wins the hit test, the way noteDigger files its channels.
    """

    def __init__(self, note: Note):
        super().__init__()
        self.note = note
        self.filtered = False
        self.lyric_text = ""
        self.lyric_runs: tuple[tuple[str, bool], ...] = ()
        self.fill, self.edge_light, self.edge_dark = theme.note_shades(QColor(theme.NOTE_PALETTE[0]))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._sync()

    @property
    def pitch(self) -> int:
        return self.note.pitch

    @property
    def start(self) -> float:
        return self.note.start

    @property
    def duration(self) -> float:
        return self.note.duration

    @property
    def channel(self) -> int:
        return self.note.channel

    @property
    def id(self) -> int:
        return self.note.id

    @property
    def end(self) -> float:
        return self.note.end

    def set_duration(self, duration: float) -> None:
        self.note.set_duration(duration)
        self._sync()

    def set_range(self, start: float, pitch: int) -> None:
        self.note.set_range(start, pitch)
        self._sync()

    def _sync(self) -> None:
        self.setRect(0.0, 0.0, self.duration, 1.0 - 2 * NOTE_INSET)
        self.setPos(self.start, PITCH_MAX - self.pitch + NOTE_INSET)
        self.setZValue(self.channel)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        selected = self.isSelected()
        colors = theme.canvas()
        body = colors.note_selected if selected else self.fill
        light = colors.note_selected_edge if selected else self.edge_light
        rect = self.rect().adjusted(0, NOTE_VISUAL_INSET - NOTE_INSET, 0, NOTE_INSET - NOTE_VISUAL_INSET)
        transform = painter.transform()
        px, py = 1.0 / transform.m11(), 1.0 / transform.m22()  # one device pixel in scene units
        left, top = rect.left() + px / 2, rect.top() + py / 2
        right, bottom = rect.right() - px / 2, rect.bottom() - py / 2

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(body)
        painter.drawRect(rect)
        painter.setPen(QPen(light, 0))
        painter.drawLine(QPointF(left, top), QPointF(right + px / 2, top))
        painter.drawLine(QPointF(left, top), QPointF(left, bottom + py / 2))
        painter.setPen(QPen(QColor(theme.NOTE_OUTLINE), 0))
        painter.drawLine(QPointF(left - px / 2, bottom), QPointF(right + px / 2, bottom))
        painter.drawLine(QPointF(right, top - py / 2), QPointF(right, bottom + py / 2))
        if self.filtered:
            pen = QPen(QColor(theme.LYRIC_BAD), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
        if self.lyric_text:
            painter.setClipRect(rect)
            painter.scale(px, py)
            font = QFont()
            font.setPixelSize(max(9, round(rect.height() / py * 0.72)))
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            x = 3
            right = rect.width() / px - 3
            runs = self.lyric_runs or ((self.lyric_text, False),)
            for text, zero in runs:
                room = max(0, int(right - x))
                if not room:
                    break
                visible = metrics.elidedText(text, Qt.TextElideMode.ElideRight, room)
                painter.setPen(
                    QColor(theme.LYRIC_ZERO)
                    if zero
                    else colors.note_selected_edge
                    if selected
                    else QColor(theme.LYRIC_TEXT)
                )
                painter.drawText(
                    QRectF(x, rect.top() / py, room, rect.height() / py), int(Qt.AlignmentFlag.AlignVCenter), visible
                )
                x += metrics.horizontalAdvance(visible)
                if visible != text:
                    break
        painter.restore()


class _SelectionBox(QWidget):
    """The rectangle dragged over the roll, in the accent the palette carries.

    Painted, not styled: it belongs to the roll, which this app draws itself.
    """

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        accent = self.palette().highlight().color()
        fill = QColor(accent)
        fill.setAlpha(40)
        painter.fillRect(self.rect(), fill)
        painter.setPen(accent)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class Drag(Enum):
    """The pointer gesture in flight, from press to release."""

    PAN = "pan"
    SEEK = "seek"
    SELECT = "select"
    DRAW = "draw"
    MOVE = "move"
    TRIM = "trim"


class Edge(Enum):
    """Which end of a note a trim gesture moves."""

    START = "start"
    END = "end"


@dataclass
class Gesture:
    """The gesture the button is holding, the one value the pointer handlers read and write.

    `origin` is the scene point the press landed on, `target` the note it grabbed, `edge` the end a
    trim moves, `snapshot` the geometry the gesture measures from, `press` the click/drag boundary
    `blocks.Press`, and `created` the note a draw opened, which a cancel takes away again.
    """

    kind: Drag
    origin: QPointF
    target: NoteItem | None = None
    edge: Edge | None = None
    snapshot: dict[NoteItem, tuple[float, int, float]] = field(default_factory=dict)
    press: Press | None = None
    view_origin: QPoint = field(default_factory=QPoint)
    rubber_origin: QPoint = field(default_factory=QPoint)
    created: NoteItem | None = None
    candidate: tuple | None = None
    collision_index: dict[tuple[int, int], list[Note]] | None = None
    collision_ignored: frozenset[Note] = frozenset()


class PianoRollView(QGraphicsView):
    """The roll: the note grid, the interaction with it and the drawn extras (spectrum, cursor).

    It also lights the lyric NOTE the playhead sits in and the sounds that map to it, at the
    playhead's position whether or not the transport runs, reading the NOTE blocks the mapping draws
    (`lyric_times`) and the note ids each sound was placed on (`mapped`).
    """

    view_changed = Signal()
    viewport_changed = Signal()
    zoom_changed = Signal()
    notes_changed = Signal()
    gesture_started = Signal()
    gesture_finished = Signal()
    channels_changed = Signal()
    active_channel_changed = Signal(int)
    hover_changed = Signal(object)
    note_preview = Signal(int)
    seek_requested = Signal(float)
    lyrics_changed = Signal()
    lyric_text_changed = Signal(str)
    lyric_highlight_changed = Signal()

    GRAB_PX = 7
    MIN_ZOOM_X, MAX_ZOOM_X = 12.0, 900.0
    MIN_ZOOM_Y, MAX_ZOOM_Y = 8.0, 64.0

    def __init__(self, parent=None):
        scene = QGraphicsScene()
        scene.setSceneRect(QRectF(0.0, 0.0, LENGTH_BEATS, PITCH_COUNT))
        super().__init__(scene, parent)
        self._scene = scene
        self.snap = 0.25
        self._interaction = Interaction.viewing()
        self.overtone_highlight = True
        self.note_lyrics_visible = True
        self.playing = False
        self.division = "beats"
        self.initial_center = (8.0, float(PITCH_MAX - 60))  # where the view opens, unless a session says otherwise
        self.hover_pitch: int | None = None
        self._preview_pitch: int | None = None
        self.playhead: float | None = None
        self._bpm = 120.0
        self.gain = 240.0
        self.contrast = 1.0
        self.spectrum: SpectrumImage | None = None
        self._zoom_x = 48.0
        self._zoom_y = 16.0
        self._grid_offset = 0.0  # seconds the drawn grid slides by; the notes and the playhead keep their time
        self._gesture: Gesture | None = None  # the pointer gesture in flight, from press to release
        self.document = Document(channels=[Channel(channel=0, color=theme.NOTE_PALETTE[0])])
        self._items: list[NoteItem] = []
        self._clipboard: tuple[tuple[int, float, float, int], ...] = ()
        self.active_channel = 0
        self._stack = QUndoStack(self)
        self._stack.setUndoLimit(HISTORY_LIMIT)
        self._gesture_before: _RollState | None = None
        self._gesture_text = ""
        self._history_depth = 0
        self._lines: tuple[SoundLine, ...] = ()
        self._lyric_times: tuple[tuple[tuple[float | None, float | None], ...], ...] = ()
        self._lyric_red: tuple[tuple[bool, ...], ...] = ()
        self._lyric_zero: tuple[tuple[bool, ...], ...] = ()
        self._lyric_group: tuple[tuple[int, ...], ...] = ()
        self._lyric_raw: tuple[tuple[tuple[float | None, float | None], ...], ...] = ()
        self._lyric_mapped: tuple[tuple[tuple[int, ...], ...], ...] = ()
        self._lyric_operations: tuple = ()
        self.lyric_evidence: tuple = ("", (), (), ())
        self._lyric_editable = True
        self.lyric_text = ""
        self.lyric_mapping_pending = False
        self._highlight_note: int | None = None
        self._highlight_sounds: tuple[tuple[int, int], ...] = ()

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.viewport().setMouseTracking(True)  # the row under the mouse is highlighted
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setTransform(QTransform.fromScale(self._zoom_x, self._zoom_y))

        self._rubber = _SelectionBox(self.viewport())
        self._rubber.hide()
        self._initialized = False

    @property
    def bpm(self) -> float:
        return self._bpm

    @bpm.setter
    def bpm(self, value: float) -> None:
        """Tempo of the beat grid. Notes hold their place in the audio, so a new tempo only
        re-divides the grid they sit on: their beat coordinates and the horizontal zoom are
        rescaled together, which leaves the audio and the notes where they are on screen."""
        tempo = max(1.0, float(value))
        scale = tempo / self._bpm  # seconds = beats * 60 / bpm, so the beats scale with the tempo
        if scale != 1.0:
            self._zoom_x = min(self.MAX_ZOOM_X, max(self.MIN_ZOOM_X, self._zoom_x / scale))
            self.setTransform(QTransform.fromScale(self._zoom_x, self._zoom_y))
            for note in self.notes():
                note.set_range(note.start * scale, note.pitch)
                note.set_duration(note.duration * scale)
        self._bpm = tempo
        self._update_scene()
        self.refresh()

    def content_beats(self) -> float:
        """Scrollable length: the default canvas, the spectrum or the notes, whichever is longest."""
        end = float(LENGTH_BEATS)
        if self.spectrum is not None:
            end = max(end, self.spectrum.spectrum.frames * self.frame_width())
        for note in self.notes():
            end = max(end, note.end)
        return end + CONTENT_MARGIN

    def _update_scene(self) -> None:
        self._scene.setSceneRect(QRectF(0.0, 0.0, self.content_beats(), PITCH_COUNT))

    def add_note(self, pitch: int, start: float, duration: float, channel: int | None = None) -> NoteItem:
        note = Note(pitch, start, duration, self.active_channel if channel is None else channel)
        with self._edit("Draw note"):
            self.document.add_note(note)
            item = self._add_item(note)
            self._update_scene()
            self.notes_changed.emit()
        return item

    def set_notes(self, notes, next_id: int | None = None) -> None:
        """Replace every note in one go: a project or an extraction arrives all at once.

        A note may carry a fourth element, the channel it plays on, and a fifth, the stable id it was
        saved under; `next_id` is the id the project file says is free next. Without it, the current
        allocator is kept, including when undo restores older notes.
        """
        with self._edit("Replace notes"):
            self._drop_items()
            self.document.replace_notes(
                (
                    Note(item[0], item[1], item[2], item[3] if len(item) > 3 else 0, item[4] if len(item) > 4 else 0)
                    for item in notes
                ),
                next_id,
            )
            for note in self.document.notes:
                self._add_item(note)
            self._sync_channel_visuals()
            self._update_scene()
            self.notes_changed.emit()

    def replace(self, channels, notes, text: str = "Replace notes") -> None:
        """Put a whole document in over the running one - a MIDI import, say - as a single step."""
        with self._edit(text):
            self.set_channels(channels)
            self.set_notes(notes)

    def clear_notes(self) -> None:
        with self._edit("Clear notes"):
            self._drop_items()
            self.document.clear_notes()
            self._update_scene()
            self.notes_changed.emit()
            self.view_changed.emit()

    # --- lyrics -----------------------------------------------------------

    @property
    def lyric_lines(self) -> tuple[SoundLine, ...]:
        return self._lines

    @property
    def lyric_times(self) -> tuple[tuple[tuple[float | None, float | None], ...], ...]:
        return self._lyric_times

    @property
    def lyric_red(self) -> tuple[tuple[bool, ...], ...]:
        """Whether the mapping doubts each sound, laid out as `lyric_times` is."""
        return self._lyric_red

    @property
    def lyric_zero(self) -> tuple[tuple[bool, ...], ...]:
        """Whether each sound covers no note, the sounds it draws grey."""
        return self._lyric_zero

    @property
    def lyric_group(self) -> tuple[tuple[int, ...], ...]:
        """The note each sound shares with its neighbours, -1 where it stands on its own."""
        return self._lyric_group

    @property
    def lyric_mapped(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """The stable ids of the notes each sound maps to, empty where it covers none."""
        return self._lyric_mapped

    @property
    def highlight_note(self) -> int | None:
        """The lyric NOTE the playhead sits inside, or None between sounds."""
        return self._highlight_note

    @property
    def highlight_sounds(self) -> tuple[tuple[int, int], ...]:
        """The sounds the lyric NOTE maps from, the ones the strip lights with the NOTE."""
        return self._highlight_sounds

    @property
    def lyric_raw(self) -> tuple[tuple[tuple[float | None, float | None], ...], ...]:
        """The editable onsets and reference durations the project keeps."""
        return self._lyric_raw

    @property
    def lyric_editable(self) -> bool:
        """Whether the strip may edit the KRC mapping."""
        return self._lyric_editable

    @lyric_editable.setter
    def lyric_editable(self, editable: bool) -> None:
        self._lyric_editable = bool(editable)
        self.lyrics_changed.emit()

    @property
    def lyric_operations(self) -> tuple:
        """The mapping's operations, kept with the undo state so an edit is one step."""
        return self._lyric_operations

    @lyric_operations.setter
    def lyric_operations(self, operations) -> None:
        self._lyric_operations = tuple(operations or ())

    @contextmanager
    def lyric_edit(self, text: str):
        """One user edit of the lyrics as one undo step, the way a note edit is recorded."""
        with self._edit(text):
            yield

    def set_lyrics(self, lines, times) -> None:
        """Take a whole aligned `.krc` over: every line's sounds and their times, as one step."""
        with self._edit("Align lyrics"):
            self.load_lyrics(lines, times)

    def load_lyrics(
        self, lines, times, red=None, raw=None, zero=None, group=None, mapped=None, editable=True, operations=None
    ) -> None:
        """The lyrics a `.krc` or a project brings in, with no undo step of their own.

        `times` are the spans the mapping draws, `raw` the `(onset, raw_length)` evidence whose
        onsets the strip edits, `red` the sounds the mapping doubts, `zero` those that cover no note,
        `group` the note each sound shares, `mapped` the stable ids of the notes each sound maps to
        and `editable` whether the strip may move `raw` at all. Without `red`/`zero` every sound is
        taken as sound, without `raw` evidence is captured from `times`, and without `mapped` no
        note is taken as a lyric NOTE.
        """
        self._lines = tuple(lines)
        self._lyric_editable = bool(editable)
        if operations is not None:
            self._lyric_operations = tuple(operations)
        self._lyric_times = tuple(tuple(span) for span in times)
        self._lyric_raw = tuple(tuple(sound) for sound in (from_spans(times) if raw is None else raw))
        self._lyric_red = (
            tuple(tuple(bool(flag) for flag in row) for row in red)
            if red is not None
            else tuple(tuple(False for _span in row) for row in times)
        )
        self._lyric_zero = (
            tuple(tuple(bool(flag) for flag in row) for row in zero)
            if zero is not None
            else tuple(tuple(False for _span in row) for row in times)
        )
        self._lyric_group = (
            tuple(tuple(int(flag) for flag in row) for row in group)
            if group is not None
            else tuple(tuple(-1 for _span in row) for row in times)
        )
        self._lyric_mapped = (
            tuple(tuple(tuple(int(note) for note in chunk) for chunk in row) for row in mapped)
            if mapped is not None
            else tuple(tuple(() for _span in row) for row in times)
        )
        self._sync_channel_visuals()
        self._sync_lyric_highlight()
        self.lyrics_changed.emit()
        self.view_changed.emit()

    def _lyric_highlight_at(self, seconds: float | None) -> tuple[int | None, tuple[tuple[int, int], ...]]:
        """The lyric NOTE the playhead sits in and every sound that maps to it, from the strip's own spans.

        A sound's block is the NOTE span the mapping derives, so the block holding `seconds` names the
        NOTE; every sound the mapping put on that NOTE lights with it, a `merge` run as a whole.
        """
        if seconds is None:
            return None, ()
        for row, spans in enumerate(self._lyric_times):
            mapped = self._lyric_mapped[row] if row < len(self._lyric_mapped) else ()
            for column, (start, end) in enumerate(spans):
                if start is None or end is None or not start <= seconds < end:
                    continue
                note_ids = set(mapped[column]) if column < len(mapped) else set()
                if not note_ids:
                    return None, ()
                shared = tuple(
                    (line, index)
                    for line, notes in enumerate(self._lyric_mapped)
                    for index, ids in enumerate(notes)
                    if note_ids & set(ids)
                )
                return min(note_ids), shared
        return None, ()

    def _sync_lyric_highlight(self) -> None:
        """Light the lyric NOTE under the playhead and the sounds it maps from, playing or paused."""
        note, sounds = self._lyric_highlight_at(self.playhead)
        if note == self._highlight_note and sounds == self._highlight_sounds:
            return
        self._highlight_note = note
        self._highlight_sounds = sounds
        self.lyric_highlight_changed.emit()

    def set_sound_onset(self, row: int, column: int, seconds: float, base=None) -> bool:
        """Move one onset between its channel neighbours without changing reference durations.

        `base` is the row at the start of the gesture, so dragging home restores its evidence.
        Only confirmed operations containing the edited Sound lose their confirmation.
        """
        if not 0 <= row < len(self._lyric_raw) or not math.isfinite(seconds):
            return False
        sounds = list(self._lyric_raw[row] if base is None else base)
        if not 0 <= column < len(sounds):
            return False
        channel = self.lyric_lines[row].sounds[column].channel
        neighbours = [
            ((line, at), onset)
            for line, spans in enumerate(self._lyric_raw)
            for at, (onset, _length) in enumerate(sounds if line == row else spans)
            if onset is not None and self.lyric_lines[line].sounds[at].channel == channel
        ]
        low = next((onset for ref, onset in reversed(neighbours) if ref < (row, column)), 0.0)
        high = next((onset for ref, onset in neighbours if ref > (row, column)), math.inf)
        value = min(max(0.0, float(seconds), low), high)
        sounds[column] = (value, sounds[column][1])
        if tuple(sounds) == self._lyric_raw[row]:
            return False
        self._lyric_raw = self._lyric_raw[:row] + (tuple(sounds),) + self._lyric_raw[row + 1 :]
        self._lyric_operations = tuple(
            operation
            for operation in self._lyric_operations
            if not (operation.confirmed and any(ref.line == row and ref.index == column for ref in operation.sounds))
        )
        self.lyrics_changed.emit()
        return True

    @property
    def gesture_active(self) -> bool:
        return self._gesture_before is not None

    def note_seconds(self) -> list[tuple[float, float]]:
        return [(self.to_seconds(note.start), self.to_seconds(note.end)) for note in self.notes()]

    def notes(self) -> list[NoteItem]:
        return list(self._items)

    def set_filtered_notes(self, ids) -> None:
        """Mark the target channel's notes a conflict kept out of the preview stream."""
        wanted = set(ids)
        for item in self._items:
            item.filtered = item.id in wanted
        self._sync_channel_visuals()

    def _add_item(self, note: Note) -> NoteItem:
        item = NoteItem(note)
        item.fill, item.edge_light, item.edge_dark = theme.note_shades(self.channel_color(note.channel))
        channel = self._channel(note.channel)
        item.setVisible(self.edit_mode and (channel.visible if channel else True))
        self._scene.addItem(item)
        self._items.append(item)
        return item

    def _drop_item(self, item: NoteItem) -> None:
        self._scene.removeItem(item)
        self._items.remove(item)

    def _drop_items(self) -> None:
        for item in list(self._items):
            self._scene.removeItem(item)
        self._items.clear()

    # --- history ----------------------------------------------------------

    @property
    def undo_stack(self) -> QUndoStack:
        return self._stack

    def undo(self) -> None:
        self._stack.undo()

    def redo(self) -> None:
        self._stack.redo()

    def _capture(self) -> _RollState:
        return _RollState(
            channels=tuple(self.channels),
            notes=tuple(
                (note.pitch, self.to_seconds(note.start), self.to_seconds(note.duration), note.channel, note.id)
                for note in self.notes()
            ),
            selected=frozenset(index for index, item in enumerate(self._items) if item.isSelected()),
            active_channel=self.active_channel,
            lyrics=self._lines,
            lyric_raw=self._lyric_raw,
            lyric_operations=self._lyric_operations,
            lyric_evidence=self.lyric_evidence,
            lyric_text=self.lyric_text,
        )

    def _push(self, before: _RollState, after: _RollState, text: str) -> bool:
        """Record one step, and say whether it changed anything worth a step."""
        if _state_data(before) == _state_data(after):
            return False
        self._stack.push(_RollEdit(self, before, after, text))
        return True

    @contextmanager
    def _edit(self, text: str):
        """Record one discrete edit as one step; nested edits join the outer one, and a gesture
        records itself in `commit_gesture`."""
        if self._history_depth or self._gesture_before is not None:
            yield
            return
        before = self._capture()
        self._history_depth += 1
        try:
            yield
        finally:
            self._history_depth -= 1
        self._push(before, self._capture(), text)

    def begin_gesture(self, text: str) -> None:
        if self._gesture_before is None:
            self._gesture_before = self._capture()
            self._gesture_text = text
            self.gesture_started.emit()

    def commit_gesture(self) -> None:
        before, text = self._gesture_before, self._gesture_text
        if before is None:
            return
        self._gesture_before = None
        self._gesture_text = ""
        if self._push(before, self._capture(), text):
            self.notes_changed.emit()  # a move or a trim is a change of the notes like any other
        self.gesture_finished.emit()

    def cancel_gesture(self) -> None:
        """Drop the gesture in flight: undo its in-memory effect and record no undo step.

        `Escape` calls this, so a move or a trim goes back to where it started and a draw takes the
        note it opened away again; `commit_gesture` on the release then finds no gesture to push.
        """
        if self._gesture_before is None:
            return
        before = self._gesture_before
        self._gesture_before = None
        self._gesture_text = ""
        gesture, self._gesture = self._gesture, None
        if gesture is None:
            self.lyric_text = before.lyric_text
            self.lyric_evidence = before.lyric_evidence
            self.lyric_text_changed.emit(before.lyric_text)
            self._lines = before.lyrics
            self._lyric_raw = before.lyric_raw
            self._lyric_operations = before.lyric_operations
            self.lyrics_changed.emit()
            self.gesture_finished.emit()
            return
        created = gesture.created is not None and gesture.created in self._items
        if created:
            self.document.remove_note(gesture.created.note)
            self._drop_item(gesture.created)
        for item, (start, pitch, duration) in gesture.snapshot.items():
            if item in self._items:
                item.set_range(start, pitch)
                item.set_duration(duration)
        self._update_scene()
        self._sync_channel_visuals()
        self.view_changed.emit()
        if created:
            self.notes_changed.emit()
        self.gesture_finished.emit()

    def _restore_state(self, state: _RollState) -> None:
        """Put a snapshot back in one rebuild, selecting the same notes by position again.

        The lyrics go in first, so the remap the notes trigger reads the times restored. The notes
        go in before the channels: a channel list only holds the numbers its notes play on, so a
        channel that is about to lose its notes would come straight back if it were set while they
        were still there.
        """
        self._history_depth += 1
        try:
            self.lyric_text = state.lyric_text
            self.lyric_evidence = state.lyric_evidence
            self.lyric_text_changed.emit(state.lyric_text)
            self._lines = state.lyrics
            self._lyric_raw = state.lyric_raw
            self._lyric_operations = state.lyric_operations
            self._lyric_mapped = ()
            self._sync_lyric_highlight()
            self.set_notes(
                (pitch, self.to_beats(start), self.to_beats(duration), channel, identifier)
                for pitch, start, duration, channel, identifier in state.notes
            )
            self.set_channels(state.channels)
            for index in state.selected:
                if index < len(self._items):
                    self._items[index].setSelected(True)
            self.set_active_channel(state.active_channel)
            self.lyrics_changed.emit()
        finally:
            self._history_depth -= 1
        self.refresh()

    # --- channels ---------------------------------------------------------

    @property
    def channels(self) -> list[Channel]:
        """The document's channel list; the panel and the players read it, nothing holds a copy."""
        return self.document.channels

    def _channel(self, number: int) -> Channel | None:
        return next((channel for channel in self.channels if channel.channel == number), None)

    def channel_color(self, number: int) -> QColor:
        channel = self._channel(number)
        color = QColor(channel.color) if channel is not None else QColor()
        if not color.isValid():
            color = QColor(theme.NOTE_PALETTE[number % len(theme.NOTE_PALETTE)])
        return color

    def set_note_lyrics_visible(self, visible: bool) -> None:
        self.note_lyrics_visible = visible
        self._sync_channel_visuals()

    def _sync_channel_visuals(self) -> None:
        """Body colour, bevel, stacking and visibility all come from the channel list; the notes only
        show while editing, the way WaveTone keeps its graph to the spectrum outside note edit mode."""
        labels = {}
        for row, line in enumerate(self._lines):
            mapped = self._lyric_mapped[row] if row < len(self._lyric_mapped) else ()
            zero = self._lyric_zero[row] if row < len(self._lyric_zero) else ()
            first = next((ids[0] for ids in mapped if ids), None)
            previous = None
            for column, sound in enumerate(line.sounds):
                ids = mapped[column] if column < len(mapped) else ()
                dropped = column < len(zero) and zero[column]
                if ids:
                    previous = ids[-1]
                elif dropped:
                    identifier = previous if previous is not None else first
                    ids = (identifier,) if identifier is not None else ()
                for position, identifier in enumerate(ids):
                    prefix = "↳ " if position else ""
                    entries = labels.setdefault(identifier, [])
                    if (
                        entries
                        and sound.rubied
                        and entries[-1][0].rubied
                        and entries[-1][0].base == sound.base
                        and entries[-1][2:] == (dropped, prefix, row, column - 1)
                    ):
                        last = entries[-1]
                        entries[-1] = (last[0], last[1] + sound.reading, dropped, prefix, row, column)
                    else:
                        entries.append((sound, sound.reading, dropped, prefix, row, column))
        for note in self.notes():
            runs = []
            for sound, reading, zero, prefix, _row, _column in labels.get(note.id, ()):
                text = reading
                if sound.rubied:
                    left, right = ("(", ")") if sound.first else ("[", "]")
                    text += f"{left}{sound.base}{right}"
                runs.append(((" " if runs else "") + prefix + text, zero))
            text = "".join(text for text, _zero in runs)
            note.lyric_runs = tuple(runs) if self.note_lyrics_visible else ()
            note.lyric_text = text if self.note_lyrics_visible else ""
            warning = (
                tr("Overlapping note: excluded from lyric mapping. Move, shorten or remove the overlapping notes.")
                if note.filtered
                else ""
            )
            note.setToolTip("\n".join(part for part in (text, warning) if part))
            note.update()
            note.fill, note.edge_light, note.edge_dark = theme.note_shades(self.channel_color(note.channel))
            channel = self._channel(note.channel)
            note.setVisible(self.edit_mode and (channel.visible if channel else True))
            note.setZValue(note.channel)

    def set_channels(self, channels) -> None:
        """Replace the channel list in one go, the way a project hands it over.

        A note whose channel is missing from the list gets a plain entry back rather than being
        dropped: a channel number is always a place a note can play on.
        """
        with self._edit("Set channels"):
            self.document.set_channels(channels or [Channel(channel=0, color=theme.NOTE_PALETTE[0])])
            for channel in list(self.channels):
                if not channel.color:
                    self.document.set_channel_field(
                        channel.channel, color=theme.NOTE_PALETTE[channel.channel % len(theme.NOTE_PALETTE)]
                    )
            if self.active_channel not in {channel.channel for channel in self.channels}:
                self.active_channel = self.channels[0].channel
            self._sync_channel_visuals()
            self.channels_changed.emit()
            self.active_channel_changed.emit(self.active_channel)

    def add_channel(self, program: int = 0) -> Channel | None:
        number = free_channel(self.channels)
        if number is None:
            return None
        channel = Channel(channel=number, color=theme.NOTE_PALETTE[number % len(theme.NOTE_PALETTE)], program=program)
        with self._edit("Add channel"):
            self.document.add_channel(channel)
            self.channels_changed.emit()
        return channel

    def remove_channel(self, number: int) -> bool:
        """Drop a channel and the notes on it; the other numbers stay as they are. The last one stays."""
        if len(self.channels) <= 1:
            return False
        with self._edit("Delete channel"):
            removed = self.document.remove_channel(number)
            gone = {id(note) for note in removed or ()}
            for item in list(self._items):
                if id(item.note) in gone:
                    self._drop_item(item)
            if self.active_channel not in {channel.channel for channel in self.channels}:
                self.active_channel = self.channels[0].channel
            self._sync_channel_visuals()
            if removed:
                self.notes_changed.emit()
            self.channels_changed.emit()
            self.active_channel_changed.emit(self.active_channel)
        return True

    def set_channel_field(self, number: int, **fields) -> None:
        with self._edit("Edit channel"):
            self.document.set_channel_field(number, **fields)
            self._sync_channel_visuals()
            self.channels_changed.emit()

    def set_channel_number(self, old: int, new: int) -> bool:
        """Move a channel and its notes onto another MIDI channel; its colour follows the number."""
        with self._edit("Change channel"):
            if not self.document.set_channel_number(old, new):
                return False
            self.document.set_channel_field(new, color=theme.NOTE_PALETTE[new % len(theme.NOTE_PALETTE)])
            if self.active_channel == old:
                self.active_channel = new
            self._sync_channel_visuals()
            self.notes_changed.emit()
            self.channels_changed.emit()
            self.active_channel_changed.emit(self.active_channel)
        return True

    def set_active_channel(self, number: int) -> None:
        if number in {channel.channel for channel in self.channels} and number != self.active_channel:
            self.active_channel = number
            self.active_channel_changed.emit(number)

    def move_selection_to_channel(self, number: int | None) -> bool:
        """File the selected notes under a channel; None makes a fresh one for them instead.

        Only what the lock leaves alone moves, the notes keep their place and take the target's
        colour and stacking order, and creating the channel is part of the same undo step.
        """
        if not self.edit_mode:
            return False
        selection = [item for item in self.selected_notes() if not self._locked(item.channel)]
        if not selection:
            return False
        if number is not None and self._locked(number):
            return False
        with self._edit("Move notes to channel"):
            if number is None:
                created = self.add_channel()
                if created is None:
                    return False
                number = created.channel
            elif self._channel(number) is None:
                return False
            for item in selection:
                item.note.channel = number
            self._sync_channel_visuals()
            self.notes_changed.emit()
            self.view_changed.emit()
        return True

    def _locked(self, number: int) -> bool:
        channel = self._channel(number)
        return channel.lock if channel is not None else False

    def selected_notes(self) -> list[NoteItem]:
        return [note for note in self.notes() if note.isSelected()]

    def delete_selection(self) -> bool:
        """Remove the selected notes as one step, leaving a locked channel's notes alone; nothing
        outside edit mode."""
        if not self.edit_mode:
            return False
        selection = [item for item in self.selected_notes() if not self._locked(item.channel)]
        with self._edit("Delete notes"):
            for item in selection:
                self.document.remove_note(item.note)
                self._drop_item(item)
            if selection:
                self.notes_changed.emit()
                self.view_changed.emit()
        return bool(selection)

    def copy_selection(self) -> bool:
        """Take the selected notes as one block, timed from the earliest of them."""
        if not self.edit_mode:
            return False
        selection = sorted(self.selected_notes(), key=lambda note: note.start)
        if not selection:
            return False
        anchor = selection[0].start
        self._clipboard = tuple((note.pitch, note.start - anchor, note.duration, note.channel) for note in selection)
        return True

    def paste_notes(self) -> bool:
        """Drop the copied block at the playhead, its earliest note on the snapped grid and the
        spacing between them quantised to the same cell, so the block arrives on the current snap.
        """
        if not self.edit_mode or not self._clipboard:
            return False
        known = {channel.channel for channel in self.channels}
        anchor = self._snap_beats((self.playhead or 0.0) * self.bpm / 60.0)
        placed = False
        with self._edit("Paste notes"):
            for pitch, offset, duration, channel in self._clipboard:
                place = channel if channel in known else self.active_channel
                if self._locked(place):
                    continue  # a locked channel takes no new notes
                if not placed:
                    self._clear_selection()
                placed = True
                note = Note(pitch, anchor + self._snap_beats(offset), duration, place)
                self.document.add_note(note)
                self._add_item(note).setSelected(True)
            self._update_scene()
            if placed:
                self.notes_changed.emit()
            self.view_changed.emit()
        return placed

    def quantize_notes(self) -> bool:
        """Put the starts and ends of the notes on the snap grid, so they sit on the current beats.

        The selection when there is one and the whole roll otherwise; a locked channel is never
        touched; a note that rounds away keeps one cell. The grid is a setting and this is the one
        command that applies it, so both stay usable outside edit mode.
        """
        cell = self._cell_beats()
        targets = [note for note in (self.selected_notes() or self.notes()) if not self._locked(note.channel)]
        moved = False
        with self._edit("Quantize notes"):
            for note in targets:
                start = self._snap_beats(note.start)
                duration = max(self._snap_beats(note.end), start + cell) - start
                if (start, duration) == (note.start, note.duration):
                    continue
                if self._would_collide(note, start, note.pitch, duration, (note.note,)):
                    continue  # the grid would fold this note onto another of its pitch; leave it be
                note.set_range(start, note.pitch)
                note.set_duration(duration)
                moved = True
            if moved:
                self.notes_changed.emit()
                self.view_changed.emit()
        return moved

    def set_spectrum(self, spectrum: NoteSpectrum | None) -> None:
        self.spectrum = SpectrumImage(spectrum) if spectrum is not None else None
        self._update_scene()
        self.refresh()

    def refresh(self) -> None:
        self.viewport().update()
        self.view_changed.emit()

    def canvas(self) -> theme.Canvas:
        """The colours the roll paints with: the fixed dark set once a spectrum covers it, else the
        desktop's own light or dark."""
        return theme.canvas("dark") if self.spectrum is not None else theme.canvas()

    @property
    def interaction(self) -> Interaction:
        return self._interaction

    @property
    def edit_mode(self) -> bool:
        return self._interaction.editing

    @property
    def tool(self) -> Tool | None:
        return self._interaction.tool

    def apply_interaction(self, state: Interaction) -> None:
        """The one place a mode and tool land: it decides what is drawn and how it answers."""
        self._interaction = state
        self._sync_channel_visuals()
        self.refresh()

    def frame_width(self) -> float:
        """Scene width of one spectrum frame, in beats."""
        return self.spectrum.spectrum.frame_ms / 1000.0 * self.bpm / 60.0

    def highlight_pitches(self) -> list[int]:
        """The rows to tint behind the notes: the row under the mouse, and - with the overtone
        highlight on - f, 2f, 3f and 4f above it, the WaveTone hint about where a note would double."""
        if self.hover_pitch is None:
            return []
        if not self.overtone_highlight:
            return [self.hover_pitch]
        pitches = [self.hover_pitch]
        for harmonic in OVERTONES:
            pitch = self.hover_pitch + round(12 * math.log2(harmonic))
            if PITCH_MIN <= pitch <= PITCH_MAX and pitch not in pitches:
                pitches.append(pitch)
        return pitches

    def set_hover_pitch(self, pitch: int | None) -> None:
        if pitch != self.hover_pitch:
            self.hover_pitch = pitch
            self.hover_changed.emit(pitch)
            self.refresh()

    def set_playhead(self, seconds: float | None) -> None:
        if seconds != self.playhead:
            viewport = self.viewport()
            for position in (self.playhead, seconds):
                if position is not None:
                    x = self.mapFromScene(QPointF(self.to_beats(position), 0.0)).x()
                    region = QRect(x - 3, 0, 7, viewport.height()).intersected(viewport.rect())
                    if not region.isEmpty():
                        viewport.update(region)
            self.playhead = seconds
            self._sync_lyric_highlight()

    def pixels_per_beat(self) -> float:
        return self._zoom_x

    @property
    def zoom(self) -> tuple[float, float]:
        """Pixels per beat and per semitone row, as the zoom controls leave them."""
        return self._zoom_x, self._zoom_y

    def set_zoom(self, zoom_x: float, zoom_y: float) -> None:
        """Start at a given zoom, the way `showEvent` centers the view where the session left it."""
        self._zoom_x = min(self.MAX_ZOOM_X, max(self.MIN_ZOOM_X, zoom_x))
        self._zoom_y = min(self.MAX_ZOOM_Y, max(self.MIN_ZOOM_Y, zoom_y))
        self.setTransform(QTransform.fromScale(self._zoom_x, self._zoom_y))
        self.refresh()
        self.viewport_changed.emit()

    def center_on(self, x: float, y: float) -> None:
        """Put (x, y) in the middle of the viewport, the way opening a project restores its view."""
        self.initial_center = (x, y)
        self.centerOn(x, y)
        self.refresh()
        self.viewport_changed.emit()

    def follow_playhead(self, seconds: float) -> None:
        """Leave the page alone until the playhead reaches its right edge, then turn it."""
        x = seconds * self.bpm / 60.0
        page = self.mapToScene(self.viewport().rect()).boundingRect()
        if page.left() <= x <= page.right() - page.width() * PAGE_TURN_MARGIN:
            return
        self.centerOn(x + page.width() * PAGE_TURN_LEAD, page.center().y())

    @property
    def seconds_per_beat(self) -> float:
        """What one scene unit is worth in time: the one conversion every caller has to get right."""
        return 60.0 / self.bpm

    def to_seconds(self, beats: float) -> float:
        """A scene-unit length or position in seconds, the unit the files and the players speak."""
        return beats * self.seconds_per_beat

    def to_beats(self, seconds: float) -> float:
        """A time in seconds as scene units, the unit the notes are drawn and kept in."""
        return seconds / self.seconds_per_beat

    @property
    def offset(self) -> float:
        """How far the drawn grid slides, in seconds; `+` to the right, `-` to the left."""
        return self._grid_offset

    @property
    def offset_beats(self) -> float:
        """Where the grid's own zero sits, in scene units: `set_offset`'s seconds as beats."""
        return self._grid_offset * self.bpm / 60.0

    def set_offset(self, seconds: float) -> None:
        """Slide the drawn grid and its snapping by `seconds`: `+` to the right, `-` to the left.

        Only the grid moves - the notes, the spectrum and the playhead stay where their timestamps
        put them.
        """
        self._grid_offset = float(seconds)
        self.refresh()
        self.viewport_changed.emit()

    def seconds_at_viewport_x(self, x: float) -> float:
        """The timeline position under a viewport x, for the widgets that share the roll's columns."""
        return max(0.0, self.mapToScene(QPoint(int(x), 0)).x()) * 60.0 / self.bpm

    def pixels_per_second(self) -> float:
        return self._zoom_x * self.bpm / 60.0

    def seconds_lines(self, rect: QRectF, minimum: float) -> list[tuple[float, float, bool]]:
        """(x, seconds, major) of the time grid across `rect`, on the 1-2-5 step that keeps the
        lines at least `minimum` pixels apart and marks every step above it as a major line.

        The clock is absolute: `editor.grid_offset_ms` slides the beat grid, not the time axis, so a
        label always names the second it sits on.
        """
        index = TIME_STEPS.index(time_step(self.pixels_per_second(), minimum))
        step, major = TIME_STEPS[index], TIME_STEPS[min(index + 1, len(TIME_STEPS) - 1)]
        beats_per_second = self.bpm / 60.0
        value = math.floor(rect.left() / (step * beats_per_second)) * step
        lines = []
        while value * beats_per_second <= rect.right():
            lines.append((value * beats_per_second, value, _is_multiple(value, major)))
            value += step
        return lines

    def division_lines(self, rect: QRectF, minimum: float, snap: bool = False) -> list[tuple[float, int]]:
        """Time-axis grid lines across `rect` as (x, level) pairs, 0 minor, 1 beat, 2 prominent.

        The division picks the basis: beats of the tempo map, or seconds. `snap` follows the snap
        grid on the beats basis, for the canvas that has no spectrum behind it.
        """
        if self.division == "seconds":
            return [(x, 2 if major else 0) for x, _seconds, major in self.seconds_lines(rect, minimum)]
        offset = self.offset_beats
        step = self.grid_step() if snap else 1.0
        value = math.floor((rect.left() - offset) / step) * step + offset
        lines = []
        while value <= rect.right():
            grid = value - offset
            level = 2 if _is_multiple(grid, BAR_BEATS) else 1 if _is_multiple(grid, 1.0) else 0
            if self._zoom_x * (BAR_BEATS if level == 2 else step) >= minimum:
                lines.append((value, level))
            value += step
        return lines

    def grid_step(self) -> float:
        for step in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, BAR_BEATS):
            if step * self._zoom_x >= MIN_GRID_SPACING:
                return step
        return BAR_BEATS

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            self.centerOn(*self.initial_center)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.view_changed.emit()
        self.viewport_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.view_changed.emit()
        self.viewport_changed.emit()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        first_row = max(0, int(math.floor(rect.top())))
        last_row = min(PITCH_COUNT, int(math.ceil(rect.bottom())) + 1)
        colors = self.canvas()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # keeps 1px grid lines on one pixel

        if self.spectrum is None:
            painter.fillRect(rect, colors.background)
            for row in range(first_row, last_row):
                color = colors.row_black if is_black_key(PITCH_MAX - row) else colors.row_white
                painter.fillRect(QRectF(rect.left(), float(row), rect.width(), 1.0), color)
            painter.setPen(QPen(colors.grid_line, 0))
            for row in range(first_row, last_row + 1):
                y = float(row)
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            for x, level in self.division_lines(rect, MIN_GRID_SPACING, snap=True):
                painter.setPen(QPen((colors.grid_line, colors.grid_beat, colors.grid_bar)[level], 0))
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            self._draw_cursor(painter, rect)
            painter.restore()
            return

        painter.fillRect(rect, colors.spectrum_background)
        self._draw_spectrum(painter, rect)
        painter.setPen(QPen(colors.spectrum_octave, 0))
        for pitch in range(MIDI_OFFSET, MIDI_OFFSET + NOTE_COUNT + 1, 12):
            y = float(PITCH_MAX - pitch + 1)  # the C row's lower edge (B sits below C)
            if first_row <= y <= last_row:
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for x, level in self.division_lines(rect, MIN_LINE_SPACING):
            painter.setPen(QPen(colors.spectrum_bar if level == 2 else colors.spectrum_beat, 0))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        self._draw_cursor(painter, rect)
        painter.restore()

    def _draw_cursor(self, painter: QPainter, rect: QRectF) -> None:
        """The rows highlighted under the mouse; the playback position goes over the notes."""
        painter.setPen(Qt.PenStyle.NoPen)
        colors = self.canvas()
        # the spectrum is dark in either canvas, so the row over it takes the band that shows there
        band = colors.spectrum_hover_band if self.spectrum is not None else colors.hover_band
        painter.setBrush(band)
        for pitch in self.highlight_pitches():
            painter.drawRect(QRectF(rect.left(), float(PITCH_MAX - pitch), rect.width(), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """The playback position, over the notes so that they never hide it."""
        if self.playhead is None:
            return
        x = self.playhead * self.bpm / 60.0
        if not rect.left() <= x <= rect.right():
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(self.canvas().playhead, 0))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.restore()

    def _draw_spectrum(self, painter: QPainter, rect: QRectF) -> None:
        width = self.frame_width()
        first = max(0, int(math.floor(rect.left() / width)))
        last = min(self.spectrum.spectrum.frames, int(math.ceil(rect.right() / width)) + 1)
        if last <= first:
            return
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        image_first = max(0, int(math.floor(visible.left() / width))) // IMAGE_BLOCK_FRAMES * IMAGE_BLOCK_FRAMES
        image_last = min(
            self.spectrum.spectrum.frames,
            math.ceil((int(math.ceil(visible.right() / width)) + 1) / IMAGE_BLOCK_FRAMES) * IMAGE_BLOCK_FRAMES,
        )
        image = self.spectrum.image(self.gain, self.contrast, image_first, image_last)
        target = QRectF(first * width, SPECTRUM_TOP, (last - first) * width, float(image.height()))
        source = QRectF(first - image_first, 0.0, last - first, float(image.height()))
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, width * self._zoom_x < 1.0 or self._zoom_y < 1.0
        )
        if self.edit_mode:
            painter.setOpacity(EDIT_DIM)  # blends with the black underneath, so only the cells fade
        painter.drawImage(target, image, source)
        painter.setOpacity(1.0)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

    # --- interaction -----------------------------------------------------

    def _note_at(self, scene_pos: QPointF) -> NoteItem | None:
        item = self.itemAt(self.mapFromScene(scene_pos))
        return item if isinstance(item, NoteItem) else None

    def pitch_at(self, y: float) -> int:
        row = min(PITCH_COUNT - 1, max(0, int(math.floor(y))))
        return PITCH_MAX - row

    def _cell_beats(self) -> float:
        """One snap cell, but never shorter than a note can be."""
        return max(self.snap, MIN_DURATION)

    def _would_collide(self, item: NoteItem, start: float, pitch: int, duration: float, ignore=(), index=None) -> bool:
        """Whether putting a note there would sound its pitch twice over the same time on its channel."""
        gesture = self._gesture
        if index is None and gesture is not None and item in gesture.snapshot:
            ignored = frozenset(ignore)
            if gesture.collision_index is None or ignored != gesture.collision_ignored:
                gesture.collision_index = {
                    voice: [note for note in notes if note not in ignored]
                    for voice, notes in self.document.index_by_voice().items()
                }
            gesture.collision_ignored = ignored
            index = gesture.collision_index
        return self.document.collides(pitch, start, duration, item.channel, ignore, index)

    def _snap_beats(self, x: float) -> float:
        offset = self.offset_beats
        return round((x - offset) / self.snap) * self.snap + offset

    def snap_movement_beats(self, x: float) -> float:
        """Whole snap cells: a movement keeps the offset a block already had on the grid."""
        return round(x / self.snap) * self.snap

    def _snap_floor_beats(self, x: float) -> float:
        offset = self.offset_beats
        return math.floor((x - offset) / self.snap) * self.snap + offset

    def _snap_ceil_beats(self, x: float) -> float:
        offset = self.offset_beats
        return math.ceil((x - offset) / self.snap) * self.snap + offset

    def _clear_selection(self) -> None:
        for note in self.notes():
            note.setSelected(False)

    def channel_menu(self, scene_pos: QPointF) -> QMenu | None:
        """The menu that moves the selected notes onto another channel, a fresh one included.

        A right click over an unselected note is built for that note alone, so a note can be filed
        without a trip to the select tool. Nothing comes back outside edit mode or with nothing to
        move, and the selection is left as it was until an entry is actually picked.
        """
        if not self.edit_mode:
            return None
        note = self._note_at(scene_pos)
        if note is not None and not note.isSelected() and not self._locked(note.channel):
            selection = [note]
        else:
            selection = [item for item in self.selected_notes() if not self._locked(item.channel)]
        if not selection:
            return None
        current = {item.channel for item in selection}
        menu = QMenu(self)
        others = [channel for channel in self.channels if channel.channel not in current and not channel.lock]
        for channel in others:
            menu.addAction(channel.label).setData(channel.channel)
        if others:
            menu.addSeparator()
        new = menu.addAction(tr("New channel"))
        new.setData(NEW_CHANNEL)
        new.setEnabled(free_channel(self.channels) is not None)
        return menu

    def contextMenuEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())
        menu = self.channel_menu(scene_pos)
        if menu is None:
            return
        action = menu.exec(event.globalPos())
        if action is None:
            return
        note = self._note_at(scene_pos)
        if note is not None and not note.isSelected() and not self._locked(note.channel):
            # the menu was built for this note alone; take the selection over only once it is picked
            self._clear_selection()
            note.setSelected(True)
        number = action.data()
        self.move_selection_to_channel(None if number == NEW_CHANNEL else number)

    def mouseDoubleClickEvent(self, event) -> None:
        # Qt files the second of two quick clicks as a double-click, and that press has to seek and
        # audition like any other: the roll has no double-click gesture for it to mean instead
        self.mousePressEvent(event)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint()
        scene_pos = self.mapToScene(pos)

        if event.button() == Qt.MouseButton.MiddleButton:
            self._gesture = Gesture(Drag.PAN, scene_pos, view_origin=pos)
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if not self.playing:
                # seeking is independent of the mode and of the tool, so every press in the roll moves
                # the playhead, over a note as well as over the grid. While the file is playing the roll
                # keeps its cursor and stays an editor; the time ruler is the one that seeks then
                self.seek_requested.emit(self.seconds_at_viewport_x(pos.x()))
            # and it sounds the row it lands on, whatever the mode, so a click is heard while editing
            # and while only listening
            self._preview_pitch = self.pitch_at(scene_pos.y())
            self.note_preview.emit(self._preview_pitch)
            if not self.edit_mode:
                self._gesture = Gesture(Drag.SEEK, scene_pos)
                return

        if event.button() == Qt.MouseButton.RightButton:
            # the context menu is what a right click opens; without this the scene would drop the
            # selection before `contextMenuEvent` ever sees it
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        note = self._note_at(scene_pos)
        self._gesture = None  # the gesture, if any, is set again below
        modifier = event.modifiers()
        ctrl = bool(modifier & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifier & Qt.KeyboardModifier.ShiftModifier)
        if note is not None and self._locked(note.channel):
            return
        if note is not None:
            self.set_active_channel(note.channel)  # picking a note files the pen under its channel

        if note is None:
            if ctrl or self.tool is Tool.SELECT:
                if not shift:
                    self._clear_selection()
                self._gesture = Gesture(Drag.SELECT, scene_pos, rubber_origin=pos)
                self._rubber.setGeometry(QRect(pos, pos))
                self._rubber.show()
                return
            if self._locked(self.active_channel):
                return
            pitch = self.pitch_at(scene_pos.y())
            start = max(0.0, self._snap_floor_beats(scene_pos.x()))
            duration = self._cell_beats()
            # the inset leaves a strip above and below a note that the hit test misses, yet the
            # click still reads as that note's pitch: without this the pen opens a note over it
            if self.document.collides(pitch, start, duration, self.active_channel):
                return
            self.begin_gesture("Draw note")
            note = self.add_note(pitch, start, duration)
            self._clear_selection()
            note.setSelected(True)
            self._gesture = Gesture(
                Drag.DRAW,
                scene_pos,
                target=note,
                snapshot={note: (note.start, note.pitch, note.duration)},
                created=note,
            )
            self.view_changed.emit()
            return

        if shift:
            # Shift splits a note into two halves: the left one moves its start, the right one its
            # end; Ctrl is what adds to the selection.
            self._clear_selection()
            note.setSelected(True)
            edge = Edge.START if scene_pos.x() < note.start + note.duration / 2 else Edge.END
            self.begin_gesture("Trim note")
            self._gesture = Gesture(
                Drag.TRIM,
                scene_pos,
                target=note,
                edge=edge,
                snapshot={note: (note.start, note.pitch, note.duration)},
            )
            return

        press = None
        if ctrl:
            note.setSelected(not note.isSelected())
        elif not note.isSelected():
            self._clear_selection()
            note.setSelected(True)
        else:
            # the press keeps the selection a drag would carry; if it stays a click, the release
            # narrows it to this one note
            press = Press(note, event.position())
            for item in self.selected_notes():
                if item.channel != note.channel:
                    item.setSelected(False)
        if not note.isSelected():
            return

        # a press on either edge changes the duration: the left one moves the start, the right one the end
        part = block_part(scene_pos.x(), note.start, note.end, self.GRAB_PX / self._zoom_x)
        kind, edge = Drag.MOVE, None
        if part == "right":
            kind, edge = Drag.TRIM, Edge.END
        elif part == "left":
            kind, edge = Drag.TRIM, Edge.START
        self.begin_gesture("Trim note" if kind is Drag.TRIM else "Move notes")
        self._gesture = Gesture(
            kind,
            scene_pos,
            target=note,
            edge=edge,
            snapshot={n: (n.start, n.pitch, n.duration) for n in self.selected_notes()},
            press=press,
        )

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        scene_pos = self.mapToScene(pos)
        gesture = self._gesture
        if gesture is not None and gesture.press is not None:
            gesture.press.follow(event.position())  # past the slop, the release stops being a click
        self.set_hover_pitch(self.pitch_at(scene_pos.y()))

        if gesture is not None and gesture.kind is not Drag.PAN:
            self._follow(pos, scene_pos)

        if gesture is None:
            note = self._note_at(scene_pos) if self.edit_mode else None
            if note is None:
                self.viewport().unsetCursor()
            else:
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                part = block_part(scene_pos.x(), note.start, note.end, self.GRAB_PX / self._zoom_x)
                shape = Qt.CursorShape.SizeHorCursor if shift or part != "move" else Qt.CursorShape.OpenHandCursor
                self.viewport().setCursor(shape)
            return

        if gesture.kind is Drag.PAN:
            delta = pos - gesture.view_origin
            gesture.view_origin = pos
            hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            return

        if gesture.kind is Drag.SELECT:
            self._rubber.setGeometry(QRect(gesture.rubber_origin, pos).normalized())
            region = self.mapToScene(self._rubber.geometry()).boundingRect()
            for item in self._scene.items(region):
                # only the channel being edited answers a frame, the way noteDigger frames its channels
                if (
                    isinstance(item, NoteItem)
                    and item.channel == self.active_channel
                    and not self._locked(item.channel)
                ):
                    item.setSelected(True)
            return

        if gesture.target is None or gesture.target not in gesture.snapshot:
            return

        if gesture.kind is Drag.TRIM:
            start, pitch, duration = gesture.snapshot[gesture.target]
            cell = self._cell_beats()  # one snap cell is the shortest the pointer may leave behind
            ignore = (gesture.target.note,)
            if gesture.edge is Edge.START:
                new_start = min(max(0.0, self._snap_beats(scene_pos.x())), start + duration - cell)
                candidate = (new_start, pitch, start + duration - new_start)
                if candidate == gesture.candidate:
                    return
                gesture.candidate = candidate
                if self._would_collide(gesture.target, new_start, pitch, start + duration - new_start, ignore):
                    return  # the edge cannot reach over a note of its own pitch
                gesture.target.set_range(new_start, pitch)
                gesture.target.set_duration(start + duration - new_start)
            else:
                new_duration = max(self._snap_beats(scene_pos.x()), start + cell) - start
                candidate = (start, pitch, new_duration)
                if candidate == gesture.candidate:
                    return
                gesture.candidate = candidate
                if self._would_collide(gesture.target, start, pitch, new_duration, ignore):
                    return
                gesture.target.set_duration(new_duration)
            self.view_changed.emit()
            return

        if gesture.kind is Drag.DRAW:
            anchor = gesture.origin.x()
            left = max(0.0, self._snap_floor_beats(min(anchor, scene_pos.x())))
            right = max(left + self._cell_beats(), self._snap_ceil_beats(max(anchor, scene_pos.x())))
            pitch = self.pitch_at(scene_pos.y())
            candidate = (left, pitch, right - left)
            if candidate == gesture.candidate:
                return
            gesture.candidate = candidate
            if self._would_collide(gesture.target, left, pitch, right - left, (gesture.target.note,)):
                return  # the note cannot be drawn over one of its own pitch
            gesture.target.set_range(left, pitch)  # the row follows the pointer too
            gesture.target.set_duration(right - left)
            return

        # the movement is what snaps, not the note's place on the grid: the step is always a whole
        # cell, after the same half cell of dragging, and a note sits off the grid stays off it
        moved = scene_pos.x() - gesture.origin.x()
        delta_x = self.snap_movement_beats(moved)
        delta_row = round(scene_pos.y() - gesture.origin.y())
        candidate = (delta_x, delta_row)
        if candidate == gesture.candidate:
            return
        gesture.candidate = candidate
        moving = frozenset(item.note for item in gesture.snapshot)
        for note, (start, pitch, _duration) in gesture.snapshot.items():
            if self._would_collide(note, start + delta_x, pitch - delta_row, _duration, moving):
                return  # the block cannot be dropped onto a note of one of its own pitches
        for note, (start, pitch, _duration) in gesture.snapshot.items():
            note.set_range(start + delta_x, pitch - delta_row)
        self.view_changed.emit()

    def _follow(self, pos: QPoint, scene_pos: QPointF) -> None:
        """A drag carries the playhead along and sounds every row it crosses, like a glissando."""
        if not self.playing:
            self.seek_requested.emit(self.seconds_at_viewport_x(pos.x()))
        pitch = self.pitch_at(scene_pos.y())
        if pitch != self._preview_pitch:
            self._preview_pitch = pitch
            self.note_preview.emit(pitch)

    def mouseReleaseEvent(self, event) -> None:
        self.commit_gesture()
        gesture, self._gesture = self._gesture, None
        if gesture is not None and gesture.press is not None and not gesture.press.dragged:
            # the press stayed a click, and a click is what picks the one note out of the selection
            self._clear_selection()
            gesture.press.block.setSelected(True)
        if self._rubber.isVisible():
            self._rubber.hide()
        self._preview_pitch = None
        self.viewport().unsetCursor()
        self._update_scene()
        self.view_changed.emit()

    def leaveEvent(self, event) -> None:
        self.set_hover_pitch(None)
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        modifier = event.modifiers()
        anchor = event.position().toPoint()
        factor = 1.0016**delta
        if modifier & Qt.KeyboardModifier.ControlModifier and modifier & Qt.KeyboardModifier.ShiftModifier:
            self._zoom(1.0, factor, anchor)
        elif modifier & Qt.KeyboardModifier.ControlModifier:
            self._zoom(factor, 1.0, anchor)
        elif modifier & Qt.KeyboardModifier.ShiftModifier:
            vbar = self.verticalScrollBar()
            vbar.setValue(vbar.value() - delta)
        else:
            hbar = self.horizontalScrollBar()
            hbar.setValue(hbar.value() - delta)
        event.accept()

    def _zoom(self, factor_x: float, factor_y: float, anchor: QPoint) -> None:
        scene_pos = self.mapToScene(anchor)
        before = self.mapFromScene(scene_pos)
        self._zoom_x = min(self.MAX_ZOOM_X, max(self.MIN_ZOOM_X, self._zoom_x * factor_x))
        self._zoom_y = min(self.MAX_ZOOM_Y, max(self.MIN_ZOOM_Y, self._zoom_y * factor_y))
        self.setTransform(QTransform.fromScale(self._zoom_x, self._zoom_y))
        delta = self.mapFromScene(scene_pos) - before
        hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
        hbar.setValue(hbar.value() + delta.x())
        vbar.setValue(vbar.value() + delta.y())
        self.view_changed.emit()
        self.viewport_changed.emit()
        self.zoom_changed.emit()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if not self.edit_mode and key in (
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
            Qt.Key.Key_A,
        ):
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selection()
            return
        if key == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            for note in self.notes():
                if not self._locked(note.channel):
                    note.setSelected(True)
            return
        if key == Qt.Key.Key_Escape and self._gesture_before is not None:
            self.cancel_gesture()
            return
        super().keyPressEvent(event)


class TimelineRuler(ViewportStrip):
    def __init__(self, view: PianoRollView):
        super().__init__(view)
        self._last_x: float | None = None
        self._press_x = 0.0
        self._moved = False
        self.setFixedHeight(RULER_HEIGHT)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        view.view_changed.connect(self.update)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        colors = theme.canvas()
        painter.fillRect(self.rect(), colors.panel)
        viewport = self.view.viewport()
        left_offset = self.origin().x()
        painter.setClipRect(QRect(int(left_offset), 0, viewport.width(), self.height()))
        font = QFont()
        font.setPixelSize(10)
        painter.setFont(font)

        left = self.view.mapToScene(QPoint(0, 0)).x()
        right = self.view.mapToScene(QPoint(viewport.width(), 0)).x()
        visible = QRectF(left, 0.0, right - left, 1.0)

        stride = 1
        while stride * BAR_BEATS * self.view.pixels_per_beat() < MEASURE_LABEL_SPACING:
            stride *= 2
        offset = self.view.offset_beats
        for bar in range(int((left - offset) // BAR_BEATS), int((right - offset) // BAR_BEATS) + 1):
            px = left_offset + self.view.mapFromScene(QPointF(bar * BAR_BEATS + offset, 0.0)).x()
            painter.setPen(QPen(colors.grid_bar, 1))
            painter.drawLine(px, RULER_TIME_ROW, px, self.height())
            if bar % stride == 0:
                painter.setPen(colors.text)
                painter.drawText(px + 4, RULER_TIME_ROW + 15, str(bar + 1))

        for x, level in self.view.division_lines(visible, MIN_GRID_SPACING, snap=True):
            px = left_offset + self.view.mapFromScene(QPointF(x, 0.0)).x()
            painter.setPen(QPen(colors.grid_line if level == 0 else colors.grid_beat, 1))
            painter.drawLine(px, self.height() - (4, 8, 10)[level], px, self.height())

        for x, seconds, _major in self.view.seconds_lines(visible, TIME_LABEL_SPACING):
            px = left_offset + self.view.mapFromScene(QPointF(x, 0.0)).x()
            painter.setPen(QPen(colors.grid_beat, 1))
            painter.drawLine(px, RULER_TIME_ROW - 6, px, RULER_TIME_ROW - 1)
            label = format_time(seconds)
            painter.setPen(colors.text)
            painter.drawText(QPointF(px - painter.fontMetrics().horizontalAdvance(label) / 2, 14), label)

        painter.setPen(QPen(colors.ruler_line, 1))
        right_edge = viewport.width() + int(left_offset)
        painter.drawLine(int(left_offset), RULER_TIME_ROW - 1, right_edge, RULER_TIME_ROW - 1)
        painter.drawLine(int(left_offset), self.height() - 1, right_edge, self.height() - 1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_x = event.position().x()
            self._press_x = self._last_x
            self._moved = False

    def mouseDoubleClickEvent(self, event) -> None:
        self.mousePressEvent(event)  # a quick second click on the ruler seeks like the first

    def mouseMoveEvent(self, event) -> None:
        position = event.position().x()
        if self._last_x is None:
            return
        if not self._moved:
            if abs(position - self._press_x) <= CLICK_SLOP_PX:
                return  # still a click: a pixel of jitter must not scroll the roll
            self._moved = True
        hbar = self.view.horizontalScrollBar()
        hbar.setValue(hbar.value() - int(position - self._last_x))
        self._last_x = position

    def mouseReleaseEvent(self, event) -> None:
        if self._last_x is not None and not self._moved:  # a click moves the cursor, a drag scrolls
            x = event.position().x() - self.origin().x()
            self.view.seek_requested.emit(self.view.seconds_at_viewport_x(x))
        self._last_x = None

    def wheelEvent(self, event) -> None:
        hbar = self.view.horizontalScrollBar()
        hbar.setValue(hbar.value() - event.angleDelta().y())
        event.accept()


class PianoKeyboard(ViewportStrip):
    """The keys at the left of the roll; clicking one auditions that note."""

    key_preview = Signal(int)

    def __init__(self, view: PianoRollView):
        super().__init__(view)
        self.setFixedWidth(66)
        view.view_changed.connect(self.update)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        colors = theme.canvas("light")  # a piano's keys stay white, whatever the desktop or the spectrum
        painter.fillRect(self.rect(), colors.panel)
        viewport = self.view.viewport()
        top_offset = self.origin().y()
        painter.setClipRect(QRect(0, int(top_offset), self.width(), viewport.height()))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)
        white = colors.key_white
        black = colors.key_black
        highlighted = set(self.view.highlight_pitches())

        for pitch in range(PITCH_MIN, PITCH_MAX + 1):
            row = PITCH_MAX - pitch
            top = top_offset + self.view.mapFromScene(QPointF(0.0, float(row))).y()
            bottom = top_offset + self.view.mapFromScene(QPointF(0.0, float(row) + 1.0)).y()
            if bottom < top_offset or top > viewport.height() + top_offset:
                continue
            color = black if is_black_key(pitch) else white
            painter.fillRect(QRect(0, top, self.width(), bottom - top), color)
            if pitch in highlighted:
                painter.fillRect(QRect(0, top, self.width(), bottom - top), colors.hover_key)
            if pitch % 12 == 0:
                painter.setPen(colors.key_text)
                painter.drawText(4, (top + bottom) // 2 + 3, note_name(pitch))

        painter.setPen(QPen(colors.key_line, 1))
        for pitch in range(PITCH_MIN, PITCH_MAX + 2):
            y = top_offset + self.view.mapFromScene(QPointF(0.0, float(PITCH_MAX - pitch))).y()
            painter.drawLine(0, y, self.width(), y)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.key_preview.emit(self._pitch_at(event.position().y()))

    def mouseDoubleClickEvent(self, event) -> None:
        self.mousePressEvent(event)  # a quick second click on a key sounds it again

    def _pitch_at(self, y: float) -> int:
        scene_y = self.view.mapToScene(QPoint(0, int(y) - self.origin().y())).y()
        return self.view.pitch_at(scene_y)

    def wheelEvent(self, event) -> None:
        vbar = self.view.verticalScrollBar()
        vbar.setValue(vbar.value() - event.angleDelta().y())
        event.accept()
