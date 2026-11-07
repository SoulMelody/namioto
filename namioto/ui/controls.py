# SPDX-License-Identifier: AGPL-3.0-only
"""Control bars: playback, edit, spectrum and mix, grouped into blocks laid out on one grid.

The bars are `QWidget` subclasses, not `QMainWindow` toolbars: Qt's toolbar reflow collapses every
row onto one line behind an overflow chevron. Blocks (`Cluster`) sit on `ControlArea`'s columns,
whose widths are what stands on them - nothing stretches and leftover room trails at the right - and
sliders sharing a block go through `Cluster.add_sliders` so their label, track and value columns
line up.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from namioto.i18n import tr
from namioto.interaction import Interaction, Tool, pick_tool, toggle_mode
from namioto.ui import icons
from namioto.ui.text import format_time

ICON_SIZE = 17
BUTTON_HEIGHT = 24
FIELD_HEIGHT = 24
CORNER_RADIUS = 4  # of a block's outline
EQ_BANDS = 10  # the engine's equaliser has ten bands, so the window offers exactly ten
EQ_RANGE = 24.0  # dB either way
EQ_FREQUENCIES = (31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def _outline_colour(palette: QPalette) -> QColor:
    """The shade a block is outlined in: nearest the page, so a block reads apart without shouting.

    A style's own frame is not an option: the older ones draw every one of them with a raised or
    sunken bevel - Qt's Windows style sinks a plain panel - which makes a row of blocks look
    pressed into the window.
    """
    dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
    return palette.color(QPalette.ColorRole.Light if dark else QPalette.ColorRole.Mid)


class _Card(QWidget):
    """A block of controls, outlined by this app rather than by the widget style."""

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(_outline_colour(self.palette()))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), CORNER_RADIUS, CORNER_RADIUS)


class Cluster(_Card):
    """A block of related controls, the WaveTone grouping, with a rule where kinds of control meet."""

    def __init__(self, name: str, spacing: int = 6):
        super().__init__()
        self.name = name
        self.body = QGridLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(spacing)
        self.body.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.setSpacing(6)
        layout.addLayout(self.body)
        self._columns: dict[int, int] = {}

    def add(self, widget: QWidget, row: int = 0) -> QWidget:
        # the column after the last control takes the room left over, so the group stays at the left
        column = self._columns.get(row, 0)
        self.body.addWidget(widget, row, column)
        self.body.setColumnStretch(column, 0)
        self.body.setColumnStretch(column + 1, 1)
        self._columns[row] = column + 1
        return widget

    def add_sliders(self, sliders: Sequence[ValueSlider]) -> None:
        """Stack sliders, on one width for the names, so names and values line up in columns."""
        width = max(slider.caption.sizeHint().width() for slider in sliders)
        for row, slider in enumerate(sliders):
            slider.caption.setFixedWidth(width)
            self.add(slider, row=row)


class _Rule(QWidget):
    """A one-pixel rule between the kinds of control that share one block."""

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _outline_colour(self.palette()))


def separator() -> QWidget:
    """A vertical rule that tells apart the kinds of control sharing one block."""
    rule = _Rule()
    rule.setFixedSize(1, 16)
    return rule


class AbsoluteSlider(QSlider):
    """A slider that jumps to the spot the track was clicked, and turns its wheel the other way up.

    Two deliberate deviations from Qt: a click lands where it was aimed instead of moving a page step,
    and the wheel walks the value down as it turns up, one notch to the step, which is the direction
    these controls read in.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setSliderDown(True)  # so the rest of the drag behaves like a handle drag
        self._jump_to(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if not self.isSliderDown():
            super().mouseMoveEvent(event)
            return
        self._jump_to(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        self.setSliderDown(False)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        notches = -round(event.angleDelta().y() / 120)
        if notches:
            self.setValue(self.value() + notches * self.singleStep())
        event.accept()

    def _jump_to(self, x: float) -> None:
        """The value under an x in the widget, the way a handle drag would land on it."""
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
        span = groove.width() - handle.width()
        position = round(x - groove.x() - handle.width() / 2)
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), position, span))


class ValueSlider(QWidget):
    """A caption, a horizontal slider and the current value."""

    value_changed = Signal(float)

    def __init__(
        self,
        caption: str,
        minimum: float,
        maximum: float,
        value: float,
        suffix: str = "",
        scale: int = 1,
        step: float = 0.0,
        slider_width: int = 80,
    ):
        super().__init__()
        self._suffix = suffix
        self._scale = scale
        self._step = max(1, round(step * scale)) if step else 1
        self._decimals = max(0, len(str(scale)) - 1)
        self.slider = AbsoluteSlider()
        self.slider.setRange(round(minimum * scale), round(maximum * scale))
        self.slider.setSingleStep(self._step)
        self.slider.setValue(round(value * scale))
        self.slider.setFixedWidth(slider_width)
        self.slider.setFixedHeight(24)
        self.slider.setPageStep(max(1, round((maximum - minimum) * scale / 10)))
        self.caption = QLabel(caption)
        self.caption.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.value_label = QLabel()
        self.value_label.setFixedWidth(self._value_width(minimum, maximum))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.caption)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.slider.valueChanged.connect(self._on_value_changed)
        self._refresh()

    def value(self) -> float:
        return self.slider.value() / self._scale

    def set_value(self, value: float) -> None:
        self.slider.setValue(round(value * self._scale))

    def _format(self, value: float) -> str:
        return f"{value:.{self._decimals}f}{self._suffix}"

    def _value_width(self, minimum: float, maximum: float) -> int:
        """Room for either end of the range, at whatever font the theme brings."""
        metrics = self.value_label.fontMetrics()
        return max(metrics.horizontalAdvance(self._format(value)) for value in (minimum, maximum))

    def _refresh(self) -> None:
        self.value_label.setText(self._format(self.value()))

    def _on_value_changed(self, value: int) -> None:
        snapped = round(value / self._step) * self._step
        if snapped != value:
            self.slider.setValue(snapped)  # dragging lands where it likes: come back onto the step
            return
        self._refresh()
        self.value_changed.emit(self.value())


def field_label(caption: str) -> QLabel:
    return QLabel(caption)


def icon_button(kind: str, tooltip: str, checkable: bool = False) -> QToolButton:
    button = QToolButton()
    button.setIcon(icons.icon(kind))
    button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
    button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setAutoRaise(True)
    button.setFixedHeight(BUTTON_HEIGHT)
    return button


def text_button(label: str, tooltip: str) -> QToolButton:
    """A raised-less button that says a letter: A and B are words no glyph draws better."""
    button = QToolButton()
    button.setText(label)
    button.setToolTip(tooltip)
    button.setAutoRaise(True)
    button.setFixedHeight(BUTTON_HEIGHT)
    return button


class TempoSuggestion(_Card):
    """The tempo the analyser found, in a balloon under the BPM field.

    A suggestion is not worth a row of its own, and it is not worth resizing the field for either, so
    it floats: the row keeps its size whether or not there is an estimate to offer. It floats *inside*
    the window rather than as a window of its own - a popup takes the keyboard with it and is gone at
    the first click anywhere else, and the roll is where the work is.
    """

    applied = Signal(float)
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # it floats over the roll, so its own body has to hide what is behind it
        self.setAutoFillBackground(True)
        self._bpm = 0.0
        self.label = QLabel()
        self.apply_button = icon_button("check", tr("Use this tempo"))
        self.dismiss_button = icon_button("cross", tr("Dismiss this estimate"))
        self.apply_button.clicked.connect(lambda: self.applied.emit(self._bpm))
        self.dismiss_button.clicked.connect(self.dismissed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.setSpacing(4)
        layout.addWidget(self.label)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.dismiss_button)

    def estimate(
        self,
        bpm: float,
        agreement: float,
        windows: int,
        source: str,
        residual: float | None = None,
    ) -> None:
        """Offer `bpm`, and say how the algorithm that measured it saw the result."""
        self._bpm = bpm
        self.label.setText(tr("≈{bpm} BPM", bpm=f"{bpm:.0f}"))
        detail = tr(source)
        if windows:
            detail += tr(
                ", over {windows} windows: {agreement} of them agree.",
                windows=windows,
                agreement=f"{agreement:.0%}",
            )
        if residual is not None:
            detail += tr("\nBeat fit residual {ms} ms.", ms=f"{1000 * residual:.0f}")
        self.setToolTip(tr("{detail}\nNothing changes until you click the tick.", detail=detail))

    def show_under(self, anchor: QWidget) -> None:
        """Float under `anchor`, inside its window, where it takes neither the focus nor the clicks."""
        window = anchor.window()
        self.setParent(window)
        self.adjustSize()
        self.move(anchor.mapTo(window, QPoint(0, anchor.height() + 4)))
        self.show()
        self.raise_()


class _SelectAll:
    """A spin box that selects its text when it is picked up, so typing replaces it."""

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.selectAll()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.selectAll()


class TempoBox(_SelectAll, QDoubleSpinBox):
    """BPM, with a decimal only when the tempo has one.

    Doubling and halving live in the context menu and on `*` and `/` rather than on two more
    buttons, which would crowd the transport bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.double_action = QAction(tr("Double tempo  (*)"), self)
        self.half_action = QAction(tr("Halve tempo  (/)"), self)
        self.double_action.triggered.connect(lambda: self.scale(2.0))
        self.half_action.triggered.connect(lambda: self.scale(0.5))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

    def textFromValue(self, value: float) -> str:
        return f"{value:.1f}".rstrip("0").rstrip(".")

    def scale(self, factor: float) -> None:
        self.setValue(self.value() * factor)

    def context_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addAction(self.double_action)
        menu.addAction(self.half_action)
        self.double_action.setEnabled(self.value() * 2.0 <= self.maximum())
        self.half_action.setEnabled(self.value() / 2.0 >= self.minimum())
        return menu

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Asterisk, Qt.Key.Key_multiply):
            self.scale(2.0)
            return
        if event.key() == Qt.Key.Key_Slash:
            self.scale(0.5)
            return
        super().keyPressEvent(event)

    def _show_menu(self, position) -> None:
        self.context_menu().exec(self.mapToGlobal(position))


class GridOffsetBox(_SelectAll, QSpinBox):
    """Milliseconds; the unit is a label beside the field, the way WaveTone shows it."""


class _Group(QObject):
    """Blocks that belong together, and the cell each one takes in the control grid.

    A group is not a widget and not a row: the blocks of one group can sit on different rows, which is
    how the wide mix blocks cover two rows of the shorter ones beside them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._placements: list[tuple[Cluster, int, int, int, int]] = []

    def place(self, cluster: Cluster, row: int, column: int, span: int = 1, rows: int = 1) -> Cluster:
        self._placements.append((cluster, row, column, span, rows))
        return cluster

    def placements(self) -> tuple[tuple[Cluster, int, int, int, int], ...]:
        return tuple(self._placements)


class ControlArea(QWidget):
    """Every block of every group, on one grid whose columns are as wide as what stands on them.

    The columns are what makes the blocks line up down the window: every block stands on them, so
    two blocks that share a column start and end on the same lines. Nothing stretches - the room a
    wider window has left over stays to the right of the last block - and a block narrower than the
    columns it covers leaves that room inside itself rather than pushing the next block along.
    """

    def __init__(self, groups: Sequence[_Group], parent=None):
        super().__init__(parent)
        self.setObjectName("controlArea")
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(6, 3, 6, 3)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(3)
        placements = [placement for group in groups for placement in group.placements()]
        widths = self._column_widths(placements)
        for column, width in enumerate(widths):
            self.grid.setColumnMinimumWidth(column, width)
        self.grid.setColumnStretch(len(widths), 1)  # the room left over stays at the right
        for cluster, row, column, span, rows in placements:
            self.grid.addWidget(cluster, row, column, rows, span)

    def _column_widths(self, placements: Sequence[tuple]) -> list[int]:
        """What every column needs, so that no block is wider than the columns it stands on.

        The blocks of one column settle their own column first; a block spanning more than one then
        claims what is left of its width from the narrowest column it covers, since a column that
        already holds a block of its own is as wide as that block needs. A shared column can still
        be wider than a block that spans it - the block under it needs that room - and all a block
        can do about it is leave the difference at its right.
        """
        columns = max((column + span for _cluster, _row, column, span, _rows in placements), default=0)
        widths = [0] * columns
        spacing = self.grid.horizontalSpacing()
        for cluster, _row, column, span, _rows in sorted(placements, key=lambda placement: placement[3]):
            covered = sum(widths[column : column + span]) + spacing * (span - 1)
            if (missing := cluster.sizeHint().width() - covered) > 0:
                # the narrowest column of the span takes it: the wider ones carry a block of their
                # own, and widening one of those would leave that block's room inside its own card
                target = min(range(column, column + span), key=lambda index: widths[index])
                widths[target] += missing
        return widths


class TransportBar(_Group):
    """Playback transport, position, the view toggles, playback speed, tempo and grid offset."""

    rewind_requested = Signal()
    play_from_start_requested = Signal()
    play_pause_requested = Signal()
    stop_requested = Signal()
    forward_requested = Signal()
    open_requested = Signal()
    save_requested = Signal()
    export_midi_requested = Signal()
    auto_page_toggled = Signal(bool)
    overtone_toggled = Signal(bool)
    harmonic_toggled = Signal(bool)
    division_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("transportBar")
        self.open = icon_button("open", tr("Open a project (.nto) — Ctrl+O"))
        self.save = icon_button("save", tr("Save the project — Ctrl+S, with Shift for Save As"))
        self.export = icon_button("export", tr("Export the notes as a MIDI file"))
        self.open.clicked.connect(self.open_requested)
        self.save.clicked.connect(self.save_requested)
        self.export.clicked.connect(self.export_midi_requested)
        self.rewind = icon_button("rewind", tr("Rewind to the beginning"))
        self.stop = icon_button("stop", tr("Stop"))
        self.play_from_start = icon_button("playstart", tr("Play from the beginning"))
        self.play_pause = icon_button("play", tr("Play from the cursor"))
        self.forward = icon_button("forward", tr("Go to the end"))
        self.rewind.clicked.connect(self.rewind_requested)
        self.stop.clicked.connect(self.stop_requested)
        self.play_from_start.clicked.connect(self.play_from_start_requested)
        self.play_pause.clicked.connect(self.play_pause_requested)
        self.forward.clicked.connect(self.forward_requested)

        self.position = QLabel("00:00.000")
        self.position.setToolTip(tr("Playback position"))
        # a fixed width keeps the layout steady as the clock runs; the font is the desktop's to pick
        self.position.setFixedWidth(QFontMetrics(self.position.font()).horizontalAdvance("00:00.000") + 6)
        self.position.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.speed = ValueSlider(tr("Speed"), 0.1, 2.0, 1.0, suffix="x", scale=100, step=0.05)
        self.speed.slider.setToolTip(
            tr("Playback speed in 5% steps, 0.10x to 2.00x: the song is rerendered, so the pitch stays")
        )
        self.speed_reset = icon_button("restore", tr("Reset the playback speed to 1.00x"))
        self.speed_reset.clicked.connect(lambda: self.speed.set_value(1.0))

        self.bpm = TempoBox()
        self.bpm.setRange(20.0, 300.0)
        self.bpm.setDecimals(1)
        self.bpm.setValue(120.0)
        self.bpm.setToolTip(
            tr("Tempo of the beat grid in BPM, until a tempo map is analysed\nRight-click to double or halve it")
        )
        self.bpm.setFixedWidth(self.bpm.sizeHint().width())
        self.bpm.setFixedHeight(FIELD_HEIGHT)
        self.bpm.setKeyboardTracking(False)

        self.detect = icon_button("refresh", tr("Estimate the tempo of the loaded audio"))
        self.detect.setEnabled(False)
        self.suggestion = TempoSuggestion()
        self.settings_button = icon_button("gear", tr("Settings: the advanced options the bars have no control for"))

        self.grid_offset = GridOffsetBox()
        self.grid_offset.setRange(-500, 500)
        self.grid_offset.setValue(0)
        self.grid_offset.setToolTip(
            tr("Grid offset: slides the grid lines, - left and + right; notes and playback keep their timestamps")
        )
        self.grid_offset.setFixedWidth(self.grid_offset.sizeHint().width())
        self.grid_offset.setFixedHeight(FIELD_HEIGHT)
        self.grid_offset.setKeyboardTracking(False)

        self.auto_page = icon_button(
            "page",
            tr("Auto page turn: take the next page of the roll once the playhead reaches the right"),
            checkable=True,
        )
        self.overtone = icon_button(
            "overtone",
            tr("Overtone highlight: paint f, 2f, 3f and 4f of the row under the mouse, the way WaveTone marks them"),
            checkable=True,
        )
        self.harmonic = icon_button(
            "harmonic",
            tr(
                "Overtone removal: take the partials a note puts above itself out of the spectrum, so an "
                "octave above it stops looking like a note of its own"
            ),
            checkable=True,
        )
        self.auto_page.toggled.connect(self.auto_page_toggled)
        self.overtone.toggled.connect(self.overtone_toggled)
        self.harmonic.toggled.connect(self.harmonic_toggled)
        self.channels = icon_button(
            "channels", tr("Channels: colours, mute and instruments, one card per channel"), checkable=True
        )
        self.note_lyrics = icon_button("note_lyrics", tr("Show lyrics on MIDI notes"), checkable=True)
        self.note_lyrics.setChecked(True)
        self.division = icon_button(
            "beat",
            tr("Time division: checked follows the beats of the tempo map, unchecked follows seconds"),
            checkable=True,
        )
        self.division.setChecked(True)  # beats by default; a click flips it to seconds
        self.division.toggled.connect(lambda checked: self.division_changed.emit("beats" if checked else "seconds"))

        project = Cluster("project")
        project.add(self.open)
        project.add(self.save)
        project.add(self.export)
        project.add(self.settings_button)

        playback = Cluster("playback")
        for button in (self.rewind, self.stop, self.play_from_start, self.play_pause, self.forward):
            playback.add(button)
        playback.add(separator())
        playback.add(self.position)
        playback.add(separator())
        playback.add(self.auto_page)
        playback.add(self.overtone)
        playback.add(self.harmonic)
        playback.add(separator())
        playback.add(self.channels)
        playback.add(self.note_lyrics)
        playback.add(self.division)

        bpm = Cluster("bpm")
        bpm.add(self.bpm)
        bpm.add(field_label("BPM"))
        bpm.add(self.detect)
        bpm.add(separator())
        bpm.add(self.grid_offset)
        bpm.add(field_label("ms"))

        speed = Cluster("speed")
        speed.add(self.speed)
        speed.add(self.speed_reset)

        self.place(project, 0, 0)
        self.place(playback, 0, 1, 2)
        self.place(bpm, 1, 2)
        self.place(speed, 0, 5, rows=2)

    def set_playing(self, playing: bool) -> None:
        """Playing and pausing share one button the way WaveTone shows it, so it turns into a
        pause button while the sound runs."""
        self.play_pause.setIcon(icons.icon("pause" if playing else "play"))
        self.play_pause.setToolTip(tr("Pause playback (Space)") if playing else tr("Play from the cursor (Space)"))

    def set_position(self, seconds: float) -> None:
        self.position.setText(format_time(seconds))


class EditBar(_Group):
    """Tools, snapping and the transcription button. The mode and the tool are one value, and this
    bar is only its input and its display."""

    interaction_changed = Signal(object)
    quantize_requested = Signal()
    transcribe_requested = Signal()
    lyrics_requested = Signal()
    align_requested = Signal()

    def __init__(self, snap_choices, parent=None):
        super().__init__(parent)
        self.setObjectName("editBar")
        self._interaction = Interaction.viewing()
        self.mode = icon_button(
            "edit",
            tr("Edit mode: draw, move and select notes (off: a click in the roll moves the playhead)"),
            checkable=True,
        )
        self.mode.clicked.connect(self._toggle_mode)
        self.pen = icon_button("pen", tr("Pen: click or drag an empty row to draw a note"), checkable=True)
        self.select = icon_button(
            "select", tr("Select: drag a box, ctrl-click a note to add, drag a note to move"), checkable=True
        )
        self.tools = QButtonGroup(self)
        self.tools.setExclusive(True)
        self.tools.addButton(self.pen)
        self.tools.addButton(self.select)
        self.pen.clicked.connect(lambda: self._pick_tool(Tool.PEN))
        self.select.clicked.connect(lambda: self._pick_tool(Tool.SELECT))

        self.snap = QComboBox()
        glyph = icons.icon("snap")
        for label, beats in snap_choices:
            self.snap.addItem(glyph, label, beats)
        self.snap.setCurrentIndex(self.snap.findText("1/8"))
        self.snap.setToolTip(tr("Snap grid for the pen tool: the note the grid is divided by"))
        self.snap.setFixedWidth(self.snap.sizeHint().width())
        self.snap.setFixedHeight(FIELD_HEIGHT)

        self.quantize = icon_button(
            "quantize",
            tr("Quantize: put the starts and ends of the notes on the snap grid, the selection if there is one"),
        )
        self.quantize.clicked.connect(self.quantize_requested)
        self.transcribe = icon_button("transcribe", tr("Transcribe the loaded audio"))
        self.transcribe.setEnabled(False)
        self.transcribe.clicked.connect(self.transcribe_requested)
        self.lyrics = icon_button("lyrics", tr("Lyrics: edit the text the project sings"))
        self.lyrics.setEnabled(False)
        self.lyrics.clicked.connect(self.lyrics_requested)
        self.align = icon_button("align", tr("Align the lyrics: put a time on every note from the word it sings"))
        self.align.setEnabled(False)
        self.align.clicked.connect(self.align_requested)

        tools = Cluster("tools")
        tools.add(self.mode)
        tools.add(self.pen)
        tools.add(self.select)
        tools.add(separator())
        tools.add(self.snap)
        tools.add(self.quantize)
        tools.add(self.transcribe)
        tools.add(self.lyrics)
        tools.add(self.align)

        self.place(tools, 1, 0, 2)
        self._render()

    def set_interaction(self, state: Interaction) -> None:
        """Take a mode and tool over, wherever they came from, and tell the world about it."""
        self._interaction = state
        self._render()
        self.interaction_changed.emit(state)

    def _render(self) -> None:
        editing = self._interaction.editing
        self.mode.setChecked(editing)
        if not editing:
            self.tools.setExclusive(False)  # an exclusive group keeps its last button checked
            self.pen.setChecked(False)
            self.select.setChecked(False)
            self.tools.setExclusive(True)
            return
        self.pen.setChecked(self._interaction.tool is Tool.PEN)
        self.select.setChecked(self._interaction.tool is Tool.SELECT)

    def _toggle_mode(self) -> None:
        self.set_interaction(toggle_mode(self._interaction))

    def _pick_tool(self, tool: Tool) -> None:
        self.set_interaction(pick_tool(self._interaction, tool))


class MixBar(_Group):
    """Spectrum display parameters and the audio/MIDI mix."""

    pitch_changed = Signal(float)
    eq_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mixBar")
        self.gain = ValueSlider(tr("Gain"), 10, 600, 240)
        self.gain.slider.setToolTip(tr("Spectrum gain: how much energy it takes to reach full red"))
        self.contrast = ValueSlider(tr("Contrast"), 0.2, 4.0, 1.0, scale=10)
        self.contrast.slider.setToolTip(tr("Spectrum contrast: exponent applied to the energy"))
        self.audio_volume = ValueSlider(tr("Audio"), 0, 100, 80, suffix="%")
        self.audio_volume.slider.setToolTip(tr("Volume of the analysed audio track"))
        self.midi_volume = ValueSlider("MIDI", 0, 100, 80, suffix="%")
        self.midi_volume.slider.setToolTip(tr("Volume of the note playback"))

        # pitch and the equaliser shape the audio, not the notes: the engine resamples for the one
        # and filters for the other, and both read as one value each, so one signal carries each
        self.pitch = ValueSlider(tr("Pitch"), -24, 24, 0, step=1)
        self.pitch.slider.setToolTip(tr("Audio pitch in semitones: the engine resamples, so the tempo stays"))
        self._bands: tuple[float, ...] = (0.0,) * EQ_BANDS
        self.eq = text_button(tr("EQ"), tr("Equalizer: ten bands over the audio playback"))
        self.pitch.value_changed.connect(self.pitch_changed)
        self.eq.clicked.connect(self._open_equalizer)

        spectrum = Cluster("spectrum")
        # the tone shares the spectrum block's slider columns rather than a block of its own: another
        # column would push the whole bar wider than the window, and these columns are already there
        spectrum.add_sliders((self.gain, self.contrast, self.pitch))
        spectrum.add(self.eq, row=2)

        volume = Cluster("volume")
        volume.add_sliders((self.audio_volume, self.midi_volume))

        self.place(spectrum, 0, 3, rows=2)
        self.place(volume, 0, 4, rows=2)

    @property
    def eq_bands(self) -> tuple[float, ...]:
        return self._bands

    def set_pitch(self, semitones: float) -> None:
        """Fill the slider in without telling the player: a value coming from somewhere else."""
        self.pitch.set_value(semitones)

    def set_eq_bands(self, bands) -> None:
        self._bands = _normalised_bands(bands)

    def _open_equalizer(self) -> None:
        dialog = EqualizerPopup(self._bands, self)
        dialog.bands_changed.connect(self._adopt_bands)
        dialog.exec()

    def _adopt_bands(self, bands) -> None:
        self._bands = _normalised_bands(bands)
        self.eq_changed.emit(self._bands)


def _normalised_bands(bands) -> tuple[float, ...]:
    """A curve of exactly `EQ_BANDS` numbers: an absent or short one pads with zero, which is flat."""
    values = [float(value) for value in bands][:EQ_BANDS]
    return tuple(values) + (0.0,) * (EQ_BANDS - len(values))


def _band_caption(hz: int) -> str:
    """A band's frequency, short enough for a slider's caption column: 31, 62 … 2k, 16k."""
    return f"{hz // 1000}k" if hz >= 1000 else str(hz)


class EqualizerPopup(QDialog):
    """The ten bands of the audio equaliser, in dB.

    Not in `MixBar`: that bar already reads as columns on one row, and ten sliders there would turn
    a glance into a scan. So the bar carries one button and the bands live in a window that exists
    only while it is open, in two columns of five so the shape stays a page rather than a strip.
    Every move emits the whole curve, because the engine takes the curve as one value.
    """

    bands_changed = Signal(object)  # tuple[float, ...]

    def __init__(self, bands, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Equalizer"))
        bands = _normalised_bands(bands)
        self.sliders: list[ValueSlider] = []

        caption = QLabel(tr("Gain per band, in dB: 0 leaves the band alone"))
        columns = QHBoxLayout()
        per_column = EQ_BANDS // 2
        for first in range(0, EQ_BANDS, per_column):
            cluster = Cluster("equalizer")
            column_sliders: list[ValueSlider] = []
            for index in range(first, min(first + per_column, EQ_BANDS)):
                slider = ValueSlider(_band_caption(EQ_FREQUENCIES[index]), -EQ_RANGE, EQ_RANGE, bands[index], step=1)
                slider.slider.setToolTip(tr("Band at {hz} Hz", hz=EQ_FREQUENCIES[index]))
                slider.value_changed.connect(self._emit)
                column_sliders.append(slider)
            cluster.add_sliders(column_sliders)
            columns.addWidget(cluster)
            self.sliders.extend(column_sliders)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        reset = buttons.addButton(tr("Reset"), QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self._reset)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(caption)
        layout.addLayout(columns)
        layout.addWidget(buttons)

    def bands(self) -> tuple[float, ...]:
        return tuple(slider.value() for slider in self.sliders)

    def _emit(self, *_args) -> None:
        self.bands_changed.emit(self.bands())

    def _reset(self) -> None:
        for slider in self.sliders:
            slider.set_value(0.0)
