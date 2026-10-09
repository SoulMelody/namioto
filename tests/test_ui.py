# SPDX-License-Identifier: AGPL-3.0-only
"""Layout checks for the control bars and the main window.

Offscreen, `QTest.mouseMove` delivers nothing, so send events to the handler instead - `roll_mouse`,
`ruler_mouse`, `draw_note` and `roll_wheel` below. A shortcut needs
`QApplication.setActiveWindow(window)` and the key sent to `window.view`. `QWidget.grab()` still
renders, so a pixel check works.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QProcess, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QContextMenuEvent,
    QFocusEvent,
    QFont,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPalette,
    QWheelEvent,
)
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QSlider,
    QTabWidget,
    QToolButton,
)

from namioto import lyrics, midi, project
from namioto import settings as store
from namioto import state as window_state
from namioto.analysis import align, devices, transcription
from namioto.analysis.bpm import BpmEstimate
from namioto.analysis.spectrum import MIDI_OFFSET, NOTE_COUNT, NoteSpectrum
from namioto.channels import Channel
from namioto.interaction import Interaction, Tool
from namioto.karaoke.operations import Drop, Match, Merge, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyricmap.raw import from_spans
from namioto.lyrics import text_key
from namioto.ui import theme
from namioto.ui.align_dialog import AlignDialog, Aligner
from namioto.ui.app import MainWindow, TempoLoader
from namioto.ui.ass_dialog import GROUPS as ASS_GROUPS
from namioto.ui.ass_dialog import AssDialog
from namioto.ui.audio import BuiltinSynth, MidiPortOut, find_port, find_synth_port
from namioto.ui.controls import Cluster, EditBar, TransportBar, ValueSlider
from namioto.ui.form import WrappedLabel, _show_device_status, field_editor
from namioto.ui.lyrics_dialog import LyricsConverter, LyricsDialog
from namioto.ui.midi_dialog import MidiImportDialog
from namioto.ui.open_dialog import OpenAudioDialog
from namioto.ui.roll import (
    CONTENT_MARGIN,
    LENGTH_BEATS,
    NOTE_INSET,
    PITCH_MAX,
    PITCH_MIN,
    RULER_HEIGHT,
    RULER_TIME_ROW,
    SNAP_CHOICES,
    SPECTRUM_TOP,
    PianoRollView,
    is_black_key,
)
from namioto.ui.settings_dialog import PAGES, SettingsDialog
from namioto.ui.spectrogram import SpectrumImage, SpectrumLoader
from namioto.ui.transcription_dialog import TranscriptionDialog

DEMO_NOTES = (
    (60, 0.0, 1.0),
    (64, 1.0, 1.0),
    (67, 2.0, 1.0),
    (72, 3.0, 1.0),
    (67, 4.0, 0.5),
    (69, 4.5, 0.5),
    (71, 5.0, 1.5),
    (64, 6.5, 0.5),
    (67, 7.0, 1.0),
    (55, 0.0, 8.0),
    (62, 0.0, 8.0),
)


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    theme.apply(app)
    return app


@pytest.fixture(autouse=True)
def isolated_transcription(tmp_path, monkeypatch):
    """The transcription dialog remembers its values in the config directory; no test may touch the
    real one, and none may read a previous test's."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


@pytest.fixture(scope="module")
def window(qt_app):
    window = MainWindow()
    window.resize(1200, 720)
    window.show()
    for pitch, start, duration in DEMO_NOTES:
        window.view.add_note(pitch, start, duration)
    qt_app.processEvents()
    yield window
    window.close()


def roll_mouse(
    window, kind, scene_pos: QPointF, modifiers=Qt.KeyboardModifier.NoModifier, button=Qt.MouseButton.LeftButton
) -> None:
    """Send a mouse event to the roll at a scene position, as a real click would arrive."""
    position = window.view.mapFromScene(scene_pos)
    event = QMouseEvent(
        kind,
        QPointF(position),
        window.view.viewport().mapToGlobal(QPointF(position)),
        button,
        button,
        modifiers,
    )
    if kind == QEvent.Type.MouseButtonPress:
        window.view.mousePressEvent(event)
    elif kind == QEvent.Type.MouseButtonDblClick:
        window.view.mouseDoubleClickEvent(event)
    elif kind == QEvent.Type.MouseMove:
        window.view.mouseMoveEvent(event)
    else:
        window.view.mouseReleaseEvent(event)


def roll_context_menu(window, scene_pos: QPointF) -> None:
    """Open the roll's context menu at a scene position, as a real right click would."""
    position = window.view.mapFromScene(scene_pos)
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse,
        position,
        window.view.viewport().mapToGlobal(position),
    )
    QApplication.sendEvent(window.view.viewport(), event)


def pick_menu(monkeypatch, data) -> None:
    """Make the next context menu answer with the entry carrying `data`."""
    monkeypatch.setattr(
        QMenu, "exec", lambda self, *args, **kwargs: next((a for a in self.actions() if a.data() == data), None)
    )


def ruler_mouse(window, kind, x: float) -> None:
    """Send a mouse event to the timeline ruler, whose columns line up with the roll's."""
    position = QPointF(x, 10.0)
    event = QMouseEvent(
        kind,
        position,
        window.ruler.mapToGlobal(position),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    if kind == QEvent.Type.MouseButtonPress:
        window.ruler.mousePressEvent(event)
    elif kind == QEvent.Type.MouseMove:
        window.ruler.mouseMoveEvent(event)
    else:
        window.ruler.mouseReleaseEvent(event)


def ruler_click(window, beats: float) -> None:
    x = window.ruler.origin().x() + window.view.mapFromScene(QPointF(beats, 0.0)).x()
    ruler_mouse(window, QEvent.Type.MouseButtonPress, x)
    ruler_mouse(window, QEvent.Type.MouseButtonRelease, x)


def sound_mouse(window, kind, x: float, y: float = 10.0) -> None:
    """Send a mouse event to the lyrics strip, whose columns line up with the roll's."""
    position = QPointF(x, y)
    event = QMouseEvent(
        kind,
        position,
        window.sound_strip.mapToGlobal(position),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    if kind == QEvent.Type.MouseButtonPress:
        window.sound_strip.mousePressEvent(event)
    elif kind == QEvent.Type.MouseMove:
        window.sound_strip.mouseMoveEvent(event)
    else:
        window.sound_strip.mouseReleaseEvent(event)


def test_clicking_the_timeline_moves_the_playhead(window) -> None:
    window.edit.pen.click()  # the ruler seeks even while editing
    window.view.set_playhead(None)

    ruler_click(window, 3.0)
    assert window.view.playhead == pytest.approx(1.5)  # three beats at 120 BPM

    hbar = window.view.horizontalScrollBar()
    hbar.setValue(0)
    window.view.set_playhead(None)
    ruler_mouse(window, QEvent.Type.MouseButtonPress, 400.0)
    ruler_mouse(window, QEvent.Type.MouseMove, 300.0)
    ruler_mouse(window, QEvent.Type.MouseButtonRelease, 300.0)
    assert hbar.value() == 100 and window.view.playhead is None  # a drag scrolls instead
    hbar.setValue(0)
    window._stop()
    window.view.set_playhead(None)


def test_a_click_with_jitter_on_the_ruler_does_not_scroll(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()
    hbar = window.view.horizontalScrollBar()
    hbar.setValue(200)
    x = 400.0

    ruler_mouse(window, QEvent.Type.MouseButtonPress, x)
    ruler_mouse(window, QEvent.Type.MouseMove, x + 2.0)  # within the click slop
    ruler_mouse(window, QEvent.Type.MouseButtonRelease, x + 2.0)

    assert hbar.value() == 200  # two pixels are a click, not a drag
    window._stop()
    window.view.set_playhead(None)


def test_a_click_with_the_select_tool_moves_the_playhead(window) -> None:
    window.edit.select.click()  # picking a tool enters edit mode
    assert window.view.edit_mode
    window.view.clear_notes()
    window.view.set_playhead(None)
    point = QPointF(3.0, 40.0)

    roll_mouse(window, QEvent.Type.MouseButtonPress, point)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, point)
    assert window.view.playhead == pytest.approx(1.5)  # the press moves the cursor
    assert not window.view.notes()

    window.edit.pen.click()
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_new_window_starts_with_editing_off(qt_app) -> None:
    window = MainWindow()
    try:
        assert not window.edit.mode.isChecked() and not window.view.edit_mode
    finally:
        window.close()


def draw_note(window, press: QPointF, release: QPointF | None = None) -> None:
    window.view.centerOn(QPointF((press.x() + (release or press).x()) / 2, press.y()))
    roll_mouse(window, QEvent.Type.MouseButtonPress, press)
    if release is not None:
        roll_mouse(window, QEvent.Type.MouseMove, release)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, release or press)


def clusters(window) -> list[tuple[str, Cluster]]:
    return [(cluster.name, cluster) for cluster in window.controls.findChildren(Cluster)]


def card(window, name: str) -> Cluster:
    return next(cluster for _, cluster in clusters(window) if cluster.name == name)


def grid_rows(window) -> dict[int, list[Cluster]]:
    rows: dict[int, list[Cluster]] = {}
    for _, cluster in clusters(window):
        rows.setdefault(cluster.geometry().top(), []).append(cluster)
    return rows


def test_the_roll_holds_the_keyboard_when_the_window_opens(window) -> None:
    assert window.focusWidget() is window.view, "the bar would open with a focus ring on its first button"


def test_bars_have_room_for_every_cluster(window) -> None:
    for name, cluster in clusters(window):
        assert cluster.height() >= cluster.sizeHint().height(), f"{name} is squeezed"
        assert cluster.geometry().bottom() < window.controls.height(), f"{name} overflows the control area"


def test_clusters_do_not_overlap(window) -> None:
    for row, boxes in grid_rows(window).items():
        boxes = sorted(boxes, key=lambda cluster: cluster.geometry().left())
        for left, right in zip(boxes, boxes[1:], strict=False):
            assert right.geometry().left() >= left.geometry().right(), f"row at {row}: clusters overlap"


def test_the_blocks_line_up_on_one_grid(window) -> None:
    area = window.controls
    left = area.grid.contentsMargins().left()
    project, playback = card(window, "project"), card(window, "playback")
    tools, bpm = card(window, "tools"), card(window, "bpm")
    assert project.geometry().left() == tools.geometry().left() == left
    assert playback.geometry().right() == bpm.geometry().right(), "the two rows end on different lines"
    assert tools.geometry().top() > project.geometry().top()

    mix = [card(window, name) for name in ("spectrum", "volume", "speed")]
    assert [cluster.geometry().left() for cluster in mix] == sorted(cluster.geometry().left() for cluster in mix)
    assert all(cluster.geometry().top() == project.geometry().top() for cluster in mix)
    assert all(cluster.geometry().bottom() == tools.geometry().bottom() for cluster in mix), (
        "the mix blocks cover both rows, so they are as tall as the two rows beside them"
    )
    assert playback.geometry().right() < mix[0].geometry().left()


def test_bars_fit_the_default_window(window) -> None:
    assert window.controls.sizeHint().width() <= window.width(), (
        f"the controls need {window.controls.sizeHint().width()}px of {window.width()}px"
    )


def test_value_sliders_keep_their_caption_next_to_them(window) -> None:
    for slider in window.mix.findChildren(ValueSlider):
        assert slider.caption.x() < slider.slider.x() < slider.value_label.x()
        assert slider.slider.width() >= slider.slider.minimumWidth()


def test_the_sliders_of_one_block_line_up_in_columns(window) -> None:
    for name in ("spectrum", "volume"):
        sliders = card(window, name).findChildren(ValueSlider)
        assert len(sliders) > 1
        for part in ("caption", "slider", "value_label"):
            columns = {getattr(slider, part).mapTo(window.controls, QPoint(0, 0)).x() for slider in sliders}
            assert len(columns) == 1, f"{name}: {part} starts in {len(columns)} different places"


def test_control_widths_grow_with_the_theme_font(qt_app) -> None:
    original = qt_app.font()
    try:
        font = QFont(original)
        font.setPointSizeF(original.pointSizeF() + 6)
        qt_app.setFont(font)
        slider = ValueSlider("Speed", 0.1, 2.0, 1.0, suffix="x", scale=100)
        bar = TransportBar()
        edit = EditBar(SNAP_CHOICES)
        for field in (bar.bpm, bar.grid_offset, edit.snap):
            assert field.width() >= field.sizeHint().width(), f"{type(field).__name__} is narrower than its text"
        metrics = slider.value_label.fontMetrics()
        for value in (slider.slider.minimum(), slider.slider.maximum()):
            slider.slider.setValue(value)
            assert metrics.horizontalAdvance(slider.value_label.text()) <= slider.value_label.width()
    finally:
        qt_app.setFont(original)


def test_tool_buttons_switch_the_roll_mode(window) -> None:
    window.edit.select.click()
    assert window.view.tool is Tool.SELECT
    window.edit.pen.click()
    assert window.view.tool is Tool.PEN


def test_one_action_emits_one_state_and_the_bar_renders_it(window) -> None:
    seen: list[Interaction] = []
    window.edit.interaction_changed.connect(seen.append)
    try:
        window.edit.set_interaction(Interaction.viewing())
        seen.clear()

        window.edit.select.click()  # one click, one state - no edit-mode signal beside it
        assert seen == [Interaction.editing_with(Tool.SELECT)]
        assert window.edit.select.isChecked() and not window.edit.pen.isChecked()

        window.edit.mode.click()
        assert seen == [Interaction.editing_with(Tool.SELECT), Interaction.viewing()]
        assert not (window.edit.pen.isChecked() or window.edit.select.isChecked())
    finally:
        window.edit.interaction_changed.disconnect(seen.append)


def test_mode_buttons_are_icons_not_text(window) -> None:
    buttons = (
        window.edit.mode,
        window.edit.pen,
        window.edit.select,
        window.transport.division,
        window.transport.play_pause,
    )
    for button in buttons:
        assert not button.text(), f"{button.objectName()} still has a text label"
        assert not button.icon().isNull(), f"{button.objectName()} has no icon"
        assert button.toolTip(), f"{button.objectName()} has no tooltip"


def test_scrollbars_move_the_view(window) -> None:
    hbar, vbar = window.view.horizontalScrollBar(), window.view.verticalScrollBar()
    assert not hbar.isHidden() and not vbar.isHidden()
    assert hbar.maximum() > 0 and vbar.maximum() > 0

    before = window.view.mapToScene(window.view.viewport().rect().center())
    hbar.setValue(hbar.value() + 120)
    assert window.view.mapToScene(window.view.viewport().rect().center()).x() > before.x()
    vbar.setValue(vbar.value() + 120)
    assert window.view.mapToScene(window.view.viewport().rect().center()).y() > before.y()


def test_ruler_marks_line_up_with_the_roll(window) -> None:
    window.view.clear_notes()
    ruler, viewport = window.ruler, window.view.viewport()
    ruler_image, view_image = ruler.grab().toImage(), window.view.grab().toImage()
    row = viewport.mapTo(window.view, QPoint(0, viewport.height() // 2)).y()
    ruler_marks = [
        x
        for x in range(ruler_image.width())
        if ruler_image.pixelColor(x, RULER_TIME_ROW + 2) == theme.canvas().grid_bar
    ]
    roll_marks = [x for x in range(view_image.width()) if view_image.pixelColor(x, row) == theme.canvas().grid_bar]
    offset = ruler.mapToGlobal(QPoint(0, 0)).x() - window.view.mapToGlobal(QPoint(0, 0)).x()
    assert ruler_marks and roll_marks
    assert [x + offset for x in ruler_marks] == roll_marks


def test_keyboard_rows_line_up_with_the_roll(window) -> None:
    window.view.clear_notes()
    keyboard, viewport = window.keyboard, window.view.viewport()
    keyboard_image, view_image = keyboard.grab().toImage(), window.view.grab().toImage()
    centre = viewport.mapTo(window.view, QPoint(viewport.width() // 2, 0)).x()
    # a vertical grid line covers the whole height of the roll, so the column read has to miss every one
    column = next(
        x
        for x in range(centre, centre + 8)
        if any(view_image.pixelColor(x, y) == theme.canvas().row_white for y in range(view_image.height()))
    )
    white_keys = [
        y for y in range(keyboard_image.height()) if keyboard_image.pixelColor(2, y) == theme.CANVAS["light"].key_white
    ]
    white_rows = [y for y in range(view_image.height()) if view_image.pixelColor(column, y) == theme.canvas().row_white]
    offset = keyboard.mapToGlobal(QPoint(0, 0)).y() - window.view.mapToGlobal(QPoint(0, 0)).y()
    assert white_keys and white_rows
    assert [y + offset for y in white_keys] == white_rows


def test_the_black_keys_span_the_whole_keyboard(window) -> None:
    window.view.set_hover_pitch(None)
    keyboard = window.keyboard
    image = keyboard.grab().toImage()
    origin = keyboard.origin().y()
    for pitch in range(PITCH_MIN, PITCH_MAX + 1):
        if not is_black_key(pitch):
            continue
        row = PITCH_MAX - pitch
        top = origin + window.view.mapFromScene(QPointF(0.0, float(row))).y()
        bottom = origin + window.view.mapFromScene(QPointF(0.0, float(row) + 1.0)).y()
        if top < 0 or bottom > image.height():
            continue
        y = (top + bottom) // 2
        black = theme.CANVAS["light"].key_black
        assert all(image.pixelColor(x, y) == black for x in range(keyboard.width()))
        return
    raise AssertionError("no black key is fully on screen")


def test_the_division_button_flips_between_beats_and_seconds(window) -> None:
    assert window.transport.division.isChecked()  # beats by default
    window.transport.division.click()
    assert not window.transport.division.isChecked()
    assert window.view.division == "seconds"
    window.transport.division.click()
    assert window.transport.division.isChecked()
    assert window.view.division == "beats"


def text_columns(image: QImage, top: int, bottom: int) -> set[int]:
    """Columns holding label glyphs in a ruler row; the rows are otherwise flat colours."""
    flat = {
        theme.canvas().panel.name(),
        theme.canvas().grid_line.name(),
        theme.canvas().grid_beat.name(),
        theme.canvas().grid_bar.name(),
        theme.canvas().ruler_line.name(),
    }
    return {x for x in range(image.width()) for y in range(top, bottom) if QColor(image.pixel(x, y)).name() not in flat}


def test_ruler_shows_time_and_measures_whatever_the_division(window) -> None:
    for division in ("beats", "seconds"):
        window.view.division = division
        window.view.refresh()
        image = window.ruler.grab().toImage()
        assert text_columns(image, 0, RULER_TIME_ROW), f"no time labels with the {division} division"
        assert text_columns(image, RULER_TIME_ROW, RULER_HEIGHT), f"no measure numbers with {division}"
    window.view.division = "beats"


def test_ruler_time_row_ignores_the_division(window) -> None:
    labels = {}
    for division in ("beats", "seconds"):
        window.view.division = division
        window.view.refresh()
        labels[division] = text_columns(window.ruler.grab().toImage(), 0, RULER_TIME_ROW)
    window.view.division = "beats"
    assert labels["beats"] and labels["beats"] == labels["seconds"]


def test_spectrum_grid_lines_follow_the_division() -> None:
    view = PianoRollView()
    view.resize(900, 500)
    view.bpm = 120.0
    view.set_spectrum(make_spectrum(frames=400, value=0.0))  # black cells, so only the lines show
    view.centerOn(20.0, float(PITCH_MAX - 60) + 0.5)
    columns = {}
    for division in ("beats", "seconds"):
        view.division = division
        view.refresh()
        image = view.grab().toImage()
        y = device_point(view, 0.0, float(PITCH_MAX - 60) + 0.5).y()
        columns[division] = {
            x
            for x in range(image.width())
            if image.pixelColor(x, y) in (theme.canvas().spectrum_beat, theme.canvas().spectrum_bar)
        }
    assert columns["beats"] and columns["seconds"]
    assert columns["beats"] != columns["seconds"]


def test_defaults_of_the_control_bars(window) -> None:
    assert window.transport.bpm.value() == 120.0
    assert window.transport.grid_offset.value() == 0
    assert window.transport.position.text() == "00:00.000"
    assert window.transport.speed.value() == 1.0
    assert window.mix.gain.value() == 240.0
    assert window.mix.contrast.value() == 1.0
    assert window.mix.audio_volume.value() == 80.0
    assert window.edit.snap.currentData() == 0.5 and window.view.snap == 0.5  # 1/8 by default
    assert window.transport.division.isChecked()


def test_transport_position_is_a_clock(window) -> None:
    window.transport.set_position(63.25)
    assert window.transport.position.text() == "01:03.250"
    window.transport.set_position(0.0)


def test_the_tempo_hides_a_zero_decimal(window) -> None:
    for value, text in ((120.0, "120"), (96.4, "96.4"), (100.0, "100"), (97.5, "97.5")):
        window.transport.bpm.setValue(value)
        assert window.transport.bpm.text() == text
    window.transport.bpm.lineEdit().setText("96.4")  # typing still goes in as a value
    window.transport.bpm.interpretText()
    assert window.transport.bpm.value() == 96.4
    window.transport.bpm.setValue(120.0)


def test_the_tempo_and_grid_offset_fields_select_their_text(window) -> None:
    for field in (window.transport.bpm, window.transport.grid_offset):
        field.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))
        assert field.lineEdit().selectedText() == field.cleanText() != ""
        field.lineEdit().deselect()
        field.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(6.0, 6.0),
                QPointF(field.mapToGlobal(QPoint(6, 6))),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert field.lineEdit().selectedText() == field.cleanText()


def test_the_grid_offset_field_shows_its_unit_beside_it(window) -> None:
    field = window.transport.grid_offset
    assert field.suffix() == ""
    unit = next(label for label in window.controls.findChildren(QLabel) if label.text() == "ms")
    # both in the control area's coordinates: the label is a sibling of the field, not a child of it
    assert unit.mapTo(window.controls, QPoint(0, 0)).x() >= field.mapTo(window.controls, field.rect().topRight()).x()
    image = field.grab().toImage()
    lit = [
        x for x in range(image.width()) if any(image.pixelColor(x, y).lightness() > 120 for y in range(image.height()))
    ]
    assert lit and min(lit) < image.width() * 0.25  # the number stays at the left of the field


def test_the_tempo_can_be_doubled_and_halved(window) -> None:
    box = window.transport.bpm
    box.setValue(120.0)
    box.double_action.trigger()
    assert box.value() == 240.0
    box.half_action.trigger()
    assert box.value() == 120.0
    box.setValue(300.0)
    assert not box.context_menu().actions()[0].isEnabled()  # the range stops the doubling
    assert box.context_menu().actions()[1].isEnabled()
    box.setValue(120.0)

    menu = box.context_menu()
    assert [action.text() for action in menu.actions()] == ["Double tempo  (*)", "Halve tempo  (/)"]
    assert all(action.isEnabled() for action in menu.actions())

    box.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Asterisk, Qt.KeyboardModifier.NoModifier))
    assert box.value() == 240.0
    box.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Slash, Qt.KeyboardModifier.NoModifier))
    assert box.value() == 120.0


def fake_estimate(bpm: float = 96.0, windows: int = 10, agree: int = 6) -> BpmEstimate:
    return BpmEstimate(
        bpm=bpm,
        algorithm="librosa",
        windows=windows,
        agreement=agree / windows if windows else 0.0,
        residual=0.02,
    )


def test_tempo_estimate_is_only_a_suggestion(window) -> None:
    window.transport.bpm.setValue(120.0)
    window._on_tempo_loaded(window._audio_generation, fake_estimate())
    assert window.transport.suggestion.isVisible()
    assert window.transport.bpm.value() == 120.0  # nothing is applied by itself
    assert "60% of them agree" in window.transport.suggestion.toolTip()

    window.transport.suggestion.apply_button.click()
    assert window.transport.bpm.value() == 96.0
    assert not window.transport.suggestion.isVisible()
    window.transport.bpm.setValue(120.0)


def test_tempo_already_in_the_field_is_not_offered_again(window) -> None:
    window.transport.bpm.setValue(120.0)
    window._on_tempo_loaded(window._audio_generation, fake_estimate(bpm=120.0, agree=10))
    assert not window.transport.suggestion.isVisible()


def test_a_tempo_of_its_own_keeps_the_suggestion_away(window) -> None:
    window.transport.bpm.setValue(96.0)  # the user typed one, or took an earlier estimate
    window._on_tempo_loaded(window._audio_generation, fake_estimate(bpm=140.0))
    assert not window.transport.suggestion.isVisible()
    assert window.transport.bpm.value() == 96.0
    window.transport.bpm.setValue(120.0)


def test_a_manual_detect_offers_its_tempo_over_a_tempo_of_its_own(window, monkeypatch) -> None:
    window.transport.bpm.setValue(96.0)
    window.audio_path = "song.wav"
    window.transport.detect.setEnabled(True)
    monkeypatch.setattr(TempoLoader, "start", lambda self: None)

    window.transport.detect.click()
    assert window._tempo_manual
    window._on_tempo_loaded(window._audio_generation, fake_estimate(bpm=140.0))
    assert window.transport.suggestion.isVisible()
    assert window.transport.bpm.value() == 96.0  # still a suggestion
    window.transport.bpm.setValue(120.0)


def test_tempo_suggestion_can_be_dismissed_without_applying(window) -> None:
    window.transport.bpm.setValue(120.0)
    window._on_tempo_loaded(window._audio_generation, fake_estimate(bpm=100.0))
    window.transport.suggestion.dismiss_button.click()
    assert not window.transport.suggestion.isVisible()
    assert window.transport.bpm.value() == 120.0


def test_typing_a_tempo_drops_the_suggestion(window) -> None:
    window._on_tempo_loaded(window._audio_generation, fake_estimate())
    window.transport.bpm.setValue(140.0)
    assert not window.transport.suggestion.isVisible()
    window.transport.bpm.setValue(120.0)


def test_the_tempo_suggestion_floats_without_widening_the_bar(window) -> None:
    before = window.controls.sizeHint().width()
    window._on_tempo_loaded(window._audio_generation, fake_estimate(windows=10, agree=2))
    assert window.transport.suggestion.isVisible()
    assert window.controls.sizeHint().width() == before, "the suggestion is not worth a wider row"
    window.transport.suggestion.hide()


def test_the_tempo_suggestion_does_not_take_the_keyboard(window) -> None:
    suggestion = window.transport.suggestion
    QApplication.setActiveWindow(window)
    window.view.setFocus()
    window._on_tempo_loaded(window._audio_generation, fake_estimate())

    assert suggestion.isVisible()
    assert not suggestion.isWindow(), "a window of its own takes the keyboard and closes on a click"
    assert suggestion.parentWidget() is window
    assert QApplication.activeWindow() is window
    assert window.view.hasFocus(), "the roll keeps the keyboard the bar gave it"
    suggestion.hide()


def test_working_in_the_roll_leaves_the_suggestion_up(window) -> None:
    window._on_tempo_loaded(window._audio_generation, fake_estimate())
    assert window.transport.suggestion.isVisible()

    window.edit.set_interaction(Interaction.editing_with(Tool.PEN))  # a drawing gesture, in the spectrum
    draw_note(window, QPointF(4.0, 60.0), QPointF(8.0, 60.0))
    window.edit.set_interaction(Interaction.viewing())
    draw_note(window, QPointF(4.0, 60.0))  # and a plain click, which moves the playhead
    assert window.transport.suggestion.isVisible()

    window.view.set_notes(())
    window.transport.suggestion.dismiss_button.click()
    assert not window.transport.suggestion.isVisible()


def drawn_pixels(window, image: QImage, button: QToolButton) -> list[QColor]:
    """The pixels one button draws, read out of the window's own grab."""
    top = button.mapTo(window, QPoint(0, 0))
    return [image.pixelColor(top.x() + x, top.y() + y) for x in range(button.width()) for y in range(button.height())]


def test_a_button_that_cannot_be_clicked_reads_as_off(window) -> None:
    button = window.transport.detect
    button.setEnabled(True)
    lit = drawn_pixels(window, window.grab().toImage(), button)
    button.setEnabled(False)
    assert drawn_pixels(window, window.grab().toImage(), button) != lit, "it has to look unclickable"


def test_tempo_loader_reports_a_bad_file(window, tmp_path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_text("not audio")
    messages: list[str] = []
    loader = TempoLoader(broken)
    loader.failed.connect(messages.append)
    loader.run()
    assert len(messages) == 1 and "Error" in messages[0]


def test_a_spectrum_of_a_replaced_audio_is_left_out(own_window) -> None:
    own_window._audio_generation = 2
    own_window._on_spectrum_loaded(1, object())
    assert own_window.view.spectrum is None


def test_a_song_of_a_replaced_audio_is_left_out(own_window) -> None:
    own_window._audio_generation = 2
    own_window._on_song_loaded(1, object(), 44100)
    assert not own_window.song.is_loaded


def test_a_tempo_of_a_replaced_audio_is_not_offered(own_window) -> None:
    own_window.transport.bpm.setValue(120.0)
    own_window._audio_generation = 2
    own_window._on_tempo_loaded(1, fake_estimate(bpm=96.0))
    assert not own_window.transport.suggestion.isVisible()


def test_closing_waits_for_the_audio_loaders(own_window, monkeypatch) -> None:
    fake_loaders(monkeypatch)
    own_window.load_audio("/tmp/song.wav")
    waited: list[bool] = []

    class Waited:
        def wait(self) -> None:
            waited.append(True)

    own_window._workers = {Waited()}
    own_window.project_dirty = False
    own_window.close()
    assert waited


def test_hover_marks_the_row_and_its_overtones(window) -> None:
    window.view.overtone_highlight = True  # in either mode, editing or not
    window.view.clear_notes()
    window.view.centerOn(QPointF(8.0, float(PITCH_MAX - 66)))  # G3 and its overtones in view
    window.view.set_hover_pitch(55)  # G3
    assert window.view.highlight_pitches() == [55, 67, 74, 79]  # 2f, 3f and 4f above it
    assert window.cursor_note.text() == "G3   196.00 Hz"

    def row_brightness(pitch: int) -> int:
        return pixel_at(window.view, window.view.grab().toImage(), 8.0, PITCH_MAX - pitch + 0.5).lightness()

    rows = (54, 55, 67, 74, 79)
    marked = {pitch: row_brightness(pitch) for pitch in rows}
    window.view.set_hover_pitch(None)
    plain = {pitch: row_brightness(pitch) for pitch in rows}
    assert all(abs(marked[pitch] - plain[pitch]) > 5 for pitch in (55, 67, 74, 79)), "the row is marked"
    assert marked[54] == plain[54]  # the row above stays as it was
    assert window.cursor_note.text() == ""
    window.view.overtone_highlight = False


def test_hover_marks_the_row_over_the_spectrum_in_either_canvas(window, qt_app) -> None:
    window.view.overtone_highlight = False
    window.view.clear_notes()
    window.view.set_spectrum(make_spectrum(frames=400, value=0.0))  # black cells: only the band shows
    original = qt_app.palette()
    try:
        for name, window_colour in (("light", "#ffffff"), ("dark", "#202020")):
            palette = qt_app.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(window_colour))
            qt_app.setPalette(palette)
            assert theme.apply(qt_app) == name
            window.view.centerOn(QPointF(8.3, float(PITCH_MAX - 62)))

            def row_lightness(pitch: int) -> int:
                return pixel_at(window.view, window.view.grab().toImage(), 8.3, PITCH_MAX - pitch + 0.5).lightness()

            window.view.set_hover_pitch(62)
            marked = row_lightness(62)
            window.view.set_hover_pitch(None)
            plain = row_lightness(62)
            assert marked > plain, f"the hovered row has to show over the spectrum in the {name} canvas"
    finally:
        window.view.set_hover_pitch(None)
        window.view.set_spectrum(None)
        qt_app.setPalette(original)
        theme.apply(qt_app)


def test_hover_turns_the_key_of_that_row_red_in_either_mode(window) -> None:
    window.view.centerOn(QPointF(8.0, float(PITCH_MAX - 65)))
    window.view.overtone_highlight = True

    def red_rows(pitch: int) -> set[int]:
        window.view.set_hover_pitch(pitch)
        image = window.keyboard.grab().toImage()
        window.view.set_hover_pitch(None)
        return {y for y in range(image.height()) if image.pixelColor(2, y) == theme.canvas().hover_key}

    def red_bands(pitch: int) -> int:
        rows = sorted(red_rows(pitch))
        return sum(1 for index, y in enumerate(rows) if index == 0 or y != rows[index - 1] + 1)

    window.view.overtone_highlight = False
    if window.view.edit_mode:
        window.edit.mode.click()
    assert red_bands(55) == 1 and red_bands(54) == 1  # G3 is a white key, F#3 a black one: both mark

    window.view.set_hover_pitch(None)
    plain = window.keyboard.grab().toImage()
    assert not {y for y in range(plain.height()) if plain.pixelColor(2, y) == theme.canvas().hover_key}

    window.view.overtone_highlight = True  # the overtones mark the keyboard as well, in either mode
    assert red_bands(55) == 4
    window.edit.pen.click()
    assert red_bands(55) == 4  # and the same while editing
    window.edit.mode.click()
    window.view.overtone_highlight = False


def test_playhead_is_drawn_at_the_play_position(window) -> None:
    window.view.set_playhead(2.0)  # 2 s at 120 BPM = 4 beats
    image = window.view.grab().toImage()
    row = window.view.viewport().mapTo(window.view, QPoint(0, window.view.viewport().height() // 2)).y()
    marks = [x for x in range(image.width()) if image.pixelColor(x, row) == theme.canvas().playhead]
    window.view.set_playhead(None)
    assert marks
    assert min(abs(x - device_point(window.view, 4.0, 0.0).x()) for x in marks) <= 1


class FakeOutput:
    """Stands in for a playback backend, so the transport can be tested without sound."""

    silent = False

    def __init__(self) -> None:
        self.gain = 1.0
        self.duration = 0.0
        self.position = 0.0
        self.is_playing = False
        self.calls: list[str] = []
        self.programs: list[tuple] = []
        self.previews: list[int] = []

    def set_program(self, notes, speed, channels=()) -> None:
        self.programs.append((tuple(notes), speed))
        self.duration = max((start + duration for _pitch, start, duration, *_rest in notes), default=0.0) + 0.5
        self.calls.append("set_program")

    def preview(self, pitch: int, seconds: float = 0.6) -> None:
        self.previews.append(pitch)

    def play(self, seconds: float = 0.0) -> None:
        self.position = seconds
        self.is_playing = True
        self.calls.append("play")

    def pause(self) -> None:
        self.is_playing = False
        self.calls.append("pause")

    def stop(self) -> None:
        self.is_playing = False
        self.position = 0.0
        self.calls.append("stop")

    def seek(self, seconds: float) -> None:
        self.position = seconds
        self.calls.append("seek")


class FakeSong:
    """Stands in for the audio file layer, so the transport can be tested without a device."""

    def __init__(self, duration: float = 30.0) -> None:
        self.gain = 1.0
        self.speed = 1.0
        self.is_loaded = True
        self.duration = duration
        self.position = 0.0
        self.is_playing = False
        self.calls: list[str] = []

    def play(self, seconds: float = 0.0) -> None:
        self.position = seconds
        self.is_playing = True
        self.calls.append("play")

    def pause(self) -> None:
        self.is_playing = False
        self.calls.append("pause")

    def stop(self) -> None:
        self.is_playing = False
        self.position = 0.0
        self.calls.append("stop")

    def seek(self, seconds: float) -> None:
        self.position = seconds
        self.calls.append("seek")


def test_a_speed_change_reaches_the_notes_while_they_play(window, monkeypatch) -> None:
    song, notes = FakeSong(), FakeOutput()
    monkeypatch.setattr(window, "song", song)
    monkeypatch.setattr(window, "player", notes)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 2.0)
    window.transport.speed.set_value(1.0)
    window.transport.play_pause.click()
    assert notes.is_playing and notes.programs[-1][1] == 1.0
    song.position = 1.0  # the song is the master clock, and it is a second in

    window.transport.speed.set_value(1.5)
    window._apply_speed()  # what the settle timer does
    assert notes.programs[-1][1] == 1.5  # the notes are handed over at the new speed
    assert notes.is_playing and notes.position == pytest.approx(1.0)  # and carry on from there

    song.position = 1.4
    window.transport.speed.set_value(0.8)
    window._apply_speed()
    assert notes.programs[-1][1] == 0.8 and notes.position == pytest.approx(1.4)
    assert song.speed == 0.8  # the song takes the same speed, with nothing to rerender first
    assert notes.calls.count("set_program") == 3

    window.transport.speed.set_value(1.0)
    window._stop()
    window.view.clear_notes()


def test_the_roll_waits_for_a_pause_before_it_seeks(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 1.0)
    window.view.centerOn(QPointF(2.5, 20.0))

    window.transport.play_pause.click()
    assert window.view.playhead == pytest.approx(0.0)

    scene_pos = QPointF(2.5, 20.0)
    roll_mouse(window, QEvent.Type.MouseButtonPress, scene_pos)  # playing: the roll keeps its cursor
    roll_mouse(window, QEvent.Type.MouseButtonRelease, scene_pos)
    assert window.view.playhead == pytest.approx(0.0)
    assert fake.calls.count("seek") == 0  # the click did not reach the transport

    ruler_click(window, 4.0)  # the ruler seeks while the file runs
    assert window.view.playhead == pytest.approx(2.0)
    assert fake.calls.count("seek") == 1
    assert fake.is_playing and window.view.playing

    window.transport.play_pause.click()  # pause
    assert not window.view.playing
    roll_mouse(window, QEvent.Type.MouseButtonPress, scene_pos)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, scene_pos)
    assert window.view.playhead == pytest.approx(1.25)  # now the roll seeks again

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_the_volume_sliders_reach_their_layers(window, monkeypatch) -> None:
    song, notes = FakeSong(), FakeOutput()
    monkeypatch.setattr(window, "song", song)
    monkeypatch.setattr(window, "player", notes)

    window.mix.audio_volume.set_value(30)
    window.mix.midi_volume.set_value(20)

    assert song.gain == pytest.approx(0.3)  # what the audio file is streamed at
    assert notes.gain == pytest.approx(0.2)  # what the notes sound at


def test_transport_buttons_drive_the_player(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 1.0)

    window.transport.play_pause.click()
    assert fake.calls == ["set_program", "play"] and fake.is_playing
    assert fake.programs == [(((69, 0.0, 0.5, 0),), 1.0)]  # one beat at 120 BPM, handed over in seconds
    assert window.view.playhead is not None

    window.transport.play_pause.click()  # the same button pauses
    assert not fake.is_playing
    window.transport.forward.click()
    assert fake.position == pytest.approx(fake.duration)
    window.transport.rewind.click()
    assert fake.position == 0.0
    window.transport.stop.click()
    assert not fake.is_playing and fake.position == 0.0

    window.view.set_playhead(None)
    window.view.clear_notes()


def test_the_play_button_shows_what_it_will_do(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 2.0)

    def icon_image() -> QImage:
        return window.transport.play_pause.icon().pixmap(24, 24).toImage()

    play_icon = icon_image()
    window.transport.play_pause.click()
    assert window.transport.play_pause.toolTip() == "Pause playback (Space)"
    assert icon_image() != play_icon  # it became a pause button
    window.transport.play_pause.click()
    assert window.transport.play_pause.toolTip() == "Play from the cursor (Space)"
    assert icon_image() == play_icon
    window.view.clear_notes()


def test_play_from_the_beginning_rewinds_first(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 4.0)
    window.transport.play_pause.click()
    fake.position = 1.0  # pretend a second of it has played

    window.transport.play_from_start.click()
    assert "seek" in fake.calls and fake.position == 0.0 and fake.is_playing
    window.transport.stop.click()
    window.view.clear_notes()


def test_space_plays_and_pauses(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 2.0)
    QApplication.setActiveWindow(window)  # an offscreen window is never active on its own

    QTest.keyClick(window, Qt.Key.Key_Space)
    assert fake.is_playing
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert not fake.is_playing
    window.view.clear_notes()


def test_the_audio_file_plays_alongside_the_notes(window, monkeypatch) -> None:
    song, notes = FakeSong(), FakeOutput()
    monkeypatch.setattr(window, "song", song)
    monkeypatch.setattr(window, "player", notes)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 1.0)
    window.transport.speed.set_value(0.5)
    song.position = 4.0  # the playhead already sits inside the file

    window.transport.play_pause.click()
    assert song.calls == ["play"] and song.position == pytest.approx(4.0) and song.is_playing
    assert notes.calls == ["set_program", "play"] and notes.position == pytest.approx(4.0)
    assert notes.programs[0][1] == 0.5  # the note layer takes the same speed

    song.position = 6.5
    window._show_position()
    assert window.view.playhead == pytest.approx(6.5)  # the file leads the readout
    assert window.transport.position.text() == "00:06.500"

    window.transport.stop.click()
    assert song.calls == ["play", "stop"] and song.position == 0.0 and not song.is_playing
    window.transport.speed.set_value(1.0)
    window.view.clear_notes()


def test_the_audio_file_keeps_playing_when_the_notes_end(window, monkeypatch) -> None:
    song, notes = FakeSong(), FakeOutput()
    monkeypatch.setattr(window, "song", song)
    monkeypatch.setattr(window, "player", notes)
    window.position_timer.start()
    song.is_playing = True

    window._on_playback_finished()  # the note layer ran out first
    assert window.position_timer.isActive()

    song.is_playing = False
    window._on_playback_finished()  # and now the file is over too
    assert not window.position_timer.isActive()
    window.view.set_playhead(None)


def test_clicking_the_roll_moves_the_playhead(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()  # outside edit mode the roll is a seek bar
    assert not window.view.edit_mode
    scene_pos = QPointF(2.0, 40.0)  # scene units: two beats across, forty rows down

    roll_mouse(window, QEvent.Type.MouseButtonPress, scene_pos)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, scene_pos)
    assert window.view.playhead == pytest.approx(1.0)  # two beats at 120 BPM
    assert window.player.position == pytest.approx(1.0)

    window.edit.pen.click()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_the_pen_draws_where_the_playhead_lands(window) -> None:
    window.edit.pen.click()  # picking a tool enters edit mode
    assert window.view.edit_mode and window.view.tool is Tool.PEN
    window.view.set_playhead(None)
    window.view.clear_notes()
    left = QPointF(2.0, 40.0)
    right = QPointF(4.0, 40.0)

    draw_note(window, left, right)
    assert len(window.view.notes()) == 1
    assert window.view.playhead == pytest.approx(2.0)  # the drag carried the cursor to 4 beats
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_clicking_a_key_previews_the_pitch_under_it(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    keyboard = window.keyboard
    for y in (keyboard.height() // 4, keyboard.height() // 2, keyboard.height() * 3 // 4):
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(20.0, float(y)),
            keyboard.mapToGlobal(QPointF(20.0, float(y))),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.keyboard.mousePressEvent(event)
    assert len(fake.previews) == 3
    assert fake.previews == sorted(fake.previews, reverse=True)  # lower on screen is a lower note


def test_clicking_a_note_previews_it(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 1.0)
    window.view.centerOn(QPointF(2.5, float(PITCH_MAX - 69) + 0.5))
    position = window.view.mapFromScene(QPointF(2.5, float(PITCH_MAX - 69) + 0.5))
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(position),
        window.view.viewport().mapToGlobal(QPointF(position)),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.view.mousePressEvent(event)
    assert fake.previews == [69]
    window.view.mouseReleaseEvent(event)
    window.view.clear_notes()


def test_the_second_of_two_quick_clicks_still_sounds_and_seeks(window, monkeypatch) -> None:
    """Qt hands the second quick click over as a double-click, and that press counts as well."""
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    if window.view.edit_mode:
        window.edit.mode.click()
    window.view.clear_notes()
    row = float(PITCH_MAX - 60) + 0.5

    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
        window.view.set_playhead(None)
        roll_mouse(window, kind, QPointF(2.5, row))
        assert fake.previews[-1] == 60
        assert window.view.playhead == pytest.approx(1.25)  # 2.5 beats at 120 BPM
        roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))

    assert fake.previews == [60, 60]
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_drag_in_view_mode_scrubs_the_playhead(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    if window.view.edit_mode:
        window.edit.mode.click()
    window.view.clear_notes()
    window.view.set_playhead(None)
    window.view.centerOn(QPointF(3.0, float(PITCH_MAX - 60) + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.0, float(PITCH_MAX - 60) + 0.5))
    for beats, pitch in ((3.0, 62), (4.0, 64)):
        roll_mouse(window, QEvent.Type.MouseMove, QPointF(float(beats), float(PITCH_MAX - pitch) + 0.5))
        assert window.view.playhead == pytest.approx(beats / 2.0)  # two beats a second at 120 BPM
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.0, float(PITCH_MAX - 64) + 0.5))

    assert fake.previews == [60, 62, 64]  # the drag sounds the rows it crosses
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_drag_glides_the_playhead_and_sounds_every_row_it_crosses(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.edit.pen.click()  # the pen draws and auditions while it slides
    window.view.clear_notes()
    window.view.set_playhead(None)
    window.view.centerOn(QPointF(2.5, float(PITCH_MAX - 60) + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, float(PITCH_MAX - 60) + 0.5))
    for pitch in (62, 64, 64, 65):  # two moves inside one row must not sound it twice
        roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.5 + (pitch - 60) * 0.5, float(PITCH_MAX - pitch) + 0.5))
    assert fake.previews == [60, 62, 64, 65]  # a glissando up the rows
    assert window.view.playhead == pytest.approx(2.5)  # the drag reached 5 beats at 120 BPM
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, float(PITCH_MAX - 65) + 0.5))

    assert fake.previews == [60, 62, 64, 65]  # no stray preview after the release
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(7.0, float(PITCH_MAX - 70) + 0.5))
    assert fake.previews == [60, 62, 64, 65]  # and none while merely hovering either
    assert window.view.playhead == pytest.approx(2.5)

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_click_sounds_the_row_it_lands_on_in_either_mode(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    if window.view.edit_mode:
        window.edit.mode.click()  # nobody has picked a tool: this is the listening mode
    window.view.clear_notes()
    window.view.add_note(64, 2.0, 1.0)
    empty_row = float(PITCH_MAX - 60) + 0.5
    note_row = float(PITCH_MAX - 64) + 0.5

    draw_note(window, QPointF(2.5, empty_row))
    assert fake.previews == [60]  # an empty row sounds the pitch it is on
    draw_note(window, QPointF(2.5, note_row))
    assert fake.previews == [60, 64]  # a note sounds its own

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_press_on_a_note_moves_the_playhead_as_well(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.edit.pen.click()
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 1.0)
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(2.5, row))
    window.view.set_playhead(None)

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    assert window.view.playhead == pytest.approx(1.25)  # 2.5 beats at 120 BPM, note or not
    assert fake.previews == [69]  # and the note is still the one being edited
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(6.0, row))
    assert (note.start, note.end) == (5.5, 6.5)
    assert window.view.playhead == pytest.approx(3.0)  # the playhead travels with the drag
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(6.0, row))

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_drawing_a_note_covers_the_cells_it_passed_through(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    row = float(PITCH_MAX - 69) + 0.5
    press = QPointF(8.3, row)  # a third of the way into the cell that starts at 8.0
    draw_note(window, press, QPointF(9.6, row))
    notes = window.view.notes()
    assert len(notes) == 1
    note = notes[0]
    assert (note.pitch, note.start, note.end) == (69, 8.0, 10.0)
    assert note.start <= press.x() <= note.end  # the note is where the click was
    assert fake.previews == [69]
    window.view.clear_notes()


def test_dragging_left_draws_the_note_to_the_left(window) -> None:
    window.view.clear_notes()
    row = float(PITCH_MAX - 69) + 0.5
    draw_note(window, QPointF(9.4, row), QPointF(7.6, row))
    note = window.view.notes()[0]
    assert (note.start, note.end) == (7.5, 9.5)
    window.view.clear_notes()


def test_a_click_without_a_drag_draws_one_snap_cell(window) -> None:
    row = float(PITCH_MAX - 69) + 0.5
    for with_move in (False, True):  # Qt hands out move events even when the mouse barely moved
        window.view.clear_notes()
        press = QPointF(8.3, row)
        draw_note(window, press, press if with_move else None)
        note = window.view.notes()[0]
        assert (note.start, note.end) == (8.0, 8.5)
    window.view.clear_notes()


def test_drawing_slides_the_note_to_the_row_the_pointer_moves_to(window) -> None:
    window.view.clear_notes()
    draw_note(
        window,
        QPointF(8.3, float(PITCH_MAX - 69) + 0.5),
        QPointF(9.6, float(PITCH_MAX - 76) + 0.5),  # slid seven rows up while drawing
    )
    note = window.view.notes()[0]
    assert (note.pitch, note.start, note.end) == (76, 8.0, 10.0)
    window.view.clear_notes()


def test_dragging_either_edge_of_a_note_changes_its_duration(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.05, row))  # within the left grab band
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(1.0, row))
    assert (note.start, note.end) == (1.0, 4.0)  # the start moved, the end stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(1.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.95, row))  # within the right grab band
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row))
    assert (note.start, note.end) == (1.0, 5.0)  # the end moved, the start stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row))
    window.view.clear_notes()


def test_shift_dragging_a_note_moves_the_edge_that_was_grabbed(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0, so it is split at 3.0
    row = float(PITCH_MAX - 69) + 0.5
    start = Qt.KeyboardModifier.ShiftModifier

    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row), start)  # left half
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(1.6, row), start)
    assert (note.start, note.end) == (1.5, 4.0)  # the start moved, the end stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(1.6, row), start)

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.5, row), start)  # right half
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.9, row), start)
    assert (note.start, note.end) == (1.5, 5.0)  # the end moved, the start stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.9, row), start)

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.9, row), start)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(9.0, row), start)  # drag the start past the end
    assert note.duration == pytest.approx(window.view.snap)  # one snap cell, never nothing
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(9.0, row), start)
    window.view.clear_notes()


def test_a_note_drag_snaps_the_movement_and_keeps_the_note_off_the_grid(window) -> None:
    window.edit.pen.click()  # picking a tool enters edit mode
    window.view.clear_notes()
    note = window.view.add_note(69, 2.2, 1.0)  # a fifth of a cell off the 1/2-beat grid
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    body = QPointF(2.7, row)

    roll_mouse(window, QEvent.Type.MouseButtonPress, body)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.9, row))  # under half a cell: nothing at all
    assert (note.start, note.end) == (2.2, 3.2)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.0, row))  # past it: one whole cell
    assert (note.start, note.end) == (2.7, 3.7)  # and the note keeps its own offset
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.1, row))  # still one cell further along
    assert (note.start, note.end) == (2.7, 3.7)
    roll_mouse(window, QEvent.Type.MouseMove, body)  # and back again
    assert (note.start, note.end) == (2.2, 3.2)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, body)
    window.view.clear_notes()


def test_a_note_trim_that_comes_back_leaves_the_note_as_it_was(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    before = window.view.undo_stack.count()

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.95, row))  # the right edge
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(6.0, row))
    assert (note.start, note.end) == (2.0, 6.0)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.95, row))  # and the drag hands it back
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(3.95, row))
    assert (note.start, note.end) == (2.0, 4.0)
    assert window.view.undo_stack.count() == before
    window.view.clear_notes()


def test_a_drag_cannot_drop_a_note_onto_its_own_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    right = window.view.add_note(69, 4.0, 1.0)  # spans 4.0 to 5.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(4.5, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.5, row))  # one cell left: 3.0 to 4.0, touching
    assert (right.start, right.end) == (3.0, 4.0)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.0, row))  # another: 2.5 to 3.5 would overlap
    assert (right.start, right.end) == (3.0, 4.0)  # the drag stays at the last free place
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(1.5, row))  # far enough left to clear it again
    assert (right.start, right.end) == (1.0, 2.0)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(1.5, row))
    window.view.clear_notes()


def test_a_note_may_be_dragged_onto_another_channel(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    first = window.view.add_note(69, 2.0, 1.0, 0)
    window.view.add_note(69, 4.0, 1.0, 1)  # the same pitch over the same time, another channel
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.5, row))
    assert (first.start, first.end) == (4.0, 5.0)  # another channel is no obstacle
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.5, row))
    window.view.clear_notes()


def test_a_trim_cannot_reach_over_a_note_of_its_own_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    right = window.view.add_note(69, 4.0, 2.0)  # spans 4.0 to 6.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(4.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(4.05, row))  # the left edge
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.5, row))  # 3.5 to 6.0, still clear
    assert (right.start, right.end) == (3.5, 6.0)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.0, row))  # 2.0 to 6.0 would cross the first
    assert (right.start, right.end) == (3.5, 6.0)  # the edge stops where it last fit
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.0, row))
    window.view.clear_notes()


def test_a_trim_reaches_a_grid_line_a_hair_past_its_neighbour(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    left = window.view.add_note(69, 2.0, 0.5)  # spans 2.0 to 2.5
    # the neighbour lands a hair past the 3.0 line, the way a note comes back from a project file
    # that rounded its seconds; the snap line the grid offers must still be reachable
    window.view.add_note(69, 2.9999, 1.0)
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(2.5, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.45, row))  # the right edge
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.9, row))  # 2.0 to 3.0, the next snap line
    assert (left.start, left.end) == (2.0, 3.0)  # the hair it overlaps is below the model's slack
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.9, row))
    window.view.clear_notes()


def test_the_pen_opens_a_cell_that_overlaps_its_neighbour_by_a_hair(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 0.5)  # spans 2.0 to 2.5
    window.view.add_note(69, 2.9999, 1.0)  # a hair off the 3.0 line, as a project file leaves it
    row = float(PITCH_MAX - 69) + 0.5

    draw_note(window, QPointF(2.7, row), QPointF(2.7, row))  # one cell: 2.5 to 3.0
    assert (window.view.notes()[-1].start, window.view.notes()[-1].end) == (2.5, 3.0)
    window.view.clear_notes()


def test_the_pen_cannot_draw_over_a_note_of_its_own_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    row = float(PITCH_MAX - 69) + 0.5

    draw_note(window, QPointF(5.0, row), QPointF(2.0, row))  # back over the first note
    drawn = window.view.notes()[-1]
    assert (drawn.start, drawn.end) == (5.0, 5.5)  # it never left its first cell
    window.view.clear_notes()


def test_the_pen_cannot_open_a_note_over_one_of_its_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.set_channels((Channel(channel=0),))
    window.view.add_note(69, 2.1, 1.0)  # spans 2.1 to 3.1, off the grid
    row = float(PITCH_MAX - 69) + 0.5  # the note's row, so a click misses the note but reads its pitch
    window.view.centerOn(QPointF(2.1, row))

    draw_note(window, QPointF(2.0, row))  # off the note, yet the cell it would open reaches into it
    assert len(window.view.notes()) == 1  # the pen refuses instead of doubling the pitch
    draw_note(window, QPointF(1.5, row))  # a cell that stops at the note's edge
    assert len(window.view.notes()) == 2
    window.view.clear_notes()


def test_clicking_a_selected_note_leaves_only_it_selected(window) -> None:
    window.view.clear_notes()
    left = window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    right = window.view.add_note(69, 5.0, 1.0)  # spans 5.0 to 6.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(4.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    assert window.view.selected_notes() == [left, right]

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))  # a click, no drag
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))
    assert window.view.selected_notes() == [left]

    # and a press on one of them that does drag still carries the whole selection
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(5.5, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row))
    assert (left.start, left.end) == (1.5, 2.5)  # the whole selection moved by the one cell
    assert (right.start, right.end) == (4.5, 5.5)

    window.view.undo()
    window.view.clear_notes()


def test_ctrl_click_is_what_adds_to_the_selection(window) -> None:
    window.view.clear_notes()
    first = window.view.add_note(69, 2.0, 1.0)
    second = window.view.add_note(72, 4.0, 1.0)
    first_row = float(PITCH_MAX - 69) + 0.5
    second_row = float(PITCH_MAX - 72) + 0.5
    window.view.centerOn(QPointF(4.0, first_row))
    for scene_pos, modifiers in (
        (QPointF(2.5, first_row), Qt.KeyboardModifier.NoModifier),
        (QPointF(4.5, second_row), Qt.KeyboardModifier.ControlModifier),
    ):
        roll_mouse(window, QEvent.Type.MouseButtonPress, scene_pos, modifiers)
        roll_mouse(window, QEvent.Type.MouseButtonRelease, scene_pos, modifiers)
    assert first.isSelected() and second.isSelected()
    window.view.clear_notes()


def test_clicking_a_note_makes_its_channel_active_and_drops_the_others(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    window.view.clear_notes()
    on_zero = window.view.add_note(60, 2.0, 1.0, 0)
    on_one = window.view.add_note(64, 4.0, 1.0, 1)
    on_zero.setSelected(True)
    window.view.set_active_channel(0)
    window.view.centerOn(QPointF(4.5, PITCH_MAX - 64 + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(4.5, PITCH_MAX - 64 + 0.5))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.5, PITCH_MAX - 64 + 0.5))
    assert window.view.active_channel == 1
    assert on_one.isSelected() and not on_zero.isSelected()
    reset_channels(window)


def test_a_marquee_selects_only_the_active_channel(window) -> None:
    window.edit.select.click()
    window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    window.view.clear_notes()
    on_zero = window.view.add_note(60, 2.0, 2.0, 0)
    on_one = window.view.add_note(62, 2.0, 2.0, 1)
    window.view.set_active_channel(0)
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 61 + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(1.0, PITCH_MAX - 64 + 0.5))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, PITCH_MAX - 58 + 0.5))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, PITCH_MAX - 58 + 0.5))
    assert on_zero.isSelected() and not on_one.isSelected()
    reset_channels(window)
    window.edit.mode.click()


def test_a_drawn_note_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()

    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    assert len(window.view.notes()) == 1
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert window.view.notes() == []
    assert window.view.undo_stack.canRedo()
    window.view.redo()
    assert len(window.view.notes()) == 1
    window.view.clear_notes()


def test_moving_notes_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 2.0)
    window.view.undo_stack.clear()

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row - 2))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row - 2))
    assert window.view.undo_stack.count() == 1
    assert (window.view.notes()[0].start, window.view.notes()[0].pitch) == (4.0, 71)

    window.view.undo()
    assert (window.view.notes()[0].start, window.view.notes()[0].pitch) == (2.0, 69)
    window.view.redo()
    assert (window.view.notes()[0].start, window.view.notes()[0].pitch) == (4.0, 71)
    window.view.clear_notes()


def test_trimming_a_note_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0
    window.view.undo_stack.clear()

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.95, row))  # within the right grab band
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row))
    assert window.view.undo_stack.count() == 1
    assert window.view.notes()[0].end == 5.0

    window.view.undo()
    assert window.view.notes()[0].end == 4.0
    window.view.clear_notes()


def test_deleting_the_selection_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 0.0, 1.0)
    window.view.add_note(62, 1.0, 1.0)
    window.view.undo_stack.clear()

    for item in window.view.notes():
        item.setSelected(True)
    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
    assert window.view.notes() == []
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert [note.pitch for note in window.view.notes()] == [60, 62]
    window.view.clear_notes()


def test_escape_takes_back_a_note_that_was_just_drawn(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()
    window.view.centerOn(QPointF(3.0, 40.0))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.0, 40.0))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.0, 40.0))
    assert len(window.view.notes()) == 1  # the gesture opened a note
    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.0, 40.0))

    assert window.view.notes() == []  # and Escape took it back again
    assert window.view.undo_stack.count() == 0  # with no undo step
    window.view.clear_notes()


def test_ctrl_d_deletes_the_selection(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()
    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    assert len(window.view.notes()) == 1

    QApplication.setActiveWindow(window)  # an offscreen window is never active on its own
    QTest.keyClick(window, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []

    window.view.undo()
    assert len(window.view.notes()) == 1
    window.view.clear_notes()


def test_the_roll_shortcuts_need_the_roll(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 1.0).setSelected(True)
    QApplication.setActiveWindow(window)

    window.transport.bpm.setFocus()
    QTest.keyClick(window.transport.bpm, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert len(window.view.notes()) == 1  # a field keeps its own Ctrl+D

    window.view.setFocus()
    QTest.keyClick(window.view, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []  # in the roll, Ctrl+D is the notes'
    window.view.clear_notes()


def test_the_note_menu_lists_the_other_channels_and_a_new_one(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.add_channel()
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    seen: dict[str, list] = {}

    def fake_exec(self, *args, **kwargs):
        seen["entries"] = [(action.text(), action.data()) for action in self.actions() if not action.isSeparator()]
        return None

    monkeypatch.setattr(QMenu, "exec", fake_exec)
    roll_context_menu(window, QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert seen["entries"] == [(Channel(channel=1).label, 1), ("New channel", -1)]
    assert [item.channel for item in window.view.notes()] == [0]  # and nothing moves until an entry is picked
    window.view.clear_notes()


def test_the_note_menu_leaves_a_locked_channel_out(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0), Channel(channel=1, lock=True)))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))

    menu = window.view.channel_menu(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert [action.data() for action in menu.actions() if not action.isSeparator()] == [-1]
    window.view.clear_notes()


def test_a_cancelled_note_menu_leaves_the_selection(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.add_note(64, 8.0, 1.0, 0)
    window.view.centerOn(QPointF(8.5, PITCH_MAX - 64 + 0.5))
    keeper = window.view.notes()[0]
    keeper.setSelected(True)

    pick_menu(monkeypatch, None)  # the menu opens over the other note, but nothing is picked
    roll_context_menu(window, QPointF(8.5, PITCH_MAX - 64 + 0.5))

    assert [item.start for item in window.view.selected_notes()] == [2.0]
    window.view.clear_notes()


def test_a_note_moves_to_another_channel_from_the_menu(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.add_channel()
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    window.view.undo_stack.clear()

    pick_menu(monkeypatch, 1)
    roll_context_menu(window, QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert [item.channel for item in window.view.notes()] == [1]
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert [item.channel for item in window.view.notes()] == [0]
    window.view.clear_notes()


def test_a_note_moves_to_a_fresh_channel_in_one_step(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    window.view.undo_stack.clear()

    pick_menu(monkeypatch, -1)
    roll_context_menu(window, QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert [channel.channel for channel in window.view.channels] == [0, 1]
    assert [item.channel for item in window.view.notes()] == [1]
    assert window.view.undo_stack.count() == 1  # creating the channel and the move are one step
    window.view.undo()
    assert [channel.channel for channel in window.view.channels] == [0]
    assert [item.channel for item in window.view.notes()] == [0]
    window.view.clear_notes()


def test_clearing_every_note_can_be_undone(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 0.0, 1.0)
    window.view.add_note(62, 1.0, 1.0)
    window.view.undo_stack.clear()

    window.view.clear_notes()
    assert window.view.notes() == []
    window.view.undo()
    assert [note.pitch for note in window.view.notes()] == [60, 62]
    window.view.clear_notes()


def test_selecting_a_note_is_not_an_undo_step(window) -> None:
    window.edit.select.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 1.0)
    window.view.undo_stack.clear()

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))
    assert note.isSelected()
    assert window.view.undo_stack.count() == 0
    window.view.clear_notes()


def test_copy_and_paste_drops_the_selection_at_the_playhead() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.add_note(60, 2.0, 1.0)
    view.add_note(64, 3.0, 0.5)
    for note in view.notes():
        note.setSelected(True)
    view.set_playhead(3.0)  # six beats at 120 BPM
    view.undo_stack.clear()

    assert view.copy_selection()
    assert view.paste_notes()
    assert [(note.pitch, note.start, note.duration) for note in view.notes()] == [
        (60, 2.0, 1.0),
        (64, 3.0, 0.5),
        (60, 6.0, 1.0),
        (64, 7.0, 0.5),
    ]
    assert [note.pitch for note in view.selected_notes()] == [60, 64]  # the paste is what is selected
    assert view.undo_stack.count() == 1
    view.undo()
    assert len(view.notes()) == 2


def test_paste_snaps_the_anchor_and_the_spacing() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5  # the 1/8 default
    view.add_note(60, 2.0, 1.0)
    view.add_note(62, 3.3, 0.5)  # off the grid, so the paste has to align it
    for note in view.notes():
        note.setSelected(True)
    view.set_playhead(1.05)  # 2.1 beats, which snap to 2.0

    assert view.copy_selection()
    assert view.paste_notes()
    assert [(note.pitch, note.start) for note in view.notes()[2:]] == [(60, 2.0), (62, 3.5)]


def test_paste_without_a_copy_does_nothing() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_playhead(1.0)
    assert not view.paste_notes()
    assert view.notes() == []


def test_copy_and_paste_need_edit_mode() -> None:
    view = PianoRollView()
    view.set_channels((Channel(channel=0),))
    view.add_note(60, 2.0, 1.0).setSelected(True)
    assert not view.copy_selection()
    assert not view.paste_notes()


def test_quantize_puts_the_notes_on_the_snap_grid() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 0.6, 0.7)  # starts 0.5, ends 1.5
    view.add_note(62, 1.9, 0.2)  # both ends round to the same cell, so one cell is the length
    view.undo_stack.clear()

    assert view.quantize_notes()
    assert [(note.start, note.duration) for note in view.notes()] == [(0.5, 1.0), (2.0, 0.5)]
    assert view.undo_stack.count() == 1
    view.undo()
    assert [(note.start, note.duration) for note in view.notes()] == [(0.6, 0.7), (1.9, 0.2)]


def test_quantize_leaves_a_note_off_its_neighbour_rather_than_on_it() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 1.1, 0.2)  # both 60s round to the cell 1.0-1.5
    view.add_note(60, 1.2, 0.2)
    view.add_note(62, 3.1, 0.2)  # another pitch, free to snap

    assert view.quantize_notes()
    assert [(note.pitch, round(note.start, 2), round(note.duration, 2)) for note in view.notes()] == [
        (60, 1.1, 0.2),  # the grid would fold it onto the other 60, so it stays
        (60, 1.2, 0.2),
        (62, 3.0, 0.5),
    ]


def test_quantize_takes_only_the_selection_when_there_is_one() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 0.6, 1.0).setSelected(True)
    view.add_note(62, 1.6, 1.0)

    assert view.quantize_notes()
    assert [(note.pitch, note.start) for note in view.notes()] == [(60, 0.5), (62, 1.6)]


def test_quantize_leaves_a_locked_channel_where_it_is() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0, lock=True),))
    view.snap = 0.5
    view.add_note(60, 0.6, 1.0)

    assert not view.quantize_notes()
    assert view.notes()[0].start == 0.6


def test_a_note_already_on_the_grid_is_not_an_edit() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 1.0, 1.0)
    view.undo_stack.clear()

    assert not view.quantize_notes()
    assert view.undo_stack.count() == 0


def test_quantize_works_outside_edit_mode() -> None:
    view = PianoRollView()
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 0.6, 1.0)

    assert view.edit_mode is False
    assert view.quantize_notes()
    assert view.notes()[0].start == 0.5


def test_the_quantize_button_quantizes_the_roll(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.snap = 0.5
    window.view.add_note(60, 0.6, 1.0)
    window.view.undo_stack.clear()

    window.edit.quantize.click()
    assert window.view.notes()[0].start == 0.5
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    window.view.clear_notes()


def test_the_snap_grid_and_quantize_stay_usable_outside_edit_mode() -> None:
    bar = EditBar(SNAP_CHOICES)
    assert bar.snap.isEnabled() and bar.quantize.isEnabled()
    bar.mode.click()
    assert bar.snap.isEnabled() and bar.quantize.isEnabled()


def test_the_quantize_button_asks_the_roll_to_quantize() -> None:
    bar = EditBar(SNAP_CHOICES)
    asked: list[bool] = []
    bar.quantize_requested.connect(lambda: asked.append(True))
    bar.quantize.click()
    assert asked == [True]


def test_ctrl_c_and_ctrl_v_carry_the_selection(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 1.0).setSelected(True)
    window.view.set_playhead(2.0)  # four beats at 120 BPM

    QApplication.setActiveWindow(window)  # an offscreen window is never active on its own
    QTest.keyClick(window, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(window, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert [(note.pitch, note.start) for note in window.view.notes()] == [(69, 2.0), (69, 4.0)]
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_new_edit_drops_the_redo_branch(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()

    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    window.view.undo()
    assert window.view.undo_stack.canRedo()
    draw_note(window, QPointF(6.0, 40.0), QPointF(7.0, 40.0))
    assert not window.view.undo_stack.canRedo()
    window.view.clear_notes()


def test_channel_edits_and_a_removal_can_be_undone(window) -> None:
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_channel(program=4)
    window.view.add_note(60, 0.0, 1.0, 0)
    window.view.add_note(64, 1.0, 1.0, 1)
    window.view.undo_stack.clear()

    window.view.set_channel_field(1, mute=True)
    assert window.view.channels[1].mute is True
    window.view.undo()
    assert window.view.channels[1].mute is False

    assert window.view.remove_channel(0)
    assert [note.pitch for note in window.view.notes()] == [64]
    window.view.undo()
    assert [(note.pitch, note.channel) for note in window.view.notes()] == [(60, 0), (64, 1)]
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()


def test_a_whole_document_replacement_is_one_undo_step(window) -> None:
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()

    window.view.replace((Channel(channel=0),), [(60, 0.0, 1.0, 0), (64, 1.0, 1.0, 0)], "Import MIDI")
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert window.view.notes() == []


def test_undo_keeps_a_note_on_the_audio_across_a_tempo_change(window) -> None:
    window.edit.pen.click()
    window.transport.bpm.setValue(120.0)
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 2.0)
    window.view.undo_stack.clear()
    seconds = 2.0 * window.view.seconds_per_beat

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.0, row))

    window.transport.bpm.setValue(60.0)  # beats halve, seconds do not
    window.view.undo()
    restored = window.view.notes()[0]
    assert restored.start * window.view.seconds_per_beat == pytest.approx(seconds)
    window.transport.bpm.setValue(120.0)
    window.view.clear_notes()


def test_the_time_conversion_runs_both_ways(window) -> None:
    window.transport.bpm.setValue(120.0)
    assert window.view.to_seconds(2.0) == pytest.approx(1.0)
    assert window.view.to_beats(1.0) == pytest.approx(2.0)
    window.transport.bpm.setValue(60.0)
    assert window.view.to_seconds(0.5) == pytest.approx(0.5)
    assert window.view.to_beats(2.0) == pytest.approx(2.0)
    window.transport.bpm.setValue(120.0)


def test_ctrl_z_and_ctrl_shift_z_drive_the_history(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()
    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))

    QApplication.setActiveWindow(window)  # an offscreen window is never active on its own
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert len(window.view.notes()) == 1
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []
    QTest.keyClick(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert len(window.view.notes()) == 1
    window.view.clear_notes()


def test_opening_a_project_starts_the_history_over(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.add_note(60, 0.0, 1.0)
    assert own_window.view.undo_stack.canUndo()

    path = tmp_path / "song.nto"
    project.save(
        project.Project(settings=project.default_settings(), notes=(project.Note(1.0, 0.5, 62),)),
        path,
    )
    assert own_window.load_project(path)
    assert not own_window.view.undo_stack.canUndo()
    assert not own_window.view.undo_stack.canRedo()


def test_dragging_a_box_fills_the_selection_under_any_style(window) -> None:
    QApplication.setStyle("Windows")  # its own rubber band ignores a stylesheet, and the box is ours
    window.edit.select.click()
    window.view.centerOn(QPointF(5.0, 63.0))
    try:
        roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.0, 60.0))
        roll_mouse(window, QEvent.Type.MouseMove, QPointF(8.0, 66.0))
        rubber = window.view._rubber
        assert rubber.isVisible()
        area = rubber.geometry().intersected(window.view.viewport().rect())
        image = window.view.viewport().grab().toImage()
        accent = rubber.palette().highlight().color()
        canvas = image.pixelColor(area.left() - 20, area.center().y())
        inside = image.pixelColor(area.center())
        border = image.pixelColor(area.topLeft())

        def from_accent(colour: QColor) -> int:
            """How far one pixel is from the palette's accent, across the three channels."""
            return (
                abs(colour.red() - accent.red())
                + abs(colour.green() - accent.green())
                + abs(colour.blue() - accent.blue())
            )

        # the accent is whatever the palette carries - blue, grey, red - so the pixels are measured
        # against it rather than against a colour this test would be guessing
        assert inside != canvas, "the region is filled"
        assert from_accent(inside) < from_accent(canvas), "the fill is the accent over the canvas"
        assert from_accent(border) < from_accent(canvas), "the outline is the accent, not the style's own"
        roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(8.0, 66.0))
    finally:
        QApplication.setStyle("Fusion")
        window.edit.mode.click()


def roll_wheel(window, delta: int, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    view = window.view
    position = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        position,
        QPointF(view.viewport().mapToGlobal(QPointF(position).toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.wheelEvent(event)


def slider_wheel(slider, delta: int) -> None:
    position = QPointF(slider.width() / 2, slider.height() / 2)
    event = QWheelEvent(
        position,
        QPointF(slider.mapToGlobal(position.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    slider.wheelEvent(event)


def test_a_click_on_a_slider_track_lands_where_it_was_aimed(window) -> None:
    speed = window.transport.speed
    slider = speed.slider

    speed.set_value(1.0)
    QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width() * 3 // 4, slider.height() // 2))
    assert speed.value() > 1.5  # near the right end, not one page step (0.20) from where it was

    QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width() // 10, slider.height() // 2))
    assert speed.value() < 0.3  # and near the left end
    speed.set_value(1.0)


def test_the_slider_wheel_walks_the_other_way(window) -> None:
    speed = window.transport.speed
    speed.set_value(1.0)

    slider_wheel(speed.slider, 120)  # wheel up
    assert speed.value() == pytest.approx(0.95)  # turns the value down by one step (5%)
    slider_wheel(speed.slider, 120)
    slider_wheel(speed.slider, -120)  # wheel down
    assert speed.value() == pytest.approx(0.95)
    slider_wheel(speed.slider, -120)
    assert speed.value() == pytest.approx(1.0)
    speed.set_value(1.0)


def test_the_speed_slider_lands_on_five_percent_steps(window) -> None:
    speed = window.transport.speed
    seen: list[float] = []
    speed.value_changed.connect(seen.append)

    speed.slider.setValue(153)  # what a drag would hand over
    assert speed.value() == pytest.approx(1.55) and speed.value_label.text() == "1.55x"

    speed.slider.setValue(41)
    assert speed.value() == pytest.approx(0.40)
    speed.slider.triggerAction(QSlider.SliderAction.SliderSingleStepAdd)
    assert speed.value() == pytest.approx(0.45)

    speed.slider.setValue(0)
    assert speed.value() == pytest.approx(0.10)  # the slowest it goes
    speed.set_value(1.0)  # the reset button, on the grid as well
    assert speed.value() == pytest.approx(1.0)
    assert seen == [
        pytest.approx(1.55),
        pytest.approx(0.40),
        pytest.approx(0.45),
        pytest.approx(0.10),
        pytest.approx(1.0),
    ]
    assert all(abs(value * 20 - round(value * 20)) < 1e-9 for value in seen)  # every value is a step


def test_the_wheel_scrolls_the_timeline_and_shift_the_pitches(window) -> None:
    window.view.horizontalScrollBar().setValue(200)
    window.view.verticalScrollBar().setValue(200)
    horizontal = window.view.horizontalScrollBar().value()
    vertical = window.view.verticalScrollBar().value()

    roll_wheel(window, -120)
    assert window.view.horizontalScrollBar().value() == horizontal + 120
    assert window.view.verticalScrollBar().value() == vertical

    roll_wheel(window, -120, Qt.KeyboardModifier.ShiftModifier)
    assert window.view.horizontalScrollBar().value() == horizontal + 120
    assert window.view.verticalScrollBar().value() == vertical + 120


def test_the_roll_only_edits_in_edit_mode(window) -> None:
    row = float(PITCH_MAX - 69) + 0.5
    window.view.clear_notes()
    note = window.view.add_note(69, 8.0, 1.0)
    note.setSelected(True)

    if not window.view.edit_mode:
        window.edit.mode.click()
    window.edit.mode.click()  # leaving edit mode
    assert not window.view.edit_mode
    assert window.view.tool is None

    draw_note(window, QPointF(8.3, row), QPointF(9.6, row))
    assert [n.pitch for n in window.view.notes()] == [69]  # the pen drew nothing
    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
    assert len(window.view.notes()) == 1  # and delete did nothing either
    window.view.set_hover_pitch(69)
    assert window.view.highlight_pitches() == [69]  # only the row under the mouse, no overtones
    window.view.set_hover_pitch(None)
    window._stop()

    window.edit.mode.click()  # back in
    assert window.view.edit_mode and window.view.tool is Tool.PEN
    draw_note(window, QPointF(12.3, row), QPointF(13.6, row))  # elsewhere: the first row is taken
    assert len(window.view.notes()) == 2
    window.view.clear_notes()


def test_picking_a_tool_turns_on_edit_mode(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()
    assert not window.view.edit_mode
    window.edit.select.click()
    assert window.view.edit_mode and window.edit.mode.isChecked() and window.view.tool is Tool.SELECT
    window.edit.mode.click()
    window.edit.mode.click()  # entering the mode always lands on the pen
    assert window.view.tool is Tool.PEN and window.edit.pen.isChecked()


def test_playing_an_empty_roll_says_so(window) -> None:
    window.view.clear_notes()
    window.transport.play_pause.click()
    assert not window.player.is_playing
    assert "Nothing to play" in window.statusBar().currentMessage()


def test_changing_the_tempo_keeps_the_audio_at_the_same_size_on_screen(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 4.0, 2.0)
    window.view.centerOn(QPointF(20.0, 0.5))
    before = window.view.mapFromScene(QPointF(note.start, 0.0)).x()
    per_second = window.view.pixels_per_second()
    window.view.bpm = 93.0
    assert window.view.mapFromScene(QPointF(note.start, 0.0)).x() == before  # nothing moves
    assert window.view.pixels_per_second() == pytest.approx(per_second)
    window.view.bpm = 120.0
    window.view.clear_notes()


def test_changing_the_tempo_keeps_the_notes_where_they_are_in_time(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 4.0, 2.0)  # at 120 BPM: 2 s in, 1 s long
    window.view.bpm = 60.0  # the grid halves, the note does not move in the audio
    assert (note.start, note.duration) == pytest.approx((2.0, 1.0))
    window.view.bpm = 93.0  # and it survives a tempo that does not divide the old one
    assert note.start * 60.0 / 93.0 == pytest.approx(2.0)
    assert note.duration * 60.0 / 93.0 == pytest.approx(1.0)
    window.view.bpm = 120.0
    assert (note.start, note.duration) == pytest.approx((4.0, 2.0))
    window.view.clear_notes()


def test_note_times_follow_the_tempo(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 0.0, 2.0)  # two beats, one second at 120 BPM
    window.view.bpm = 60.0  # one beat per second now, so it is one beat and still one second
    window.transport.play_pause.click()
    assert fake.duration == pytest.approx(1.5)  # one second plus the release tail
    window.transport.stop.click()
    window.view.bpm = 120.0
    window.view.clear_notes()


def test_midi_volume_scales_the_output(window) -> None:
    window.mix.midi_volume.set_value(40.0)
    assert window.player.gain == pytest.approx(0.4)
    window.mix.midi_volume.set_value(80.0)


def test_a_silent_player_takes_the_midi_slider_out_of_reach(window) -> None:
    volume = window.mix.midi_volume.value()
    window.player.silent = True
    window._sync_midi_volume()
    assert window.mix.midi_volume.value() == 0
    assert not window.mix.midi_volume.isEnabled()
    window.player.silent = False
    window.mix.midi_volume.set_value(volume)
    window._sync_midi_volume()
    assert window.mix.midi_volume.isEnabled()


def test_a_silent_player_keeps_the_midi_volume_out_of_the_project(window) -> None:
    window.mix.midi_volume.set_value(70.0)
    window.player.silent = True
    window._sync_midi_volume()
    window._remember_configuration()
    assert window.project_settings.playback.midi_volume == 70
    window.player.silent = False
    window.mix.midi_volume.set_value(70.0)
    window._sync_midi_volume()


def test_the_built_in_synth_keeps_the_position_without_playing() -> None:
    synth = BuiltinSynth()
    synth.set_program([(69, 0.0, 1.0)], 1.0)
    assert synth.duration == pytest.approx(1.5)  # the note plus its release tail
    assert synth.position == 0.0 and not synth.is_playing
    synth.seek(0.75)
    assert synth.position == pytest.approx(0.75)
    synth.stop()
    assert synth.position == 0.0
    synth.set_program([(69, 0.0, 1.0)], 2.0)
    assert synth.duration == pytest.approx(1.5)  # the timeline itself does not change


def test_a_synth_port_is_preferred_over_the_loopback() -> None:
    loopback = "Midi Through:Midi Through Port-0 14:0"
    assert find_synth_port([]) is None
    assert find_synth_port([loopback]) is None
    assert find_synth_port([loopback, "TiMidity:TiMidity port 0 129:0"]) == 1
    assert find_synth_port(["Midi Through:Midi Through Port-0 14:0", "FLUID Synth (qsynth)"]) == 1


class FakePort:
    def __init__(self) -> None:
        self.messages: list[list[int]] = []
        self.times: list[float] = []

    def send_message(self, message) -> None:
        self.messages.append(list(message))
        self.times.append(time.monotonic())


def test_the_port_player_schedules_notes_and_silences_them_on_stop() -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.set_program([(69, 0.0, 1.0), (76, 0.5, 1.0)], 1.0)
    assert [0xC0, 0x00] in port.messages  # the synth is asked for its piano first
    assert [0xB0, 0x07, 127] in port.messages  # and told how loud to be
    assert player.duration == pytest.approx(1.5)

    player.play()
    assert player.is_playing
    time.sleep(0.15)
    assert [0x90, 69, 100] in port.messages  # the first note is on
    assert [0x80, 69, 0] not in port.messages  # and not yet off

    player.stop()
    assert not player.is_playing
    assert [0x80, 69, 0] in port.messages  # stopping silences what was sounding


def test_the_port_player_reports_when_the_notes_are_done() -> None:
    def app_events() -> None:
        QApplication.instance().processEvents()

    player = MidiPortOut(FakePort())
    player.set_program([(69, 0.0, 0.1)], 1.0)
    done: list[bool] = []
    player.finished.connect(lambda: done.append(True))
    player.play()
    for _ in range(100):
        if done:
            break
        app_events()
        time.sleep(0.02)
    assert done
    assert not player.is_playing
    assert player.position == pytest.approx(player.duration)


def test_a_preview_plays_the_note_and_releases_it() -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.preview(72, seconds=0.05)
    assert [0x90, 72, 100] in port.messages
    time.sleep(0.15)
    assert [0x80, 72, 0] in port.messages


def make_spectrum(frames: int = 4, value: float = 1.0) -> NoteSpectrum:
    table = np.full((frames, NOTE_COUNT), value, dtype=np.float32)
    return NoteSpectrum(table=table, frame_ms=50.0, sample_rate=44100, fft_points=8192, hop=2205, a4=440.0, sigma=1.0)


def device_point(view: PianoRollView, scene_x: float, scene_y: float) -> QPoint:
    return view.viewport().mapTo(view, view.mapFromScene(QPointF(scene_x, scene_y)))


def pixel_at(view: PianoRollView, image: QImage, scene_x: float, scene_y: float) -> QColor:
    return image.pixelColor(device_point(view, scene_x, scene_y))


def row_colors(image: QImage, row: int) -> list[int]:
    return [image.pixel(x, row) for x in range(image.width())]


def test_spectrum_image_follows_the_colour_ramp() -> None:
    table = np.array([[0.0], [0.2], [0.5], [1.0], [9.0]], dtype=np.float32).repeat(NOTE_COUNT, axis=1)
    spectrum = NoteSpectrum(table, 50.0, 44100, 8192, 2205, 440.0, 1.0)
    image = SpectrumImage(spectrum).image(240.0, 1.0)

    assert (image.width(), image.height()) == (5, NOTE_COUNT)
    assert image.pixel(0, 40) == 0xFF000000  # silence is black
    assert image.pixel(4, 40) == 0xFFFF0000  # full energy saturates to red

    quiet, middle, hot = (image.pixelColor(column, 40) for column in (1, 2, 3))
    assert quiet.blue() > quiet.red()  # the ramp starts as blue
    assert middle.green() > middle.red()  # climbs through green
    assert (hot.red(), hot.green(), hot.blue()) == (255, 2, 0)  # and ends red


def test_spectrum_image_is_cached_until_the_parameters_change() -> None:
    spectrum = SpectrumImage(make_spectrum())
    first = spectrum.image(240.0, 1.0)
    assert spectrum.image(240.0, 1.0) is first
    assert spectrum.image(300.0, 1.0) is not first


def test_spectrum_row_zero_is_the_highest_band() -> None:
    table = np.zeros((2, NOTE_COUNT), dtype=np.float32)
    table[:, NOTE_COUNT - 1] = 9.0  # B7, the top band
    image = SpectrumImage(NoteSpectrum(table, 50.0, 44100, 8192, 2205, 440.0, 1.0)).image(240.0, 1.0)
    assert image.pixel(0, 0) == 0xFFFF0000
    assert image.pixel(0, NOTE_COUNT - 1) == 0xFF000000


def test_spectrum_frames_map_onto_the_beat_grid() -> None:
    view = PianoRollView()
    view.bpm = 120.0
    view.set_spectrum(make_spectrum())
    assert view.frame_width() == pytest.approx(0.1)  # 50 ms at 120 BPM
    view.bpm = 60.0
    assert view.frame_width() == pytest.approx(0.05)


def test_timeline_grows_with_the_spectrum() -> None:
    view = PianoRollView()
    assert view.content_beats() == pytest.approx(LENGTH_BEATS + CONTENT_MARGIN)

    view.set_spectrum(make_spectrum(frames=2000))  # 2000 frames of 50 ms is 100 s, 200 beats at 120 BPM
    assert view.content_beats() == pytest.approx(200.0 + CONTENT_MARGIN)
    assert view.sceneRect().width() == pytest.approx(view.content_beats())

    view.bpm = 60.0  # the same audio spans half as many beats
    assert view.content_beats() == pytest.approx(100.0 + CONTENT_MARGIN)

    view.set_spectrum(None)
    assert view.sceneRect().width() == pytest.approx(LENGTH_BEATS + CONTENT_MARGIN)


def test_timeline_covers_a_five_minute_song() -> None:
    view = PianoRollView()
    view.set_spectrum(make_spectrum(frames=6000))  # 5 minutes of 50 ms frames
    assert view.content_beats() * 60 / view.bpm >= 299.0


def test_timeline_grows_with_notes() -> None:
    view = PianoRollView()
    view.add_note(60, 100.0, 4.0)
    assert view.content_beats() == pytest.approx(104.0 + CONTENT_MARGIN)
    view.clear_notes()
    assert view.content_beats() == pytest.approx(LENGTH_BEATS + CONTENT_MARGIN)


def test_spectrum_sits_on_the_pitch_axis() -> None:
    assert SPECTRUM_TOP == PITCH_MAX - (MIDI_OFFSET + NOTE_COUNT - 1) == 1


def test_octave_lines_sit_on_the_b_and_c_boundary() -> None:
    view = PianoRollView()
    view.resize(900, 500)
    view.set_spectrum(make_spectrum(frames=400, value=0.0))  # black cells, so only the lines show
    view.centerOn(20.0, float(PITCH_MAX - MIDI_OFFSET + 1))
    image = view.grab().toImage()

    def line_fraction(scene_y: float) -> float:
        y = device_point(view, 0.0, scene_y).y()
        row = [image.pixelColor(x, y) for x in range(3, image.width() - 3, 7)]
        return sum(colour == theme.canvas().spectrum_octave for colour in row) / len(row)

    assert line_fraction(PITCH_MAX - MIDI_OFFSET + 1) > 0.8  # below C1, where B0 ends
    assert line_fraction(PITCH_MAX - MIDI_OFFSET) < 0.05  # the C1 / C#1 edge stays black
    assert line_fraction(PITCH_MAX - MIDI_OFFSET - 12 + 1) > 0.8  # still there an octave up


def test_roll_paints_the_spectrum(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()  # full colour, so the edit-mode fade has to be off
    window.view.set_spectrum(make_spectrum(frames=200, value=9.0))
    image = window.view.grab().toImage()
    assert not image.isNull()
    assert any(
        image.pixelColor(x, y).red() > 200 and image.pixelColor(x, y).green() < 40
        for x in range(200, image.width(), 40)
        for y in range(20, image.height(), 20)
    )
    window.view.set_spectrum(None)


def test_editing_fades_the_spectrum_behind_the_notes(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()
    window.view.set_hover_pitch(None)
    window.view.set_spectrum(make_spectrum(frames=200, value=9.0))
    window.view.centerOn(QPointF(2.0, 40.5))

    plain = pixel_at(window.view, window.view.grab().toImage(), 2.0, 40.5)
    window.edit.pen.click()  # the spectrum steps back so a note draws attention over it
    dimmed = pixel_at(window.view, window.view.grab().toImage(), 2.0, 40.5)
    assert dimmed.lightness() < plain.lightness()

    window.edit.mode.click()
    assert pixel_at(window.view, window.view.grab().toImage(), 2.0, 40.5).lightness() == plain.lightness()
    window.view.set_spectrum(None)


def test_notes_are_drawn_over_the_spectrum(window) -> None:
    window.edit.pen.click()  # notes are only drawn while editing, and the spectrum fades then
    window.view.clear_notes()
    window.view.set_spectrum(make_spectrum(frames=400, value=9.0))
    note = window.view.add_note(60, 1.0, 4.0)
    middle = (note.start + note.duration / 2, PITCH_MAX - note.pitch + 0.5)
    window.view.centerOn(QPointF(*middle))
    image = window.view.grab().toImage()
    assert pixel_at(window.view, image, *middle) == note.fill
    beside = pixel_at(window.view, image, note.start - 0.5, middle[1])
    assert beside != note.fill and beside.red() > 0  # the spectrum is behind the note, dimmed by the mode
    window.view.clear_notes()
    window.view.set_spectrum(None)
    window.edit.mode.click()


def test_notes_are_flat_red_with_a_bevel(window) -> None:
    window.edit.pen.click()  # a note is only on screen while editing
    window.view.clear_notes()
    window.view.set_spectrum(make_spectrum(frames=400, value=0.0))  # black cells, nothing else is red
    note = window.view.add_note(60, 2.0, 2.0)
    top = PITCH_MAX - note.pitch + NOTE_INSET
    bottom = PITCH_MAX - note.pitch + 1.0 - NOTE_INSET
    window.view.centerOn(QPointF(note.start + note.duration / 2, (top + bottom) / 2))
    image = window.view.grab().toImage()
    top_left = device_point(window.view, note.start, top)
    bottom_right = device_point(window.view, note.end, bottom)

    mid_x = (top_left.x() + bottom_right.x()) // 2
    column = [image.pixelColor(mid_x, y) for y in range(top_left.y(), bottom_right.y())]
    assert note.edge_light in column[:2]  # the light bevel is on top
    assert note.edge_dark in column[-2:]  # the dark one below
    assert {colour.name() for colour in column[2:-2]} == {note.fill.name()}

    row = [image.pixelColor(x, (top_left.y() + bottom_right.y()) // 2) for x in range(top_left.x(), bottom_right.x())]
    assert note.edge_light in row[:2]  # and light on the left, dark on the right
    assert note.edge_dark in row[-2:]
    assert {colour.name() for colour in row[2:-2]} == {note.fill.name()}
    window.edit.mode.click()


def test_selected_notes_use_the_wavetone_highlight(window) -> None:
    window.edit.pen.click()  # a note is only on screen while editing
    window.view.clear_notes()
    note = window.view.add_note(60, 2.0, 2.0)
    top = PITCH_MAX - note.pitch + NOTE_INSET
    bottom = PITCH_MAX - note.pitch + 1.0 - NOTE_INSET
    window.view.centerOn(QPointF(note.start + note.duration / 2, (top + bottom) / 2))
    note.setSelected(True)
    image = window.view.grab().toImage()
    top_left = device_point(window.view, note.start, top)
    bottom_right = device_point(window.view, note.end, bottom)
    assert pixel_at(window.view, image, note.start + 1.0, (top + bottom) / 2) == theme.canvas().note_selected

    mid_x = (top_left.x() + bottom_right.x()) // 2
    column = [image.pixelColor(mid_x, y) for y in range(top_left.y(), bottom_right.y())]
    assert {colour.name() for colour in column} == {
        theme.canvas().note_selected.name(),
        theme.canvas().note_selected_edge.name(),
    }
    window.edit.mode.click()


def test_notes_are_drawn_only_in_edit_mode(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()
    window.view.clear_notes()
    note = window.view.add_note(60, 1.0, 2.0)
    assert not note.isVisible()  # outside it the roll is the graph, the way WaveTone keeps it

    window.edit.pen.click()
    assert note.isVisible()

    window.edit.mode.click()
    assert not note.isVisible()
    window.view.clear_notes()


def test_the_item_is_a_view_of_the_document_note(window) -> None:
    window.view.clear_notes()
    item = window.view.add_note(60, 1.0, 2.0)
    note = item.note
    assert window.view.document.notes == [note] and window.view.notes() == [item]

    note.set_range(4.0, 67)  # the model moves and the item follows it
    assert (item.start, item.pitch) == (4.0, 67)
    item.set_duration(3.0)  # and the item writes back into the model
    assert note.duration == 3.0

    window.view.clear_notes()
    assert window.view.document.notes == [] and window.view.notes() == []


def test_spectrum_loader_reports_a_bad_file(window, tmp_path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_text("not audio")
    messages: list[str] = []
    loader = SpectrumLoader(broken)
    loader.failed.connect(messages.append)
    loader.run()
    assert len(messages) == 1 and "Error" in messages[0]


@pytest.fixture
def own_window(tmp_path, monkeypatch):
    """A window with a settings file of its own, for the tests that read or write one."""
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(tmp_path / "settings.json"))
    opened = MainWindow()
    opened.resize(1200, 720)
    yield opened
    opened.project_dirty = False  # a test may leave unsaved notes: closing must not ask about them
    opened.close()


def wait_for_lyric_mapping(window) -> None:
    deadline = time.monotonic() + 1.0
    while window._lyric_map_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    assert window._lyric_map_thread is None


def row_writer(dialog, section: str, name: str):
    """The write half of one row of the settings window, to change it the way a widget would."""
    for row_section, field, _read, write in dialog._rows:
        if (row_section, field.name) == (section, name):
            return write
    raise AssertionError(f"the settings window has no row for {section}.{name}")


def test_the_gear_button_opens_the_settings_window(own_window, monkeypatch) -> None:
    opened: list[SettingsDialog] = []
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: opened.append(self) or 0)
    own_window.transport.settings_button.click()
    assert len(opened) == 1
    assert opened[0].parent() is own_window


def test_the_settings_window_lists_every_visible_field(own_window) -> None:
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    names = {(section, field.name) for section, field, _read, _write in dialog._rows}
    expected = {
        (section.name, field.name) for section in store.SECTIONS if section.name in PAGES for field in section.fields
    }
    assert names == expected  # `editor` has no page: its switches are what the program remembers itself
    pages = [dialog.findChild(QTabWidget).tabText(index) for index in range(dialog.findChild(QTabWidget).count())]
    assert pages == ["General", "Devices", "Tempo", "Lyrics", "Network", "MIDI", "Remote"]
    dialog.close()


def test_applying_the_settings_window_reaches_the_window_and_the_file(own_window) -> None:
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    dialog.applied.connect(own_window.settings_store.apply)  # the window wires this up when it opens it
    row_writer(dialog, "tempo", "window_seconds")(20.0)
    row_writer(dialog, "midi", "wavetone")(False)
    dialog.apply()

    assert own_window.settings.tempo.window_seconds == 20.0
    assert own_window.settings.midi.wavetone is False
    saved = json.loads(store.default_path().read_text())
    assert saved["tempo"]["window_seconds"] == 20.0
    assert saved["midi"]["wavetone"] is False
    dialog.close()


def test_the_analysis_options_are_a_project_s_so_the_window_has_no_row(own_window) -> None:
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    assert not [row for row in dialog._rows if row[0] == "analysis"]
    dialog.close()


def test_a_wrapped_status_line_grows_to_fit_its_text(qt_app) -> None:
    label = WrappedLabel()
    label.setText("没有可用的 GPU 后端；选择 GPU 时会回退到 CPU")
    label.setFixedWidth(200)
    label.show()
    qt_app.processEvents()

    needed = label.heightForWidth(200)
    assert needed > label.fontMetrics().lineSpacing()  # the text really does wrap
    assert label.minimumHeight() == needed  # and the row is forced tall enough for it
    label.close()


def test_the_gpu_row_says_whether_its_runtime_is_installed(qt_app, monkeypatch) -> None:
    monkeypatch.setattr(devices, "installed", lambda: ("CPUExecutionProvider",))
    field = store.FIELD_SPECS[("devices", "gpu")]
    combo, _read, write = field_editor(store.Settings().devices.gpu, field)
    status = QLabel()

    _show_device_status(combo, status)
    assert "No GPU backend is available" in status.text()

    write("cuda")
    _show_device_status(combo, status)
    assert "NVIDIA CUDA is not available" in status.text()

    monkeypatch.setattr(devices, "installed", lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"))
    _show_device_status(combo, status)
    assert "NVIDIA CUDA is available" in status.text()


def test_restoring_defaults_puts_every_widget_back(own_window) -> None:
    own_window.settings.ass.font_size = 72
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    row_writer(dialog, "tempo", "window_seconds")(8.0)
    row_writer(dialog, "midi", "wavetone")(False)
    assert store.get_value(dialog.values(), "midi", "wavetone") is False

    dialog.restore_defaults()
    values = dialog.values()
    assert store.get_value(values, "midi", "wavetone") is True
    assert store.get_value(values, "tempo", "window_seconds") == 12.0
    assert values.ass.font_size == 72
    dialog.close()


def test_closing_the_window_remembers_its_state_and_the_document(own_window) -> None:
    own_window.view.set_zoom(96.0, 20.0)
    own_window.close()

    saved = window_state.load()
    assert saved.geometry
    assert own_window.project_settings.editor.zoom_x == 96.0
    assert own_window.project_settings.editor.zoom_y == 20.0
    assert own_window.project_settings.view.center_x > 0.0


def test_a_new_window_starts_from_the_remembered_defaults() -> None:
    window_state.save(window_state.State(project={"spectrum": {"gain": 300.0}, "editor": {"zoom_x": 96.0}}))
    opened = MainWindow()
    assert opened.mix.gain.value() == 300.0
    assert opened.view.zoom == (96.0, 16.0)
    opened.project_dirty = False
    opened.close()


def test_an_explicit_change_is_what_the_next_project_inherits(own_window) -> None:
    own_window.mix.gain.set_value(300.0)
    own_window.transport.grid_offset.setValue(120)
    own_window.view._zoom(1.5, 1.5, QPoint(0, 0))  # a wheel zoom is the user's; set_zoom is not
    own_window.close()

    saved = window_state.load().project
    assert saved["spectrum"]["gain"] == 300.0
    assert saved["editor"]["zoom_x"] == 72.0  # 48 * 1.5
    assert saved["editor"]["zoom_y"] == 24.0  # 16 * 1.5
    assert "grid_offset_ms" not in saved["editor"]  # the grid offset belongs to the song too
    assert "view" not in saved


def test_opening_a_project_leaves_the_remembered_defaults_alone(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.spectrum.gain = 300.0
    other.editor.zoom_x = 96.0
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)
    assert own_window.load_project(path) is True
    own_window.close()

    assert window_state.load().project == {}


def test_the_preferences_are_read_when_the_window_starts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "settings.json"
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(path))
    saved = store.Settings()
    store.set_value(saved, "editor", "auto_page", True)
    store.save(saved)

    opened = MainWindow()
    assert opened.transport.auto_page.isChecked() is True
    opened.close()


def test_a_new_document_starts_from_the_document_defaults(own_window) -> None:
    document = project.default_settings()
    assert own_window.view.contrast == document.spectrum.contrast
    assert own_window.view.snap == document.editor.snap
    assert own_window.transport.bpm.value() == document.tempo.bpm
    assert own_window.view.zoom == (document.editor.zoom_x, document.editor.zoom_y)


def test_the_command_line_seeds_the_run_without_writing_itself_back(own_window) -> None:
    own_window.apply_overrides(gain=300.0, contrast=2.0)
    assert own_window.view.gain == 300.0
    own_window.close()

    document = project.default_settings()
    assert own_window.project_settings.spectrum.gain == document.spectrum.gain
    assert own_window.project_settings.spectrum.contrast == document.spectrum.contrast


def test_every_bar_setting_has_one_binding(own_window) -> None:
    bound = {(binding.section, binding.name) for binding in own_window._bindings}
    # the document's values are all on a bar, save the ones the roll, the window and the command line own
    elsewhere = {
        ("editor", "zoom_x"),
        ("editor", "zoom_y"),
        ("view", "center_x"),
        ("view", "center_y"),
        ("analysis", "channels"),
        ("analysis", "t_num"),
        ("analysis", "fft_points"),
        ("analysis", "a4"),
    }
    assert set(project.FIELD_SPECS) - bound == elsewhere
    # the two switches with no page are the program's, and are still on the bars
    assert {("editor", "auto_page"), ("editor", "overtone_highlight")} <= bound


def test_a_command_line_channel_beats_the_document(own_window) -> None:
    own_window.project_settings.analysis.channels = "side"
    own_window.overrides["channels"] = "left"
    own_window.overrides["t_num"] = None
    options = own_window._analysis_options()
    assert options["channels"] == "left"
    assert options["t_num"] == 40.0
    assert options["fft_points"] == 8192
    assert options["a4"] == 440.0


def test_the_spectrum_loader_takes_every_analysis_parameter(tmp_path) -> None:
    loader = SpectrumLoader(tmp_path / "song.wav", channels="side", t_num=25.0, fft_points=2048, a4=442.0)
    assert (loader.channels, loader.t_num, loader.fft_points, loader.a4) == ("side", 25.0, 2048, 442.0)
    assert loader.path == tmp_path / "song.wav"


class _Signal:
    """A signal nothing is connected to: the loaders a file starts are faked in the tests below."""

    def connect(self, *_args) -> None:
        pass


def fake_loaders(monkeypatch) -> list[dict]:
    """Replace the loaders a file starts with ones that only record what they were asked to run."""
    captured: list[dict] = []

    class FakeLoader:
        def __init__(self, *args, parent=None, **kwargs):
            self.progress = self.loaded = self.failed = self.finished = _Signal()
            captured.append(kwargs)

        def start(self) -> None:
            pass

        def wait(self) -> None:
            pass

        def deleteLater(self) -> None:
            pass

    for name in ("SpectrumLoader", "SongLoader", "TempoLoader"):
        monkeypatch.setattr(f"namioto.ui.app.{name}", FakeLoader)
    return captured


def test_loading_a_file_hands_the_document_to_the_analysers(own_window, monkeypatch) -> None:
    captured = fake_loaders(monkeypatch)
    own_window.project_settings.analysis.fft_points = 4096
    own_window.project_settings.analysis.a4 = 441.0
    store.set_value(own_window.settings, "tempo", "window_seconds", 8.0)
    own_window.overrides["channels"] = "left"

    own_window.load_audio("/tmp/song.wav")

    analysis, _song, tempo = captured
    assert analysis == {"channels": "left", "t_num": 40.0, "fft_points": 4096, "a4": 441.0}
    assert tempo == {"algorithm": "wavetone", "window_seconds": 8.0, "window_hop_seconds": 6.0}
    assert own_window.state.last_audio_dir == "/tmp"
    assert own_window.edit.transcribe.isEnabled()


def test_the_tempo_loader_follows_the_settings(own_window, monkeypatch) -> None:
    own_window.audio_path = "song.wav"
    store.set_value(own_window.settings, "tempo", "estimator", "librosa")
    store.set_value(own_window.settings, "tempo", "window_seconds", 8.0)
    store.set_value(own_window.settings, "tempo", "window_hop_seconds", 3.0)
    monkeypatch.setattr(TempoLoader, "start", lambda self: None)
    own_window._start_tempo()
    assert own_window.tempo_loader.algorithm == "librosa"
    assert own_window.tempo_loader.window_seconds == 8.0
    assert own_window.tempo_loader.window_hop_seconds == 3.0


def test_a_song_tempo_never_reaches_the_preferences(own_window) -> None:
    own_window.transport.bpm.setValue(93.0)
    own_window.transport.grid_offset.setValue(120)
    own_window.close()

    saved = store.load()
    assert not hasattr(saved, "playback")  # the offset is the document's, and it is not written here
    assert not hasattr(saved.tempo, "bpm")
    assert own_window.project_settings.tempo.bpm == 93.0


def test_another_song_starts_from_the_default_tempo_and_grid_offset(own_window, monkeypatch) -> None:
    fake_loaders(monkeypatch)
    own_window.transport.bpm.setValue(93.0)
    own_window.transport.grid_offset.setValue(120)

    own_window.load_audio("/tmp/other.wav")
    assert own_window.transport.bpm.value() == 120.0
    assert own_window.transport.grid_offset.value() == 0

    own_window.transport.bpm.setValue(93.0)
    own_window.load_audio("/tmp/other.wav")  # the same file again: a re-analysis keeps what was typed
    assert own_window.transport.bpm.value() == 93.0


def test_a_project_brings_its_tempo_and_grid_offset_back(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.tempo.bpm = 93.0
    other.editor.grid_offset_ms = 120
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)

    assert own_window.load_project(path) is True
    assert own_window.transport.bpm.value() == 93.0
    assert own_window.transport.grid_offset.value() == 120
    own_window.settings_store.flush()
    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert not hasattr(own_window.settings, "playback")
    assert "bpm" not in written["tempo"]  # the song's tempo stays in the document


def test_the_overtone_highlight_can_be_turned_off(window) -> None:
    window.edit.set_interaction(Interaction.viewing())
    window.view.overtone_highlight = True
    window.view.set_hover_pitch(60)
    assert window.view.highlight_pitches() == [60, 72, 79, 84]  # no edit mode needed

    window.view.overtone_highlight = False
    assert window.view.highlight_pitches() == [60]

    window.edit.set_interaction(Interaction.editing_with(Tool.PEN))
    assert window.view.highlight_pitches() == [60]  # and the switch still governs it while editing
    window.edit.set_interaction(Interaction.viewing())
    window.view.set_hover_pitch(None)


def test_the_zoom_can_be_set_from_outside(window) -> None:
    window.view.set_zoom(96.0, 32.0)
    assert window.view.zoom == (96.0, 32.0)
    window.view.set_zoom(1.0, 500.0)  # both ends are held to what the roll can draw
    assert window.view.zoom == (window.view.MIN_ZOOM_X, window.view.MAX_ZOOM_Y)
    window.view.set_zoom(48.0, 16.0)


def test_a_midi_port_is_matched_by_name() -> None:
    ports = ("Midi Through:Midi Through Port-0 14:0", "TiMidity:TiMidity port 0 128:0")
    assert find_port(ports, "TiMidity:TiMidity port 0 128:0") == 1
    assert find_port(ports, "TiMidity") == 1  # the client number changes between sessions
    assert find_port(ports, "Nonesuch") == 1  # nothing matches, so the first synth is used
    assert find_port(("Midi Through:0",), "") is None


def test_the_transport_carries_the_project_buttons() -> None:
    bar = TransportBar()
    seen: list[str] = []
    bar.open_requested.connect(lambda: seen.append("open"))
    bar.save_requested.connect(lambda: seen.append("save"))
    bar.export_midi_requested.connect(lambda: seen.append("export midi"))
    bar.export_krc_requested.connect(lambda: seen.append("export lyrics"))
    bar.export_ass_requested.connect(lambda: seen.append("export subtitle"))
    bar.open.click()
    bar.save.click()
    bar.export_midi_action.trigger()
    bar.export_krc_action.trigger()
    bar.export_ass_action.trigger()
    assert seen == ["open", "save", "export midi", "export lyrics", "export subtitle"]


def test_the_edit_bar_carries_the_wand() -> None:
    bar = EditBar(SNAP_CHOICES)
    bar.transcribe.setEnabled(True)  # no audio in this bar, so it starts off
    seen: list[str] = []
    bar.transcribe_requested.connect(lambda: seen.append("transcribe"))
    bar.transcribe.click()
    assert seen == ["transcribe"]


class FakeProcess:
    """A child that has already stopped, so the dialog sees what a real one leaves behind."""

    def __init__(self, exitcode: int = 0):
        self.exitcode = exitcode
        self.terminated = False

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout=None) -> None:
        pass


def fake_job(messages, exitcode: int = 0):
    """The spawn seam replaced by a queue that is already full and a child that already stopped."""
    channel: queue.Queue = queue.Queue()
    for message in messages:
        channel.put(message)
    process = FakeProcess(exitcode)

    def start_job(_audio, _parameters):
        return process, channel

    return start_job


def parameter_writer(dialog, name: str):
    """The write half of one row of a dialog's parameter form, to change it the way a widget would."""
    for field, _read, write in dialog._fields:
        if field.name == name:
            return write
    raise AssertionError(f"this dialog has no {name} field")


def test_the_transcribe_button_needs_audio(own_window) -> None:
    assert own_window.edit.transcribe.isEnabled() is False


def test_the_transcribe_button_opens_the_dialog(own_window, monkeypatch) -> None:
    own_window.audio_path = "/tmp/song.wav"
    own_window.edit.transcribe.setEnabled(True)
    opened: list[TranscriptionDialog] = []
    monkeypatch.setattr(TranscriptionDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    own_window.edit.transcribe.click()

    assert len(opened) == 1
    assert opened[0].parent() is own_window
    assert opened[0].audio == "/tmp/song.wav"


def test_the_transcription_dialog_carries_the_grid_offset(own_window, monkeypatch) -> None:
    own_window.audio_path = "/tmp/song.wav"
    own_window.transport.grid_offset.setValue(125)
    opened: list[TranscriptionDialog] = []
    monkeypatch.setattr(TranscriptionDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    own_window._open_transcription()

    assert opened[0].offset == 0.125


def test_the_transcription_dialog_snaps_the_raw_notes_on_its_grid(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.51, 0.98, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, offset=0.25)
    parameter_writer(dialog, "quantize")(4)  # 1/16 notes: a 0.125 s cell at 120 BPM

    dialog._start()
    dialog._poll()

    assert dialog.notes() == [(0.5, 1.0, 60.0)]  # the 0.25 s offset grid, not the absolute one
    assert "offset 250.0 ms" in dialog.log.toPlainText()


def test_the_transcription_dialog_offers_every_parameter(own_window) -> None:
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    assert [field.name for field, _read, _write in dialog._fields] == [item.name for item in transcription.PARAMETERS]
    dialog.close()


def test_the_transcription_dialog_remembers_what_was_typed(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    parameter_writer(dialog, "size")("large")
    parameter_writer(dialog, "language")("zh")

    dialog._start()
    dialog._poll()

    saved = transcription.load_parameters()
    assert saved["size"] == "large"
    assert saved["language"] == "zh"
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_the_transcription_run_shows_its_log_and_progress(own_window, monkeypatch) -> None:
    messages = [
        ("log", "GAME model small"),
        ("progress", "parts", 1, 2),
        ("progress", "parts", 2, 2),
        ("done", [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)]),
    ]
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job(messages))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert "GAME model small" in dialog.log.toPlainText()
    assert dialog.progress_label.text() == "2 notes"
    assert dialog.notes() == [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)]
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_a_cached_run_is_offered_instead_of_running(own_window, monkeypatch) -> None:
    transcription.save_run("/tmp/song.wav", transcription.default_parameters(), [(0.0, 0.5, 60.0)])
    asked: list[bool] = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: asked.append(True) or QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", lambda *args: pytest.fail("must not run"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()

    assert asked == [True]
    assert dialog.notes() == [(0.0, 0.5, 60.0)]
    assert "saved run reused" in dialog.log.toPlainText()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_a_declined_cache_still_runs(own_window, monkeypatch) -> None:
    transcription.save_run("/tmp/song.wav", transcription.default_parameters(), [(0.0, 0.5, 60.0)])
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(1.0, 1.5, 64.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert dialog.notes() == [(1.0, 1.5, 64.0)]


def test_a_crashed_transcription_leaves_the_window_alone(own_window, monkeypatch) -> None:
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_job", fake_job([("log", "GAME model small")], exitcode=1)
    )
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert "exit code 1" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_a_failed_transcription_shows_the_traceback(own_window, monkeypatch) -> None:
    monkeypatch.setattr(
        "namioto.ui.transcription_dialog.start_job",
        fake_job([("error", "Traceback\nRuntimeError: no model")]),
    )
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    dialog._poll()

    assert "RuntimeError: no model" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()


def test_a_child_that_cannot_start_is_reported(own_window, monkeypatch) -> None:
    def refuse(*args):
        raise OSError("no processes left")

    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", refuse)
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()

    assert "no processes left" in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()


def test_a_missing_runtime_stops_the_transcription(own_window, monkeypatch) -> None:
    monkeypatch.setattr(devices, "validate", lambda: devices.MISSING_RUNTIME)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", lambda *args: pytest.fail("must not run"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()

    assert devices.MISSING_RUNTIME in dialog.log.toPlainText()
    assert dialog.run_button.isEnabled()


def test_choosing_the_active_target_does_not_ask_until_the_run(own_window, monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: pytest.fail("must not ask"))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, active_has_notes=True)

    parameter_writer(dialog, "target")("active")

    assert dialog.target() == "active"  # picked quietly; the question belongs to the run
    dialog.close()


def test_the_active_channel_is_only_overwritten_on_an_explicit_yes(own_window, monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, active_has_notes=True)
    parameter_writer(dialog, "target")("active")

    dialog._start()
    dialog._poll()

    assert dialog.target() == "new"  # a declined overwrite falls back to a channel of its own


def test_an_empty_active_channel_takes_the_notes_without_asking(own_window, monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: pytest.fail("must not ask"))
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    parameter_writer(dialog, "target")("active")

    dialog._start()
    dialog._poll()

    assert dialog.notes() == [(0.0, 0.5, 60.0)]


def test_a_remembered_active_target_is_checked_before_it_overwrites(own_window, monkeypatch) -> None:
    values = dict(transcription.default_parameters(), target="active")
    monkeypatch.setattr(transcription, "load_parameters", lambda: dict(values))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("done", [(0.0, 0.5, 60.0)])]))

    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window, active_has_notes=True)
    assert dialog.target() == "active"  # remembered, and only questioned when the run is asked for

    dialog._start()
    dialog._poll()

    assert dialog.target() == "new"


def test_the_button_closes_the_window_when_idle(own_window) -> None:
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    assert dialog.close_button.text() == "Close"

    dialog._close_or_cancel()

    assert dialog.result() == QDialog.DialogCode.Rejected


def test_cancelling_a_run_keeps_the_window_and_offers_another(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("log", "GAME model small")]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)

    dialog._start()
    assert dialog.close_button.text() == "Cancel"

    dialog._close_or_cancel()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.run_button.isEnabled()
    assert dialog.close_button.text() == "Close"
    assert "Cancelled" in dialog.log.toPlainText()


def test_the_log_stays_hidden_until_there_is_something_to_say(own_window, monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.transcription_dialog.start_job", fake_job([("log", "GAME model small")]))
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    assert dialog.log.isHidden()

    dialog._start()

    assert not dialog.log.isHidden()


def test_the_window_is_pinned_to_its_content(own_window) -> None:
    """Hyprland resizes a native toplevel only while min == max, so the dialog follows its content
    by staying rigid."""
    dialog = TranscriptionDialog("/tmp/song.wav", 120.0, parent=own_window)
    collapsed = dialog.size().height()
    assert dialog.minimumSize() == dialog.maximumSize()

    dialog.findChild(QToolButton).click()
    QApplication.processEvents()
    QApplication.processEvents()

    assert dialog.size().height() > collapsed
    assert dialog.minimumSize() == dialog.maximumSize() == dialog.size()


def test_a_transcription_lands_on_a_channel_of_its_own(own_window) -> None:
    before = len(own_window.view.channels)

    own_window._adopt_transcription([(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)], "new")

    channels = own_window.view.channels
    assert len(channels) == before + 1
    assert channels[-1].name == "GAME"
    assert [note.pitch for note in own_window.view.notes()] == [60, 62]
    assert {note.channel for note in own_window.view.notes()} == {channels[-1].channel}

    own_window.view.undo()
    assert own_window.view.notes() == []
    assert len(own_window.view.channels) == before


def test_a_transcription_over_the_active_channel_replaces_only_that_channel(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes([(60, 0.0, 1.0, 0), (62, 0.0, 1.0, 1)])
    own_window.view.set_active_channel(0)

    own_window._adopt_transcription([(1.0, 1.5, 64.0)], "active")

    assert len(own_window.view.channels) == 2
    assert sorted((note.pitch, note.channel) for note in own_window.view.notes()) == [(62, 1), (64, 0)]


@pytest.fixture
def lyrics_window(own_window, tmp_path):
    """A window with a project open, so the lyrics have a file to sit beside."""
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    return own_window


def lyrics_dialog(lyrics_window, **settings):
    config = store.clone(lyrics_window.settings).lyrics
    for name, value in settings.items():
        setattr(config, name, value)
    return LyricsDialog(lyrics_window.lyrics_path(), config, parent=lyrics_window)


def test_the_lyrics_button_waits_for_a_project(own_window) -> None:
    assert own_window.lyrics_path() is None
    assert own_window.edit.lyrics.isEnabled() is False


def test_the_lyrics_button_opens_the_window(lyrics_window, monkeypatch) -> None:
    opened: list[LyricsDialog] = []
    monkeypatch.setattr(LyricsDialog, "exec", lambda self: opened.append(self) or QDialog.DialogCode.Rejected)

    lyrics_window.edit.lyrics.click()

    assert len(opened) == 1
    assert opened[0].path == lyrics_window.lyrics_path()
    assert opened[0].parent() is lyrics_window


def test_the_window_opens_with_the_lyrics_already_there(lyrics_window) -> None:
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.result.toPlainText() == "歌[うた]"
    dialog.close()


def test_the_plain_text_panel_opens_when_there_are_no_lyrics(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.plain_open is True
    assert dialog.plain_body.isVisibleTo(dialog) is True
    dialog.close()


def test_the_plain_text_panel_stays_folded_over_existing_lyrics(lyrics_window) -> None:
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.plain_open is False
    assert dialog.plain_body.isVisibleTo(dialog) is False
    dialog.close()


def test_an_unset_api_points_at_the_prompt(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.convert_button.isEnabled() is False
    assert dialog.copy_button.isEnabled() is True
    assert dialog.hint.isVisibleTo(dialog) is True
    dialog.close()


def test_a_set_api_hides_the_prompt_hint(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    assert dialog.convert_button.isEnabled() is True
    assert dialog.hint.isVisibleTo(dialog) is False
    dialog.close()


def test_closing_with_unsaved_lyrics_asks(lyrics_window, monkeypatch) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("歌[うた]")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)

    dialog.reject()

    assert lyrics.load(lyrics_window.lyrics_path()) == ""  # Discard wrote nothing


def test_closing_saves_the_lyrics_when_asked(lyrics_window, monkeypatch) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("歌[うた]")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)

    dialog.reject()

    assert lyrics.load(lyrics_window.lyrics_path()) == "歌[うた]"


def test_cancelling_the_close_keeps_the_unsaved_lyrics(lyrics_window, monkeypatch) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("歌[うた]")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    finished: list[int] = []
    dialog.finished.connect(finished.append)

    dialog.reject()

    assert finished == []
    assert lyrics.load(lyrics_window.lyrics_path()) == ""


def test_importing_a_krc_fills_the_lyrics_box(lyrics_window, tmp_path, monkeypatch) -> None:
    other = tmp_path / "other.krc"
    other.write_text("季節[き,せつ]", encoding="utf-8")
    dialog = lyrics_dialog(lyrics_window)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), "Lyrics file (*.krc)"))

    dialog.load_krc_button.click()

    assert dialog.result.toPlainText() == "季節[き,せつ]"
    assert "Imported" in dialog.status_label.text()
    dialog.close()


def test_loading_plain_text_fills_the_source_box(lyrics_window, tmp_path, monkeypatch) -> None:
    other = tmp_path / "words.txt"
    other.write_text("季節は移ろい", encoding="utf-8")
    dialog = lyrics_dialog(lyrics_window)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), "Text file (*.txt)"))

    dialog.load_text_button.click()

    assert dialog.source.toPlainText() == "季節は移ろい"
    assert dialog.result.toPlainText() == ""  # plain text is the source, not the lyrics
    dialog.close()


def test_copying_the_prompt_puts_it_on_the_clipboard(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.source.setPlainText("君の名は")
    dialog.copy_button.click()
    assert QGuiApplication.clipboard().text() == lyrics.build_prompt("君の名は")
    assert "web model" in dialog.status_label.text()
    dialog.close()


def test_saving_writes_the_lyrics_beside_the_project(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("季節[き,せつ]は移[うつ]ろい")
    dialog.save_button.click()
    assert lyrics.load(lyrics_window.lyrics_path()) == "季節[き,せつ]は移[うつ]ろい"
    assert "Saved to song.krc" in dialog.status_label.text()
    dialog.close()


def test_the_api_button_waits_for_the_settings(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    assert dialog.convert_button.isEnabled() is False
    dialog.close()

    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    assert dialog.convert_button.isEnabled() is True
    dialog.close()


def test_a_missing_source_is_not_sent(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(lyrics, "convert", lambda *args, **kwargs: pytest.fail("must not call"))
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.convert_button.click()
    assert "Paste the lyrics" in dialog.status_label.text()
    dialog.close()


def test_the_converter_asks_the_model_and_reports_it_back(lyrics_window, monkeypatch) -> None:
    asked: dict = {}

    def fake(source, **fields):
        asked.update(fields, source=source)
        return "歌[うた]"

    monkeypatch.setattr(lyrics, "convert", fake)
    dialog = lyrics_dialog(
        lyrics_window,
        api_base="https://api.example.com/v1",
        api_key="k",
        model="m",
        temperature=0.5,
        timeout=30.0,
    )
    converter = LyricsConverter("歌", dialog.config, dialog.path)
    dialog.converter = converter
    converter.converted.connect(dialog._converted)
    converter.load()

    assert asked["source"] == "歌"
    assert asked["base_url"] == "https://api.example.com/v1"
    assert asked["api_key"] == "k"
    assert asked["model"] == "m"
    assert asked["temperature"] == 0.5
    assert asked["timeout"] == 30.0
    assert asked["stream"] is True
    assert callable(asked["on_delta"])
    assert dialog.result.toPlainText() == "歌[うた]"
    assert "Converted" in dialog.status_label.text()
    dialog.close()


def test_a_wrong_answer_is_sent_back_for_another_try(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(LyricsConverter, "start", lambda self: None)
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.source.setPlainText("歌")
    dialog.result.setPlainText("元の歌詞")
    converter = LyricsConverter("歌", dialog.config, dialog.path)
    dialog.converter = converter
    converter.converted.connect(dialog._converted)

    converter.converted.emit("別[べつ]の歌詞[か,し]")  # the model rewrote the lyrics

    assert dialog.result.toPlainText() == "元の歌詞"  # the box keeps the last good lyrics
    assert dialog.converter is not converter  # a correction request replaced the first one
    assert "別[べつ]" in dialog.converter.retry
    assert "changed the lyrics" in dialog.converter.retry
    assert "asking the model" in dialog.status_label.text()
    dialog.close()


def test_the_last_bad_answer_warns_and_keeps_the_correction(lyrics_window, monkeypatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warned.append(message))
    monkeypatch.setattr(LyricsConverter, "start", lambda self: None)
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.source.setPlainText("歌")
    converter = LyricsConverter("歌", dialog.config, dialog.path)
    dialog.converter = converter
    converter.converted.connect(dialog._converted)

    for _ in range(3):
        dialog.converter.converted.emit("別[べつ]の歌詞[か,し]")

    assert dialog.result.toPlainText() == ""
    assert warned
    assert "copy the prompt" in dialog.status_label.text()
    dialog.close()


def test_copying_after_a_refusal_carries_the_correction(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window)
    dialog.source.setPlainText("歌")
    dialog._rejected = ("別[べつ]の歌詞[か,し]", "the answer changed the lyrics instead of annotating them")

    dialog.copy_button.click()

    text = QGuiApplication.clipboard().text()
    assert "別[べつ]" in text
    assert "changed the lyrics" in text
    assert "web model" in dialog.status_label.text()
    dialog.close()


def test_a_manual_answer_without_readings_is_refused_at_save(lyrics_window, monkeypatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warned.append(message))
    dialog = lyrics_dialog(lyrics_window)
    dialog.source.setPlainText("歌")
    dialog.result.setPlainText("歌")  # every kanji still wants a reading

    dialog.save_button.click()

    assert lyrics.load(lyrics_window.lyrics_path()) == ""
    assert warned
    assert dialog._rejected is not None
    dialog.copy_button.click()
    assert "no reading" in QGuiApplication.clipboard().text()
    dialog.close()


def test_a_failed_conversion_shows_why(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog._failed("HTTP 401 Unauthorized")
    assert "HTTP 401" in dialog.status_label.text()
    assert dialog.convert_button.isEnabled() is True
    dialog.close()


def test_a_broken_krc_is_not_saved(lyrics_window, monkeypatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: warned.append(message))
    dialog = lyrics_dialog(lyrics_window)
    dialog.result.setPlainText("見[み")

    dialog.save_button.click()

    assert lyrics.load(lyrics_window.lyrics_path()) == ""
    assert "Not saved" in dialog.status_label.text()
    assert warned
    dialog.close()


def test_the_output_box_is_only_shown_once_the_api_is_asked(lyrics_window, monkeypatch) -> None:
    monkeypatch.setattr(LyricsConverter, "start", lambda self: None)
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    assert dialog.log.isHidden() is True

    dialog.source.setPlainText("歌")
    dialog.convert_button.click()

    assert dialog.log.isHidden() is False
    assert dialog.log.toPlainText() == ""  # it starts empty, waiting for the model
    dialog.close()


def test_the_streamed_output_is_logged_as_it_comes(lyrics_window) -> None:
    dialog = lyrics_dialog(lyrics_window, api_base="https://api.example.com/v1", api_key="k", model="m")
    dialog.log.show()

    dialog._on_delta("reasoning", "考え")
    dialog._on_delta("reasoning", "て")
    dialog._on_delta("content", "歌[うた]を")

    text = dialog.log.toPlainText()
    assert "考えて" in text
    assert text.index("考えて") < text.index("歌[うた]を")  # the reasoning runs before the answer
    assert text.count("歌[うた]を") == 1
    dialog.close()


def test_the_api_key_is_typed_back_hidden() -> None:
    spec = store.FIELD_SPECS[("lyrics", "api_key")]
    widget, read, write = field_editor("hunter2", spec)
    assert isinstance(widget, QLineEdit)
    assert widget.echoMode() == QLineEdit.EchoMode.Password
    write("secret")
    assert read() == "secret"
    widget.deleteLater()


def test_the_lyrics_open_in_an_external_editor(lyrics_window, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(QProcess, "startDetached", lambda program, args: calls.append((program, args)) or True)
    lyrics_window.settings.lyrics.editor = "code --wait"

    lyrics_window._open_lyrics_editor()

    assert calls == [("code", ["--wait", str(lyrics_window.lyrics_path())])]


def test_an_outside_change_is_reloaded(lyrics_window) -> None:
    lyrics_window.statusBar().clearMessage()
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")

    lyrics_window._on_lyrics_file_changed()

    assert lyrics_window.lyrics_text == "歌[うた]"
    assert "reloaded" in lyrics_window.statusBar().currentMessage().lower()


def test_a_save_of_our_own_is_not_read_back(lyrics_window) -> None:
    lyrics_window._on_lyrics_saved("歌[うた]")
    lyrics.save(lyrics_window.lyrics_path(), "歌[うた]")
    lyrics_window.statusBar().clearMessage()

    lyrics_window._on_lyrics_file_changed()

    assert lyrics_window.statusBar().currentMessage() == ""


def test_the_watcher_moves_with_the_project(lyrics_window, tmp_path) -> None:
    other = tmp_path / "other.nto"
    other.write_text("{}")

    lyrics_window.project_path = other
    lyrics_window._watch_lyrics()

    assert lyrics_window.lyrics_path() == tmp_path / "other.krc"
    assert lyrics_window.lyrics_text == ""


def test_saving_a_project_takes_the_notes_and_the_values_with_it(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(120.0)
    own_window.view.set_notes([(64, 2.0, 1.0), (67, 3.0, 0.5)])  # beats, as the roll holds them
    own_window.mix.gain.set_value(300.0)
    path = tmp_path / "song.nto"
    assert own_window.save_project(path) is True
    assert own_window.project_path == path
    assert own_window.project_dirty is False
    assert own_window._document_name() == "song"
    saved = project.load(path)  # beat 2 at 120 BPM is one second in, and it is seconds that are kept
    assert saved.notes == (project.Note(1.0, 0.5, 64, 0, 1), project.Note(1.5, 0.25, 67, 0, 2))
    assert saved.settings.tempo.bpm == 120.0
    assert saved.settings.spectrum.gain == 300.0
    assert saved.settings.editor.snap == own_window.view.snap
    assert "Saved song.nto" in own_window.statusBar().currentMessage()


def test_loading_a_project_brings_the_notes_and_the_values_back(own_window, tmp_path, monkeypatch) -> None:
    other = project.default_settings()
    other.tempo.bpm = 120.0  # not 60: at 120 BPM a beat and a second differ, so the conversion shows
    other.analysis.a4 = 432.0
    other.spectrum.gain = 300.0
    other.editor.snap = 0.25
    path = tmp_path / "song.nto"
    opened = project.Project(
        settings=other,
        audio="vocal.wav",
        notes=(project.Note(2.0, 0.5, 64), project.Note(3.5, 0.5, 67)),
    )
    (tmp_path / "vocal.wav").write_bytes(b"")
    project.save(opened, path)
    played: list[str] = []  # the analysis and the audio device are not what this checks
    monkeypatch.setattr(type(own_window), "load_audio", lambda _self, path: played.append(path))

    assert own_window.load_project(path) is True
    assert sorted(note.pitch for note in own_window.view.notes()) == [64, 67]
    beats = sorted((note.start, note.duration) for note in own_window.view.notes())
    assert beats[0] == pytest.approx((4.0, 1.0))  # 2 s at 120 BPM is beat 4, and half a second is a beat
    assert beats[1] == pytest.approx((7.0, 1.0))
    assert own_window.view.bpm == 120.0
    assert own_window.project_settings.analysis.a4 == 432.0
    assert own_window.view.gain == 300.0
    assert own_window.view.snap == 0.25
    assert played == [str(tmp_path / "vocal.wav")]
    assert own_window.project_dirty is False
    assert own_window.project_path == path
    assert own_window._document_name() == "song"
    assert "Opened song.nto — 2 notes" in own_window.statusBar().currentMessage()

    own_window.view.set_notes([(60, 0.0, 1.0)])
    assert own_window._document_name() == "song*"


def test_a_project_without_its_audio_still_opens(own_window, tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(
        project.Project(
            settings=project.default_settings(),
            audio="gone.wav",
            notes=(project.Note(1.0, 0.5, 60),),
        ),
        path,
    )
    assert own_window.load_project(path) is True
    assert [note.pitch for note in own_window.view.notes()] == [60]
    assert own_window.audio_path is None
    assert own_window.song.is_loaded is False
    assert "audio not found" in own_window.statusBar().currentMessage()


def test_a_file_that_is_not_a_project_says_so(own_window, tmp_path) -> None:
    path = tmp_path / "not.nto"
    path.write_text("{}")
    assert own_window.load_project(path) is False
    assert "could not be opened" in own_window.statusBar().currentMessage()
    assert own_window.project_path is None


def test_opening_a_project_keeps_its_values_out_of_the_preferences(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.spectrum.gain = 300.0
    other.analysis.a4 = 432.0
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)

    assert own_window.load_project(path) is True
    assert own_window.project_settings.spectrum.gain == 300.0
    assert own_window.mix.gain.value() == 300.0
    assert not (tmp_path / "settings.json").exists()  # opening is not saving

    own_window.settings_store.flush()
    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert "spectrum" not in written  # a document's values never become the program's
    assert "analysis" not in written
    assert "playback" not in written


def test_a_change_made_with_a_project_open_stays_in_the_document(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.spectrum.gain = 300.0
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)
    assert own_window.load_project(path) is True

    own_window.mix.gain.set_value(340.0)  # the document's gain, not the program's
    own_window.settings_store.flush()
    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert own_window.project_settings.spectrum.gain == 340.0
    assert "spectrum" not in written


def test_drawing_marks_the_document_that_has_a_name(own_window) -> None:
    assert own_window.project_dirty is False
    own_window.transport.bpm.setValue(90.0)
    assert own_window.project_dirty is True
    assert own_window._document_name() == "Untitled*"  # nothing to ask about until it has a file


def test_auto_save_is_off_until_it_is_turned_on(own_window, tmp_path) -> None:
    path = tmp_path / "song.nto"
    own_window.project_path = path
    own_window.project_dirty = False

    own_window.view.set_notes([(64, 0.0, 1.0)])

    assert own_window.settings.general.auto_save is False
    assert own_window.project_dirty is True
    assert own_window.autosave_timer.isActive() is False
    assert path.exists() is False


def test_auto_save_writes_the_project_once_editing_stops(own_window, tmp_path) -> None:
    own_window.settings.general.auto_save = True
    path = tmp_path / "song.nto"
    assert own_window.save_project(path) is True

    own_window.view.set_notes([(64, 0.0, 1.0)])
    assert own_window.project_dirty is True
    assert own_window.autosave_timer.interval() == 3000
    assert own_window.autosave_timer.isActive() is True

    own_window.autosave_timer.stop()
    own_window._autosave()

    assert own_window.project_dirty is False
    assert project.load(path).notes == (project.Note(0.0, 0.5, 64, 0, 1),)


def test_losing_focus_writes_the_project_too(own_window, tmp_path, monkeypatch) -> None:
    own_window.settings.general.auto_save = True
    path = tmp_path / "song.nto"
    own_window.save_project(path)
    own_window.view.set_notes([(60, 0.0, 1.0)])
    own_window.autosave_timer.stop()

    monkeypatch.setattr(own_window, "isActiveWindow", lambda: False)
    own_window.changeEvent(QEvent(QEvent.Type.ActivationChange))

    assert own_window.project_dirty is False
    assert project.load(path).notes == (project.Note(0.0, 0.5, 60, 0, 1),)


def test_auto_save_leaves_a_document_without_a_file_alone(own_window) -> None:
    own_window.settings.general.auto_save = True
    own_window.view.set_notes([(60, 0.0, 1.0)])
    assert own_window.project_path is None

    own_window.autosave_timer.stop()
    own_window._autosave()

    assert own_window.project_dirty is True  # a sketch waits for Save As


def test_a_modal_dialog_holds_auto_save_back(own_window, tmp_path, monkeypatch) -> None:
    own_window.settings.general.auto_save = True
    path = tmp_path / "song.nto"
    own_window.save_project(path)
    own_window.view.set_notes([(60, 0.0, 1.0)])
    own_window.autosave_timer.stop()

    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: object()))
    own_window._autosave()

    assert own_window.project_dirty is True
    assert project.load(path).notes == ()  # the prompt, not the timer, is what decides


def test_moving_a_note_counts_as_a_change(own_window) -> None:
    note = own_window.view.add_note(69, 2.0, 2.0)
    own_window.project_dirty = False

    own_window.view.begin_gesture("Move notes")
    note.set_range(3.0, 69)
    own_window.view.commit_gesture()

    assert own_window.project_dirty is True


def test_unsaved_notes_are_asked_about_once(own_window, monkeypatch) -> None:
    asked: list[tuple] = []

    def warning(*args, **_kwargs):
        asked.append(args)
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "warning", warning)
    assert own_window._confirm_discard() is True  # a sketch is not worth interrupting anyone over
    assert asked == []

    own_window.view.set_notes([(64, 0.0, 1.0)])
    own_window.project_path = Path("song.nto")
    assert own_window._confirm_discard() is True
    assert len(asked) == 1
    assert "song.nto" in asked[0][2]

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    assert own_window._confirm_discard() is False
    own_window.project_dirty = False


def test_the_chosen_file_is_left_alone_when_the_notes_are_kept(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_notes([(64, 0.0, 1.0)])
    own_window.project_path = Path("song.nto")
    other = tmp_path / "other.nto"
    project.save(project.Project(settings=project.default_settings()), other)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), ""))

    own_window._on_open()

    assert own_window.project_path == Path("song.nto")  # the prompt was answered no, so nothing opened
    assert [note.pitch for note in own_window.view.notes()] == [64]
    own_window.project_dirty = False


def test_a_chosen_audio_file_is_left_alone_when_the_notes_are_kept(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_notes([(64, 0.0, 1.0)])
    own_window.project_path = Path("song.nto")
    audio = tmp_path / "other.wav"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(audio), ""))

    own_window._on_open()

    assert own_window.project_path == Path("song.nto")  # the prompt was answered no, so nothing opened
    assert own_window.audio_path is None  # and the sound was not analysed behind its back
    own_window.project_dirty = False


def test_saving_as_adds_the_suffix_when_it_is_missing(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_notes([(64, 0.0, 1.0)])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "mysong"), ""))
    assert own_window._on_save() is True
    assert own_window.project_path == tmp_path / "mysong.nto"
    assert (tmp_path / "mysong.nto").exists()


def test_saving_as_suggests_the_name_of_the_audio_file(own_window, monkeypatch, tmp_path) -> None:
    own_window.audio_path = str(tmp_path / "vocal.wav")
    own_window.state.last_audio_dir = str(tmp_path)
    asked: list[str] = []

    def choose(_parent, _caption, suggested, *_filters):
        asked.append(suggested)
        return (str(tmp_path / "vocal.nto"), "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose)
    assert own_window._on_save_as() is True
    assert Path(asked[0]) == tmp_path / "vocal.nto"


def test_a_project_beside_the_audio_is_opened_instead(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    project.save(
        project.Project(
            settings=project.default_settings(),
            audio="song.wav",
            notes=(project.Note(1.0, 0.5, 62),),
        ),
        tmp_path / "song.nto",
    )
    fake_loaders(monkeypatch)  # the project's own audio is analysed, not the file handed in

    assert own_window.open_audio(str(audio)) is True

    assert own_window.project_path == tmp_path / "song.nto"
    assert [note.pitch for note in own_window.view.notes()] == [62]
    assert own_window.audio_path == str(audio)


def test_a_broken_project_beside_the_audio_stops_the_open(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"")
    (tmp_path / "song.nto").write_text("{}")
    captured = fake_loaders(monkeypatch)

    assert own_window.open_audio(str(audio)) is False

    assert own_window.project_path is None
    assert own_window.audio_path is None  # the project is the document, so a broken one opens nothing
    assert not captured  # and the sound is not analysed behind its back


def test_a_project_is_picked_up_from_the_command_line(qt_app, tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(
        project.Project(settings=project.default_settings(), notes=(project.Note(1.0, 0.5, 62),)),
        path,
    )
    window = MainWindow()
    window.open_file(str(path))
    try:
        assert window.project_path == path
        assert [note.pitch for note in window.view.notes()] == [62]
    finally:
        window.project_dirty = False
        window.close()


def reset_channels(window) -> None:
    window.view.clear_notes()
    window.view.set_channels((Channel(channel=0),))


def test_notes_land_on_the_active_channel_and_wear_its_colour(window) -> None:
    window.edit.pen.click()
    window.view.clear_notes()
    window.view.add_channel(program=4)
    window.view.set_active_channel(1)
    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    drawn = window.view.notes()[0]
    assert drawn.channel == 1
    on_first = window.view.add_note(60, 0.0, 1.0, 0)
    assert on_first.fill != drawn.fill  # each channel paints its own colour
    reset_channels(window)


def test_a_locked_channel_cannot_be_edited(window) -> None:
    window.edit.pen.click()
    window.view.clear_notes()
    note = window.view.add_note(60, 2.0, 2.0)
    scene_pos = QPointF(3.0, PITCH_MAX - 60 + 0.5)
    window.view.set_channel_field(0, lock=True)

    assert window.view.channel_menu(scene_pos) is None  # the lock leaves nothing to move
    assert window.view.notes() == [note]

    draw_note(window, QPointF(6.0, 30.0))  # and the pen stays silent on it too
    assert window.view.notes() == [note]
    window.view.set_channel_field(0, lock=False)
    window.view.clear_notes()


def test_a_locked_channel_survives_select_all_and_delete(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0, lock=True), Channel(channel=1)))
    window.view.clear_notes()
    locked = window.view.add_note(60, 2.0, 1.0, 0)
    window.view.add_note(64, 4.0, 1.0, 1)
    window.view.undo_stack.clear()

    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier))
    assert not locked.isSelected()  # select-all skips a locked channel
    window.view.delete_selection()
    assert [note.pitch for note in window.view.notes()] == [60]
    window.view.undo()
    assert [note.pitch for note in window.view.notes()] == [60, 64]
    reset_channels(window)


def test_pasting_leaves_a_locked_channel_alone(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 1.0, 0).setSelected(True)
    window.view.set_playhead(8.0)
    assert window.view.copy_selection()
    window.view.set_channel_field(0, lock=True)

    assert window.view.paste_notes() is False  # a locked channel takes no new notes
    assert len(window.view.notes()) == 1
    window.view.set_channel_field(0, lock=False)
    window.view.clear_notes()


def test_an_invisible_channel_hides_its_notes(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(60, 2.0, 2.0)
    assert note.isVisible()
    window.view.set_channel_field(0, visible=False, lock=True)  # hiding also locks, noteDigger style
    assert not note.isVisible()
    window.view.set_channel_field(0, visible=True, lock=False)
    assert note.isVisible()
    window.view.clear_notes()


def test_removing_a_channel_takes_its_notes_and_leaves_the_numbers_alone(window) -> None:
    window.view.clear_notes()
    window.view.add_channel()
    window.view.add_note(60, 0.0, 1.0, 0)
    survivor = window.view.add_note(64, 1.0, 1.0, 1)
    assert window.view.remove_channel(0)
    assert [note.pitch for note in window.view.notes()] == [64]
    assert survivor.channel == 1
    reset_channels(window)


def test_a_muted_channel_is_left_out_of_the_program(window) -> None:
    window.view.clear_notes()
    window.view.add_channel(program=4)
    window.view.add_note(60, 0.0, 1.0, 0)
    window.view.add_note(64, 0.0, 1.0, 1)
    notes, channels = window._program()
    assert len(notes) == 2 and len(channels) == 2

    window.view.set_channel_field(1, mute=True)
    notes, channels = window._program()
    assert [note[0] for note in notes] == [60]
    assert [channel[0] for channel in channels] == [0]
    reset_channels(window)


def test_the_channels_button_toggles_the_sidebar(window) -> None:
    window.channel_panel.setVisible(False)
    window.transport.channels.click()
    assert window.channel_panel.isVisible()
    window.transport.channels.click()
    assert not window.channel_panel.isVisible()


def test_a_channel_switch_marks_the_exceptional_state(own_window) -> None:
    own_window.view.set_channels(
        (
            Channel(channel=0, visible=True),
            Channel(channel=1, lock=True, mute=True),
            Channel(channel=2, visible=False, lock=True),
        )
    )
    cards = own_window.channel_panel._cards
    assert cards[0].lock_button.isChecked() is False
    assert cards[0].eye_button.isChecked() is False  # visible is the normal state
    assert cards[1].lock_button.isChecked() is True
    assert cards[1].mute_button.isChecked() is True
    assert cards[1].eye_button.isChecked() is False
    assert cards[2].lock_button.isChecked() is True
    assert cards[2].eye_button.isChecked() is True  # hidden


def test_a_channel_change_refreshes_its_card_in_place(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    panel = own_window.channel_panel
    card = panel._cards[0]

    view.set_channel_field(0, lock=True)
    assert panel._cards[0] is card  # a field edit refreshes the card instead of rebuilding it
    assert card.lock_button.isChecked() is True

    view.set_active_channel(1)
    assert panel._cards[0] is card  # and so does making another channel active
    assert card.property("active") is False
    assert panel._cards[1].property("active") is True

    view.add_channel()
    assert panel._cards[0] is not card  # the set of channels changed, so the cards are rebuilt


def test_a_channel_id_shows_before_its_name(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0, name="Lead"), Channel(channel=2)))
    cards = own_window.channel_panel._cards
    assert cards[0].id.text() == "(1)"
    assert cards[0].name.text() == "Lead"
    assert cards[2].id.text() == "(3)"
    assert cards[2].name.text() != ""  # an unnamed channel keeps a word to read


def test_the_channel_id_wears_the_note_colour(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0, color="#ff8800"),))
    card = own_window.channel_panel._cards[0]
    assert "#ff8800" in card.id.styleSheet()


def test_a_channel_moves_to_another_id_with_its_notes(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=3)))
    view.set_notes([(60, 0.0, 1.0, 3)])
    assert view.set_channel_number(3, 5) is True
    assert [channel.channel for channel in view.channels] == [0, 5]
    assert [note.channel for note in view.notes()] == [5]
    assert view.set_channel_number(5, 0) is False  # 1 is already taken
    assert view.set_channel_number(5, 5) is True  # the same number is a quiet no-op


def test_moving_a_channel_moves_its_colour_too(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    assert view.channel_color(1).name() == QColor(theme.NOTE_PALETTE[1]).name()
    assert view.set_channel_number(1, 4) is True
    assert view.channel_color(4).name() == QColor(theme.NOTE_PALETTE[4]).name()
    assert own_window.channel_panel._cards[4].id.styleSheet() == f"color: {QColor(theme.NOTE_PALETTE[4]).name()};"


def test_the_card_menu_can_change_the_channel_id(own_window, monkeypatch) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    card = own_window.channel_panel._cards[0]
    monkeypatch.setattr(
        QMenu, "exec", lambda self, *args, **kwargs: next(a for a in self.actions() if a.text() == "Channel ID…")
    )
    monkeypatch.setattr(QInputDialog, "getInt", lambda *args, **kwargs: (5, True))
    card.contextMenuEvent(
        QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(0, 0), card.mapToGlobal(QPoint(0, 0)))
    )
    assert [channel.channel for channel in view.channels] == [1, 4]


def test_the_active_channel_wears_the_selection_colour(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    panel = own_window.channel_panel
    card = panel._cards[0]
    card.resize(288, 62)  # an unshown window never lays the cards out on its own
    highlight = panel.palette().color(QPalette.ColorRole.Highlight)

    def edge_distance() -> int:
        """How far the card's top edge is from the palette's selection colour, at its middle."""
        colour = card.grab().toImage().pixelColor(card.width() // 2, 1)
        return sum((getattr(colour, part)() - getattr(highlight, part)()) ** 2 for part in ("red", "green", "blue"))

    view.set_active_channel(1)
    plain = edge_distance()
    view.set_active_channel(0)
    assert edge_distance() < plain  # the outline is drawn only while the channel is active


def test_the_auto_page_and_overtone_toggles_start_off_and_reach_the_settings(own_window) -> None:
    assert own_window.transport.auto_page.isChecked() is False
    assert own_window.transport.overtone.isChecked() is False
    assert own_window.view.overtone_highlight is False

    own_window.transport.overtone.click()
    assert own_window.view.overtone_highlight is True
    own_window.transport.auto_page.click()
    assert store.get_value(own_window.settings, "editor", "overtone_highlight") is True
    assert store.get_value(own_window.settings, "editor", "auto_page") is True


def test_the_auto_page_turn_follows_the_playhead(own_window, monkeypatch) -> None:
    own_window.show()
    QApplication.processEvents()
    view = own_window.view
    view.set_zoom(48.0, 16.0)
    view.clear_notes()
    view.add_note(60, 0.0, 900.0)  # the page can only turn as far as the sound goes
    monkeypatch.setattr(own_window, "_is_playing", lambda: True)
    own_window.transport.auto_page.setChecked(True)

    def page() -> tuple[float, float]:
        rect = view.mapToScene(view.viewport().rect()).boundingRect()
        return rect.left(), rect.right()

    monkeypatch.setattr(own_window, "_position", lambda: 1.0)
    own_window._show_position()
    left, right = page()
    assert left <= 1.0 * view.bpm / 60.0 <= right

    monkeypatch.setattr(own_window, "_position", lambda: 300.0)
    own_window._show_position()
    left, right = page()
    playhead = 300.0 * view.bpm / 60.0
    assert left <= playhead <= right
    assert playhead - left < (right - left) * 0.5  # the playhead lands near the left of the fresh page

    own_window.transport.auto_page.setChecked(False)
    monkeypatch.setattr(own_window, "_position", lambda: 500.0)
    own_window._show_position()
    assert page() == pytest.approx((left, right))  # with the toggle off nobody turns the page


def accept_import(monkeypatch, mode: str, mapping=()) -> None:
    """Answer the import dialog without one: the flow under test is the import, not the widget."""
    monkeypatch.setattr(MidiImportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(MidiImportDialog, "mode", lambda self: mode)
    monkeypatch.setattr(MidiImportDialog, "mapping", lambda self: list(mapping))


@pytest.fixture
def midi_window(own_window, tmp_path):
    """A window with a project open: a MIDI is imported into one, never on its own."""
    own_window.project_path = tmp_path / "song.nto"
    return own_window


def test_importing_a_midi_brings_in_its_notes_and_channels(midi_window, tmp_path) -> None:
    path = tmp_path / "song.mid"
    midi.write(
        path,
        (Channel(channel=0, program=52), Channel(channel=1)),
        (project.Note(1.0, 0.5, 60, 0), project.Note(1.0, 0.5, 48, 1)),
        120.0,
    )
    assert midi_window.import_midi(path) is True

    assert [channel.channel for channel in midi_window.view.channels] == [0, 1]
    assert [channel.program for channel in midi_window.view.channels] == [52, 0]
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(48, 1), (60, 0)]
    assert midi_window.transport.bpm.value() == 120.0
    assert midi_window.project_dirty is True
    assert "Imported 2 notes" in midi_window.statusBar().currentMessage()


def test_a_midi_without_a_project_is_refused(own_window, tmp_path) -> None:
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 60, 0),), 120.0)

    assert own_window.import_midi(path) is False
    assert own_window.view.notes() == []
    assert "Open a song first" in own_window.statusBar().currentMessage()


def test_importing_over_notes_asks_before_replacing(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_notes([(60, 0.0, 1.0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 62, 0),), 120.0)
    monkeypatch.setattr(MidiImportDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

    assert midi_window.import_midi(path) is False
    assert [note.pitch for note in midi_window.view.notes()] == [60]  # nothing was touched


def test_replacing_is_what_the_dialog_can_choose(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Old"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 62, 0),), 120.0)
    accept_import(monkeypatch, "replace")

    assert midi_window.import_midi(path) is True
    # a replacement is the file's channels whole: a name the project gave the old one does not stay
    assert [channel.channel for channel in midi_window.view.channels] == [0]
    assert [channel.name for channel in midi_window.view.channels] == [""]
    assert [note.pitch for note in midi_window.view.notes()] == [62]


def test_merging_adds_the_file_channels_to_the_roll(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Voice"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(
        path,
        (Channel(channel=0), Channel(channel=2, program=33, volume=90)),
        (project.Note(1.0, 0.5, 64, 0), project.Note(1.0, 0.5, 40, 2)),
        140.0,
    )
    accept_import(monkeypatch, "merge", [0, -1])

    assert midi_window.import_midi(path) is True
    assert [channel.channel for channel in midi_window.view.channels] == [0, 2]
    assert (midi_window.view.channels[1].program, midi_window.view.channels[1].volume) == (33, 90)
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(40, 2), (60, 0), (64, 0)]
    # the file's 140 BPM does not touch the grid: a second stays a second on the audio (120 BPM here)
    assert sorted((note.pitch, round(note.start, 3)) for note in midi_window.view.notes()) == [
        (40, 2.0),
        (60, 0.0),
        (64, 2.0),
    ]
    assert midi_window.transport.bpm.value() == 120.0  # the grid stays on the audio, not on the file


def test_merging_onto_an_empty_channel_takes_the_file_channel_over(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=5, name="Placeholder"), Channel(channel=0, name="Used")))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])  # channel 5 carries nothing
    path = tmp_path / "song.mid"
    midi.write(
        path,
        (Channel(channel=2, program=81, volume=90),),
        (project.Note(1.0, 0.5, 62, 2),),
        120.0,
    )
    accept_import(monkeypatch, "merge", [5])

    assert midi_window.import_midi(path) is True
    # an empty channel is a free place: the file's channel takes it over, the way a brand new one would
    assert [channel.channel for channel in midi_window.view.channels] == [0, 5]
    landed = next(channel for channel in midi_window.view.channels if channel.channel == 5)
    assert (landed.program, landed.volume) == (81, 90)
    assert landed.name == "Placeholder"  # the name belongs to the project, and it stays
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(60, 0), (62, 5)]


def test_a_new_channel_keeps_the_number_the_file_played_on(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Voice"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=6),), (project.Note(1.0, 0.5, 62, 6),), 120.0)
    accept_import(monkeypatch, "merge", [-1])

    assert midi_window.import_midi(path) is True
    # nothing else plays on channel 6, so the file's new channel keeps that number
    assert [channel.channel for channel in midi_window.view.channels] == [0, 6]
    assert midi_window.view.channels[1].name == ""  # a MIDI channel carries no name
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(60, 0), (62, 6)]


def test_a_new_channel_takes_a_free_number_when_the_files_own_is_taken(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Voice"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 62, 0),), 120.0)
    accept_import(monkeypatch, "merge", [-1])

    assert midi_window.import_midi(path) is True
    # channel 0 already carries the roll's notes, so the file's new channel takes the lowest free one
    assert [channel.channel for channel in midi_window.view.channels] == [0, 1]
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(60, 0), (62, 1)]


def test_the_import_dialog_prefills_the_mapping_by_position() -> None:
    imported = midi.Imported(channels=(Channel(channel=2), Channel(channel=7)), notes=(), bpm=120.0)
    dialog = MidiImportDialog(imported, (Channel(channel=0), Channel(channel=2)), "song.mid")
    assert dialog.mapping() == [0, 2]  # both roll channels carry no notes, so the file lands on them in order
    assert dialog.mode() == "merge"  # the additive choice is what a bare accept takes
    dialog.close()


def test_a_file_channel_moves_past_a_roll_channel_that_already_has_notes() -> None:
    imported = midi.Imported(channels=(Channel(channel=0), Channel(channel=1), Channel(channel=2)), notes=(), bpm=120.0)
    channels = tuple(Channel(name=name, channel=index) for index, name in enumerate("ABCD"))
    # A and C carry notes, so the file's first channel takes the empty place after the second's own
    dialog = MidiImportDialog(imported, channels, "song.mid", occupied={0, 2})
    assert dialog.mapping() == [3, 1, -1]
    dialog.close()


def test_the_import_dialog_keeps_the_roll_within_sixteen_channels() -> None:
    imported = midi.Imported(channels=(Channel(channel=15),), notes=(), bpm=120.0)
    # every channel carries notes, so nothing is free to land on and the mapping reaches for a new one
    channels = tuple(Channel(name=f"T{index}", channel=channel) for index, channel in enumerate([0, *range(15)]))
    dialog = MidiImportDialog(imported, channels, "song.mid", occupied=range(16))
    assert dialog.mapping() == [-1]
    assert dialog.merge_button.isEnabled() is False  # a new channel would be the seventeenth
    dialog._targets[0].setCurrentIndex(0)  # pointed at an existing channel instead
    assert dialog.merge_button.isEnabled() is True
    dialog.close()


def test_a_wavetone_file_loses_its_lead_in_only_when_the_setting_says_so(midi_window, monkeypatch, tmp_path) -> None:
    path = tmp_path / "wavetone.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 60, 0),), 120.0, wavetone=True)

    midi_window.import_midi(path)
    assert [round(note.start, 3) for note in midi_window.view.notes()] == [2.0]

    store.set_value(midi_window.settings, "midi", "wavetone", False)
    accept_import(monkeypatch, "replace")  # the roll holds a note now, so the dialog would stand in the way
    midi_window.import_midi(path)
    assert [round(note.start, 3) for note in midi_window.view.notes()] == [6.0]


def test_a_midi_that_cannot_be_read_says_so(midi_window, tmp_path) -> None:
    broken = tmp_path / "broken.mid"
    broken.write_bytes(b"not a MIDI file at all")
    assert midi_window.import_midi(broken) is False
    assert "could not be read" in midi_window.statusBar().currentMessage()


def test_exporting_writes_the_roll_out_as_midi(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=2, program=81),))
    own_window.view.set_notes(((60, 0.0, 1.0, 2), (64, 1.0, 0.5, 2)))
    path = tmp_path / "out.mid"
    assert own_window.export_midi(path) is True

    imported = midi.read(path, wavetone=True)  # the WaveTone compatibility the settings start with
    assert [(note.pitch, round(note.start, 3)) for note in imported.notes] == [(60, 0.0), (64, 0.5)]
    assert [(channel.channel, channel.program) for channel in imported.channels] == [(2, 81)]
    assert [round(note.start, 3) for note in midi.read(path).notes] == [2.0, 2.5]  # the same, a bar late
    assert own_window.project_path is None  # exporting is not saving
    assert "Exported out.mid" in own_window.statusBar().currentMessage()


def test_a_hidden_channel_is_exported_like_any_other(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1, visible=False)))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 0.0, 1.0, 1)))
    path = tmp_path / "hidden.mid"
    own_window.export_midi(path)

    assert [note.pitch for note in midi.read(path).notes] == [60, 62]


def test_opening_a_midi_file_imports_it(midi_window, monkeypatch, tmp_path) -> None:
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(0.5, 0.5, 60, 0),), 120.0)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))

    midi_window._on_open()
    assert [note.pitch for note in midi_window.view.notes()] == [60]


def test_the_open_dialog_shows_every_file_type_by_default(own_window, monkeypatch) -> None:
    shown: list[str] = []

    def fake(_parent, _caption, _directory, filters, *_args, **_kwargs):
        shown.append(filters)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake)
    own_window._on_open()

    assert shown[0].split(";;")[0] == "All files (*)"
    # a MIDI is a project's, so without one the chooser does not offer it
    assert "MIDI" not in shown[0]


def test_the_open_dialog_offers_a_midi_once_a_project_is_open(midi_window, monkeypatch) -> None:
    shown: list[str] = []

    def fake(_parent, _caption, _directory, filters, *_args, **_kwargs):
        shown.append(filters)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake)

    midi_window._on_open()

    assert "MIDI" in shown[0]


def test_opening_an_audio_file_analyses_it(own_window, monkeypatch, tmp_path) -> None:
    path = tmp_path / "song.flac"
    path.write_bytes(b"")
    fake_loaders(monkeypatch)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(OpenAudioDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    own_window._on_open()

    assert own_window.audio_path == str(path)
    assert own_window.project_path == tmp_path / "song.nto"  # a sound is a project, and it has a file


def open_editor(dialog, name: str):
    """The widget one row of the open dialog edits, to change it the way a user would."""
    return next(widget for field, widget, _read in dialog._rows if field == name)


def test_the_open_dialog_starts_from_the_document_and_the_audio(own_window, tmp_path) -> None:
    document = project.default_settings()
    document.analysis.channels = "left"
    document.analysis.fft_points = 4096
    dialog = OpenAudioDialog(str(tmp_path / "song.flac"), document, parent=own_window)
    try:
        assert dialog.path.text() == str(tmp_path / "song.nto")
        assert dialog.document().analysis.channels == "left"
        assert dialog.document().analysis.fft_points == 4096
    finally:
        dialog.close()


def test_the_open_dialog_reads_an_edited_widget_back(own_window, tmp_path) -> None:
    dialog = OpenAudioDialog(str(tmp_path / "song.flac"), project.default_settings(), parent=own_window)
    try:
        open_editor(dialog, "a4").setValue(442.0)
        assert dialog.document().analysis.a4 == 442.0
    finally:
        dialog.close()


def test_the_open_dialog_adds_the_project_suffix(own_window, tmp_path) -> None:
    dialog = OpenAudioDialog(str(tmp_path / "song.flac"), project.default_settings(), parent=own_window)
    try:
        dialog.path.setText(str(tmp_path / "mysong"))
        assert dialog.target() == tmp_path / "mysong.nto"
    finally:
        dialog.close()


def test_opening_audio_keeps_the_analysis_habits_the_user_changed(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)

    def accept(self):
        open_editor(self, "fft_points").setValue(4096)
        open_editor(self, "a4").setValue(442.0)  # the song's tuning is not a habit
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(OpenAudioDialog, "exec", accept)

    assert own_window.open_audio(str(audio)) is True

    assert own_window.state.project["analysis"] == {"fft_points": 4096}
    assert own_window.project_settings.analysis.fft_points == 4096
    assert own_window.project_settings.analysis.a4 == 442.0


def test_the_command_line_analysis_values_start_the_dialog_and_stop_there(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)
    own_window.overrides["channels"] = "left"
    seen: list[str] = []

    def accept(self):
        seen.append(self.document().analysis.channels)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(OpenAudioDialog, "exec", accept)

    assert own_window.open_audio(str(audio)) is True

    assert seen == ["left"]  # the window opened on the value the command line asked for
    assert own_window.overrides == {}  # and the document owns it from there
    assert own_window.project_settings.analysis.channels == "left"


def test_opening_new_audio_does_not_save_the_previous_document(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)
    own_window.view.set_channels((Channel(channel=5),))
    own_window.view.add_note(60, 0.0, 1.0, 5).setSelected(True)
    own_window.view.delete_selection()
    own_window.view.undo()
    assert own_window.view.undo_stack.count() > 0
    own_window.lyrics_text = "あい"
    own_window._stored_lyrics = project.Lyrics(text="あい", key=text_key("あい"))
    assert own_window.save_project(tmp_path / "previous.nto") is True
    monkeypatch.setattr(OpenAudioDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    assert own_window.open_audio(str(audio)) is True

    saved = project.load(tmp_path / "song.nto")
    assert saved.notes == ()
    assert [channel.channel for channel in saved.channels] == [0]
    assert saved.lyrics is None
    assert own_window.view.notes() == []
    assert own_window.lyrics_text == ""
    assert own_window._stored_lyrics is None
    assert own_window.view.undo_stack.count() == 0


def test_cancelling_the_open_dialog_leaves_the_audio_alone(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    captured = fake_loaders(monkeypatch)
    own_window.view.set_channels((Channel(channel=5),))
    own_window.view.add_note(60, 0.0, 1.0, 5).setSelected(True)
    own_window.view.delete_selection()
    own_window.view.undo()
    own_window.lyrics_text = "あい"
    stored = project.Lyrics(text="あい", key=text_key("あい"))
    own_window._stored_lyrics = stored
    own_window.audio_path = str(tmp_path / "previous.flac")
    assert own_window.save_project(tmp_path / "previous.nto") is True
    stored = own_window._stored_lyrics
    settings = own_window.project_settings
    notes = own_window.view.notes()
    undo_count = own_window.view.undo_stack.count()
    monkeypatch.setattr(OpenAudioDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

    assert own_window.open_audio(str(audio)) is False

    assert own_window.audio_path == str(tmp_path / "previous.flac")
    assert own_window.project_path == tmp_path / "previous.nto"
    assert own_window.project_settings is settings
    assert own_window.view.notes() == notes
    assert [channel.channel for channel in own_window.view.channels] == [5]
    assert own_window.lyrics_text == "あい"
    assert own_window._stored_lyrics is stored
    assert own_window.view.undo_stack.count() == undo_count
    assert not audio.with_suffix(".nto").exists()
    assert not captured


def test_the_command_line_takes_audio_and_projects_but_not_a_midi_alone(qt_app, monkeypatch, tmp_path) -> None:
    fake_loaders(monkeypatch)
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"")
    midi_path = tmp_path / "tune.mid"
    midi.write(midi_path, (Channel(channel=0),), (project.Note(0.5, 0.5, 60, 0),), 120.0)
    project_path = tmp_path / "work.nto"
    project.save(
        project.Project(settings=project.default_settings(), notes=(project.Note(1.0, 0.5, 62),)),
        project_path,
    )
    monkeypatch.setattr(OpenAudioDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    opened = [MainWindow(), MainWindow(), MainWindow()]
    for window, path in zip(opened, (audio, midi_path, project_path), strict=True):
        window.open_file(str(path))
    try:
        assert opened[0].audio_path == str(audio)
        assert opened[0].project_path == tmp_path / "clip.nto"  # opening a sound names its project
        # a MIDI is part of a project, so it is not a document the command line can open on its own
        assert opened[1].project_path is None
        assert [note.pitch for note in opened[1].view.notes()] == []
        assert opened[2].project_path == project_path
        assert [note.pitch for note in opened[2].view.notes()] == [62]
    finally:
        for window in opened:
            window.project_dirty = False
            window.close()


def test_the_export_button_writes_a_midi_file(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "MIDI file (*.mid)")
    )

    assert own_window._on_export_midi() is True
    target = tmp_path / "exported.mid"  # the export adds its own suffix
    assert [note.pitch for note in midi.read(target, wavetone=True).notes] == [60]
    assert own_window.project_path is None  # an export leaves the document where it was


@pytest.fixture
def subtitle_window(own_window, tmp_path):
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい", key=text_key("あい"), raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    return own_window


def test_the_export_button_writes_an_ass_subtitle(subtitle_window, monkeypatch, tmp_path) -> None:
    own_window = subtitle_window

    def options(dialog):
        dialog.editors["font_size"].setValue(72)
        dialog.editors["offset_ms"].setValue(1000)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AssDialog, "exec", options)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "ASS subtitle (*.ass)")
    )
    before = own_window.view.notes()
    assert own_window._on_export_ass() is True
    target = tmp_path / "exported.ass"  # the export adds its own suffix
    text = target.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "Dialogue: 0," in text
    assert "Style: K1,sans-serif,72," in text
    assert "0:00:03.00,K1" in text
    assert own_window.view.notes() == before
    assert own_window.settings.ass.font_size == 72
    assert store.load().ass.font_size == 72
    assert store.load().ass.offset_ms == 1000
    assert own_window.project_path == tmp_path / "song.nto"  # an export leaves the document where it was


@pytest.mark.parametrize("stop", ("options", "path", "write", "build"))
def test_cancelled_or_failed_subtitle_export_does_not_remember_options(subtitle_window, monkeypatch, tmp_path, stop):
    original = store.to_dict(subtitle_window.settings)
    saved = store.default_path().read_bytes() if store.default_path().exists() else None
    seen = []

    def options(dialog):
        seen.append("options")
        dialog.editors["font_size"].setValue(24)
        return QDialog.DialogCode.Rejected if stop == "options" else QDialog.DialogCode.Accepted

    def path(*args, **kwargs):
        seen.append("path")
        return ("" if stop == "path" else str(tmp_path / "out.ass"), "")

    def fail(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(AssDialog, "exec", options)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", path)
    if stop == "write":
        monkeypatch.setattr("namioto.ui.app.write_text", fail)
    if stop == "build":

        def fail_build(*args, **kwargs):
            raise ValueError("invalid subtitle")

        monkeypatch.setattr("namioto.ui.app.generate_ass", fail_build)
    assert not subtitle_window._on_export_ass()
    assert seen == (["options"] if stop == "options" else ["options", "path"])
    assert store.to_dict(subtitle_window.settings) == original
    assert (store.default_path().read_bytes() if store.default_path().exists() else None) == saved
    assert not (tmp_path / "out.ass").exists()


def test_ass_options_cover_the_single_track_settings_without_mutating_them(own_window):
    original = store.to_dict(own_window.settings)
    dialog = AssDialog(own_window.settings, own_window)
    assert set(dialog.editors) == set(vars(own_window.settings.ass))
    assert set(dialog.editors) == {name for _, names in ASS_GROUPS for name in names}
    assert dialog.export_button.isEnabled()
    dialog.editors["font_size"].setValue(40)
    dialog.reject()
    assert store.to_dict(own_window.settings) == original


def test_ass_automatic_colours_follow_the_overlay_and_manual_blur(own_window):
    from namioto.karaoke import AssSettings

    dialog = AssDialog(own_window.settings, own_window)
    before = dialog.editors["overlay_blur_color"].text()
    dialog.editors["overlay_color"].setText("FF0000")
    assert dialog.editors["overlay_blur_color"].text() != before
    assert dialog.values().ass.overlay_blur_color == ""
    assert not dialog.editors["overlay_blur_color"].isEnabled()
    dialog.automatic["overlay_blur_color"].setChecked(False)
    assert dialog.editors["overlay_blur_color"].isEnabled()
    dialog.editors["overlay_blur_color"].setText("123456")
    expected = AssSettings(overlay_color="FF0000", overlay_blur_color="123456").blur_colours()[1]
    assert dialog.editors["base_blur_color"].text() == expected
    dialog.editors["overlay_color"].setText("00FF00")
    assert dialog.values().ass.overlay_blur_color == "123456"
    assert dialog.values().ass.base_blur_color == ""
    dialog.restore_defaults()
    assert vars(dialog.values().ass) == vars(store.Settings().ass)
    assert all(check.isChecked() for check in dialog.automatic.values())


def test_ass_numeric_options_enforce_their_lower_bounds(own_window):
    dialog = AssDialog(own_window.settings, own_window)
    for name in ("font_size", "guide_dot_duration_ms"):
        dialog.editors[name].setValue(0)
        assert dialog.editors[name].value() == 1
    for name in ("border", "border_furi", "margin_h", "margin_v", "blur", "blur_scale", "clip_size"):
        dialog.editors[name].setValue(-1)
        assert dialog.editors[name].value() == 0
    for name in ("ruby_offset", "offset_ms"):
        dialog.editors[name].setValue(-10)
        assert dialog.editors[name].value() == -10


def test_ass_options_validate_and_pick_rgb_colours(own_window, monkeypatch):
    from PyQt6.QtWidgets import QColorDialog

    dialog = AssDialog(own_window.settings, own_window)
    dialog.editors["lead_time_ms"].setValue(0)
    assert not dialog.export_button.isEnabled()
    assert "Fade in" in dialog.hint.text()
    dialog._accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.editors["fade_in_ms"].setValue(0)
    assert dialog.export_button.isEnabled()
    dialog.editors["overlay_color"].setText("invalid")
    assert not dialog.export_button.isEnabled()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#12ABEF"))
    dialog._choose_colour(dialog.editors["overlay_color"])
    assert dialog.editors["overlay_color"].text() == "12ABEF"
    assert dialog.export_button.isEnabled()
    dialog.editors["font"].setText("a,b")
    assert not dialog.export_button.isEnabled()
    dialog.restore_defaults()
    assert dialog.export_button.isEnabled()


def test_a_subtitle_needs_lyrics_and_notes(own_window) -> None:
    assert own_window._on_export_ass() is False
    assert "no lyrics" in own_window.statusBar().currentMessage()
    own_window.lyrics_text = "あ"
    own_window._load_sounds()  # the lyrics are readable; now the subtitle needs a note to sit on
    assert own_window._on_export_ass() is False
    assert "no notes" in own_window.statusBar().currentMessage()


def test_a_narrow_sound_still_names_itself(window) -> None:
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    window.sound_strip._update_hover((0, 1))
    assert window.sound_strip.toolTip() == "い"
    window.sound_strip._toggle_raw(True)
    window.sound_strip._update_hover((0, 1))
    assert window.sound_strip.toolTip() == "い\nReference duration: 1.000 s"
    window.sound_strip._toggle_raw(False)
    window.view.load_lyrics((), ())
    window.sound_strip.setVisible(False)


def test_the_sound_strip_shows_every_line(window) -> None:
    window.view.load_lyrics(natural_sounds("あい\nうえ"), [[(0.0, 1.0), (1.0, 2.0)], [(3.0, 4.0), (4.0, 5.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    # the second line's blocks show too, not just the first line's
    assert window.sound_strip._sound_at(window.sound_strip._x(3.5)) == (1, 0)
    assert not window.sound_strip.grab().isNull()
    window.view.load_lyrics((), ())
    window.sound_strip.setVisible(False)


def test_the_sound_strip_renders_the_current_line(window) -> None:
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    image = window.sound_strip.grab()
    assert not image.isNull() and image.width() > 0
    window.view.load_lyrics((), ())
    window.sound_strip.setVisible(False)


def test_dragging_a_sound_onset_preserves_all_reference_durations(window) -> None:
    window.sound_strip._toggle_raw(True)
    times = [[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]]
    window.view.load_lyrics(natural_sounds("あいう"), times)
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    before = window.view.undo_stack.count()

    sound_mouse(window, QEvent.Type.MouseButtonPress, window.sound_strip._x(1.0))
    sound_mouse(window, QEvent.Type.MouseMove, window.sound_strip._x(1.5))
    sound_mouse(window, QEvent.Type.MouseButtonRelease, window.sound_strip._x(1.5))

    assert window.view.lyric_raw[0] == ((0.0, 1.0), (1.5, 1.0), (2.0, 1.0))
    assert window.view.undo_stack.count() == before + 1
    wait_for_lyric_mapping(window)

    window.view.undo()
    wait_for_lyric_mapping(window)
    assert window.view.lyric_raw[0] == ((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    window.sound_strip._toggle_raw(False)


def test_the_last_sound_onset_has_no_reference_end_limit(window) -> None:
    window.sound_strip._toggle_raw(True)
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()

    sound_mouse(window, QEvent.Type.MouseButtonPress, window.sound_strip._x(1.0))
    sound_mouse(window, QEvent.Type.MouseMove, window.sound_strip._x(5.0))
    sound_mouse(window, QEvent.Type.MouseButtonRelease, window.sound_strip._x(5.0))

    assert window.view.lyric_raw[0] == ((0.0, 1.0), (5.0, 1.0))
    window.sound_strip._toggle_raw(False)


def test_a_zero_duration_final_sound_can_be_dragged_right_and_undone(own_window) -> None:
    own_window.sound_strip._toggle_raw(True)
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.load_lyrics(natural_sounds("ーす"), [[(0.0, 1.0), (1.0, 2.0)]], raw=[[(0.0, 1.0), (1.0, 0.0)]])
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()
    strip = own_window.sound_strip
    before = view.undo_stack.count()

    assert strip._boundary_x(0, 1) == pytest.approx(strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.5))
    wait_for_lyric_mapping(own_window)

    assert view.lyric_raw == (((0.0, 1.0), (1.5, 0.0)),)
    assert view.undo_stack.count() == before + 1
    view.undo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_raw == (((0.0, 1.0), (1.0, 0.0)),)
    view.redo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_raw == (((0.0, 1.0), (1.5, 0.0)),)


def test_sound_onsets_stop_at_global_neighbours_across_empty_lines(own_window) -> None:
    view = own_window.view
    raw = [[(1.0, 0.2)], [], [(2.0, 0.0)], [(3.0, 0.1)]]
    view.load_lyrics(natural_sounds("あ\n！？\nい\nう"), (), raw=raw)

    assert view.set_sound_onset(2, 0, 5.0)
    assert view.lyric_raw[2] == ((3.0, 0.0),)
    assert view.set_sound_onset(2, 0, 0.0)
    assert view.lyric_raw[2] == ((1.0, 0.0),)
    assert view.set_sound_onset(0, 0, -1.0)
    assert view.lyric_raw[0] == ((0.0, 0.2),)
    assert view.set_sound_onset(3, 0, 10.0)
    assert view.lyric_raw[3] == ((10.0, 0.1),)


def test_an_onset_drag_cannot_cross_the_next_onset_in_its_line(own_window) -> None:
    view = own_window.view
    view.load_lyrics(natural_sounds("あいう"), (), raw=[[(0.0, 9.0), (1.0, 0.1), (2.0, 0.2)]])
    assert view.set_sound_onset(0, 1, 5.0)
    assert view.lyric_raw == (((0.0, 9.0), (2.0, 0.1), (2.0, 0.2)),)
    assert view.set_sound_onset(0, 1, -5.0)
    assert view.lyric_raw == (((0.0, 9.0), (0.0, 0.1), (2.0, 0.2)),)


def test_only_the_operation_of_the_moved_onset_loses_confirmation(own_window) -> None:
    view = own_window.view
    previous = Match(SoundRef(0, 0), (1,), confirmed=True)
    moved = Match(SoundRef(0, 1), (2,), confirmed=True)
    following = Drop(SoundRef(0, 2), confirmed=True)
    view.load_lyrics(
        natural_sounds("あいう"),
        (),
        raw=[[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]],
        operations=[previous, moved, following],
    )
    assert not view.set_sound_onset(0, 1, 1.0)
    assert view.lyric_operations == (previous, moved, following)
    assert view.set_sound_onset(0, 1, 1.5)
    assert view.lyric_operations == (previous, following)
    assert view.lyric_raw == (((0.0, 1.0), (1.5, 1.0), (2.0, 1.0)),)


def test_the_sound_strip_has_no_line_end_handle(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.load_lyrics(natural_sounds("あ"), [[(0.0, 2.0)]], raw=[[(0.0, 1.0)]])
    strip = own_window.sound_strip
    assert len(strip._layout(0)) == 1
    assert strip._boundary_at(strip._x(1.0)) is None
    assert strip._boundary_at(strip._x(2.0)) is None
    assert not view.set_sound_onset(0, 1, 3.0)


def test_a_lyric_drag_is_smooth_but_magnets_to_the_drawn_grid(window) -> None:
    window.view.set_zoom(48.0, 16.0)
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    strip = window.sound_strip
    step = window.view.grid_step() * window.view.seconds_per_beat
    line = 1.0

    # near a drawn line the time sticks to it; halfway between two lines it stays where it is put
    assert strip._magnet_seconds(line + step * 0.1, strip._x(line)) == pytest.approx(line)
    between = line + step * 0.5
    assert strip._magnet_seconds(between, strip._x(between)) == pytest.approx(between)


def test_hovering_a_shared_note_lights_the_whole_group(window) -> None:
    times = [[(0.0, 1.0), (1.0, 2.0)]]
    window.view.load_lyrics(natural_sounds("あい"), times, group=[[0, 0]])
    assert window.sound_strip._group_run((0, 0)) == (0, 0, 1)
    assert window.sound_strip._group_run((0, 1)) == (0, 0, 1)
    assert window.sound_strip._group_run(None) is None


def test_the_lyric_note_lights_with_the_sounds_that_map_to_it(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so the spans read in seconds
    view = own_window.view
    view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]], mapped=[[(1,), (2,)]])

    view.set_playhead(0.5)  # the highlight follows the playhead, playing or paused
    assert view.highlight_note == 1
    assert view.highlight_sounds == ((0, 0),)

    view.set_playhead(1.5)
    assert view.highlight_note == 2
    assert view.highlight_sounds == ((0, 1),)

    view.set_playhead(2.5)  # past the last note: nothing is sung there
    assert view.highlight_note is None
    assert view.highlight_sounds == ()


def test_a_merge_lights_the_whole_run(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.set_notes(((60, 0.0, 1.0, 0),))
    view.load_lyrics(
        natural_sounds("あい"), [[(0.0, 0.5), (0.5, 1.0)]], raw=[[(0.0, 0.5), (0.5, 0.5)]], mapped=[[(1,), (1,)]]
    )

    view.set_playhead(0.25)
    assert view.highlight_note == 1
    assert view.highlight_sounds == ((0, 0), (0, 1))

    view.playing = False  # a pause keeps the highlight where the playhead rests
    assert view.highlight_note == 1
    assert view.highlight_sounds == ((0, 0), (0, 1))


def test_the_strip_paints_the_lyric_highlight(own_window) -> None:
    own_window.transport.bpm.setValue(60.0)
    view = own_window.view
    view.set_notes(((60, 0.0, 1.0, 0),))
    view.load_lyrics(natural_sounds("あ"), [[(0.0, 1.0)]], mapped=[[(1,)]])
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()

    view.set_playhead(0.5)
    QApplication.processEvents()
    image = own_window.sound_strip.grab().toImage()
    pixels = [image.pixelColor(x, y).rgb() for x in range(image.width()) for y in range(image.height())]
    assert QColor(theme.LYRIC_HIGHLIGHT).rgb() in pixels  # the NOTE block takes the blue
    assert QColor(theme.LYRIC_SELECT).rgb() in pixels  # its text takes the group colour, not blue

    view.set_playhead(None)
    own_window.sound_strip.setVisible(False)


def test_the_sound_strip_lays_a_row_out_once_per_refresh(window) -> None:
    window.view.load_lyrics(natural_sounds("あい\nうえ"), [[(0.0, 1.0), (1.0, 2.0)], [(3.0, 4.0), (4.0, 5.0)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()
    strip = window.sound_strip
    strip._refresh()
    x = strip._x(3.5)
    strip._boundary_at(x)  # fills the cache

    calls: list[float] = []
    original = strip._x
    strip._x = lambda seconds, origin=None: (calls.append(seconds), original(seconds, origin))[1]
    try:
        strip._boundary_at(x)
        assert not calls  # a second hit test reuses the rows already laid out
        strip._refresh()
        strip._boundary_at(x)
        assert calls  # a refresh lays them out again
    finally:
        strip._x = original
        window.view.load_lyrics((), ())
        window.sound_strip.setVisible(False)


def test_drop_labels_never_land_on_each_other(window) -> None:
    window.view.load_lyrics(natural_sounds("あいう"), [[(0.0, 1.0), (None, None), (None, None)]])
    window.sound_strip.setVisible(True)
    QApplication.processEvents()

    first = window.sound_strip._boundary_x(0, 1)
    second = window.sound_strip._boundary_x(0, 2)
    assert first is not None and second is not None
    blocks = window.sound_strip._blocks()
    assert blocks[(0, 1)].top() > blocks[(0, 0)].bottom()
    assert blocks[(0, 2)].left() > blocks[(0, 1)].right()


def test_mapped_labels_ignore_raw_onsets(window) -> None:
    window.view.load_lyrics(natural_sounds("あい"), [[(0.0, 1.0), (1.0, 2.0)]], raw=[[(0.2, 0.1), (0.8, 0.1)]])
    strip = window.sound_strip
    assert strip._boundary_x(0, 1) == pytest.approx(strip._x(1.0))
    assert strip._raw_boundary_at(strip._x(0.8)) == (0, 1)


class _Metrics:
    """A stand-in for QFontMetrics: ten pixels a character, whatever the character."""

    def horizontalAdvance(self, text: str) -> int:
        return len(text) * 10


def test_a_label_is_dropped_when_it_does_not_fit(window) -> None:
    strip = window.sound_strip
    rubied = natural_sounds("世界[せ,かい]")[0].sounds[0]  # label せ(世), reading せ
    plain = natural_sounds("あい")[0].sounds[0]  # label あ

    assert strip._fit_label(rubied, _Metrics(), 50) == "せ(世)"
    assert strip._fit_label(rubied, _Metrics(), 20) == "せ"  # the base in brackets gives way first
    assert strip._fit_label(rubied, _Metrics(), 5) == ""
    assert strip._fit_label(plain, _Metrics(), 5) == ""


def test_without_alignment_the_mapping_does_not_run(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 2.0, 1.0, 0)))
    wait_for_lyric_mapping(own_window)

    assert own_window._lyric_error  # incomplete_alignment: the sounds have no times yet
    assert own_window.view.lyric_times == (((None, None), (None, None)),)


def test_an_alignment_puts_each_sound_on_the_note_its_time_covers(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))  # one note under both sounds
    own_window._lyric_key = ""
    own_window._stored_lyrics = project.Lyrics(key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._load_sounds()
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)  # the merge tiles the one note
    assert own_window.view.lyric_red == ((False, False),)


def test_mapping_the_lyrics_to_the_active_channel(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 0.0, 1.0, 1), (64, 1.0, 1.0, 1)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい", key=text_key("あい"), raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window._lyric_channel == 0  # the stored mapping names channel 0
    assert own_window.view.lyric_times == (((0.0, 1.0), (None, None)),)  # channel 0 has one note

    own_window.view.set_active_channel(1)
    own_window._map_to_channel()
    wait_for_lyric_mapping(own_window)

    assert own_window._lyric_channel == 1
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)  # channel 1 has two


def _mapped_pair(own_window, tmp_path, raw=((0.0, 1.0), (1.0, 1.0))) -> None:
    """A one-to-one mapping of あ, い onto two notes, as the context-menu tests start from."""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい", key=text_key("あい"), raw=(raw,))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)


def test_note_lyrics_merge_same_ruby_base(own_window) -> None:
    view = own_window.view
    view.set_notes(((60, 0, 1, 0),))
    note = view.notes()[0]
    view.load_lyrics(natural_sounds("今日[きょう]"), [[(0.0, 0.5), (0.5, 1.0)]], mapped=[[(note.id,), (note.id,)]])
    assert note.lyric_text == "きょう(今日)"
    assert note.lyric_runs == (("きょう(今日)", False),)
    assert len(view.lyric_lines[0].sounds) == 2


def test_note_lyrics_append_zero_to_previous_note(own_window) -> None:
    view = own_window.view
    view.set_notes(((60, 0, 1, 0), (62, 1, 1, 0)))
    first, second = view.notes()
    view.load_lyrics(
        natural_sounds("あいうえ"),
        [[(0.0, 1.0), (None, None), (None, None), (1.0, 2.0)]],
        zero=[[False, True, True, False]],
        mapped=[[(first.id,), (), (), (second.id,)]],
    )
    assert first.lyric_text == "あ い う"
    assert first.lyric_runs == (("あ", False), (" い", True), (" う", True))
    assert second.lyric_text == "え"
    assert view.lyric_mapped[0][1:3] == ((), ())


def test_note_lyrics_leading_zero_uses_first_mapped_note(own_window) -> None:
    view = own_window.view
    view.set_notes(((60, 0, 1, 0),))
    note = view.notes()[0]
    view.load_lyrics(
        natural_sounds("あい"),
        [[(None, None), (0.0, 1.0)]],
        zero=[[True, False]],
        mapped=[[(), (note.id,)]],
    )
    assert note.lyric_runs == (("あ", True), (" い", False))


def test_note_lyrics_toggle_preserves_mapping(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view = own_window.view
    labels = [note.lyric_text for note in view.notes()]
    operations = view.lyric_operations
    before = view.undo_stack.count()
    assert own_window.edit.note_lyrics.isChecked()
    assert any(labels)
    own_window.edit.note_lyrics.click()
    assert all(not note.lyric_text for note in view.notes())
    assert [note.toolTip() for note in view.notes()] == labels
    view._sync_channel_visuals()
    assert all(not note.lyric_text for note in view.notes())
    own_window.edit.note_lyrics.click()
    assert [note.lyric_text for note in view.notes()] == labels
    assert view.lyric_operations == operations
    assert view.undo_stack.count() == before


def test_note_lyric_font_is_bold_and_tracks_vertical_zoom(own_window, monkeypatch) -> None:
    own_window.view.set_notes(((60, 0, 1, 0),))
    note = own_window.view.notes()[0]
    note.lyric_text = "あ"
    fonts = []
    original = QPainter.setFont

    def capture_font(painter, font):
        fonts.append(QFont(font))
        original(painter, font)

    monkeypatch.setattr(QPainter, "setFont", capture_font)
    for zoom_x, zoom_y in ((100, 16), (100, 60), (100, 120), (200, 120)):
        image = QImage(400, 200, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.scale(zoom_x, zoom_y)
        note.paint(painter, None)
        painter.end()

    assert all(font.bold() for font in fonts)
    assert fonts[0].pixelSize() >= 12
    assert fonts[0].pixelSize() < fonts[1].pixelSize() < fonts[2].pixelSize()
    assert fonts[2].pixelSize() > 14
    assert fonts[3].pixelSize() == fonts[2].pixelSize()


def test_boundary_drag_changes_mapping_without_editing_raw(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    raw = view.lyric_raw
    ids = tuple(note.id for note in view.notes())
    before = view.undo_stack.count()
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.8))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.8))
    wait_for_lyric_mapping(own_window)
    assert view.lyric_operations == (Match(SoundRef(0, 0), ids, True), Drop(SoundRef(0, 1), True))
    assert view.lyric_raw == raw
    assert view.undo_stack.count() == before + 1
    assert [note.lyric_text for note in view.notes()] == ["あ", "↳ あ い"]
    assert view.notes()[1].lyric_runs == (("↳ あ", False), (" い", True))
    view.undo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_mapped == (((ids[0],), (ids[1],)),)
    view.redo()
    wait_for_lyric_mapping(own_window)
    assert view.lyric_zero == ((False, True),)
    operations = view.lyric_operations
    assert own_window.save_project(own_window.project_path)
    saved = project.load(own_window.project_path).lyrics
    assert saved.raw == raw
    assert saved.operations == operations


def test_lyric_mapping_preview_is_cancelled_by_escape(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    baseline = view.lyric_operations
    before = view.undo_stack.count()
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.8))
    QTest.keyClick(strip, Qt.Key.Key_Escape)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_operations == baseline
    assert view.undo_stack.count() == before
    assert not view.gesture_active


def test_lyric_drop_layer_drag_can_merge_or_displace(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    view.lyric_operations = ()
    view.set_notes(((60, 0.0, 2.0, 0),))
    wait_for_lyric_mapping(own_window)
    raw = view.lyric_raw
    second = strip._blocks()[(0, 1)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, second.center().x(), second.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, second.center().x(), 55)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, second.center().x(), 55)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_zero == ((False, True),)
    dropped = strip._blocks()[(0, 1)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, dropped.center().x(), dropped.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.0), 15)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.0), 15)
    wait_for_lyric_mapping(own_window)
    assert isinstance(view.lyric_operations[0], Merge)
    assert view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)
    assert view.lyric_raw == raw
    assert view.notes()[0].lyric_text == "あ い"
    view.undo()
    wait_for_lyric_mapping(own_window)
    dropped = strip._blocks()[(0, 1)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, dropped.center().x(), dropped.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(0.3), 15)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(0.3), 15)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_zero == ((True, False),)
    assert view.lyric_raw == raw


def test_invalid_lyric_drag_preserves_confirmed_neighbours(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    own_window._confirm_operation(0, 1)
    wait_for_lyric_mapping(own_window)
    view, strip = own_window.view, own_window.sound_strip
    baseline = view.lyric_operations
    before = view.undo_stack.count()
    first = strip._blocks()[(0, 0)]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, first.center().x(), first.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, first.center().x(), 55)
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, first.center().x(), 55)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_operations == baseline
    assert view.undo_stack.count() == before
    assert not view.gesture_active


def test_middle_group_member_cannot_be_dropped_from_the_menu(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60)
    own_window.view.set_notes(((60, 0, 3, 0),))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="あいう", key=text_key("あいう"), raw=(((0.0, 1.0), (1.0, 1.0), (2.0, 1.0)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    baseline = own_window.view.lyric_operations
    assert isinstance(baseline[0], Merge)
    before = own_window.view.undo_stack.count()
    own_window._on_lyric_action("drop", 0, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == baseline
    assert own_window.view.undo_stack.count() == before


def test_advanced_raw_drag_escape_restores_evidence(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    view, strip = own_window.view, own_window.sound_strip
    raw, operations = view.lyric_raw, view.lyric_operations
    before = view.undo_stack.count()
    strip._toggle_raw(True)
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.3))
    QTest.keyClick(strip, Qt.Key.Key_Escape)
    wait_for_lyric_mapping(own_window)
    assert view.lyric_raw == raw
    assert view.lyric_operations == operations
    assert view.undo_stack.count() == before


def test_dropping_a_sound_marks_it_zero_and_keeps_the_rest(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    own_window.sound_strip.lyric_action_requested.emit("drop", 0, 1)
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_zero == ((False, True),)
    assert any(
        isinstance(operation, Drop) and operation.confirmed and operation.sound.index == 1
        for operation in own_window.view.lyric_operations
    )
    assert own_window.project_dirty is True


def test_a_lyric_edit_is_one_undo_step(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path)
    own_window.sound_strip.lyric_action_requested.emit("drop", 0, 1)
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_zero == ((False, True),)

    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_zero == ((False, False),)
    assert not any(operation.confirmed for operation in own_window.view.lyric_operations)


def test_a_cross_container_merge_is_green_and_exports_a_continuation(own_window, tmp_path) -> None:
    from namioto import project

    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="泣[な]い", key=text_key("泣[な]い"), raw=(((0.0, 0.5), (0.5, 0.5)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == (Merge((SoundRef(0, 0), SoundRef(0, 1)), 1),)
    assert own_window.view.lyric_red == ((False, False),)
    assert own_window.export_krc(tmp_path / "exported.krc")
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "泣[な]い.+"


def test_read_mode_uses_continuations_without_changing_edit_evidence(own_window, tmp_path) -> None:
    from namioto import project

    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        text="泣[な]い.+", key=text_key("泣[な]い.+"), raw=(((0.0, 1.0), (1.0, 1.0)),)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_mapped == (((1,), (2,)),)
    raw = own_window.view.lyric_raw
    own_window._set_lyric_mode("read")
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_times == (((0.0, 0.5), (0.5, 1.0)),)
    assert own_window.view.lyric_mapped == (((1,), (1,)),)
    assert own_window.view.lyric_raw == raw


def test_merging_two_sounds_confirms_the_pair(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.0, 0.5), (0.5, 0.5)))
    own_window.sound_strip.lyric_action_requested.emit("merge", 0, 1)
    wait_for_lyric_mapping(own_window)

    assert any(
        isinstance(operation, Merge) and operation.confirmed and len(operation.sounds) == 2
        for operation in own_window.view.lyric_operations
    )
    assert own_window.view.lyric_times == (((0.0, 0.5), (0.5, 1.0)),)


def test_a_merge_can_be_dissolved_again(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.0, 0.5), (0.5, 0.5)))
    own_window.sound_strip.lyric_action_requested.emit("merge", 0, 1)
    wait_for_lyric_mapping(own_window)
    assert any(operation.confirmed for operation in own_window.view.lyric_operations)
    own_window.sound_strip.lyric_action_requested.emit("dissolve", 0, 0)
    wait_for_lyric_mapping(own_window)

    assert not any(operation.confirmed for operation in own_window.view.lyric_operations)


def test_a_grouped_run_is_marked_on_the_strip(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_group == ((0, 0),)


def test_shared_sound_blocks_have_a_thin_internal_separator(window) -> None:
    text = "あい"
    window.transport.bpm.setValue(60.0)
    window.view.load_lyrics(
        natural_sounds(text),
        [[(0.0, 2.0), (1.0, 2.0)]],
        raw=[[(0.0, 1.5), (1.5, 0.5)]],
        zero=[[False, False]],
        group=[[1, 1]],
    )
    window.view.set_zoom(300.0, 16.0)
    window.view.horizontalScrollBar().setValue(0)
    window.view.verticalScrollBar().setValue(0)
    window.sound_strip.setVisible(True)
    QApplication.processEvents()

    strip = window.sound_strip
    strip._hover = None  # a live pointer may have tinted the group; the plain body is what it tested
    body = theme.lyric_shades(True)[0]
    seam = round(strip._x(1.0))  # い's left edge, inside あ's block
    image = strip.grab().toImage()
    mid = strip.height() // 2
    colors = [image.pixelColor(seam + dx, mid).name() for dx in (-1, 0, 1)]
    assert colors == [body.name(), QColor(theme.LYRIC_ZERO).name(), body.name()]


def test_a_sound_that_covers_no_note_is_marked_grey(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あか\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(key=text_key("あか\n"), model="mms", raw=(((0.0, 1.0), (3.0, 0.5)),))
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_zero == ((False, True),)


@pytest.mark.parametrize("model", ["mms", "yohane"])
def test_pattern_weights_apply_to_both_aligners_without_changing_evidence(own_window, tmp_path, model) -> None:
    text = "泣[な]い"
    raw = (((0.0, 0.74), (0.74, 0.26)),)
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), model=model, raw=raw)
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_operations == (Merge((SoundRef(0, 0), SoundRef(0, 1)), 1),)
    assert own_window.view.lyric_raw == raw
    assert own_window.export_krc(tmp_path / "exported.krc")
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "泣[な]い.+"


def test_short_sounds_on_one_note_fall_to_zero(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("きにく\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(
        key=text_key("きにく\n"), model="mms", raw=(((0.0, 0.9), (0.9, 0.05), (0.95, 0.05)),)
    )
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))  # one note under three sounds: the first keeps it
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window.view.lyric_raw == (((0.0, 0.9), (0.9, 0.05), (0.95, 0.05)),)
    assert own_window.view.lyric_zero == ((False, True, True),)
    assert own_window.view.lyric_group == ((-1, -1, -1),)


def test_a_block_is_the_note_the_sound_maps_to(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("タに\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(key=text_key("タに\n"), model="mms", raw=(((0.0, 0.8), (0.8, 0.5)),))
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 0.4, 0)))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    # タ sits on note 0 and に on note 1: a block is the whole note a sound maps to, not its raw span
    assert own_window.view.lyric_zero == ((False, False),)
    assert own_window.view.lyric_times[0][0] == (0.0, 1.0)
    # に's raw span (0.8, 1.3) crosses the boundary, but its block is note 1 whole
    assert own_window.view.lyric_times[0][1] == (1.0, 1.4)
    assert own_window.view.lyric_raw[0][1][0] == 0.8


def test_a_lyric_drag_is_kept_in_the_project(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    project_path = tmp_path / "song.nto"
    own_window.project_path = project_path
    own_window._stored_lyrics = project.Lyrics(key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._lyric_key = ""
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window.sound_strip._toggle_raw(True)
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()

    sound_mouse(own_window, QEvent.Type.MouseButtonPress, own_window.sound_strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, own_window.sound_strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, own_window.sound_strip._x(1.5))
    wait_for_lyric_mapping(own_window)

    assert own_window.project_dirty is True
    assert own_window.save_project(project_path) is True
    assert project.load(project_path).lyrics.raw == (((0.0, 1.0), (1.5, 1.0)),)


def test_notes_on_another_channel_are_not_mapped(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=1),))
    own_window.view.set_notes(((60, 0.0, 1.0, 1), (62, 1.0, 1.0, 1)))

    assert own_window.view.lyric_times == (((None, None), (None, None)),)


def test_the_grid_offset_only_slides_the_drawn_grid(own_window) -> None:
    view = own_window.view
    view.set_channels(())
    view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.transport.bpm.setValue(120.0)
    rect = QRectF(0.0, 0.0, 3.0, 1.0)
    assert [x for x, _level in view.division_lines(rect, 0.0)] == [0.0, 1.0, 2.0, 3.0]

    own_window.transport.grid_offset.setValue(125)  # +0.125 s = a quarter beat at 120 BPM
    assert [x for x, _level in view.division_lines(rect, 0.0)] == [-0.75, 0.25, 1.25, 2.25]
    assert view.notes()[0].start == 0.0  # the note keeps its own time
    assert view._snap_beats(0.6) == pytest.approx(0.75)  # snapping follows the drawn lines
    assert view._snap_floor_beats(0.6) == pytest.approx(0.25)
    assert view._snap_ceil_beats(0.6) == pytest.approx(0.75)


def test_the_grid_offset_leaves_the_time_clock_absolute(own_window) -> None:
    view = own_window.view
    own_window.transport.bpm.setValue(120.0)
    rect = QRectF(0.0, 0.0, 3.0, 1.0)
    assert [x for x, _seconds, _major in view.seconds_lines(rect, 40.0)] == [0.0, 1.0, 2.0, 3.0]

    own_window.transport.grid_offset.setValue(125)  # +0.125 s = a quarter beat at 120 BPM
    assert [x for x, _seconds, _major in view.seconds_lines(rect, 40.0)] == [0.0, 1.0, 2.0, 3.0]
    view.division = "seconds"
    assert [x for x, _level in view.division_lines(rect, 40.0)] == [0.0, 1.0, 2.0, 3.0]


def test_the_grid_offset_leaves_the_playhead_at_its_timestamp(own_window, monkeypatch) -> None:
    own_window.transport.grid_offset.setValue(-300)
    monkeypatch.setattr(own_window, "_position", lambda: 1.0)
    own_window._show_position()
    assert own_window.view.playhead == 1.0


def test_the_strip_hides_until_the_times_arrive(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window.lyrics_text = "あい\n"
    own_window._load_sounds()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_times == (((None, None), (None, None)),)
    assert own_window.sound_strip.isHidden()

    own_window._stored_lyrics = project.Lyrics(key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),))
    own_window._lyric_key = ""
    own_window._load_sounds()
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0)),)
    assert not own_window.sound_strip.isHidden()


def test_the_project_keeps_the_lyrics_text_without_the_krc(own_window, tmp_path) -> None:
    project_path = tmp_path / "song.nto"
    sidecar = tmp_path / "song.krc"
    sidecar.write_text("あい\n", encoding="utf-8")
    own_window.project_path = project_path
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    assert own_window.lyrics_text == "あい\n"

    assert own_window.save_project(project_path) is True
    saved = project.load(project_path)
    assert saved.lyrics.text == "あい\n"

    sidecar.unlink()
    own_window._stored_lyrics = saved.lyrics
    own_window._watch_lyrics()
    assert own_window.lyrics_text == "あい\n"  # the project itself is the baseline

    own_window._watch_lyrics(materialize=True)
    assert sidecar.read_text(encoding="utf-8") == "あい\n"  # a copy for the external editor


def test_the_lyrics_mode_is_kept_in_the_project(own_window, tmp_path) -> None:
    project_path = tmp_path / "song.nto"
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = project_path
    own_window._stored_lyrics = None
    own_window._watch_lyrics()

    own_window._set_lyric_mode("read")

    assert own_window._lyric_mode == "read"
    assert own_window.project_dirty is True
    assert own_window.save_project(project_path) is True
    assert project.load(project_path).lyrics.mode == "read"


def test_exporting_krc_without_a_mapping_writes_nothing(own_window, monkeypatch, tmp_path) -> None:
    own_window.lyrics_text = "あい\n"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "Lyrics file (*.krc)")
    )

    assert own_window._on_export_krc() is False
    assert not (tmp_path / "exported.krc").exists()
    assert own_window.project_path is None


def test_exporting_krc_writes_the_mapping_back(own_window, monkeypatch, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))  # one note under both sounds
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), raw=(((0.0, 0.5), (0.5, 0.5)),))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "Lyrics file (*.krc)")
    )

    assert own_window._on_export_krc() is True
    assert (tmp_path / "exported.krc").read_text(encoding="utf-8") == "(あい).1"


def test_exporting_krc_without_lyrics_says_so(own_window, monkeypatch) -> None:
    own_window.lyrics_text = ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: pytest.fail("must not ask"))

    assert own_window._on_export_krc() is False
    assert "no lyrics" in own_window.statusBar().currentMessage().lower()


def test_read_mode_lays_the_krcs_own_dot_n_onto_the_notes(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あ.2\n", key=text_key("あ.2\n"), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window._lyric_mode == "read"
    assert own_window.view.lyric_times == (((0.0, 2.0),),)  # one sound held over both notes
    assert own_window.view.lyric_mapped == (((1, 2),),)  # the ids the playback highlight reads
    assert own_window.view.lyric_editable is False
    assert own_window.edit.align.isEnabled() is False


def test_read_mode_draws_a_dot_n_sound_where_it_holds_off(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい.0う\n", key=text_key("あい.0う\n"), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()

    assert own_window.view.lyric_zero == ((False, True, False),)
    blocks = own_window.sound_strip._blocks()
    assert blocks[(0, 1)].top() > blocks[(0, 0)].bottom()
    assert own_window.sound_strip._boundary_x(0, 1) == pytest.approx(own_window.sound_strip._x(1.0))


def test_read_mode_reads_a_ruby_part_by_part(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(tuple((60 + i, float(i), 1.0, 0) for i in range(5)))
    own_window.project_path = tmp_path / "song.nto"
    text = "確信犯[かく,(しん).1,はん]\n"
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    # the `.1` group's two sounds share the third note; か, く, は and ん take one each
    assert own_window.view.lyric_times == (((0.0, 1.0), (1.0, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0)),)


def test_a_broken_outside_change_keeps_the_read_only_lyrics(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    text = "あ.0\n"
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), mode="read")
    own_window._watch_lyrics()
    before = own_window.view.lyric_lines
    assert before

    lyrics.save(own_window.lyrics_path(), "あ.0)\n")  # a syntax error in the file beside us
    own_window._on_lyrics_file_changed()

    assert own_window.lyrics_text == text
    assert own_window.view.lyric_lines == before
    assert "could not be read" in own_window.statusBar().currentMessage().lower()


def test_a_broken_read_only_lyrics_file_opens_empty(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    own_window.project_path = tmp_path / "song.nto"
    text = "あ.0)\n"  # a syntax error
    own_window._stored_lyrics = project.Lyrics(text=text, key=text_key(text), mode="read")
    own_window._watch_lyrics()  # must not take the window down

    assert not own_window.view.lyric_lines


def test_read_mode_refuses_to_align(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), mode="read")
    own_window._watch_lyrics()

    own_window._open_align()

    assert "edit mode" in own_window.statusBar().currentMessage().lower()


def test_the_lock_button_switches_the_timeline_mode(lyrics_window) -> None:
    assert lyrics_window.edit.lyric_lock.isEnabled() is True
    assert lyrics_window.edit.lyric_lock.isChecked() is False
    assert lyrics_window._lyric_mode == "edit"

    lyrics_window.edit.lyric_lock.click()

    assert lyrics_window._lyric_mode == "read"
    assert lyrics_window.edit.align.isEnabled() is False


def test_a_read_only_project_sets_the_lock(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), mode="read")
    own_window._watch_lyrics()

    assert own_window.edit.lyric_lock.isChecked() is True
    assert own_window.edit.align.isEnabled() is False


def test_a_read_only_strip_does_not_drag(own_window, tmp_path) -> None:
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい\n", key=text_key("あい\n"), mode="read")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    own_window.sound_strip.setVisible(True)
    QApplication.processEvents()
    before = own_window.view.lyric_raw

    sound_mouse(own_window, QEvent.Type.MouseButtonPress, own_window.sound_strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, own_window.sound_strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, own_window.sound_strip._x(1.5))

    assert own_window.view.lyric_raw == before


def test_read_mode_preserves_edit_evidence_and_uses_faithful_positions(own_window, tmp_path) -> None:
    _mapped_pair(own_window, tmp_path, raw=((0.1, 0.2), (0.8, 0.4)))
    before = own_window.view.lyric_raw
    own_window._set_lyric_mode("read")
    wait_for_lyric_mapping(own_window)
    strip = own_window.sound_strip
    assert own_window.view.lyric_raw == before
    assert strip._boundary_x(0, 1) == pytest.approx(strip._x(1.0))
    own_window._set_lyric_mode("edit")
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_raw == before
    assert strip._boundary_x(0, 1) == pytest.approx(strip._x(1.0))


def test_the_aligned_times_are_kept_in_the_project(own_window, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.view.set_lyrics(natural_sounds("あい\n"), [[(0.0, 1.0), (1.0, 2.0)]])
    own_window._lyric_key = text_key("あい\n")
    own_window._lyric_model = "mms"

    assert own_window.save_project(tmp_path / "song.nto") is True
    saved = project.load(tmp_path / "song.nto")
    assert saved.lyrics == project.Lyrics(
        text="あい\n", key=text_key("あい\n"), model="mms", raw=(((0.0, 1.0), (1.0, 1.0)),)
    )
    assert own_window.project_dirty is False


def test_the_align_dialog_hands_the_times_over(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append((times, model)))
    dialog._done(([[(0.0, 1.0), (1.0, 2.0)]], "mms", [], []))
    assert got == [([[(0.0, 1.0), (1.0, 1.0)]], "mms")]
    assert dialog.run.isEnabled()


def test_a_missing_runtime_stops_the_alignment(own_window, monkeypatch) -> None:
    monkeypatch.setattr(devices, "validate", lambda: devices.MISSING_RUNTIME)
    monkeypatch.setattr(Aligner, "start", lambda self: pytest.fail("must not run"))
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)

    dialog._start()

    assert devices.MISSING_RUNTIME in dialog.log.toPlainText()
    assert dialog.run.isEnabled()


def test_the_align_dialog_snaps_to_the_beat_grid_when_asked(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "quantize")(1)  # 1/4 notes, one beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.0, 0.5), (0.5, pytest.approx(0.5))]]]


def test_the_align_dialog_can_quantize_to_eighth_notes(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "quantize")(2)  # 1/8 notes, half a beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.0, 0.5), (0.5, pytest.approx(0.5))]]]


def test_the_align_dialog_quantizes_onto_the_offset_grid(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window, offset=0.25)
    parameter_writer(dialog, "quantize")(1)  # 1/4 notes, one beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.25, 0.5), (0.75, pytest.approx(0.5))]]]


def test_a_cached_alignment_is_reused_and_resnapped(own_window, monkeypatch) -> None:
    align.save_alignment("/tmp/vocal.wav", "mms", "cpu", "あん\n", [[(0.1, 0.6), (0.6, 1.1)]], [], [])
    monkeypatch.setattr(align, "align", lambda *args, **kwargs: pytest.fail("must not run the model"))
    dialog = AlignDialog("/tmp/vocal.wav", "あん\n", 120.0, parent=own_window, offset=0.25)
    parameter_writer(dialog, "quantize")(1)  # 1/4 notes, one beat
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))

    dialog._start()

    assert dialog._thread is None  # the model never ran
    assert got == [[[(0.25, 0.5), (0.75, pytest.approx(0.5))]]]
    assert "saved alignment reused" in dialog.log.toPlainText()


def test_a_cached_whole_song_alignment_is_reused_when_chunking_is_off(own_window, monkeypatch) -> None:
    align.save_alignment("/tmp/vocal.wav", "mms", "cpu", "あん\n", [[(0.1, 0.6), (0.6, 1.1)]], [], [], chunk="off")
    monkeypatch.setattr(align, "align", lambda *args, **kwargs: pytest.fail("must not run the model"))
    dialog = AlignDialog("/tmp/vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "chunk")("off")
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))

    dialog._start()

    assert dialog._thread is None
    assert got == [[[(0.1, 0.5), (0.6, pytest.approx(0.5))]]]
    assert "saved alignment reused" in dialog.log.toPlainText()


def test_the_align_dialog_remembers_what_was_chosen(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    parameter_writer(dialog, "model")("yohane")
    parameter_writer(dialog, "quantize")(4)
    parameter_writer(dialog, "chunk")("off")
    dialog.reject()

    again = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    assert again.parameters() == {"model": "yohane", "device": "cpu", "quantize": 4, "chunk": "off"}


def test_the_align_dialog_runs_chunked_unless_told_otherwise(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    assert dialog.parameters()["chunk"] == align.DEFAULT_MODE


def test_the_align_dialog_leaves_the_times_alone_by_default(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あん\n", 120.0, parent=own_window)
    got: list = []
    dialog.aligned.connect(lambda times, model, flagged: got.append(times))
    dialog._done(([[(0.1, 0.6), (0.6, 1.1)]], "mms", [], []))
    assert got == [[[(0.1, 0.5), (0.6, pytest.approx(0.5))]]]


def test_the_align_dialog_shows_a_failure(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)
    dialog._fail("FileNotFoundError: no mms aligner")
    assert "FileNotFoundError" in dialog.log.toPlainText()
    assert dialog.run.isEnabled()


def test_the_align_dialog_keeps_its_progress_beside_the_problems(own_window) -> None:
    dialog = AlignDialog("vocal.wav", "あい\n", 120.0, parent=own_window)
    dialog.log.setPlainText("Aligning over 204.0s of audio…")
    dialog._done(([[(0.0, 1.0), (1.0, 2.0)]], "mms", ["あい: empty"], [True]))
    assert "Aligning over 204.0s of audio…" in dialog.log.toPlainText()
    assert "あい: empty" in dialog.log.toPlainText()


def test_the_aligner_logs_a_download_a_tenth_at_a_time(qt_app) -> None:
    aligner = Aligner("vocal.wav", "あい\n", "yohane")
    logged: list[str] = []
    aligner.message.connect(logged.append)
    aligner._downloading(0, 1000)
    aligner._downloading(50, 1000)
    aligner._downloading(100, 1000)
    assert logged == ["Downloading the yohane model… 0%", "Downloading the yohane model… 10%"]


def test_the_aligner_reports_the_lines_it_doubts() -> None:
    lines = natural_sounds("あい\n")
    found = align.AlignedSegment(0.0, 2.0, (align.Token("a"), align.Token("i")))
    flagged, reported = Aligner._problems(found, lines)
    assert flagged == [True]
    assert reported == ["あい: empty"]


def test_the_align_button_opens_the_dialog(own_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あん\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window.audio_path = str(tmp_path / "vocal.wav")

    opened: list = []
    monkeypatch.setattr(AlignDialog, "exec", lambda self: opened.append(self) or 0)
    own_window.transport.grid_offset.setValue(125)
    own_window._open_align()
    assert len(opened) == 1
    assert opened[0].offset == 0.125  # the dialog quantises on the grid that is drawn


def test_saving_lyrics_lets_align_see_them_at_once(own_window, monkeypatch, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window.audio_path = str(tmp_path / "vocal.wav")
    own_window._watch_lyrics()  # no .krc yet, so nothing to align
    assert own_window.view.lyric_lines == ()

    own_window._on_lyrics_saved("あん\n")  # the dialog wrote the file and told the window
    assert [sound.reading for line in own_window.view.lyric_lines for sound in line.sounds] == ["あ", "ん"]

    opened: list = []
    monkeypatch.setattr(AlignDialog, "exec", lambda self: opened.append(self) or 0)
    own_window._open_align()
    assert len(opened) == 1


def test_the_align_button_says_why_it_cannot_open(own_window, tmp_path) -> None:
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    own_window.audio_path = str(tmp_path / "vocal.wav")
    own_window._open_align()
    assert "no lyrics" in own_window.statusBar().currentMessage()

    (tmp_path / "song.krc").write_text("世界\n", encoding="utf-8")
    own_window._watch_lyrics()
    own_window._open_align()
    assert "could not be read" in own_window.statusBar().currentMessage()


class _FakeAligner(QObject):
    """One pass of the aligner, answering at once with a hand-made alignment."""

    aligned = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()
    calls: list = []

    def __init__(self, audio, text, model, provider, chunk, parent=None):
        super().__init__(parent)
        _FakeAligner.calls.append((audio, text, model, provider, chunk))

    def start(self) -> None:
        self.aligned.emit(([[(0.0, 1.0), (1.0, 2.0)]], "mms", [], [True]))
        self.finished.emit()


def test_auto_align_reuses_the_cached_pass(lyrics_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    lyrics_window._watch_lyrics()
    lyrics_window.audio_path = str(tmp_path / "vocal.wav")
    monkeypatch.setattr(align, "is_installed", lambda model, language="ja": True)
    monkeypatch.setattr(align, "has_emissions", lambda *args, **kwargs: True)
    _FakeAligner.calls = []
    monkeypatch.setattr("namioto.ui.app.Aligner", _FakeAligner)

    lyrics_window._auto_align()

    assert _FakeAligner.calls == [(str(tmp_path / "vocal.wav"), "あい\n", "mms", "cpu", align.DEFAULT_MODE)]
    assert lyrics_window.view.lyric_raw == (((0.0, 1.0), (1.0, 1.0)),)
    assert lyrics_window._lyric_flags == (True,)  # the aligner's doubt is kept with the times
    assert lyrics_window._auto_align_thread is None


def test_auto_align_quantizes_onsets_without_changing_reference_durations(lyrics_window, monkeypatch) -> None:
    lyrics_window.lyrics_text = "あい"
    lyrics_window._load_sounds()
    lyrics_window.transport.bpm.setValue(60.0)
    monkeypatch.setattr(align, "load_parameters", lambda: {"quantize": 1})
    lyrics_window._auto_align_key = text_key("あい")
    spans = [[(0.1, 0.3), (0.6, 0.9)]]
    lyrics_window._on_auto_aligned((spans, "mms", [], []))
    assert lyrics_window.view.lyric_raw == (((0.0, pytest.approx(0.2)), (1.0, pytest.approx(0.3))),)
    assert [length for _onset, length in lyrics_window.view.lyric_raw[0]] == [
        length for _onset, length in from_spans(spans)[0]
    ]


def test_auto_align_waits_when_the_pass_is_not_cached(lyrics_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    lyrics_window._watch_lyrics()
    lyrics_window.audio_path = str(tmp_path / "vocal.wav")
    monkeypatch.setattr(align, "is_installed", lambda model, language="ja": True)
    monkeypatch.setattr(align, "has_emissions", lambda *args, **kwargs: False)
    started = []
    monkeypatch.setattr("namioto.ui.app.Aligner", lambda *args, **kwargs: started.append(args))

    lyrics_window._auto_align()

    assert started == []
    assert "align again" in lyrics_window.statusBar().currentMessage()


def test_auto_align_can_be_turned_off(lyrics_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あい\n", encoding="utf-8")
    lyrics_window._watch_lyrics()
    lyrics_window.audio_path = str(tmp_path / "vocal.wav")
    lyrics_window.settings.lyrics.auto_align = False
    started = []
    monkeypatch.setattr("namioto.ui.app.Aligner", lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(align, "has_emissions", lambda *args, **kwargs: True)

    lyrics_window._auto_align()

    assert started == []


def test_exporting_is_not_gated_on_the_lyrics(own_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あん\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)  # a beat is a second, so notes read in seconds
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 2.0, 0),))  # one note where two sounds need two
    own_window.view.set_lyrics(natural_sounds("あん\n"), [[(0.0, 1.0), (1.0, 2.0)]])
    own_window._lyric_key = text_key("あん\n")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "out"), "MIDI"))

    assert own_window._on_export_midi() is True


def test_exporting_leaves_the_krc_alone(own_window, monkeypatch, tmp_path) -> None:
    (tmp_path / "song.krc").write_text("あん\n", encoding="utf-8")
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = None
    own_window._watch_lyrics()
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0), (64, 2.0, 1.0, 0)))
    own_window.view.set_lyrics(natural_sounds("あん\n"), [[(0.0, 1.0), (1.0, 3.0)]])
    own_window._lyric_key = text_key("あん\n")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "out"), "MIDI"))

    assert own_window._on_export_midi() is True
    assert (tmp_path / "song.krc").read_text(encoding="utf-8") == "あん\n"


class _ManualLyricMapper(QObject):
    mapped = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()
    instances: list = []

    def __init__(self, revision, request, parent=None):
        super().__init__(parent)
        self.revision = revision
        self.request = request

    def start(self):
        self.instances.append(self)

    def complete(self):
        from namioto.ui.lyric_map import map_lyrics

        self.mapped.emit((self.revision, map_lyrics(**self.request)))
        self.finished.emit()

    def wait(self):
        pass


def _manual_mapping(monkeypatch):
    _ManualLyricMapper.instances = []
    monkeypatch.setattr("namioto.ui.app.LyricMapper", _ManualLyricMapper)
    return _ManualLyricMapper.instances


def test_mapping_computation_runs_outside_the_gui_thread(own_window, tmp_path, monkeypatch):
    from threading import get_ident

    from namioto.ui import lyric_map

    main = get_ident()
    called = []
    original = lyric_map.map_lyrics

    def record(**request):
        called.append(get_ident())
        return original(**request)

    monkeypatch.setattr(lyric_map, "map_lyrics", record)
    _mapped_pair(own_window, tmp_path)
    assert called
    assert all(thread != main for thread in called)
    own_window._on_lyric_action("drop", 0, 1)
    assert own_window.view.lyric_mapping_pending
    wait_for_lyric_mapping(own_window)
    assert all(thread != main for thread in called)
    assert own_window.view.lyric_zero == ((False, True),)


def test_a_same_channel_map_click_forces_a_full_solve(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    dirty = own_window.project_dirty
    own_window.edit.map_channel.click()
    assert len(workers) == 1
    assert workers[0].request["previous"] is None
    assert own_window.project_dirty == dirty
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending


def test_saving_during_a_local_mapping_does_not_request_a_full_solve(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    previous = own_window._lyric_map_result
    workers = _manual_mapping(monkeypatch)
    own_window.view.begin_gesture("Move lyrics")
    own_window.view.set_sound_onset(0, 1, 1.1)
    own_window.view.commit_gesture()
    revision = own_window._lyric_map_revision
    assert own_window.save_project(tmp_path / "saved.nto")
    assert own_window._lyric_map_revision == revision
    assert own_window._lyric_map_pending is None
    assert workers[0].request["previous"] is previous
    assert project.load(tmp_path / "saved.nto").lyrics.raw == (((0.0, 1.0), (1.1, 1.0)),)
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending


def test_loading_same_lyrics_replaces_evidence_and_only_starts_one_full_mapping(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    raw = (((0.1, 1.0), (1.2, 1.0)),)
    path = tmp_path / "other.nto"
    project.save(
        project.Project(
            notes=(project.Note(0.0, 1.0, 60, id=41), project.Note(1.0, 1.0, 62, id=99)),
            lyrics=project.Lyrics(text="あい", key=text_key("あい"), raw=raw),
        ),
        path,
    )
    workers = _manual_mapping(monkeypatch)
    assert own_window.load_project(path)
    assert len(workers) == 1
    assert workers[0].request["previous"] is None
    assert own_window.view.lyric_raw == raw
    assert [note.id for note in own_window.view.notes()] == [41, 99]
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending
    assert not own_window._lyric_error


def test_a_mapping_result_cannot_overwrite_a_drag_in_progress(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    view = own_window.view
    original = view.lyric_operations
    before = view.undo_stack.count()
    own_window._remap_lyrics_async()
    view.begin_gesture("Move lyrics")
    view.set_sound_onset(0, 1, 1.1)
    workers[0].complete()
    assert view.lyric_raw == (((0.0, 1.0), (1.1, 1.0)),)
    assert view.lyric_operations == original
    assert view.lyric_mapping_pending
    assert len(workers) == 1
    view.commit_gesture()
    assert len(workers) == 2
    assert workers[1].request["raw"] == [[(0.0, 1.0), (1.1, 1.0)]]
    workers[1].complete()
    assert not view.lyric_mapping_pending
    assert view.undo_stack.count() == before + 1


def test_continuous_edits_only_queue_the_latest_mapping(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._remap_lyrics_async()
    for onset in (1.1, 1.2, 1.3):
        own_window.view.begin_gesture("Move lyrics")
        own_window.view.set_sound_onset(0, 1, onset)
        own_window.view.commit_gesture()
    assert len(workers) == 1
    assert own_window._lyric_map_pending[1]["raw"] == [[(0.0, 1.0), (1.3, 1.0)]]
    workers[0].failed.emit("obsolete failure")
    assert own_window.view.lyric_mapping_pending
    assert not own_window._lyric_error
    workers[0].complete()
    assert len(workers) == 2
    assert own_window.view.lyric_mapping_pending
    assert workers[1].request["raw"] == [[(0.0, 1.0), (1.3, 1.0)]]
    workers[1].complete()
    assert own_window.view.lyric_raw == (((0.0, 1.0), (1.3, 1.0)),)
    assert not own_window.view.lyric_mapping_pending


def test_pending_mapping_blocks_lyric_actions_and_exports_but_not_saving_or_midi(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    before = own_window.view.undo_stack.count()
    operations = own_window.view.lyric_operations
    own_window._remap_lyrics()
    assert own_window.view.lyric_mapping_pending
    assert not own_window.transport.export_krc_action.isEnabled()
    assert not own_window.transport.export_ass_action.isEnabled()
    assert own_window.transport.export_midi_action.isEnabled()
    assert not own_window.export_krc(tmp_path / "pending.krc")
    assert not own_window.export_ass(tmp_path / "pending.ass")
    assert not (tmp_path / "pending.krc").exists()
    assert not (tmp_path / "pending.ass").exists()
    assert own_window._verification().counts() == {"mapping_pending": 1}
    own_window._on_lyric_action("merge", 0, 1)
    assert own_window.view.lyric_operations == operations
    assert own_window.view.undo_stack.count() == before
    assert own_window.save_project(tmp_path / "pending.nto")
    assert own_window.export_midi(tmp_path / "pending.mid")
    workers[0].complete()
    assert own_window.transport.export_krc_action.isEnabled()
    assert own_window.transport.export_ass_action.isEnabled()
    assert own_window.export_krc(tmp_path / "ready.krc")
    assert own_window.view.undo_stack.count() == before


@pytest.mark.parametrize("finish", ["commit_gesture", "cancel_gesture"])
def test_an_unchanged_gesture_does_not_leave_mapping_pending(own_window, tmp_path, monkeypatch, finish):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._remap_lyrics_async()
    own_window.view.begin_gesture("Move lyrics")
    workers[0].complete()
    getattr(own_window.view, finish)()
    assert len(workers) == 2
    workers[1].complete()
    assert not own_window.view.lyric_mapping_pending


def test_read_mode_disables_onset_dragging_before_its_mapping_finishes(own_window, tmp_path, monkeypatch):
    _mapped_pair(own_window, tmp_path)
    workers = _manual_mapping(monkeypatch)
    own_window._set_lyric_mode("read")
    assert own_window.view.lyric_mapping_pending
    assert not own_window.view.lyric_editable
    before = own_window.view.lyric_raw
    strip = own_window.sound_strip
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(1.5))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(1.5))
    assert own_window.view.lyric_raw == before
    assert not own_window.view.gesture_active
    workers[0].complete()
    assert not own_window.view.lyric_mapping_pending


def test_changed_conflict_filtering_invalidates_the_cached_mapping(monkeypatch):
    from namioto.ui import lyric_map

    text = "\n".join(["あ"] * 5)
    request = {
        "text": text,
        "lines": natural_sounds(text),
        "raw": [[(row * 4.0, 1.0)] for row in range(5)],
        "notes": [(row * 4.0, row * 4.0 + 1.0, 60, row + 1) for row in range(5)],
    }
    previous = lyric_map.map_lyrics(**request)
    assert previous.state is not None
    cached = []
    original = lyric_map.solve

    def record(lines, raw, notes, anchors, state):
        cached.append(state)
        return original(lines, raw, notes, anchors, state)

    monkeypatch.setattr(lyric_map, "solve", record)
    request["notes"].append((0.0, 1.0, 62, 99))
    result = lyric_map.map_lyrics(**request, previous=previous)
    assert cached == [None]
    assert [note.id for note in result.filtered] == [99]
    assert result.operations == previous.operations
    assert not result.error


def test_time_edit_undo_and_redo_preserve_sparse_note_ids_and_anchors(own_window, tmp_path):
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_notes(((60, 0.0, 1.0, 0, 41), (62, 1.0, 1.0, 0, 99)))
    own_window.project_path = tmp_path / "song.nto"
    anchor = Match(SoundRef(0, 0), (41,), confirmed=True)
    own_window._stored_lyrics = project.Lyrics(
        text="あい", key=text_key("あい"), raw=(((0.0, 1.0), (1.0, 1.0)),), operations=(anchor,)
    )
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)
    view = own_window.view
    view.begin_gesture("Move lyrics")
    view.set_sound_onset(0, 1, 1.1)
    view.commit_gesture()
    wait_for_lyric_mapping(own_window)
    for restore, onset in ((view.undo, 1.0), (view.redo, 1.1)):
        restore()
        wait_for_lyric_mapping(own_window)
        assert [note.id for note in view.notes()] == [41, 99]
        assert view.lyric_raw[0][1][0] == onset
        assert anchor in view.lyric_operations
        assert not own_window._lyric_error
