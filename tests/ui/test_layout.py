# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QFocusEvent, QFont, QImage, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QToolButton

from namioto.analysis.bpm import BpmEstimate
from namioto.interaction import Interaction, Tool
from namioto.ui import theme
from namioto.ui.app import MainWindow, TempoLoader
from namioto.ui.controls import Cluster, EditBar, TransportBar, ValueSlider
from namioto.ui.roll import (
    PITCH_MAX,
    PITCH_MIN,
    RULER_HEIGHT,
    RULER_TIME_ROW,
    SNAP_CHOICES,
    PianoRollView,
    is_black_key,
)

from .helpers import (
    card,
    clusters,
    device_point,
    draw_note,
    fake_loaders,
    focus,
    make_spectrum,
    roll_mouse,
    ruler_click,
    ruler_mouse,
)


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
    focus(window)
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
