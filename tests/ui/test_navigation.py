# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import time

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSlider

from namioto.interaction import Tool
from namioto.ui.audio import MidiPortOut, SilentPlayer, find_synth_port
from namioto.ui.roll import PITCH_MAX

from .helpers import FakeOutput, draw_note


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


def test_the_silent_player_keeps_the_position_without_playing() -> None:
    player = SilentPlayer()
    player.set_program([(69, 0.0, 1.0)], 1.0)
    assert player.duration == pytest.approx(1.0)
    assert player.silent and player.position == 0.0 and not player.is_playing
    player.seek(0.75)
    assert player.position == pytest.approx(0.75)
    player.stop()
    assert player.position == 0.0
    player.set_program([(69, 0.0, 1.0)], 2.0)
    assert player.duration == pytest.approx(1.0)


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


def test_the_port_player_schedules_notes_and_silences_them_on_stop(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.set_program([(69, 0.0, 1.0), (76, 0.5, 1.0)], 1.0)
    assert [0xC0, 0x00] in port.messages  # the synth is asked for its piano first
    assert [0xB0, 0x07, 127] in port.messages  # and told how loud to be
    assert player.duration == pytest.approx(1.5)

    player.play()
    assert player.is_playing
    pump(0.15)
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


def test_a_preview_plays_the_note_and_releases_it(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.preview(72, seconds=0.05)
    assert [0x90, 72, 100] in port.messages
    pump(0.15)
    assert [0x80, 72, 0] in port.messages
