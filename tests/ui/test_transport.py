# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from namioto.ui import theme
from namioto.ui.roll import PITCH_MAX, PianoRollView

from .helpers import FakeOutput, device_point, make_spectrum, pixel_at, roll_mouse, ruler_click


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


def test_playhead_updates_only_old_and_new_columns(qt_app, monkeypatch) -> None:
    view = PianoRollView()
    view.resize(800, 400)
    view.show()
    qt_app.processEvents()
    updates = []
    monkeypatch.setattr(view.viewport(), "update", lambda rect: updates.append(rect))
    try:
        for seconds in (2.0, 3.0, None):
            previous = view.playhead
            updates.clear()
            view.set_playhead(seconds)
            positions = [position for position in (previous, seconds) if position is not None]
            assert len(updates) == len(positions)
            for rect, position in zip(updates, positions, strict=True):
                x = view.mapFromScene(QPointF(view.to_beats(position), 0)).x()
                assert rect.width() == 7
                assert rect.height() == view.viewport().height()
                assert rect.contains(QPoint(x, view.viewport().height() // 2))
        updates.clear()
        view.set_playhead(None)
        assert not updates
        view.set_playhead(10000.0)
        assert not updates
    finally:
        monkeypatch.undo()
        view.close()


def test_playhead_paints_only_dirty_columns(qt_app) -> None:
    class RecordingView(PianoRollView):
        def paintEvent(self, event):
            regions.append(event.region())
            super().paintEvent(event)

    regions = []
    view = RecordingView()
    view.resize(800, 400)
    view.show()
    qt_app.processEvents()
    try:
        for seconds in (2.0, 3.0, None):
            previous = view.playhead
            regions.clear()
            view.set_playhead(seconds)
            qt_app.processEvents()
            assert regions
            positions = [position for position in (previous, seconds) if position is not None]
            for position in positions:
                x = view.mapFromScene(QPointF(view.to_beats(position), 0)).x()
                assert any(region.contains(QPoint(x, 10)) for region in regions)
            assert all(
                sum(rect.width() * rect.height() for rect in region) <= 14 * view.viewport().height()
                for region in regions
            )
    finally:
        view.close()


def test_playhead_is_drawn_at_the_play_position(window) -> None:
    window.view.set_playhead(2.0)  # 2 s at 120 BPM = 4 beats
    image = window.view.grab().toImage()
    row = window.view.viewport().mapTo(window.view, QPoint(0, window.view.viewport().height() // 2)).y()
    marks = [x for x in range(image.width()) if image.pixelColor(x, row) == theme.canvas().playhead]
    window.view.set_playhead(None)
    assert marks
    assert min(abs(x - device_point(window.view, 4.0, 0.0).x()) for x in marks) <= 1


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
