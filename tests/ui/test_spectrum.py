# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication

from namioto.analysis.spectrum import MIDI_OFFSET, NOTE_COUNT, NoteSpectrum, stft_notes
from namioto.ui import theme
from namioto.ui.roll import CONTENT_MARGIN, LENGTH_BEATS, NOTE_VISUAL_INSET, PITCH_MAX, SPECTRUM_TOP, PianoRollView
from namioto.ui.spectrogram import SpectrumImage, SpectrumLoader

from .helpers import device_point, make_spectrum, pixel_at


def test_filtered_note_has_a_cosmetic_border_and_explanatory_tooltip(window, monkeypatch) -> None:
    view = window.view
    note = view.add_note(69, 0.0, 2.0)
    view.set_filtered_notes((note.id,))
    assert "Overlapping note" in note.toolTip()
    view._sync_channel_visuals()
    assert "excluded from lyric mapping" in note.toolTip()

    pens = []
    original = QPainter.setPen

    def capture_pen(painter, pen):
        if isinstance(pen, QPen):
            pens.append(pen)
        return original(painter, pen)

    monkeypatch.setattr(QPainter, "setPen", capture_pen)
    image = QImage(400, 200, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        painter.scale(100, 20)
        note.paint(painter, None)
    finally:
        painter.end()
    assert pens[-1].isCosmetic()
    assert pens[-1].widthF() == 2

    view.set_filtered_notes(())
    assert note.toolTip() == ""


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


def test_silent_audio_spectrum_renders_black() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        spectrum = stft_notes(np.zeros(44100, dtype=np.float32), 44100)
        image = SpectrumImage(spectrum).image(240.0, 1.0)
    assert (image.width(), image.height()) == (40, NOTE_COUNT)
    assert all(image.pixel(x, y) == 0xFF000000 for x in range(40) for y in range(NOTE_COUNT))


def test_spectrum_image_is_cached_until_the_parameters_change() -> None:
    spectrum = SpectrumImage(make_spectrum())
    first = spectrum.image(240.0, 1.0)
    assert spectrum.image(240.0, 1.0) is first
    assert spectrum.image(300.0, 1.0) is not first


def test_a_playhead_move_does_not_bake_the_spectrum_again(own_window) -> None:
    """The canvas is the expensive half, so the cursor moving must not repaint it."""
    own_window.show()
    view = own_window.view
    view.set_spectrum(make_spectrum(frames=4000))
    view.center_on(0.0, 40.5)
    QApplication.processEvents()
    renderer = view.spectrum
    baked = renderer._image
    assert baked is not None

    for step in range(40):
        view.set_playhead(view.to_seconds(step * 0.25))
        QApplication.processEvents()
    assert renderer._image is baked  # a moved cursor is drawn over the canvas, not into it

    view.gain = view.gain * 2  # a display parameter really does bake again
    view.refresh()
    QApplication.processEvents()
    assert renderer._image is not baked


@pytest.mark.parametrize(("gain", "contrast"), [(240.0, 1.0), (300.0, 0.2), (10.0, 4.0)])
def test_spectrum_frame_range_matches_full_image(gain, contrast) -> None:
    table = np.random.default_rng(0).uniform(-0.1, 1.0, (20, NOTE_COUNT)).astype(np.float32)
    original = table.copy()
    data = NoteSpectrum(table, 25.0, 44100, 8192, 1102, 440.0, 1.0)
    full = SpectrumImage(data).image(gain, contrast)
    renderer = SpectrumImage(data)
    cropped = renderer.image(gain, contrast, 3, 12)
    assert cropped == full.copy(3, 0, 9, NOTE_COUNT)
    assert renderer.image(gain, contrast, 3, 12) is cropped
    assert renderer.image(gain, contrast, 4, 13) is not cropped
    assert np.array_equal(table, original)


def test_roll_colours_only_visible_spectrum_frames(own_window) -> None:
    own_window.show()
    view = own_window.view
    spectrum = make_spectrum(frames=5000)
    spectrum.table[1003] = 9.0
    view.set_spectrum(spectrum)
    view.center_on(100.0, 40.5)
    QApplication.processEvents()
    rendered = view.grab().toImage()
    beats = 1003.5 * view.frame_width()
    assert pixel_at(view, rendered, beats, 40.5) == QColor(255, 0, 0)
    renderer = view.spectrum
    first = renderer._parameters[2]
    last = renderer._parameters[3]
    assert first > 0
    assert first % 256 == 0
    assert last % 256 == 0
    assert last - first < renderer.spectrum.frames
    image = renderer._image
    assert image.width() == last - first
    view.set_playhead(view.to_seconds(100.0))
    QApplication.processEvents()
    assert renderer._image is image
    bar = view.horizontalScrollBar()
    bar.setValue(bar.value() + 1)
    QApplication.processEvents()
    assert renderer._image is image


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

    plain = pixel_at(window.view, window.view.grab().toImage(), 2.25, 40.5)
    window.edit.pen.click()  # the spectrum steps back so a note draws attention over it
    dimmed = pixel_at(window.view, window.view.grab().toImage(), 2.25, 40.5)
    assert dimmed.lightness() < plain.lightness()

    window.edit.mode.click()
    assert pixel_at(window.view, window.view.grab().toImage(), 2.25, 40.5).lightness() == plain.lightness()
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
    assert beside != note.fill
    assert 0 < beside.red() <= 115
    window.view.clear_notes()
    window.view.set_spectrum(None)
    window.edit.mode.click()


def test_notes_have_a_light_bevel_and_dark_outline(window) -> None:
    window.edit.pen.click()  # a note is only on screen while editing
    window.view.clear_notes()
    window.view.set_spectrum(make_spectrum(frames=400, value=0.0))  # black cells, nothing else is red
    note = window.view.add_note(60, 2.0, 2.0)
    top = PITCH_MAX - note.pitch + NOTE_VISUAL_INSET
    bottom = PITCH_MAX - note.pitch + 1.0 - NOTE_VISUAL_INSET
    window.view.centerOn(QPointF(note.start + note.duration / 2, (top + bottom) / 2))
    image = window.view.grab().toImage()
    top_left = device_point(window.view, note.start, top)
    bottom_right = device_point(window.view, note.end, bottom)

    mid_x = (top_left.x() + bottom_right.x()) // 2
    column = [image.pixelColor(mid_x, y) for y in range(top_left.y(), bottom_right.y())]
    assert note.contains(QPointF(note.duration / 2, 0.01))
    assert note.rect().height() > 1.0 - 2 * NOTE_VISUAL_INSET
    outline = QColor(theme.NOTE_OUTLINE)
    assert note.edge_light in column[:2]
    assert outline in column[-2:]
    assert {colour.name() for colour in column[2:-2]} == {note.fill.name()}

    row = [image.pixelColor(x, (top_left.y() + bottom_right.y()) // 2) for x in range(top_left.x(), bottom_right.x())]
    assert note.edge_light in row[:2]
    assert outline in row[-2:]
    assert {colour.name() for colour in row[2:-2]} == {note.fill.name()}
    window.edit.mode.click()


def test_selected_notes_use_the_wavetone_highlight(window) -> None:
    window.edit.pen.click()  # a note is only on screen while editing
    window.view.clear_notes()
    note = window.view.add_note(60, 2.0, 2.0)
    top = PITCH_MAX - note.pitch + NOTE_VISUAL_INSET
    bottom = PITCH_MAX - note.pitch + 1.0 - NOTE_VISUAL_INSET
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
        theme.NOTE_OUTLINE,
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


def test_note_visual_sync_repaints_only_changed_lyrics_or_colours(own_window, monkeypatch):
    view = own_window.view
    note = view.add_note(60, 0.0, 1.0)
    calls = []
    monkeypatch.setattr(note, "update", lambda: calls.append(True))
    view._sync_channel_visuals()
    assert calls == []
    view.set_channel_field(0, color="#ff8800")
    assert note.fill.name() == "#ff8800"
    assert calls == [True]
    view._sync_channel_visuals()
    assert calls == [True]

