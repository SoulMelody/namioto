# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the two canvases, the palette that picks between them and the style setting."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from namioto.analysis.spectrum import NOTE_COUNT, NoteSpectrum
from namioto.ui import icons, theme
from namioto.ui.app import MainWindow
from namioto.ui.roll import RULER_HEIGHT, PianoKeyboard, PianoRollView, TimelineRuler


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture(autouse=True)
def the_desktops_colours_again(qt_app):
    """The suite's other checks read the canvas of whatever palette is in force."""
    yield
    theme.apply(qt_app)


def test_the_two_canvases_name_the_same_colours() -> None:
    assert set(theme.CANVAS["light"].__dict__) == set(theme.CANVAS["dark"].__dict__)
    assert theme.CANVAS["light"] != theme.CANVAS["dark"]


def test_the_canvas_follows_the_palette_the_desktop_handed_out(qt_app) -> None:
    colours = qt_app.palette()
    try:
        for name, window in (("light", "#ffffff"), ("dark", "#202020")):
            palette = qt_app.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(window))
            qt_app.setPalette(palette)
            assert theme.apply(qt_app) == name
            assert theme.current() == name
            assert theme.canvas() is theme.CANVAS[name]
    finally:
        qt_app.setPalette(colours)


def a_spectrum() -> NoteSpectrum:
    return NoteSpectrum(
        table=np.zeros((2, NOTE_COUNT), dtype=np.float32),
        frame_ms=50.0,
        sample_rate=44100,
        fft_points=8192,
        hop=2205,
        a4=440.0,
        sigma=1.0,
    )


def test_the_roll_uses_the_fixed_dark_canvas_once_a_spectrum_covers_it(qt_app) -> None:
    view = PianoRollView()
    saved = qt_app.palette()
    try:
        palette = qt_app.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
        qt_app.setPalette(palette)
        assert theme.apply(qt_app) == "light"
        assert view.canvas() is theme.CANVAS["light"]

        view.set_spectrum(a_spectrum())
        assert view.canvas() is theme.CANVAS["dark"]

        view.set_spectrum(None)
        assert view.canvas() is theme.CANVAS["light"]
    finally:
        qt_app.setPalette(saved)
        view.close()


def test_the_time_ruler_keeps_the_palette_canvas_over_a_spectrum(qt_app) -> None:
    view = PianoRollView()
    ruler = TimelineRuler(view)
    ruler.resize(400, RULER_HEIGHT)
    saved = qt_app.palette()
    try:
        palette = qt_app.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
        qt_app.setPalette(palette)
        assert theme.apply(qt_app) == "light"
        view.set_spectrum(a_spectrum())

        image = ruler.grab().toImage()
        tones = {image.pixelColor(x, y).name() for x in range(image.width()) for y in range(image.height())}
        assert theme.CANVAS["light"].panel.name() in tones
        assert theme.CANVAS["dark"].panel.name() not in tones
    finally:
        qt_app.setPalette(saved)
        ruler.close()
        view.close()


def test_the_keyboard_keeps_one_fixed_canvas(qt_app) -> None:
    view = PianoRollView()
    keyboard = PianoKeyboard(view)
    keyboard.resize(66, 300)
    saved = qt_app.palette()
    try:
        palette = qt_app.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#202020"))
        qt_app.setPalette(palette)
        assert theme.apply(qt_app) == "dark"

        image = keyboard.grab().toImage()
        tones = {image.pixelColor(x, y).name() for x in range(image.width()) for y in range(image.height())}
        assert theme.CANVAS["light"].key_white.name() in tones
        assert theme.CANVAS["dark"].key_white.name() not in tones
    finally:
        qt_app.setPalette(saved)
        keyboard.close()
        view.close()


def test_the_style_setting_puts_the_style_it_names_in_force(qt_app) -> None:
    colours = qt_app.palette()

    theme.apply_style("Windows")
    assert qt_app.style().objectName() == "windows"
    assert qt_app.palette() == colours, "the style draws, the desktop keeps the colours"

    theme.apply_style("")
    assert qt_app.style().objectName() == theme.platform_style(), "an empty name puts the desktop's back"


def test_an_override_does_not_become_the_remembered_desktop_style(qt_app, monkeypatch) -> None:
    theme.apply_style("Fusion")  # stand in for the style the desktop handed out
    monkeypatch.setattr(theme, "_desktop_style", "")
    assert qt_app.style().objectName() == "fusion"

    theme.apply_style("Windows")

    assert theme.platform_style() == "fusion", "the desktop's own, not the style just put in force"
    theme.apply_style("")


def test_an_icon_takes_its_colours_from_the_palette_in_force(qt_app) -> None:
    play = icons.icon("play")  # built before any palette was applied, and still follows it

    def most_of(pixmap) -> str:
        image = pixmap.toImage()
        tones: dict[str, int] = {}
        for x in range(image.width()):
            for y in range(image.height()):
                colour = image.pixelColor(x, y)
                if colour.alpha() > 200:
                    tones[colour.name()] = tones.get(colour.name(), 0) + 1
        return max(tones, key=lambda name: tones[name])

    palette = qt_app.palette()
    try:
        for colour in ("#112233", "#aabbcc"):
            palette.setColor(QPalette.ColorRole.WindowText, QColor(colour))
            palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#556677"))
            qt_app.setPalette(palette)
            assert most_of(play.pixmap(24, 24)) == colour
            off = most_of(play.pixmap(24, 24, QIcon.Mode.Disabled))
            assert off == "#556677", "a disabled button has to read as off"
    finally:
        qt_app.setPalette(palette)


def row_colours(window) -> set[str]:
    image = window.view.grab().toImage()
    return {image.pixelColor(300, y).name() for y in range(image.height())}


def test_the_window_draws_the_roll_in_the_canvas_the_palette_asks_for(qt_app) -> None:
    window = MainWindow()
    try:
        window.resize(1200, 720)
        window.show()
        qt_app.processEvents()
        assert theme.canvas().row_white.name() in row_colours(window)

        palette = qt_app.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#202020"))
        qt_app.setPalette(palette)
        window._on_color_scheme(Qt.ColorScheme.Dark)  # the platform says the desktop went dark
        assert theme.current() == "dark"
        assert theme.CANVAS["dark"].row_white.name() in row_colours(window)
        assert theme.CANVAS["light"].row_white.name() not in row_colours(window)
    finally:
        window.close()
