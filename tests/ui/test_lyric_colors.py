# SPDX-License-Identifier: AGPL-3.0-only
"""One project palette for lyric painting, export and colour-picker previews."""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QPushButton

from namioto import project
from namioto.lyric_colors import DEFAULT_COLORS
from namioto.ui.settings_dialog import SettingsDialog


def test_color_picker_updates_swatch_without_applying(own_window, monkeypatch):
    dialog = SettingsDialog(own_window.settings, lyric_colors=own_window.view.lyric_colors)
    editor = dialog.color_editors[0]
    picker = editor.parentWidget().findChild(QPushButton)
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#123456"))
    picker.click()
    assert editor.text() == "#123456"
    assert picker.icon().pixmap(20, 20).toImage().pixelColor(10, 10) == QColor("#123456")
    dialog.reject()
    assert own_window.view.lyric_color(0) == QColor(DEFAULT_COLORS[0])


def test_palette_is_applied_when_project_settings_are_restored(own_window):
    settings = project.default_settings()
    settings.lyric_colors.channel_3 = "#123456"
    own_window.project_settings = settings
    own_window._apply_project_settings()
    assert own_window.view.lyric_color(3) == QColor("#123456")
    exported = own_window._ass_settings()
    assert exported.channel_colours()[3] == "123456"
    from namioto.karaoke import generate_ass

    ass = generate_ass("@0{@c3{あ}}", [[(0, 1)]], settings=exported)
    assert "[3]={'563412'" in ass
    assert "\\-namioto_3_0_1000_1000" in ass
    own_window.project_settings = project.default_settings()
    own_window._apply_project_settings()
    assert own_window.view.lyric_color(3) == QColor(DEFAULT_COLORS[3])


def test_lyric_editor_uses_color_override(own_window):
    from PySide6.QtGui import QFont, QImage, QPainter

    from namioto.karaoke.parser import parse
    from namioto.ui.lyric_editor import draw_units, unit_rects

    seen = []
    own_window.view.lyric_color = lambda number: seen.append(number) or QColor("#123456")
    unit = parse("@1{@c2{あ}}\n").chapters[0].lines[0].units[0]
    image = QImage(200, 80, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        draw_units(painter, [unit], unit_rects([unit], QFont()), own_window.view, own_window.palette())
    finally:
        painter.end()
    assert seen == [2, 2]
