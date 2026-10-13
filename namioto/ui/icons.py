# SPDX-License-Identifier: AGPL-3.0-only
"""Icons: one glyph font, so the bars and the settings window draw the same shapes.

Every icon is a Material Design glyph through `qtawesome` - nothing is drawn by hand. `icon()` hands
out a `QIconEngine` that asks the palette for its shades each time it draws, so a style or palette
switch needs no walking of the widget tree.
"""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPalette, QPixmap

from namioto.ui import theme

GLYPHS = {
    "rewind": "mdi.rewind",
    "forward": "mdi.fast-forward",
    "play": "mdi.play",
    "pause": "mdi.pause",
    "playstart": "mdi.skip-previous",
    "stop": "mdi.stop",
    "pen": "mdi.pen",
    "select": "mdi.selection",
    "edit": "mdi.pencil-ruler",
    "beat": "mdi.metronome",
    "page": "mdi.book-open-page-variant",
    "overtone": "mdi.sine-wave",
    "snap": "mdi.music-note-eighth",
    "quantize": "mdi.grid",
    "check": "mdi.check",
    "cross": "mdi.close",
    "gear": "mdi.cog",
    "refresh": "mdi.refresh",
    "restore": "mdi.restore",
    "open": "mdi.folder-open",
    "save": "mdi.content-save",
    "export": "mdi.file-music-outline",
    "import": "mdi.file-import-outline",
    "loadtext": "mdi.file-document-outline",
    "external": "mdi.open-in-new",
    "translate": "mdi.translate",
    "copy": "mdi.content-copy",
    "transcribe": "mdi.auto-fix",
    "lyrics": "mdi.text-box-outline",
    "note_lyrics": "mdi.music-note-outline",
    "lyric_scope": "mdi.filter-outline",
    "align": "mdi.timeline-clock",
    "map": "mdi.arrow-decision-outline",
    "channels": "mdi.layers",
    "lock": "mdi.lock",
    "unlock": "mdi.lock-open-variant",
    "eye": "mdi.eye",
    "eyeoff": "mdi.eye-off",
    "mute": "mdi.volume-off",
    "sound": "mdi.volume-high",
}


def icon(kind: str, color: str = "") -> QIcon:
    """The glyph named by our own word for it, so a call site never spells out a font name.

    The colours are not in the icon: it paints itself in the ones the palette holds, so a button
    keeps up with the desktop's own light or dark look instead of wearing what was current when it
    was built. The disabled shade comes with it: a plain pixmap has no greyed mode of its own, so
    without it a button that is off looks exactly like one that is on.
    """
    return QIcon(_Glyph(kind, color))


def _palette_colour(group: QPalette.ColorGroup) -> str:
    """The colour the widgets around an icon wear, so the desktop's own scheme reaches the icon too."""
    return theme.running_app().palette().color(group, QPalette.ColorRole.WindowText).name()


def _drawn(kind: str, color: str = "") -> QIcon:
    """The qtawesome icon behind a glyph, built from the colours in force right now.

    qtawesome keeps its icons on the font instance it loaded them from and reloads the fonts when
    the application's font database is reset; a cache of our own would outlive that reload and keep
    painting the stale glyph as a missing-character box, so the cache is qtawesome's own, which is
    per font instance and does not survive the reload.
    """
    text = color or _palette_colour(QPalette.ColorGroup.Active)
    dim = _palette_colour(QPalette.ColorGroup.Disabled)
    return qta.icon(GLYPHS[kind], color=text, color_disabled=dim)


class _Glyph(QIconEngine):
    """An icon that asks the theme for its colours every time it is drawn."""

    def __init__(self, kind: str, color: str = ""):
        super().__init__()
        self.kind = kind
        self.color = color

    def paint(self, painter: QPainter, rect, mode: QIcon.Mode, state: QIcon.State) -> None:
        _drawn(self.kind, self.color).paint(painter, rect, Qt.AlignmentFlag.AlignCenter, mode, state)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return _drawn(self.kind, self.color).pixmap(size, mode, state)

    def actualSize(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QSize:
        return _drawn(self.kind, self.color).actualSize(size, mode)

    def clone(self) -> QIconEngine:
        return _Glyph(self.kind, self.color)
