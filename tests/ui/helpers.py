# SPDX-License-Identifier: AGPL-3.0-only
"""Shared UI gestures and fakes.

Offscreen, QTest.mouseMove delivers nothing, so send events to the handlers instead.
Shortcuts need QApplication.setActiveWindow(window) and keys sent to window.view.
QWidget.grab() still renders, so pixel checks work.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from namioto import project
from namioto.analysis.spectrum import NOTE_COUNT, NoteSpectrum
from namioto.channels import Channel
from namioto.lyrics import text_key
from namioto.ui.controls import Cluster
from namioto.ui.roll import PianoRollView


def wheel(widget, notches: int) -> None:
    """Turn the wheel over a widget, as a driver would; a notch scrolls that widget's own axis."""
    centre = QPointF(widget.rect().center())
    event = QWheelEvent(
        centre,
        widget.mapToGlobal(centre),
        QPoint(0, 0),
        QPoint(0, notches),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    widget.wheelEvent(event)


def roll_mouse(
    window, kind, scene_pos: QPointF, modifiers=Qt.KeyboardModifier.NoModifier, button=Qt.MouseButton.LeftButton
) -> None:
    """Send a mouse event to the roll at a scene position, as a real click would arrive."""
    roll_mouse_here(window, kind, window.view.mapFromScene(scene_pos), modifiers, button)


def roll_mouse_here(
    window, kind, position: QPoint, modifiers=Qt.KeyboardModifier.NoModifier, button=Qt.MouseButton.LeftButton
) -> None:
    """Send a mouse event to the roll at a position on the screen, where a pointer stays put."""
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


def drag_sound(window, ref, x, y=15.0):
    rect = window.sound_strip._blocks()[ref]
    sound_mouse(window, QEvent.Type.MouseButtonPress, rect.center().x(), rect.center().y())
    sound_mouse(window, QEvent.Type.MouseMove, x, y)
    sound_mouse(window, QEvent.Type.MouseButtonRelease, x, y)
    wait_for_lyric_mapping(window)


def sound_mouse(window, kind, x: float, y: float = 10.0, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    """Send a mouse event to the lyrics strip, whose columns line up with the roll's."""
    position = QPointF(x, y)
    event = QMouseEvent(
        kind,
        position,
        window.sound_strip.mapToGlobal(position),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )
    if kind == QEvent.Type.MouseButtonPress:
        window.sound_strip.mousePressEvent(event)
    elif kind == QEvent.Type.MouseMove:
        window.sound_strip.mouseMoveEvent(event)
    else:
        window.sound_strip.mouseReleaseEvent(event)


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


def make_spectrum(frames: int = 4, value: float = 1.0) -> NoteSpectrum:
    table = np.full((frames, NOTE_COUNT), value, dtype=np.float32)
    return NoteSpectrum(table=table, frame_ms=50.0, sample_rate=44100, fft_points=8192, hop=2205, a4=440.0, sigma=1.0)


def device_point(view: PianoRollView, scene_x: float, scene_y: float) -> QPoint:
    return view.viewport().mapTo(view, view.mapFromScene(QPointF(scene_x, scene_y)))


def pixel_at(view: PianoRollView, image: QImage, scene_x: float, scene_y: float) -> QColor:
    return image.pixelColor(device_point(view, scene_x, scene_y))


def wait_for_lyric_mapping(window) -> None:
    deadline = time.monotonic() + 1.0
    while window._lyric_map_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    assert window._lyric_map_thread is None


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


def parameter_writer(dialog, name: str):
    """The write half of one row of a dialog's parameter form, to change it the way a widget would."""
    for field, _read, write in dialog._fields:
        if field.name == name:
            return write
    raise AssertionError(f"this dialog has no {name} field")


def reset_channels(window) -> None:
    window.view.clear_notes()
    window.view.set_channels((Channel(channel=0),))


def _mapped_pair(own_window, tmp_path, raw=((0.0, 1.0), (1.0, 1.0))) -> None:
    """A one-to-one mapping of あ, い onto two notes, as the context-menu tests start from."""
    own_window.transport.bpm.setValue(60.0)
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あい", key=text_key("あい"), raw=(raw,))
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)


class _ManualLyricMapper(QObject):
    mapped = Signal(object)
    failed = Signal(str)
    finished = Signal()
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
