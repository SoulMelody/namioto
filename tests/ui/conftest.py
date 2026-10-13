# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from namioto import project
from namioto.channels import Channel
from namioto.lyrics import text_key
from namioto.ui import theme
from namioto.ui.app import MainWindow

from .helpers import wait_for_lyric_mapping

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


@pytest.fixture
def own_window(tmp_path, monkeypatch):
    """A window with a settings file of its own, for the tests that read or write one."""
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(tmp_path / "settings.json"))
    opened = MainWindow()
    opened.resize(1200, 720)
    yield opened
    opened.project_dirty = False  # a test may leave unsaved notes: closing must not ask about them
    opened.close()


@pytest.fixture
def lyrics_window(own_window, tmp_path):
    """A window with a project open, so the lyrics have a file to sit beside."""
    own_window.project_path = tmp_path / "song.nto"
    own_window._watch_lyrics()
    return own_window


@pytest.fixture
def midi_window(own_window, tmp_path):
    """A window with a project open: a MIDI is imported into one, never on its own."""
    own_window.project_path = tmp_path / "song.nto"
    return own_window


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
