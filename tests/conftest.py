# SPDX-License-Identifier: AGPL-3.0-only
"""Shared test setup: the suite runs headless, so it needs no display and no session of its own.

`QT_QPA_PLATFORM=offscreen` is assigned before Qt is imported - an assignment, never `setdefault`,
because a desktop session already exports `wayland;xcb`. One `QApplication` is built for the whole
run (session-wide, autouse): building a second one takes the first one's widgets down. `rtmidi` is
swapped for a fake port and `XDG_CONFIG_HOME`/`XDG_DATA_HOME` point at a temp directory, so no test
opens a real synth or touches the developer's own settings. No test may need a display or an audio
device.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

# assigned, not setdefault: a desktop session usually exports QT_QPA_PLATFORM=wayland;xcb, and the
# tests must not care which session, or whether there is one at all
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# the suite asserts English text, so the language the interface follows the machine for must be
# decided here rather than by whoever runs it
os.environ["LANG"] = "C"
os.environ.pop("LC_ALL", None)
os.environ.pop("LC_MESSAGES", None)


@pytest.fixture(scope="session", autouse=True)
def isolated_settings(tmp_path_factory):
    """No test reads or writes the settings file of whoever is running the suite.

    Session wide, so that it is in place before the module-wide windows are built: a window reads the
    settings the moment it is constructed.
    """
    root = tmp_path_factory.mktemp("settings")
    os.environ["NAMIOTO_SETTINGS"] = str(root / "settings.json")
    os.environ["NAMIOTO_STATE"] = str(root / "state.json")
    return root / "settings.json"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path):
    """Every test gets a window state of its own: the reusable project defaults live there now, so a
    window one test closes must not seed the next window."""
    os.environ["NAMIOTO_STATE"] = str(tmp_path / "state.json")


@pytest.fixture(scope="session", autouse=True)
def isolated_directories(tmp_path_factory):
    """No test reads the machine's config or data directory: GAME's models and the transcription
    store live there, and a suite that touched them would use - and rewrite - the developer's own."""
    root = tmp_path_factory.mktemp("xdg")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    return root


@pytest.fixture(scope="session", autouse=True)
def no_blocking_dialogs():
    """Unanswered dialogs and menus fail instead of blocking the headless run.

    Instance calls dispatch through the class so tests can replace `exec` despite PySide6's
    instance-level lookup. Unpatched modal widgets raise an error naming the widget.
    """
    from PySide6.QtWidgets import QDialog, QMenu

    def refuse(self, *_args, **_kwargs):
        raise AssertionError(f"{type(self).__name__}.exec() would block a headless test: patch it")

    patcher = pytest.MonkeyPatch()
    for widget_class in (QDialog, QMenu):
        original_init = widget_class.__init__

        def initialize(self, *args, _init=original_init, **kwargs):
            _init(self, *args, **kwargs)
            self.exec = lambda *args, **kwargs: type(self).exec(self, *args, **kwargs)

        patcher.setattr(widget_class, "__init__", initialize)
        patcher.setattr(widget_class, "exec", refuse)
    yield
    patcher.undo()


class FakeMidiOut:
    """The MIDI backend, with one synth in its port list and nowhere to send to.

    It is a class because `rtmidi.MidiOut` is called as one, once per player: each gets its own. The
    port is named like the synth a desktop usually runs, since that is the name the player looks for.
    """

    opened: int | None = None

    def get_ports(self) -> list[str]:
        return ["TiMidity:Fake TiMidity port 0 128:0"]

    def get_port_name(self, index: int) -> str:
        return self.get_ports()[index]

    def open_port(self, index: int, name: str = "") -> None:
        self.opened = index

    def send_message(self, message) -> None:
        if self.opened is None:  # a real port refuses this, and so does this one
            raise RuntimeError("the port was not opened")

    def close_port(self) -> None:
        self.opened = None


@pytest.fixture(scope="session", autouse=True)
def quiet_midi():
    """No test may open a real synthesiser.

    Every window builds a player, and the external synth is the backend it prefers, so without this
    the suite would open whatever port the machine has - TiMidity, on a desktop that runs it - and
    play its notes through the user's own sound system. The whole `rtmidi` module is swapped for a
    fake one, which keeps the backend the same on every machine, whether or not MIDI is installed.
    """
    fake_midi = types.ModuleType("rtmidi")
    fake_midi.MidiOut = FakeMidiOut
    patcher = pytest.MonkeyPatch()
    patcher.setitem(sys.modules, "rtmidi", fake_midi)
    yield
    patcher.undo()
