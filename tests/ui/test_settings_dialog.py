# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the one row of the settings window that has nothing to change on some machines."""

from __future__ import annotations

from pathlib import Path

from namioto import qtplugins
from namioto import settings as store
from namioto.ui.settings_dialog import SettingsDialog, system_plugins_can_help


def make_plugins(path: Path) -> Path:
    """A directory shaped like one Qt searches."""
    (path / qtplugins.PLATFORMS).mkdir(parents=True)
    return path


def row_names(monkeypatch, shown: bool) -> list[str]:
    """The rows the general page builds, with the switch's own reason for being there given."""
    monkeypatch.setattr("namioto.ui.settings_dialog.system_plugins_can_help", lambda: shown)
    window = SettingsDialog(store.Settings())
    return [field.name for _section, field, _read, _write in window._rows]


def test_the_switch_is_a_row_of_the_general_page(monkeypatch) -> None:
    assert "system_qt_plugins" in row_names(monkeypatch, shown=True)


def test_the_switch_is_left_out_where_it_could_change_nothing(monkeypatch) -> None:
    assert "system_qt_plugins" not in row_names(monkeypatch, shown=False)


def test_the_switch_is_not_offered_off_linux(monkeypatch) -> None:
    monkeypatch.setattr("namioto.ui.settings_dialog.sys.platform", "win32")
    assert system_plugins_can_help() is False


def test_the_switch_is_not_offered_when_this_qt_already_searches_the_system_directory(monkeypatch, tmp_path) -> None:
    system = make_plugins(tmp_path / "system")
    monkeypatch.setattr(qtplugins, "system_dirs", lambda: [system])
    monkeypatch.setattr("namioto.ui.settings_dialog.own_plugins_dir", lambda: system / "nested")
    assert system_plugins_can_help() is False


def test_the_switch_is_offered_when_this_qt_carries_its_own_plugins(monkeypatch, tmp_path) -> None:
    system = make_plugins(tmp_path / "system")
    monkeypatch.setattr(qtplugins, "system_dirs", lambda: [system])
    monkeypatch.setattr("namioto.ui.settings_dialog.own_plugins_dir", lambda: tmp_path / "wheel")
    assert system_plugins_can_help() is True
