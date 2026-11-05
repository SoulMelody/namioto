# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the settings window: the rows some machines do not get, and the tone and sync pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QTabWidget

from namioto import qtplugins
from namioto import settings as store
from namioto.ui.settings_dialog import PAGES, SettingsDialog, system_plugins_can_help


def page(window: SettingsDialog, name: str):
    """The widget the window shows for one section, found by the name `PAGES` gives it."""
    tabs = window.findChild(QTabWidget)
    return tabs.widget(PAGES.index(name))


def test_the_tone_and_the_sync_have_pages_right_after_midi() -> None:
    assert "audio" in PAGES and "mtc" in PAGES
    assert PAGES.index("audio") == PAGES.index("midi") + 1
    assert PAGES.index("mtc") == PAGES.index("audio") + 1


def test_the_audio_page_holds_the_pitch_and_ten_band_boxes() -> None:
    window = SettingsDialog(store.Settings())
    boxes = page(window, "audio").findChildren(QDoubleSpinBox)
    assert len(boxes) == 1 + 10  # the pitch, then one box per equaliser band


def test_the_audio_page_reads_a_pitch_and_a_curve_of_ten() -> None:
    window = SettingsDialog(store.Settings())
    reads = {field.name: read for section, field, read, _write in window._rows if section == "audio"}
    assert set(reads) == {"pitch_semitones", "eq_bands"}
    assert reads["pitch_semitones"]() == 0.0
    assert reads["eq_bands"]() == (0.0,) * 10
    assert isinstance(reads["eq_bands"](), tuple)


def test_the_audio_page_takes_a_curve_of_any_length() -> None:
    window = SettingsDialog(store.Settings())
    writes = {field.name: write for section, field, _read, write in window._rows if section == "audio"}
    writes["eq_bands"]((3.0, -6.0))
    assert window.values().audio.eq_bands == (3.0, -6.0) + (0.0,) * 8
    writes["eq_bands"]((1.0,) * 12)  # more than the ten the engine has: the rest is dropped
    assert window.values().audio.eq_bands == (1.0,) * 10


def test_the_mtc_page_lists_the_frame_rates_the_port_speaks() -> None:
    window = SettingsDialog(store.Settings())
    spec = store.FIELD_SPECS[("mtc", "framerate")]
    combos = [
        combo
        for combo in page(window, "mtc").findChildren(QComboBox)
        if tuple(combo.itemData(index) for index in range(combo.count())) == spec.choices
    ]
    assert len(combos) == 1
    combo = combos[0]
    assert combo.count() == len(spec.choices)
    assert [combo.itemText(index) for index in range(combo.count())] == list(spec.labels)


def test_the_mtc_page_keeps_a_frame_rate_and_two_ports() -> None:
    window = SettingsDialog(store.Settings())
    writes = {field.name: write for section, field, _read, write in window._rows if section == "mtc"}
    writes["framerate"]("29.97drop")
    writes["output_port"]("IAC Driver")
    values = window.values()
    assert values.mtc.framerate == "29.97drop"
    assert values.mtc.output_port == "IAC Driver"
    assert values.mtc.input_port == ""


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
