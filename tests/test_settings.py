# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the settings model: the preferences, their file, and how a bad value is brought back."""

from __future__ import annotations

import json
from pathlib import Path

import platformdirs
import pytest

from namioto import params
from namioto import settings as store
from namioto.analysis.game import models_root
from namioto.channels import GM_PROGRAMS, PROGRAM_LABELS


@pytest.fixture
def settings_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "settings.json"
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(path))
    return path


def test_missing_file_gives_the_defaults(settings_file) -> None:
    loaded = store.load()
    assert store.to_dict(loaded) == store.to_dict(store.Settings())
    assert store.get_value(loaded, "editor", "auto_page") is False
    assert store.get_value(loaded, "midi", "wavetone") is True


def test_the_default_path_is_the_config_directory(monkeypatch) -> None:
    monkeypatch.delenv("NAMIOTO_SETTINGS", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/namioto-config-probe")
    path = store.default_path()
    assert path.parent == Path(platformdirs.user_config_dir("namioto"))
    assert path.name == "settings.json"
    assert str(path).startswith("/tmp/namioto-config-probe")


def test_the_settings_do_not_live_with_the_models(settings_file) -> None:
    assert store.default_path() != models_root()
    assert not str(store.default_path()).startswith(str(models_root()))


def test_the_environment_variable_names_the_file(settings_file) -> None:
    assert store.default_path() == settings_file
    store.save(store.Settings())
    assert settings_file.is_file()


def test_a_round_trip_keeps_every_value(settings_file) -> None:
    original = store.Settings()
    store.set_value(original, "general", "auto_save", True)
    store.set_value(original, "general", "language", "zh")
    store.set_value(original, "devices", "gpu", "cuda")
    store.set_value(original, "tempo", "window_seconds", 8.0)
    store.set_value(original, "network", "proxy", "http://127.0.0.1:7890")
    store.set_value(original, "midi", "wavetone", False)
    store.set_value(original, "editor", "auto_page", True)
    store.save(original)

    loaded = store.load()
    assert store.to_dict(loaded) == store.to_dict(original)
    assert json.loads(settings_file.read_text())["version"] == store.VERSION
    assert store.get_value(loaded, "network", "proxy") == "http://127.0.0.1:7890"
    assert store.get_value(loaded, "midi", "wavetone") is False


def test_unknown_keys_are_dropped_and_missing_ones_default(settings_file) -> None:
    settings_file.write_text(json.dumps({"version": 1, "general": {"auto_save": True}, "nonesuch": {"a": 1}}))
    loaded = store.load()
    assert store.get_value(loaded, "general", "auto_save") is True
    assert store.get_value(loaded, "tempo", "window_seconds") == 12.0
    assert not hasattr(loaded, "nonesuch")
    assert "nonesuch" not in store.to_dict(loaded)


def test_a_broken_file_is_reported_and_the_defaults_are_used(settings_file) -> None:
    settings_file.write_text('{"tempo": {"window_seconds": ')
    with pytest.warns(UserWarning):
        loaded = store.load()
    assert store.to_dict(loaded) == store.to_dict(store.Settings())

    settings_file.write_text("[1, 2, 3]")
    assert store.to_dict(store.load()) == store.to_dict(store.Settings())


def test_values_of_the_wrong_type_fall_back_one_by_one(settings_file) -> None:
    settings_file.write_text(
        json.dumps(
            {
                "devices": {"gpu": 5, "power": None},
                "tempo": {"window_seconds": "long"},
                "editor": {"auto_page": 0},
            }
        )
    )
    loaded = store.load()
    assert store.get_value(loaded, "devices", "gpu") == ""
    assert store.get_value(loaded, "devices", "power") == "high-performance"
    assert store.get_value(loaded, "tempo", "window_seconds") == 12.0
    assert store.get_value(loaded, "editor", "auto_page") is False


def test_values_out_of_range_are_brought_back_in(settings_file) -> None:
    settings_file.write_text(json.dumps({"tempo": {"window_seconds": -5.0, "window_hop_seconds": 1000.0}}))
    loaded = store.load()
    assert store.get_value(loaded, "tempo", "window_seconds") == 2.0
    assert store.get_value(loaded, "tempo", "window_hop_seconds") == 60.0


def test_values_snap_to_the_step_they_are_shown_on(settings_file) -> None:
    settings_file.write_text(json.dumps({"tempo": {"window_seconds": 12.6}}))
    loaded = store.load()
    assert store.get_value(loaded, "tempo", "window_seconds") == 13.0


def test_text_is_trimmed_and_bounded(settings_file) -> None:
    settings_file.write_text(json.dumps({"network": {"proxy": "  " + "x" * 9999 + "  "}}))
    loaded = store.load()
    assert len(store.get_value(loaded, "network", "proxy")) == params.TEXT_LIMIT

    settings_file.write_text(json.dumps({"network": {"proxy": None}}))
    assert store.get_value(store.load(), "network", "proxy") == ""


def test_saving_leaves_nothing_half_written(settings_file) -> None:
    store.save(store.Settings())
    first = store.Settings()
    store.save(first)
    second = store.Settings()
    store.set_value(second, "tempo", "window_seconds", 20.0)
    store.save(second)

    assert [path.name for path in settings_file.parent.iterdir()] == ["settings.json"]
    assert store.get_value(store.load(), "tempo", "window_seconds") == 20.0


def test_a_clone_can_be_edited_without_touching_the_original() -> None:
    original = store.Settings()
    edited = store.clone(original)
    store.set_value(edited, "general", "auto_save", True)
    store.set_value(edited, "devices", "gpu", "cuda")
    assert store.get_value(original, "general", "auto_save") is False
    assert store.get_value(original, "devices", "gpu") == ""


def test_the_program_list_names_every_general_midi_preset() -> None:
    assert len(GM_PROGRAMS) == 128
    assert len(set(GM_PROGRAMS)) == 128  # one name per program, or the list cannot be picked from
    assert GM_PROGRAMS[0] == "Acoustic Grand Piano"
    assert GM_PROGRAMS[40] == "Violin"
    assert GM_PROGRAMS[-1] == "Gunshot"
    assert PROGRAM_LABELS[0] == "0: Acoustic Grand Piano"
    assert PROGRAM_LABELS[40] == "40: Violin"
    assert PROGRAM_LABELS[-1] == "127: Gunshot"
    assert [label.split(":")[0] for label in PROGRAM_LABELS] == [str(index) for index in range(128)]


def test_a_choice_that_is_not_one_of_them_falls_back(settings_file) -> None:
    settings_file.write_text(json.dumps({"tempo": {"estimator": "magic"}}))
    assert store.get_value(store.load(), "tempo", "estimator") == "wavetone"


def test_a_caption_that_names_its_unit_does_not_repeat_it_in_the_field() -> None:
    for spec in store.FIELD_SPECS.values():
        unit = spec.suffix.strip()
        assert not unit or unit not in spec.caption, f"{spec.name} says its unit twice"


def test_every_spec_field_is_a_field_of_its_section() -> None:
    defaults = store.Settings()
    for section in store.SECTIONS:
        for item in section.fields:
            assert store.get_value(defaults, section.name, item.name) == item.default
            assert item.caption
            assert len(item.labels) in (0, len(item.choices)), item.name
            if item.kind in ("int", "float"):
                assert item.high > item.low, item.name  # a range nobody can be inside of
                assert item.low <= item.default <= item.high, item.name
    assert len(store.FIELD_SPECS) == sum(len(section.fields) for section in store.SECTIONS)
