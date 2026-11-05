# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the window's own state file: where it sits, and what a bad one reads back as."""

from __future__ import annotations

import json
from pathlib import Path

import platformdirs
import pytest

from namioto import state as window_state


@pytest.fixture
def state_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "state.json"
    monkeypatch.setenv("NAMIOTO_STATE", str(path))
    return path


def test_the_default_path_is_the_config_directory(monkeypatch) -> None:
    monkeypatch.delenv("NAMIOTO_STATE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/namioto-state-probe")
    path = window_state.default_path()
    assert path.parent == Path(platformdirs.user_config_dir("namioto"))
    assert path.name == "state.json"
    assert str(path).startswith("/tmp/namioto-state-probe")


def test_the_environment_variable_names_the_file(state_file) -> None:
    assert window_state.default_path() == state_file
    window_state.save(window_state.State())
    assert state_file.is_file()


def test_a_round_trip_keeps_both_values(state_file) -> None:
    window_state.save(window_state.State(geometry="AAAA", last_audio_dir="/tmp/music"))
    loaded = window_state.load()
    assert loaded == window_state.State(geometry="AAAA", last_audio_dir="/tmp/music")


def test_a_missing_or_broken_file_gives_the_defaults(state_file) -> None:
    assert window_state.load() == window_state.State()

    state_file.write_text("this is not json")
    with pytest.warns(UserWarning):
        assert window_state.load() == window_state.State()

    state_file.write_text("[1, 2, 3]")
    assert window_state.load() == window_state.State()


def test_a_value_that_is_not_text_is_dropped(state_file) -> None:
    state_file.write_text(json.dumps({"geometry": 5, "last_audio_dir": None}))
    assert window_state.load() == window_state.State()


def test_the_remembered_project_defaults_round_trip(state_file) -> None:
    window_state.save(window_state.State(geometry="AAAA", project={"spectrum": {"gain": 300.0}}))
    assert window_state.load().project == {"spectrum": {"gain": 300.0}}


def test_a_project_block_that_is_not_a_section_map_is_dropped(state_file) -> None:
    state_file.write_text(json.dumps({"project": {"spectrum": "loud", "tempo": {"bpm": 93.0}}}))
    assert window_state.load().project == {"tempo": {"bpm": 93.0}}

    state_file.write_text(json.dumps({"project": [1, 2, 3]}))
    assert window_state.load().project == {}


def test_the_ab_loop_round_trips(state_file) -> None:
    window_state.save(window_state.State(loop=(1.5, 4.25)))
    assert window_state.load().loop == (1.5, 4.25)


def test_an_ab_loop_that_is_not_two_increasing_numbers_is_off(state_file) -> None:
    # the loop is session state, so an unusable one means "off" rather than a half-armed span
    for broken in ([], [1.0], [1.0, "2"], [3.0, 1.0], [2.0, 2.0], [True, False], [1.0, None]):
        state_file.write_text(json.dumps({"loop": broken}))
        assert window_state.load().loop is None, broken

    state_file.write_text(json.dumps({"loop": [1.0, float("inf")]}))
    assert window_state.load().loop is None
