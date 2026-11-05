# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the transcription side of GAME: its parameters, the run store and the child entry."""

from __future__ import annotations

import json
import pathlib
import queue

import pytest

import namioto.analysis.game as game
from namioto.analysis import devices, transcription

AUDIO = "/tmp/song.wav"


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return tmp_path


@pytest.fixture
def fake_game(monkeypatch):
    """GAME replaced by canned answers, so no test loads a model."""
    calls: dict = {}

    def resolve_model(size=None, progress=None):
        calls["size"] = size
        return pathlib.Path("/models") / str(size)

    def extract(backend, path, progress=None, **kwargs):
        calls["extract"] = {"path": path, **kwargs}
        progress(1, 2)
        progress(2, 2)
        return [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)]

    def backend(model, provider="cpu"):
        calls["backend"] = provider
        return object()

    monkeypatch.setattr(game, "resolve_model", resolve_model)
    monkeypatch.setattr(game, "OnnxBackend", backend)
    monkeypatch.setattr(game, "extract", extract)
    return calls


def drain(channel) -> list[tuple]:
    messages = []
    while True:
        try:
            messages.append(channel.get_nowait())
        except queue.Empty:
            return messages


def test_the_model_sizes_are_the_ones_game_publishes() -> None:
    assert transcription.GAME_SIZES == game.MODEL_SIZES


def test_the_devices_are_the_ones_the_registry_names() -> None:
    assert tuple(game.PROVIDERS) == devices.KEYS


def test_every_parameter_is_a_usable_field() -> None:
    names = [item.name for item in transcription.PARAMETERS]
    assert len(names) == len(set(names))
    for item in transcription.PARAMETERS:
        assert item.caption
        assert len(item.labels) in (0, len(item.choices)), item.name
        assert item.kind != "choice" or item.default in item.choices, item.name
        if item.kind in ("int", "float"):
            assert item.high > item.low, item.name
            assert item.low <= item.default <= item.high, item.name


def test_a_bad_parameter_falls_back_to_its_default() -> None:
    values = transcription.coerce_parameters({"size": "tiny", "quantize": 3, "batch_size": 999})
    assert values["size"] == "small"
    assert values["quantize"] == 0
    assert values["batch_size"] == 32
    assert transcription.coerce_parameters("nonsense") == transcription.default_parameters()


def test_the_parameters_round_trip() -> None:
    values = transcription.default_parameters()
    values.update(size="large", language="zh", quantize=4, seg_radius=0.05)
    transcription.save_parameters(values)

    loaded = transcription.load_parameters()
    assert loaded["size"] == "large"
    assert loaded["language"] == "zh"
    assert loaded["quantize"] == 4
    assert loaded["seg_radius"] == 0.05


def test_a_run_is_found_by_its_exact_inputs() -> None:
    values = transcription.coerce_parameters({"size": "medium"})
    notes = [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)]
    transcription.save_run(AUDIO, values, notes)

    assert transcription.find_run(AUDIO, values) == notes
    assert transcription.find_run(AUDIO, dict(values, language="zh")) is None
    assert transcription.find_run("/tmp/other.wav", values) is None


def test_a_changed_audio_is_not_the_same_run(tmp_path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"one")
    values = transcription.default_parameters()
    transcription.save_run(audio, values, [(0.0, 0.5, 60.0)])

    audio.write_bytes(b"a longer take")
    assert transcription.find_run(audio, values) is None


def test_the_grid_is_not_part_of_the_run() -> None:
    raw = [(0.0, 0.5, 60.0)]
    transcription.save_run(AUDIO, transcription.default_parameters(), raw)

    # the same model run, whatever the quantize choice turns out to be; the grid is the parent's
    for values in (transcription.default_parameters(), transcription.coerce_parameters({"quantize": 4})):
        assert transcription.find_run(AUDIO, values) == raw


def test_where_the_notes_go_is_not_part_of_the_run() -> None:
    notes = [(0.0, 0.5, 60.0)]
    transcription.save_run(AUDIO, dict(transcription.default_parameters(), target="new"), notes)
    assert transcription.find_run(AUDIO, dict(transcription.default_parameters(), target="active")) == notes


def test_running_the_same_thing_again_keeps_one_entry() -> None:
    values = transcription.default_parameters()
    transcription.save_run(AUDIO, values, [(0.0, 0.5, 60.0)])
    transcription.save_run(AUDIO, values, [(0.0, 0.5, 60.0), (1.0, 1.5, 62.0)])

    runs = transcription.load_runs(AUDIO)
    assert len(runs) == 1
    assert len(transcription.find_run(AUDIO, values)) == 2


def test_a_run_file_that_makes_no_sense_is_no_runs(tmp_path) -> None:
    from namioto.utils import write_json

    target = transcription.results_root() / f"{transcription.audio_key(AUDIO)}.json"
    write_json({"version": transcription.STORE_VERSION, "runs": {"x": {"notes": "nonsense"}}}, target)
    assert transcription.load_runs(AUDIO) == {}


def test_a_run_file_another_version_wrote_is_no_runs() -> None:
    from namioto.utils import write_json

    values = transcription.default_parameters()
    assert transcription.save_run(AUDIO, values, [(0.0, 0.5, 60.0)])
    assert transcription.find_run(AUDIO, values) is not None

    target = transcription.results_root() / f"{transcription.audio_key(AUDIO)}.json"
    stored = json.loads(target.read_text())
    stored["version"] = transcription.STORE_VERSION - 1
    write_json(stored, target)

    assert transcription.load_runs(AUDIO) == {}
    assert transcription.find_run(AUDIO, values) is None


def test_transcribe_reports_progress_and_the_result(fake_game) -> None:
    channel = queue.Queue()
    transcription.transcribe(AUDIO, {"size": "medium"}, channel)
    messages = drain(channel)

    assert fake_game["size"] == "medium"
    assert ("progress", "parts", 1, 2) in messages
    assert messages[-1] == ("done", [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)], [])
    assert fake_game["extract"]["language"] is None


def test_transcribe_hands_the_notes_over_raw(fake_game) -> None:
    channel = queue.Queue()
    transcription.transcribe(AUDIO, {"quantize": 4}, channel)
    messages = drain(channel)

    # the grid is the parent's to apply, so the model's own notes come back whole
    assert messages[-1] == ("done", [(0.0, 0.5, 60.0), (0.5, 1.0, 62.0)], [])


def test_transcribe_runs_on_the_device_its_choice_resolves_to(fake_game, monkeypatch) -> None:
    monkeypatch.setattr(transcription.devices, "resolve", lambda choice: "cuda" if choice == "gpu" else "cpu")
    channel = queue.Queue()
    transcription.transcribe(AUDIO, {"device": "gpu"}, channel)

    assert fake_game["backend"] == "cuda"


def test_transcribe_falls_back_to_the_cpu_for_a_gpu_that_is_not_there(fake_game, monkeypatch) -> None:
    monkeypatch.setattr(transcription.devices, "resolve", lambda choice: "cpu")
    channel = queue.Queue()
    transcription.transcribe(AUDIO, {"device": "gpu"}, channel)

    assert fake_game["backend"] == "cpu"


def test_transcribe_reports_a_failure_instead_of_raising(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("no model here")

    monkeypatch.setattr(game, "resolve_model", broken)
    channel = queue.Queue()
    transcription.transcribe(AUDIO, {}, channel)
    messages = drain(channel)

    assert messages[-1][0] == "error"
    assert "no model here" in messages[-1][1]


def test_the_engine_is_part_of_what_makes_a_run() -> None:
    singing = transcription.default_parameters()
    instruments = dict(singing, engine="muscriptor")

    transcription.save_run(AUDIO, singing, [(0.0, 0.5, 60.0)])

    assert transcription.find_run(AUDIO, instruments) is None
    assert transcription.run_key(AUDIO, singing) != transcription.run_key(AUDIO, instruments)


def test_transcribe_runs_muscriptor_when_that_engine_is_chosen(monkeypatch) -> None:
    """The instrument engine is reached only for its own choice, and its parts come back flat."""
    calls: dict = {}

    def resolve_model(size="small", progress=None):
        calls["size"] = size
        progress(1, 1)
        return pathlib.Path("/models/muscriptor") / size

    def load(folder, device="cpu"):
        calls["device"] = device
        return object()

    def extract(model, path, progress=None, instruments=None):
        calls["path"] = path
        progress(1, 1)
        from namioto.analysis.notes import Part

        return [Part("violin", 40, ((0.0, 0.5, 60.0),)), Part("cello", 42, ((0.5, 1.0, 48.0),))]

    monkeypatch.setattr("namioto.analysis.muscriptor.resolve_model", resolve_model)
    monkeypatch.setattr("namioto.analysis.muscriptor.load", load)
    monkeypatch.setattr("namioto.analysis.muscriptor.extract", extract)
    channel = queue.Queue()
    transcription.transcribe(AUDIO, {"engine": "muscriptor", "size": "medium"}, channel)

    assert calls == {"size": "medium", "device": "cpu", "path": AUDIO}
    assert drain(channel)[-1] == ("done", [(0.0, 0.5, 60.0), (0.5, 1.0, 48.0)], [("violin", 40, [(0.0, 0.5, 60.0)]), ("cello", 42, [(0.5, 1.0, 48.0)])])
