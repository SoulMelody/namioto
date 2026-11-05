# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the MuScriptor side: where a checkpoint is filed, and how its events become parts."""

from __future__ import annotations

import pytest

from namioto.analysis import muscriptor
from namioto.analysis.notes import Part


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("NAMIOTO_MUSCRIPTOR_MODEL", raising=False)
    return tmp_path


def test_an_unknown_size_is_refused() -> None:
    with pytest.raises(ValueError):
        muscriptor.model_dir("tiny")


def test_a_checkpoint_is_installed_only_when_both_files_are_there(tmp_path) -> None:
    folder = muscriptor.model_dir("small")
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert muscriptor.is_installed("small") is False

    (folder / "model.safetensors").write_bytes(b"weights")
    assert muscriptor.is_installed("small") is True
    assert muscriptor.resolve_model(size="small", download=False) == folder


def test_an_explicit_directory_wins_over_a_download(tmp_path) -> None:
    folder = tmp_path / "given"
    folder.mkdir()
    for name in muscriptor.MODEL_FILES:
        (folder / name).write_bytes(b"x")

    assert muscriptor.resolve_model(folder, size="large", download=False) == folder


def test_a_directory_missing_a_file_is_not_a_checkpoint(tmp_path) -> None:
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        muscriptor.resolve_model(folder, download=False)


def test_events_become_one_part_per_instrument(monkeypatch) -> None:
    class NoteStart:
        def __init__(self, instrument, pitch, start):
            self.instrument = instrument
            self.pitch = pitch
            self.start_time = start

    class NoteEnd:
        def __init__(self, start, end):
            self.start_event = start
            self.end_time = end

    class Progress:
        def __init__(self, completed, total):
            self.completed = completed
            self.total = total

    class Events:
        NoteStartEvent = NoteStart
        NoteEndEvent = NoteEnd
        ProgressEvent = Progress

    class Torch:
        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    starts = [NoteStart("violin", 60, 0.0), NoteStart("cello", 48, 0.5)]
    emitted = [Progress(1, 2), NoteEnd(starts[0], 0.5), NoteEnd(starts[1], 1.0)]

    class Model:
        def transcribe(self, path, **kwargs):
            return iter(emitted)

    monkeypatch.setitem(__import__("sys").modules, "torch", Torch)
    monkeypatch.setitem(__import__("sys").modules, "muscriptor.events", Events)
    seen = []
    parts = muscriptor.extract(Model(), "/tmp/song.wav", progress=lambda done, total: seen.append((done, total)))

    assert parts == [
        Part("violin", 40, ((0.0, 0.5, 60.0),)),
        Part("cello", 42, ((0.5, 1.0, 48.0),)),
    ]
    assert seen == [(1, 2)]


def test_an_instrument_the_model_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError):
        muscriptor.extract(object(), "/tmp/song.wav", instruments=("bagpipe",))


def test_a_single_codebook_key_is_remapped() -> None:
    assert muscriptor._remap("emb.0.weight") == "emb.weight"
    assert muscriptor._remap("linears.0.weight") == "linear.weight"
    assert muscriptor._remap("transformer.weight") == "transformer.weight"
    with pytest.raises(ValueError):
        muscriptor._remap("emb.1.weight")
