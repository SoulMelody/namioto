# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the Qwen recognizer side: the model's directory, and what it answers."""

from __future__ import annotations

import pathlib

import pytest

from namioto.analysis import qwen_asr


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return tmp_path


class FakeResult:
    def __init__(self, text: str):
        self.text = text


class FakeModel:
    """The one call `recognize` makes, recorded so a test can read what it was told."""

    def __init__(self, text: str = " きみのこえ "):
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.text)]


def test_the_model_directory_is_the_data_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert qwen_asr.model_dir() == tmp_path / "namioto" / "models" / "qwen3-asr-0.6b"


def test_a_directory_missing_a_file_is_refused_by_name(tmp_path) -> None:
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        qwen_asr.resolve_model(folder)

    assert "model.safetensors" in str(error.value)


def test_a_complete_directory_is_used_as_it_is(tmp_path) -> None:
    folder = tmp_path / "whole"
    folder.mkdir()
    for name in qwen_asr.CHECKPOINT.files:
        (folder / name).write_bytes(b"x")

    assert qwen_asr.resolve_model(folder) == folder
    assert qwen_asr.is_installed(folder) is True


def test_the_model_is_refused_without_its_runtime(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "qwen_asr", None)

    with pytest.raises(ModuleNotFoundError) as error:
        qwen_asr.load("/models")

    assert "qwen-asr" in str(error.value)


def test_the_text_is_trimmed_and_the_language_passed_on() -> None:
    model = FakeModel()

    assert qwen_asr.recognize(model, pathlib.Path("/tmp/a.wav"), "Japanese") == "きみのこえ"
    assert model.calls == [{"audio": "/tmp/a.wav", "language": "Japanese"}]


def test_an_empty_language_lets_the_model_identify_it() -> None:
    model = FakeModel()

    qwen_asr.recognize(model, "/tmp/a.wav")

    assert model.calls == [{"audio": "/tmp/a.wav", "language": None}]


def test_an_unknown_language_is_refused_before_the_run() -> None:
    with pytest.raises(ValueError):
        qwen_asr.recognize(FakeModel(), "/tmp/a.wav", "Klingon")
