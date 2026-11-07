# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the Qwen aligner side: the model's directory, and how its units land on words."""

from __future__ import annotations

import pathlib

import pytest

from namioto.analysis import qwen_aligner
from namioto.analysis.qwen_aligner import Unit


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("NAMIOTO_QWEN_ALIGNER_MODEL", raising=False)
    return tmp_path


def test_the_units_land_on_one_word_each() -> None:
    units = [Unit("き", 0.0, 0.4), Unit("み", 0.4, 0.9), Unit("の", 0.9, 1.4)]
    assert qwen_aligner.match_units(units, ["き", "み", "の"]) == [
        (0.0, 0.4),
        (0.4, 0.9),
        (0.9, 1.4),
    ]


def test_a_word_across_several_units_takes_the_whole_span() -> None:
    units = [Unit("き", 0.0, 0.4), Unit("み", 0.4, 0.9), Unit("の", 0.9, 1.4)]
    assert qwen_aligner.match_units(units, ["きみの"]) == [(0.0, 1.4)]


def test_a_unit_across_several_words_gives_them_its_own_span() -> None:
    # the model's tokenizer is not our word list: both words were heard across the one unit
    units = [Unit("きみ", 0.0, 1.0)]
    assert qwen_aligner.match_units(units, ["き", "み"]) == [(0.0, 1.0), (0.0, 1.0)]


def test_a_word_the_model_did_not_read_keeps_nothing() -> None:
    units = [Unit("き", 0.0, 0.4)]
    assert qwen_aligner.match_units(units, ["き", "ん"]) == [(0.0, 0.4), None]


def test_a_word_far_from_the_cursor_is_not_attached_to_it() -> None:
    # "あ" appears much later; attaching it would move the note to the wrong phrase
    units = [Unit("きみのこえがとおく", 0.0, 3.0), Unit("あ", 3.0, 3.4)]
    assert qwen_aligner.match_units(units, ["きみ", "あ"]) == [(0.0, 3.0), None]


def test_an_empty_word_is_skipped_without_consuming_anything() -> None:
    units = [Unit("き", 0.0, 0.4)]
    assert qwen_aligner.match_units(units, ["", "き"]) == [None, (0.0, 0.4)]


def test_matching_compares_letters_only() -> None:
    assert qwen_aligner.normalize(" き、み。の ") == "きみの"
    assert qwen_aligner.normalize("ＡＢ　１２") == "AB12"
    assert qwen_aligner.normalize("コーヒー") == "コーヒー"  # the long vowel mark is a letter
    assert qwen_aligner.normalize("hello, world!") == "helloworld"


def test_the_model_directory_is_the_data_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert qwen_aligner.model_dir() == tmp_path / "namioto" / "models" / "qwen3-forcedaligner-0.6b"


def test_a_directory_missing_a_file_is_refused_by_name(tmp_path) -> None:
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        qwen_aligner.resolve_model(folder, download=False)

    assert "merges.txt" in str(error.value)


def test_a_complete_directory_is_used_as_it_is(tmp_path) -> None:
    folder = tmp_path / "whole"
    folder.mkdir()
    for name in qwen_aligner.FILES:
        (folder / name).write_bytes(b"x")

    assert qwen_aligner.resolve_model(folder, download=False) == folder
    assert qwen_aligner.is_installed(folder) is True


def test_a_download_that_cannot_run_says_what_to_install(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)

    with pytest.raises((ModuleNotFoundError, ImportError)):
        qwen_aligner.install()


def test_the_model_is_refused_without_its_runtime(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "qwen_asr", None)

    with pytest.raises(ModuleNotFoundError) as error:
        qwen_aligner.load("/models")

    assert "qwen-asr" in str(error.value)


def test_an_unknown_language_is_refused_before_the_run(tmp_path) -> None:
    with pytest.raises(ValueError):
        qwen_aligner.align(object(), pathlib.Path(tmp_path / "a.wav"), "きみ", "Klingon")
