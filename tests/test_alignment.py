# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the alignment run layer: its parameters, the windows, and the child entry."""

from __future__ import annotations

import queue

import numpy as np
import pytest
import soundfile

from namioto.analysis import alignment, qwen_aligner, qwen_asr

AUDIO = "/tmp/song.wav"


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return tmp_path


def drain(channel) -> list[tuple]:
    messages = []
    while True:
        try:
            messages.append(channel.get_nowait())
        except queue.Empty:
            return messages


def test_every_parameter_is_a_usable_field() -> None:
    names = [item.name for item in alignment.PARAMETERS]
    assert len(names) == len(set(names))
    for item in alignment.PARAMETERS:
        assert item.caption
        assert len(item.labels) in (0, len(item.choices)), item.name
        assert item.kind != "choice" or item.default in item.choices, item.name
        if item.kind in ("int", "float"):
            assert item.high > item.low, item.name
            assert item.low <= item.default <= item.high, item.name


def test_the_languages_are_the_ones_the_model_names() -> None:
    offered = next(item for item in alignment.PARAMETERS if item.name == "language")
    assert offered.choices == qwen_aligner.LANGUAGES


def test_a_bad_parameter_falls_back_to_its_default() -> None:
    values = alignment.coerce_parameters({"language": "Klingon", "quantize": 3, "chunk_seconds": 9999})
    assert values["language"] == "Japanese"
    assert values["quantize"] == 0
    assert values["chunk_seconds"] == 280
    assert alignment.coerce_parameters("nonsense") == alignment.default_parameters()


def test_the_parameters_round_trip() -> None:
    values = alignment.default_parameters()
    values.update(language="Chinese", device="gpu", chunk_seconds=120, quantize=4)
    alignment.save_parameters(values)

    loaded = alignment.load_parameters()
    assert loaded["language"] == "Chinese"
    assert loaded["device"] == "gpu"
    assert loaded["chunk_seconds"] == 120
    assert loaded["quantize"] == 4


def test_a_window_never_exceeds_the_models_ceiling() -> None:
    words = ["a", "b", "c", "d"]
    starts = [0.0, 10.0, 100.0, 260.0]
    cuts = alignment.windows(words, starts, 240)

    assert [(first, last) for first, last, _start, _end in cuts] == [(0, 3), (3, 4)]
    for _first, _last, start, end in cuts:
        assert end - start <= 240


def test_a_word_is_never_split_across_windows() -> None:
    cuts = alignment.windows(["a", "b", "c"], [0.0, 30.0, 60.0], 45)
    covered = [index for first, last, _start, _end in cuts for index in range(first, last)]
    assert covered == [0, 1, 2]


def test_the_windows_need_one_start_per_word() -> None:
    with pytest.raises(ValueError):
        alignment.windows(["a", "b"], [0.0], 240)


def test_the_run_key_follows_the_words_and_leaves_quantize_out() -> None:
    values = alignment.default_parameters()
    assert alignment.run_key(AUDIO, ["き", "み"], values) == alignment.run_key(AUDIO, ["き", "み"], values)
    assert alignment.run_key(AUDIO, ["き", "み"], values) != alignment.run_key(AUDIO, ["き", "の"], values)
    assert alignment.run_key(AUDIO, ["き"], values) != alignment.run_key(
        AUDIO, ["き"], dict(values, language="Chinese")
    )
    assert alignment.run_key(AUDIO, ["き"], values) == alignment.run_key(AUDIO, ["き"], dict(values, quantize=4))


def test_a_run_is_found_by_its_exact_inputs() -> None:
    values = alignment.default_parameters()
    spans = [(0.0, 0.5), None, (1.0, 1.5)]
    alignment.save_run(AUDIO, ["き", "み", "の"], values, spans)

    assert alignment.find_run(AUDIO, ["き", "み", "の"], values) == spans
    assert alignment.find_run(AUDIO, ["き", "み"], values) is None
    assert alignment.find_run("/tmp/other.wav", ["き", "み", "の"], values) is None


def test_a_stored_run_of_another_version_is_not_reused(tmp_path) -> None:
    values = alignment.default_parameters()
    alignment.save_run(AUDIO, ["き"], values, [(0.0, 0.5)])
    path = alignment._results_path(AUDIO)
    path.write_text('{"version": 99, "runs": {}}', encoding="utf-8")

    assert alignment.find_run(AUDIO, ["き"], values) is None


def test_snapping_moves_the_onset_and_keeps_the_end() -> None:
    # at 120 BPM a 1/16 cell is 0.125 s
    assert alignment.snap_onsets([(0.51, 1.0), None], 120.0, 4) == [(0.5, 1.0), None]


def test_snapping_refuses_a_span_the_grid_would_flip() -> None:
    # the grid moves this onset to 0.5, which is past its own end, so there is no span to keep
    assert alignment.snap_onsets([(0.49, 0.5)], 120.0, 4) == [None]
    assert alignment.snap_onsets([(0.26, 0.3)], 120.0, 4) == [(0.25, 0.3)]


def test_the_child_reports_the_spans_it_found(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "song.wav"
    soundfile.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr(qwen_aligner, "resolve_model", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(qwen_aligner, "load", lambda folder, device="cpu": object())

    def fake_align(model, part, text, language):
        return [qwen_aligner.Unit(char, index * 0.5, index * 0.5 + 0.5) for index, char in enumerate(text.split())]

    monkeypatch.setattr(qwen_aligner, "align", fake_align)
    channel: queue.Queue = queue.Queue()

    alignment.align(str(audio), ["き", "み"], [0.0, 0.5], {}, channel)

    messages = drain(channel)
    assert messages[-1] == ("done", [(0.0, 0.5), (0.5, 1.0)])
    assert any(kind == "progress" for kind, *_ in messages)


def test_the_child_reports_a_missing_runtime(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "song.wav"
    soundfile.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr(qwen_aligner, "resolve_model", lambda *args, **kwargs: tmp_path)

    def missing(folder, device="cpu"):
        raise ModuleNotFoundError(qwen_aligner.MISSING)

    monkeypatch.setattr(qwen_aligner, "load", missing)
    channel: queue.Queue = queue.Queue()

    alignment.align(str(audio), ["き"], [0.0], {}, channel)

    messages = drain(channel)
    assert ("log", qwen_aligner.MISSING) in messages
    assert messages[-1][0] == "error"


def test_the_recognition_child_reports_the_text_it_heard(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "song.wav"
    soundfile.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr(qwen_asr, "resolve_model", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(qwen_asr, "load", lambda folder, device="cpu": object())
    calls: list = []

    def fake_recognize(model, path, language=""):
        calls.append((path, language))
        return "きみのこえ"

    monkeypatch.setattr(qwen_asr, "recognize", fake_recognize)
    channel: queue.Queue = queue.Queue()

    alignment.recognize(str(audio), {"asr_language": "Japanese"}, channel)

    messages = drain(channel)
    assert messages[-1] == ("done", "きみのこえ")
    assert calls == [(str(audio), "Japanese")]


def test_the_recognition_child_reports_a_missing_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(qwen_asr, "resolve_model", lambda *args, **kwargs: tmp_path)

    def missing(folder, device="cpu"):
        raise ModuleNotFoundError(qwen_asr.MISSING)

    monkeypatch.setattr(qwen_asr, "load", missing)
    channel: queue.Queue = queue.Queue()

    alignment.recognize(str(tmp_path / "song.wav"), {}, channel)

    messages = drain(channel)
    assert ("log", qwen_asr.MISSING) in messages
    assert messages[-1][0] == "error"
