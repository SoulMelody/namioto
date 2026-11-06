# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the separation side: where the stems are filed, what a run writes, and what it reports.

`pymss` is not installed in the test environment and would pull a torch build in if it were, so a
module with just the attributes `separation` reads is put in its place. That is what the child
imports, so the whole `separate` path runs without a model.
"""

from __future__ import annotations

import json
import pathlib
import queue
import sys
import types

import numpy
import pytest

from namioto.analysis import separation, transcription


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("NAMIOTO_SEPARATION_MODEL", raising=False)
    return tmp_path


@pytest.fixture
def audio(tmp_path) -> pathlib.Path:
    path = tmp_path / "song.wav"
    path.write_bytes(b"RIFF0000WAVE")
    return path


def fake_pymss(monkeypatch, results=None):
    """A stand-in for the package: what `separation` imports, and a separator that returns what it is told."""
    seen: dict[str, object] = {}
    stems = results or {"Vocals": numpy.ones((2, 4)), "Instrumental": numpy.zeros((2, 4))}

    class FakeSeparator:
        def __init__(self):
            self.config = types.SimpleNamespace(audio={"sample_rate": 44100})
            self.closed = False

        @classmethod
        def from_model_name(cls, model, **kwargs):
            seen["model"] = model
            seen["kwargs"] = kwargs
            return cls()

        def separate(self, mix, pbar=False):
            seen["mix"] = mix
            seen["pbar"] = pbar
            return dict(stems)

        def close(self):
            seen["closed"] = True

    def download_model(model, model_dir=None, **_kwargs):
        seen["download"] = (model, model_dir)
        return {"entry": None, "downloaded": [], "skipped": []}

    def load_audio(path, sr=None, mono=False):
        seen["load"] = (path, sr, mono)
        return numpy.zeros((2, 8)), sr

    def save_audio(path, data, sr, output_format, audio_params):
        seen.setdefault("saved", []).append((pathlib.Path(path).name, numpy.shape(data), sr, output_format))
        # a real file, because the manifest check is about what is on disk
        pathlib.Path(path).write_bytes(b"wav")

    module = types.ModuleType("pymss")
    module.download_model = download_model
    module.MSSeparator = FakeSeparator
    module.load_audio = load_audio
    module.save_audio = save_audio
    monkeypatch.setitem(sys.modules, "pymss", module)
    return seen


def drain(channel: queue.Queue) -> list[tuple]:
    messages = []
    while True:
        try:
            messages.append(channel.get_nowait())
        except queue.Empty:
            return messages


def write_manifest(audio: pathlib.Path, model: str, stems: dict[str, str]) -> pathlib.Path:
    """A manifest as a finished run leaves it, written by hand so a check needs no model."""
    folder = separation.stems_dir(str(audio), model)
    folder.mkdir(parents=True, exist_ok=True)
    for name in stems.values():
        (folder / name).write_bytes(b"wav")
    (folder / separation.MANIFEST).write_text(
        json.dumps(
            {
                "version": separation.STORE_VERSION,
                "audio": str(audio),
                "stamp": list(separation.file_stamp(str(audio))),
                "model": model,
                "rate": 44100,
                "stems": stems,
            }
        ),
        encoding="utf-8",
    )
    return folder


def test_an_unknown_model_is_refused_with_a_readable_message(monkeypatch, audio) -> None:
    monkeypatch.setenv("NAMIOTO_SEPARATION_MODEL", "nope.ckpt")
    channel: queue.Queue = queue.Queue()

    separation.separate(str(audio), separation.default_parameters(), channel)

    messages = drain(channel)
    assert messages[-1][0] == "error"
    assert "unknown separation model 'nope.ckpt'" in messages[-1][1]
    assert any(kind == "log" and "nope.ckpt" in text for kind, text in messages)


def test_find_stems_without_a_manifest_is_no_stems(audio) -> None:
    assert separation.find_stems(str(audio), separation.MODELS[0]) is None


def test_find_stems_after_the_audio_changed_is_no_stems(audio) -> None:
    write_manifest(audio, separation.MODELS[0], {"vocals": "vocals.wav", "instrumental": "instrumental.wav"})
    assert separation.find_stems(str(audio), separation.MODELS[0]) is not None

    audio.write_bytes(b"RIFF0000WAVE-more")  # the same path, a different file

    assert separation.find_stems(str(audio), separation.MODELS[0]) is None


def test_find_stems_with_a_missing_wav_is_no_stems(audio) -> None:
    folder = write_manifest(audio, separation.MODELS[0], {"vocals": "vocals.wav"})
    (folder / "vocals.wav").unlink()

    assert separation.find_stems(str(audio), separation.MODELS[0]) is None


def test_find_stems_returns_the_paths_of_a_good_manifest(audio) -> None:
    model = separation.MODELS[0]
    folder = write_manifest(audio, model, {"vocals": "vocals.wav", "instrumental": "instrumental.wav"})

    found = separation.find_stems(str(audio), model)

    assert found == {"vocals": folder / "vocals.wav", "instrumental": folder / "instrumental.wav"}
    assert separation.stem_key(str(audio)) in str(folder)


def test_instrumental_is_the_stem_a_two_stem_model_names() -> None:
    stems = {"vocals": numpy.ones(4), "instrumental": numpy.zeros(4)}

    name, found = separation.instrumental(stems)

    assert name == "instrumental"
    assert found is stems["instrumental"]


def test_instrumental_sums_what_is_left_after_the_voice() -> None:
    stems = {
        "vocals": numpy.full(4, 100.0),
        "bass": numpy.ones(4),
        "drums": numpy.full(4, 2.0),
        "other": numpy.full(4, 3.0),
    }

    name, found = separation.instrumental(stems)

    assert name == "instrumental"
    assert found == pytest.approx(numpy.full(4, 6.0))


def test_instrumental_with_only_vocals_names_what_it_got() -> None:
    with pytest.raises(ValueError, match="vocals"):
        separation.instrumental({"vocals": numpy.ones(4)})


def test_input_for_reads_the_stem_each_engine_hears() -> None:
    stems = {"vocals": pathlib.Path("vocals.wav"), "instrumental": pathlib.Path("instrumental.wav")}

    assert transcription.input_for("game", stems) == pathlib.Path("vocals.wav")
    assert transcription.input_for("muscriptor", stems) == pathlib.Path("instrumental.wav")


def test_input_for_a_missing_stem_names_the_engine_and_what_there_is() -> None:
    with pytest.raises(ValueError, match="game"):
        transcription.input_for("game", {"drums": pathlib.Path("drums.wav")})


def test_separate_writes_the_stems_and_the_manifest(monkeypatch, audio) -> None:
    seen = fake_pymss(monkeypatch)
    channel: queue.Queue = queue.Queue()

    separation.separate(str(audio), separation.default_parameters(), channel)

    messages = drain(channel)
    kinds = [message[0] for message in messages]
    assert kinds[-1] == "done"
    assert kinds.count("progress") >= 1
    assert messages[0][0] == "log"
    assert all(kind in ("log", "progress", "done") for kind in kinds)
    # the download reports nothing, so it says so with a zero total
    assert ("progress", "download", 0, 0) in messages
    assert seen["download"][1] == str(separation.models_root())
    assert seen["kwargs"]["download"] is False
    assert seen["kwargs"]["model_dir"] == str(separation.models_root())
    assert seen["kwargs"]["output_format"] == "wav"
    assert seen["pbar"] is False
    assert seen["closed"] is True

    model = separation.MODELS[0]
    folder = separation.stems_dir(str(audio), model)
    done = messages[-1][1]
    assert set(done) == {"vocals", "instrumental"}
    for path in done.values():
        assert pathlib.Path(path).is_file()
    manifest = json.loads((folder / separation.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["version"] == separation.STORE_VERSION
    assert manifest["model"] == model
    assert manifest["rate"] == 44100
    assert sorted(manifest["stems"]) == ["instrumental", "vocals"]
    assert separation.find_stems(str(audio), model) is not None


def test_separate_normalises_the_stem_keys(monkeypatch, audio) -> None:
    fake_pymss(monkeypatch, results={"Vocals": numpy.ones((2, 4)), "Instrumental": numpy.zeros((2, 4))})
    channel: queue.Queue = queue.Queue()

    separation.separate(str(audio), separation.default_parameters(), channel)

    assert set(drain(channel)[-1][1]) == {"vocals", "instrumental"}


def test_separate_sums_a_four_stem_model_into_an_instrumental(monkeypatch, audio) -> None:
    fake_pymss(
        monkeypatch,
        results={
            "drums": numpy.ones((2, 4)),
            "bass": numpy.ones((2, 4)),
            "other": numpy.ones((2, 4)),
            "vocals": numpy.zeros((2, 4)),
        },
    )
    channel: queue.Queue = queue.Queue()

    separation.separate(str(audio), dict(separation.default_parameters(), model=separation.MODELS[2]), channel)

    done = drain(channel)[-1][1]
    assert set(done) == {"bass", "drums", "instrumental", "other", "vocals"}


def test_a_missing_package_tells_the_user_what_to_install(monkeypatch, audio) -> None:
    monkeypatch.setitem(sys.modules, "pymss", None)  # an import of it raises ModuleNotFoundError
    channel: queue.Queue = queue.Queue()

    separation.separate(str(audio), separation.default_parameters(), channel)

    messages = drain(channel)
    assert any(kind == "log" and "namioto[separation]" in text for kind, text in messages)
    assert messages[-1][0] == "error"


def test_the_command_line_refuses_an_unknown_model(capsys) -> None:
    assert separation.main(["song.wav", "--model", "nope.ckpt"]) == 2
    assert "unknown separation model" in capsys.readouterr().err


def test_the_command_line_writes_into_the_cache(monkeypatch, audio, capsys) -> None:
    fake_pymss(monkeypatch)

    assert separation.main([str(audio)]) == 0

    printed = capsys.readouterr().out.split()
    assert len(printed) == 2
    assert all(pathlib.Path(path).is_file() for path in printed)
    assert separation.find_stems(str(audio), separation.MODELS[0]) is not None
