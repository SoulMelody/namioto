# SPDX-License-Identifier: AGPL-3.0-only
"""GAME transcription as the editor uses it: the run's parameters, their store, and the child entry.

Qt-free on purpose, and free of `namioto.analysis.game` too: importing that would pull `onnxruntime` into the
GUI's startup path. `namioto.ui.transcription_dialog` builds its form from `PARAMETERS`, and
`transcribe` is the module-level function a spawned process runs, so a crash cannot touch the GUI.

A finished run is cached by the audio's stamp and the model's own inputs alone: the tempo, the
quantize choice and the grid offset are applied where the notes are used (`game.quantized`), so
changing any of them re-snaps a stored run instead of running the model again. The parameters are
kept at `parameter_path()`, beside `settings.json`.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
import traceback
from typing import Any

from namioto import params
from namioto.analysis import devices
from namioto.params import Field
from namioto.utils import config_dir, data_dir, file_stamp, resolved, write_json

GAME_SIZES = ("small", "medium", "large")
# the codes GAME's own config.json maps for the segmenter; 0, and so the empty code, is universal
LANGUAGE_CODES = ("", "en", "ja", "yue", "zh")
LANGUAGE_LABELS = ("Universal", "English", "Japanese", "Cantonese", "Mandarin")
PARAMETER_FILE = "transcription.json"
STORE_VERSION = 3

PARAMETERS: tuple[Field, ...] = (
    Field(
        "size",
        "choice",
        "small",
        "Model",
        "GAME's ONNX model, downloaded into the data directory on first use",
        choices=GAME_SIZES,
    ),
    Field(
        "device",
        "choice",
        "cpu",
        "Device",
        "Run on the CPU, or on the GPU backend the settings name; the settings window says which",
        choices=devices.RUN_CHOICES,
        labels=devices.RUN_LABELS,
    ),
    Field(
        "language",
        "choice",
        "",
        "Language",
        "Singing language among the ones the model knows; Universal leaves it to the model",
        choices=LANGUAGE_CODES,
        labels=LANGUAGE_LABELS,
    ),
    Field(
        "quantize",
        "choice",
        0,
        "Quantize",
        "Snap the notes to the beat grid, this many cells per quarter note",
        choices=(0, 1, 2, 4, 8, 16, 32),
        labels=("Off", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128"),
    ),
    Field(
        "estimate_phase",
        "bool",
        False,
        "Estimate phase",
        "Estimate the quantization grid offset and apply it to the project when accepting the notes",
    ),
    Field("batch_size", "int", 4, "Batch", "Chunks per inference batch", low=1, high=32),
    Field(
        "seg_threshold",
        "float",
        0.2,
        "Segment threshold",
        "Boundary decoding threshold",
        low=0.0,
        high=1.0,
        step=0.05,
        decimals=2,
    ),
    Field(
        "seg_radius",
        "float",
        0.02,
        "Segment radius",
        "Boundary decoding radius",
        suffix="s",
        low=0.005,
        high=0.2,
        step=0.005,
        decimals=3,
    ),
    Field(
        "est_threshold",
        "float",
        0.2,
        "Estimate threshold",
        "Note presence threshold",
        low=0.0,
        high=1.0,
        step=0.05,
        decimals=2,
    ),
    Field(
        "d3pm_t0",
        "float",
        0.0,
        "D3PM T0",
        "Starting T of D3PM sampling",
        low=0.0,
        high=1.0,
        step=0.05,
        decimals=2,
    ),
    Field("d3pm_steps", "int", 8, "D3PM steps", "Number of D3PM sampling steps", low=1, high=64),
    Field(
        "silence_slice",
        "bool",
        True,
        "Silence slicing",
        "Cut the audio at its silences before inference",
    ),
)


def default_parameters() -> dict[str, Any]:
    return params.defaults(PARAMETERS)


def coerce_parameters(values: Any) -> dict[str, Any]:
    return params.coerce_values(PARAMETERS, values)


def parameter_path() -> pathlib.Path:
    """The preferences side of the tree: the dialog's remembered values sit next to settings.json."""
    return config_dir(PARAMETER_FILE)


def results_root() -> pathlib.Path:
    """The data side, where the models already live: transcriptions are derived, not preferences."""
    return data_dir("transcriptions")


def load_parameters() -> dict[str, Any]:
    """What the dialog opens with: the last run's values, or the defaults for anything unreadable."""
    return params.load_values(parameter_path(), PARAMETERS)


def save_parameters(values: Any) -> pathlib.Path:
    return params.save_values(parameter_path(), PARAMETERS, values)


def audio_key(path: Any) -> str:
    return hashlib.sha1(str(resolved(path)).encode("utf-8")).hexdigest()


def run_key(path: Any, parameters: Any) -> str:
    """What makes two runs the same one: the audio, the device and the model's own inputs.

    The grid is not one of them: a quantised run and a raw one are the same model run, so changing
    the tempo, the quantize choice or the grid offset re-snaps the stored notes instead of asking
    the model again.
    """
    values = coerce_parameters(parameters)
    identity = {
        "audio": list(file_stamp(path)),
        "device": devices.resolve(values["device"]),
        "parameters": {
            item.name: values[item.name]
            for item in PARAMETERS
            if item.name not in ("quantize", "estimate_phase", "device")
        },
    }
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _results_path(path: Any) -> pathlib.Path:
    return results_root() / f"{audio_key(path)}.json"


def _notes(run: dict) -> list[tuple[float, float, float]] | None:
    """The notes of one stored run, or None when the record is not one that can be used."""
    notes = run.get("notes")
    if not isinstance(notes, list):
        return None
    try:
        return [(float(onset), float(offset), float(pitch)) for onset, offset, pitch in notes]
    except (TypeError, ValueError):
        return None


def load_runs(path: Any) -> dict[str, dict]:
    """Every run saved for one audio file, keyed by run key; an unreadable file means no runs.

    A file another version wrote is no runs either: what it holds came out of a pipeline that has
    since changed, and re-running the model is cheaper than trusting notes it would not produce.
    """
    try:
        data = json.loads(_results_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
        return {}
    runs = data.get("runs")
    if not isinstance(runs, dict):
        return {}
    return {key: run for key, run in runs.items() if isinstance(run, dict) and _notes(run) is not None}


def find_run(path: Any, parameters: Any) -> list[tuple[float, float, float]] | None:
    """The notes a run with these exact inputs already found, or None when there is no such run."""
    found = load_runs(path).get(run_key(path, parameters))
    return _notes(found) if found else None


def save_run(path: Any, parameters: Any, notes) -> pathlib.Path:
    """Keep one entry per run key: running the same thing again replaces what it found last time."""
    runs = load_runs(path)
    runs[run_key(path, parameters)] = {
        "at": int(time.time()),
        "parameters": coerce_parameters(parameters),
        "notes": [
            [round(float(onset), 4), round(float(offset), 4), round(float(pitch), 4)] for onset, offset, pitch in notes
        ],
    }
    payload = {"version": STORE_VERSION, "audio": str(resolved(path)), "runs": runs}
    return write_json(payload, _results_path(path))


def transcribe(path: str, parameters: dict, queue) -> None:
    """Run GAME for one audio file, reporting through `queue`: a spawned process's whole job.

    The notes come back as the model found them, before any grid: the tempo, the quantize choice
    and the grid offset are the parent's to apply, so changing any of them re-snaps a stored run
    instead of asking the model again. The messages are `("log", text)`,
    `("progress", stage, done, total)`, `("done", notes)` and `("error", traceback)`. Nothing is
    written to disk here: the parent saves what comes back, so a crash cannot leave a half-written
    result behind.
    """
    values = coerce_parameters(parameters)
    try:
        from namioto.analysis import game

        queue.put(("log", f"GAME model {values['size']}"))
        model = game.resolve_model(
            size=values["size"],
            progress=lambda done, total: queue.put(("progress", "download", done, total)),
        )
        queue.put(("log", f"model at {model}"))
        backend = game.OnnxBackend(model, provider=devices.resolve(values["device"]))
        notes = game.extract(
            backend,
            path,
            language=values["language"] or None,
            batch_size=values["batch_size"],
            seg_threshold=values["seg_threshold"],
            seg_radius=values["seg_radius"],
            est_threshold=values["est_threshold"],
            d3pm_t0=values["d3pm_t0"],
            d3pm_steps=values["d3pm_steps"],
            silence_slice=values["silence_slice"],
            progress=lambda done, total: queue.put(("progress", "parts", done, total)),
        )
        queue.put(("log", f"{len(notes)} notes"))
        queue.put(("done", notes))
    except Exception:  # the child must report a crash rather than take the queue down with it
        queue.put(("error", traceback.format_exc()))
