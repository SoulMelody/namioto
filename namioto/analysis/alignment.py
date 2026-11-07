# SPDX-License-Identifier: AGPL-3.0-only
"""Lyric alignment as the editor uses it: the run's parameters, their store, and the child entries.

Qt-free on purpose, and free of `namioto.analysis.qwen_aligner` too: importing that would be
harmless, but the models are reached only in the spawned process, which is also where the weights
are fetched and `torch` is imported, so a crash cannot touch the GUI and opening the editor never
loads a model.

Two jobs live here because they are the two halves of one window. `align` puts a time on words the
user already has; `recognize` writes the words themselves, for a lyric nobody has typed yet. Only
the alignment is remembered in a store: a recognized text lands in the window's box where the user
edits it, so the text that gets aligned is theirs and re-running the recognizer would only undo
their corrections.

A run is keyed by the audio its alignment reads - the separated vocal when the source is a mix, the
file itself otherwise - together with the words on the notes and the parameters that change what the
model hears. The grid is not one of them: the quantize choice and the tempo are applied where the
spans are used, so changing either re-snaps a stored run instead of aligning again.

The model hears five minutes per call, so a longer file is cut into windows between words and the
windows are aligned in order; the spans of one window never start before the one before it ended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from typing import Any

from namioto import params
from namioto.analysis import devices, qwen_aligner, qwen_asr
from namioto.params import Field
from namioto.utils import config_dir, data_dir, file_stamp, resolved, write_json

LANGUAGE_CODES = qwen_aligner.LANGUAGES
# the recognizer knows more languages than the aligner, and empty is "identify it yourself"
ASR_LANGUAGE_CODES = ("", *qwen_asr.LANGUAGES)
PARAMETER_FILE = "alignment.json"
STORE_VERSION = 1
# how much audio either side of a window is handed to the model, so a word on the seam is heard
# whole by both windows it sits between
WINDOW_PAD = 0.5

PARAMETERS: tuple[Field, ...] = (
    Field(
        "language",
        "choice",
        "Japanese",
        "Language",
        "The language the lyrics are in; the model aligns text it is given, so this only picks its reading",
        choices=LANGUAGE_CODES,
    ),
    Field(
        "asr_language",
        "choice",
        "",
        "Recognition language",
        "Language for automatic lyric recognition; empty lets Qwen identify it",
        choices=ASR_LANGUAGE_CODES,
        labels=("Auto", *qwen_asr.LANGUAGES),
    ),
    Field(
        "device",
        "choice",
        "cpu",
        "Device",
        "Run on the CPU, or on a CUDA GPU; WebGPU is an ONNX provider and cannot run this model",
        choices=devices.RUN_CHOICES,
        labels=devices.RUN_LABELS,
    ),
    Field(
        "chunk_seconds",
        "int",
        240,
        "Window (s)",
        "Audio handed to the model at once; its own limit is 5 minutes",
        low=30,
        high=280,
        step=10,
    ),
    Field(
        "quantize",
        "choice",
        0,
        "Quantize",
        "Snap the aligned onsets to the beat grid, this many cells per quarter note",
        choices=(0, 1, 2, 4, 8, 16, 32),
        labels=("Off", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128"),
    ),
    Field(
        "model_dir",
        "text",
        "",
        "Weights",
        "A directory holding the checkpoint; empty uses $NAMIOTO_QWEN_ALIGNER_MODEL, then the data directory",
    ),
)


def default_parameters() -> dict[str, Any]:
    return params.defaults(PARAMETERS)


def coerce_parameters(values: Any) -> dict[str, Any]:
    return params.coerce_values(PARAMETERS, values)


def parameter_path() -> pathlib.Path:
    """The preferences side of the tree: the window's remembered values sit next to settings.json."""
    return config_dir(PARAMETER_FILE)


def results_root() -> pathlib.Path:
    """The data side, where the models already live: aligned spans are derived, not preferences."""
    return data_dir("alignments")


def load_parameters() -> dict[str, Any]:
    """What the window opens with: the last run's values, or the defaults for anything unreadable."""
    return params.load_values(parameter_path(), PARAMETERS)


def save_parameters(values: Any) -> pathlib.Path:
    return params.save_values(parameter_path(), PARAMETERS, values)


def audio_key(path: Any) -> str:
    return hashlib.sha1(str(resolved(path)).encode("utf-8")).hexdigest()


def run_key(path: Any, words: Sequence[str], parameters: Any) -> str:
    """What makes two runs the same one: the audio, the words, and what the model is told.

    The quantize choice is not one of them, and neither is the grid offset: a quantised run and a
    raw one are the same run, so re-snapping a stored one costs nothing.
    """
    values = coerce_parameters(parameters)
    identity = {
        "audio": list(file_stamp(path)),
        "words": list(words),
        "model": qwen_aligner.REVISION,
        "parameters": {
            item.name: values[item.name]
            for item in PARAMETERS
            if item.name not in ("quantize", "model_dir", "asr_language")
        },
    }
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _results_path(path: Any) -> pathlib.Path:
    return results_root() / f"{audio_key(path)}.json"


def _spans(run: dict) -> list[tuple[float, float] | None] | None:
    """The spans of one stored run, or None when the record is not one that can be used."""
    spans = run.get("spans")
    if not isinstance(spans, list):
        return None
    try:
        return [None if span is None else (float(span[0]), float(span[1])) for span in spans]
    except (TypeError, ValueError, IndexError):
        return None


def load_runs(path: Any) -> dict[str, dict]:
    """Every run saved for one audio file, keyed by run key; an unreadable file means no runs.

    A file another version wrote is no runs either: what it holds came out of a pipeline that has
    since changed, and re-running the model is cheaper than trusting spans it would not produce.
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
    return {key: run for key, run in runs.items() if isinstance(run, dict) and _spans(run) is not None}


def find_run(path: Any, words: Sequence[str], parameters: Any) -> list[tuple[float, float] | None] | None:
    """The spans an identical run already found, or None when there is no such run."""
    found = load_runs(path).get(run_key(path, words, parameters))
    return _spans(found) if found else None


def save_run(path: Any, words: Sequence[str], parameters: Any, spans) -> pathlib.Path:
    """Keep one entry per run key: running the same thing again replaces what it found last time."""
    runs = load_runs(path)
    runs[run_key(path, words, parameters)] = {
        "at": int(time.time()),
        "parameters": coerce_parameters(parameters),
        "words": list(words),
        "spans": [None if span is None else [round(float(span[0]), 4), round(float(span[1]), 4)] for span in spans],
    }
    payload = {"version": STORE_VERSION, "audio": str(resolved(path)), "runs": runs}
    return write_json(payload, _results_path(path))


def snap_onsets(spans, tempo: float, cells: int, phase: float = 0.0):
    """Put every onset on the grid, leaving each aligned end where the model put it.

    A span the grid would push past its own end is dropped rather than flipped: the note it belongs
    to then keeps the time it has, which is what a word the aligner could not place means.
    """
    unit = (60.0 / tempo) / cells
    out = []
    for span in spans:
        if span is None:
            out.append(None)
            continue
        start, end = span
        snapped = unit * round((start - phase) / unit) + phase
        out.append(None if snapped >= end else (snapped, end))
    return out


def windows(words: Sequence[str], starts: Sequence[float], chunk_seconds: float) -> list[tuple[int, int, float, float]]:
    """Cut the words into the windows the model is handed: `(first, last, audio_start, audio_end)`.

    A window is a run of words whose first starts within `chunk_seconds` of the window's start, so a
    word is never split across two calls, and the audio it covers never exceeds `chunk_seconds`
    either: the model's own ceiling is five minutes per call. The caller clamps the ranges to the
    file's length and pads them, so a word on a seam is heard whole by both windows.
    """
    if not words:
        return []
    ordered = [float(start) for start in starts]
    if len(ordered) != len(words):
        raise ValueError(f"{len(words)} words need {len(words)} starts, got {len(ordered)}")
    found: list[tuple[int, int, float, float]] = []
    first = 0
    for index, start in enumerate(ordered):
        if index and start - ordered[first] >= chunk_seconds:
            found.append((first, index, ordered[first], min(ordered[first] + chunk_seconds, start)))
            first = index
    found.append((first, len(words), ordered[first], ordered[first] + chunk_seconds))
    return found


def align(path: str, words: Sequence[str], starts: Sequence[float], parameters: dict, queue) -> None:
    """Align one audio file's words, reporting through `queue`: a spawned process's whole job.

    The messages are `("log", text)`, `("progress", stage, done, total)`, `("done", spans)` and
    `("error", traceback)`. `spans` is one `(start, end)` or `None` per word, in the order given, so
    the caller can tell a word the model heard somewhere from one it could not place at all.
    """
    values = coerce_parameters(parameters)
    try:
        try:
            queue.put(("log", f"Qwen3-ForcedAligner, {values['language']}"))
            folder = qwen_aligner.resolve_model(
                values["model_dir"] or None,
                progress=lambda done, total: queue.put(("progress", "weights", done, total)),
            )
            queue.put(("log", f"weights at {folder}"))
            model = qwen_aligner.load(folder, device=devices.resolve(values["device"]))
            found = _align_windows(model, path, words, starts, values, queue)
            placed = sum(1 for span in found if span is not None)
            queue.put(("log", f"{placed} of {len(words)} words placed"))
            queue.put(("done", found))
        except ModuleNotFoundError:
            # the traceback alone would not say what to do, so the window gets the sentence first
            queue.put(("log", qwen_aligner.MISSING))
            raise
    except Exception:  # the child must report a crash rather than take the queue down with it
        queue.put(("error", traceback.format_exc()))


def _align_windows(model, path: str, words, starts, values, queue) -> list[tuple[float, float] | None]:
    """The whole file as one list of spans: each window aligned, then put in the order given."""
    import librosa
    import soundfile

    waveform, rate = librosa.load(path, sr=None, mono=True)
    if waveform.size == 0:
        return [None] * len(words)
    duration = float(waveform.size) / float(rate)
    cuts = windows(words, starts, values["chunk_seconds"])
    spans: list[tuple[float, float] | None] = [None] * len(words)
    previous_end = 0.0
    with tempfile.TemporaryDirectory(prefix="namioto-align-") as staging:
        for index, (first, last, begin, end) in enumerate(cuts):
            end = min(duration, end)
            low = max(0.0, begin - WINDOW_PAD)
            high = min(duration, end + WINDOW_PAD)
            if high <= low:  # a window past the end of the audio has nothing to hear
                queue.put(("progress", "chunks", index + 1, len(cuts)))
                continue
            part = pathlib.Path(staging) / f"part{index:03d}.wav"
            soundfile.write(part, waveform[int(low * rate) : int(high * rate)], rate)
            try:
                units = qwen_aligner.align(model, part, " ".join(words[first:last]), values["language"])
            finally:
                part.unlink(missing_ok=True)
            units = [unit._replace(start=unit.start + low, end=unit.end + low) for unit in units]
            for offset, span in enumerate(qwen_aligner.match_units(units, words[first:last])):
                if span is None:
                    continue
                start, stop = span
                start = max(start, previous_end)
                if stop <= start:
                    continue
                spans[first + offset] = (start, stop)
                previous_end = stop
            queue.put(("progress", "chunks", index + 1, len(cuts)))
    return spans


def recognize(path: str, parameters: dict, channel) -> None:
    """Recognize the lyric in a spawned process and report text or a traceback through `channel`."""
    values = coerce_parameters(parameters)
    try:
        try:
            channel.put(("log", f"{qwen_asr.MODEL}, {values['asr_language'] or 'auto'}"))
            folder = qwen_asr.resolve_model(
                values["model_dir"] or None,
                progress=lambda done, total: channel.put(("progress", "weights", done, total)),
            )
            channel.put(("log", f"weights at {folder}"))
            model = qwen_asr.load(folder, device=devices.resolve(values["device"]))
            channel.put(("progress", "recognizing", 0, 0))
            text = qwen_asr.recognize(model, path, values["asr_language"])
            channel.put(("log", f"{len(text)} characters recognized"))
            channel.put(("done", text))
        except ModuleNotFoundError:
            # the traceback alone would not say what to do, so the window gets the sentence first
            channel.put(("log", qwen_asr.MISSING))
            raise
    except Exception:  # the child must report a crash rather than take the queue down with it
        channel.put(("error", traceback.format_exc()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="namioto-cli align",
        description="Put a time on every word of a lyric with Qwen3-ForcedAligner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", help="audio file to align against; a separated vocal works best")
    parser.add_argument(
        "--words",
        required=True,
        help="the words to place, one per note, separated by spaces (quote the whole list)",
    )
    parser.add_argument(
        "--starts",
        help="comma-separated start of each word in seconds, used only to cut the windows; "
        "empty spreads them evenly over the file",
    )
    parser.add_argument("--language", choices=LANGUAGE_CODES, default="Japanese", help="language of the lyric")
    parser.add_argument("--device", choices=devices.RUN_CHOICES, default="cpu", help="where the model runs")
    parser.add_argument("--chunk-seconds", type=int, default=240, help="audio handed to the model at once")
    parser.add_argument("--model", type=pathlib.Path, help="directory holding the checkpoint")
    parser.add_argument("--json", action="store_true", help="print the spans as JSON")
    return parser.parse_args(argv)


def report_download(done: int, total: int) -> None:
    """One line per file, so a fetch does not look like a hang."""
    print(f"  {done}/{total} files", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    words = args.words.split()
    if not words:
        print("No words to align", file=sys.stderr)
        return 2
    parameters = coerce_parameters(
        {
            "language": args.language,
            "device": args.device,
            "chunk_seconds": args.chunk_seconds,
            "model_dir": str(args.model) if args.model else "",
        }
    )
    try:
        folder = qwen_aligner.resolve_model(args.model, progress=report_download)
    except (OSError, ValueError) as error:
        print(f"No weights to use: {error}", file=sys.stderr)
        return 2
    model = qwen_aligner.load(folder, device=devices.resolve(parameters["device"]))
    starts = _starts(args.starts, words, args.audio)
    found = _align_windows(
        model,
        args.audio,
        words,
        starts,
        parameters,
        _SilentQueue(),
    )
    if args.json:
        payload = {
            "words": words,
            "spans": [None if span is None else {"onset": span[0], "offset": span[1]} for span in found],
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        placed = sum(1 for span in found if span is not None)
        print(f"{placed} of {len(words)} words placed")
        for word, span in zip(words, found, strict=True):
            print(f"  {word:12} {'—' if span is None else f'{span[0]:8.3f} -> {span[1]:8.3f} s'}")
    return 0


class _SilentQueue:
    """A queue-shaped sink, so the CLI can call the same worker the child does."""

    def put(self, _message) -> None:
        return None


def _starts(value: str | None, words: Sequence[str], audio: str) -> list[float]:
    """The coarse cut points: the ones given, or an even spread over the file."""
    if value:
        found = [float(item) for item in value.split(",") if item.strip()]
        if len(found) != len(words):
            raise ValueError(f"{len(words)} words need {len(words)} starts, got {len(found)}")
        return found
    import librosa

    duration = float(librosa.get_duration(path=audio))
    step = duration / max(1, len(words))
    return [index * step for index in range(len(words))]


if __name__ == "__main__":
    raise SystemExit(main())
