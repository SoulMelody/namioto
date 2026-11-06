# SPDX-License-Identifier: AGPL-3.0-only
"""Source separation as the transcription window uses it: the model's parameters, their store, and the child entry.

A mix hides the two things the editors hear: one singing voice and everything that is not it. GAME
transcribes a voice badly when the band plays over it, and MuScriptor hears the instruments best
without one, so a run can split the audio first and transcribe only the stem its engine wants.

Qt-free on purpose, and free of `pymss`, `torch`, `librosa` and ffmpeg at import: those are reached
inside `separate` and `main` only, which a spawned process or the CLI calls, so opening the editor
never pulls a torch build in. The weights are a separate work fetched from the pymss catalog into
`models_root()` on first use; nothing of them ships with this program.

The stems are derived results, so they sit in the data directory beside the transcriptions, one
directory per audio file and model. `manifest.json` is what makes a directory usable: it records the
audio's stamp, so the same path holding a different file is not the same recording, and it is
written only after every wav is there, so a killed run never looks like a finished one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import traceback
from collections.abc import Callable
from typing import Any

from namioto import params
from namioto.params import Field
from namioto.utils import config_dir, data_dir, file_stamp, resolved, write_json

# the catalog's own file names, pinned because a run's cache is keyed by them; only these were
# checked to load and to answer with the stems the two engines need
MODELS = (
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "voc_fv7.ckpt",
    "HTDemucs4.th",
)
MODEL_LABELS = (
    "BS-RoFormer (vocals / instrumental)",
    "Mel-Band RoFormer (vocals / instrumental)",
    "HTDemucs4 (four stems)",
)
PARAMETER_FILE = "separation.json"
MANIFEST = "manifest.json"
# the stem names the rest of the program speaks; a separator's own casing is normalised to these
STEM_VOCALS = "vocals"
STEM_INSTRUMENTAL = "instrumental"
STORE_VERSION = 1
# the variables that keep every BLAS in the child to one thread: a model run is already using all
# the cores it was given, and a library that decides otherwise oversubscribes the machine
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
MISSING = "pymss is not installed; run `pip install namioto[separation]`"

PARAMETERS: tuple[Field, ...] = (
    Field(
        "enabled",
        "bool",
        False,
        "Separate first",
        "Split the audio into stems before transcribing, and transcribe only the stem the engine hears best",
    ),
    Field(
        "model",
        "choice",
        MODELS[0],
        "Separation model",
        "Which separator splits the audio; only the vocals and instrumental stems are kept for transcription",
        choices=MODELS,
        labels=MODEL_LABELS,
    ),
    Field(
        "device",
        "choice",
        "auto",
        "Device",
        "Run the separator on the CPU, or on a CUDA GPU when one is available; Auto takes the GPU when it can",
        choices=("auto", "cpu", "cuda"),
        labels=("Auto", "CPU", "CUDA"),
    ),
)


def default_parameters() -> dict[str, Any]:
    return params.defaults(PARAMETERS)


def coerce_parameters(values: Any) -> dict[str, Any]:
    return params.coerce_values(PARAMETERS, values)


def parameter_path() -> pathlib.Path:
    """The preferences side of the tree: the window's remembered values sit next to settings.json."""
    return config_dir(PARAMETER_FILE)


def load_parameters() -> dict[str, Any]:
    """What the window opens with: the last run's values, or the defaults for anything unreadable."""
    return params.load_values(parameter_path(), PARAMETERS)


def save_parameters(values: Any) -> pathlib.Path:
    return params.save_values(parameter_path(), PARAMETERS, values)


def models_root() -> pathlib.Path:
    """Where the separator's weights live: this program's data directory, not the package's cache.

    Both the download and the load are pointed here, so one tree holds every model the program
    fetched, and a person clearing a model knows where to look.
    """
    return data_dir("models", "pymss")


def results_root() -> pathlib.Path:
    """The data side, where the models already live: stems are derived, not preferences."""
    return data_dir("stems")


def model_name(name: str | None = None) -> str:
    """The model a run asks for: `$NAMIOTO_SEPARATION_MODEL` wins, then the stored choice."""
    return os.environ.get("NAMIOTO_SEPARATION_MODEL") or name or MODELS[0]


def stem_key(audio: str) -> str:
    return hashlib.sha1(str(resolved(audio)).encode("utf-8")).hexdigest()


def stems_dir(audio: str, model: str) -> pathlib.Path:
    """One audio file's stems for one model: the file by its identity, the model by its name."""
    return results_root() / stem_key(audio) / _slug(model)


def _slug(model: str) -> str:
    """A checkpoint's file name made fit to be a directory's: dots stay, a path separator does not."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def find_stems(audio: str, model: str) -> dict[str, pathlib.Path] | None:
    """The stems a run of this model already wrote for this exact audio, or None when there are none.

    The manifest is the whole check: a version another store wrote, another model's directory, a
    changed audio or a missing wav all mean the directory cannot be used, and running the model
    again is cheaper than transcribing a stem of a recording that is no longer there.
    """
    target = stems_dir(audio, model)
    try:
        manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("version") != STORE_VERSION or manifest.get("model") != model:
        return None
    if list(manifest.get("stamp") or []) != list(file_stamp(audio)):
        return None
    stored = manifest.get("stems")
    if not isinstance(stored, dict) or not stored:
        return None
    folder = resolved(target)
    found: dict[str, pathlib.Path] = {}
    for name, file in stored.items():
        if not isinstance(file, str):
            return None
        path = target / file
        if not path.is_file() or resolved(path).parent != folder:
            return None
        found[str(name)] = path
    return found


def instrumental(stems: dict[str, Any]) -> tuple[str, Any]:
    """The one instrumental stem a transcription runs on, whatever the separator called its stems.

    A two-stem model names it; a four-stem one has no such stem at all, so what is left after the
    voice is summed into it, in a stable order so two runs of the same audio add up the same way.
    """
    import numpy

    if STEM_INSTRUMENTAL in stems:
        return STEM_INSTRUMENTAL, stems[STEM_INSTRUMENTAL]
    parts = [name for name in sorted(stems) if name != STEM_VOCALS]
    if not parts:
        available = ", ".join(sorted(stems)) or "nothing"
        raise ValueError(f"no instrumental stem to keep; the separator returned {available}")
    return STEM_INSTRUMENTAL, numpy.sum([stems[name] for name in parts], axis=0)


def separate(audio: str, parameters: dict, queue) -> None:
    """Separate one audio file, reporting through `queue`: a spawned process's whole job.

    The messages are `("log", text)`, `("progress", stage, done, total)`, `("done", {stem: path})`
    and `("error", traceback)`. The download has no progress to report, so its message carries a
    zero total and the window shows a bar that does not pretend to know how far it is.
    """
    values = coerce_parameters(parameters)
    model = model_name(values["model"])
    try:
        try:
            queue.put(("log", f"separation model {model}"))
            if model not in MODELS:
                raise ValueError(f"unknown separation model {model!r}, expected one of {', '.join(MODELS)}")
            rate, stems = _split(
                audio,
                model,
                values["device"],
                # the download has nothing to report, so the window shows a bar that does not pretend
                lambda: queue.put(("progress", "download", 0, 0)),
                lambda done, total, _message: queue.put(("progress", "separating", done, total)),
            )
            written = _write(audio, model, rate, stems)
            for name, path in written.items():
                queue.put(("log", f"{name} stem at {path}"))
            queue.put(("done", {name: str(path) for name, path in written.items()}))
        except ModuleNotFoundError:
            # the traceback alone would not say what to do, so the window gets the sentence first
            queue.put(("log", MISSING))
            raise
    except Exception:  # the child must report a crash rather than take the queue down with it
        queue.put(("error", traceback.format_exc()))


def _split(
    audio: str,
    model: str,
    device: str,
    notice: Callable[[], None],
    progress: Callable[[int, int, str], None],
) -> tuple[int, dict[str, Any]]:
    """The audio as a sample rate and its stems: the model's own step, shared by the child and the CLI.

    `notice` is called once the weights are on disk, because the download itself reports nothing:
    the caller turns that into the message its own audience reads.
    """
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    import pymss

    pymss.download_model(model, model_dir=str(models_root()))
    notice()
    separator = pymss.MSSeparator.from_model_name(
        model,
        model_dir=str(models_root()),
        download=False,
        device=device,
        output_format="wav",
        progress_callback=progress,
    )
    try:
        rate = int(separator.config.audio.get("sample_rate", 44100))
        mix, _ = pymss.load_audio(audio, sr=rate, mono=False)
        results = separator.separate(mix, pbar=False)
    finally:
        separator.close()
    # a two-stem model answers with "Vocals" and "Instrumental", a four-stem one in lowercase
    stems = {str(name).lower(): audio_array for name, audio_array in results.items()}
    name, summed = instrumental(stems)
    stems[name] = summed
    return rate, stems


def _write(
    audio: str,
    model: str,
    rate: int,
    stems: dict[str, Any],
    target: pathlib.Path | None = None,
    manifest: bool = True,
) -> dict[str, pathlib.Path]:
    """Every stem as a float wav, then the manifest: a half-written set is never offered as a cache."""
    import pymss

    folder = stems_dir(audio, model) if target is None else target
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}
    for name in sorted(stems):
        path = folder / f"{name}.wav"
        pymss.save_audio(str(path), stems[name], rate, "wav", {"wav_bit_depth": "FLOAT"})
        written[name] = path
    if not manifest:
        return written
    write_json(
        {
            "version": STORE_VERSION,
            "audio": str(resolved(audio)),
            "stamp": list(file_stamp(audio)),
            "model": model,
            "rate": rate,
            "stems": {name: path.name for name, path in written.items()},
        },
        folder / MANIFEST,
    )
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="namioto-cli separate", description="Separate an audio file into the stems a transcription engine wants"
    )
    parser.add_argument("audio", type=pathlib.Path, help="audio file to separate")
    parser.add_argument("--model", help=f"separator to run, one of {', '.join(MODELS)}; $NAMIOTO_SEPARATION_MODEL wins")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="run on the CPU, or on a CUDA GPU when one is available",
    )
    parser.add_argument("--out", type=pathlib.Path, help="directory to write the stems into")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = model_name(args.model)
    if model not in MODELS:
        expected = ", ".join(MODELS)
        print(f"No model to use: unknown separation model {model!r}, expected one of {expected}", file=sys.stderr)
        return 2
    try:
        rate, stems = _split(
            str(args.audio),
            model,
            args.device,
            lambda: print("  model ready", file=sys.stderr),
            lambda done, total, _message: print(f"  chunks {done}/{total}", file=sys.stderr),
        )
    except ModuleNotFoundError:
        print(f"No separator to use: {MISSING}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"No separator to use: {error}", file=sys.stderr)
        return 2
    # into the cache when no directory is given, so a later transcription finds the stems again; a
    # directory of the caller's own is written without a manifest, which is what makes a set a cache
    target = stems_dir(str(args.audio), model) if args.out is None else args.out
    written = _write(str(args.audio), model, rate, stems, target, manifest=args.out is None)
    for path in written.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
