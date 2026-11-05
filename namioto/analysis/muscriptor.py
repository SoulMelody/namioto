# SPDX-License-Identifier: AGPL-3.0-only
"""Instrument transcription with MuScriptor's published checkpoints.

GAME hears one singing voice. MuScriptor (the `muscriptor` package on PyPI, pinned at 0.4.0) hears
instruments, and answers with one part per instrument it kept. This module is the thin side of that:
it fetches the checkpoint the run asks for, loads it, and turns the note events back into the parts
the rest of the program already speaks. The package's own decoding stays inside it.

Nothing here is imported by the GUI. `torch` and `muscriptor` are reached only inside `load` and
`extract`, which a spawned process calls, so opening the editor never pulls them in. The checkpoints
are a separate work under CC BY-NC 4.0 and are not shipped; they are fetched from the Hugging Face
endpoint the network settings name (`namioto.net.hf_url`), or read from a directory named by
`$NAMIOTO_MUSCRIPTOR_MODEL`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections.abc import Callable, Sequence

from namioto import net
from namioto.analysis import devices
from namioto.analysis.notes import Note, Part, quantized_parts, save_parts
from namioto.utils import data_dir

MODEL_SIZES = ("small", "medium", "large")
MODEL_FILES = ("model.safetensors", "config.json")
PACKAGE_VERSION = "0.4.0"
# the instruments the released tokenizer knows, with the General MIDI program each one is written as
INSTRUMENTS: dict[str, int] = {
    "acoustic_piano": 0,
    "electric_piano": 2,
    "chromatic_percussion": 8,
    "organ": 16,
    "acoustic_guitar": 24,
    "clean_electric_guitar": 26,
    "distorted_electric_guitar": 29,
    "acoustic_bass": 32,
    "electric_bass": 33,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "orchestral_harp": 46,
    "timpani": 47,
    "string_ensemble": 48,
    "synth_strings": 50,
    "voice": 52,
    "orchestra_hit": 55,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "french_horn": 60,
    "brass_section": 61,
    "soprano_and_alto_sax": 64,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    "flutes": 73,
    "synth_lead": 80,
    "synth_pad": 88,
    "drums": 0,
}
# repo, revision and the two files, pinned to the revisions music-to-midi already verified
_CHECKPOINTS = {
    "small": (
        "MuScriptor/muscriptor-small",
        "8c127f603b807520fa465c838e9bfee8a91ada4e",
        {"model.safetensors": 411_888_600, "config.json": 124},
    ),
    "medium": (
        "MuScriptor/muscriptor-medium",
        "f32236969308476e01fd3aae67357de5feb05a2d",
        {"model.safetensors": 1_228_144_472, "config.json": 126},
    ),
    "large": (
        "MuScriptor/muscriptor-large",
        "8809fdfbed2affa7ade94a7059e746e3880720e7",
        {"model.safetensors": 5_465_642_136, "config.json": 125},
    ),
}


def _size(size: str) -> str:
    """A size nobody publishes is a mistake worth raising before touching the disk."""
    if size not in MODEL_SIZES:
        raise ValueError(f"unknown model size {size!r}, expected one of {', '.join(MODEL_SIZES)}")
    return size


def model_dir(size: str) -> pathlib.Path:
    """Where one size is filed, whether or not it has been fetched yet."""
    return data_dir("models", "muscriptor", _size(size))


def is_installed(size: str) -> bool:
    target = model_dir(size)
    return all((target / name).is_file() for name in MODEL_FILES)


def resolve_model(
    path: str | pathlib.Path | None = None,
    size: str = "small",
    *,
    download: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pathlib.Path:
    """The directory to load one size from.

    An explicit path or `$NAMIOTO_MUSCRIPTOR_MODEL` wins, then an already installed size, and only
    then a download of that size into the data directory.
    """
    chosen = _size(size)
    for candidate in (path, os.environ.get("NAMIOTO_MUSCRIPTOR_MODEL") or None):
        if candidate is not None:
            return _whole(pathlib.Path(candidate))
    target = model_dir(chosen)
    if is_installed(chosen):
        return target
    if not download:
        raise FileNotFoundError(f"no MuScriptor {chosen} checkpoint in {target}")
    return install(chosen, progress)


def install(size: str, progress: Callable[[int, int], None] | None = None) -> pathlib.Path:
    """Fetch one size's two files into the data directory, through the endpoint in force."""
    chosen = _size(size)
    repo, revision, sizes = _CHECKPOINTS[chosen]
    target = model_dir(chosen)
    target.mkdir(parents=True, exist_ok=True)
    open_url = net.opener()
    for name, expected in sizes.items():
        address = net.hf_url(repo, revision, name)
        temporary = target / f".{name}.part"
        try:
            with open_url(address) as response, temporary.open("wb") as out:
                total = int(response.headers.get("Content-Length") or expected)
                done = 0
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)
            temporary.replace(target / name)
        finally:
            temporary.unlink(missing_ok=True)
    return _whole(target)


def _whole(directory: pathlib.Path) -> pathlib.Path:
    """A directory is a checkpoint only when it holds both files."""
    folder = directory.parent if directory.is_file() else directory
    missing = [name for name in MODEL_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} is not a MuScriptor checkpoint: missing {', '.join(missing)}")
    return folder


def _torch_device(device: str):
    """The torch device a run's resolved key names; anything but CUDA stays on the CPU."""
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(directory: str | pathlib.Path, device: str = "cpu"):
    """The released transcription model, with this checkpoint's weights copied in place.

    Importing `muscriptor` is the point of no return for the dependency, so it happens here and
    nowhere a window can reach. The build follows the package's own `_build_model`: constructing it
    by hand would drift from the checkpoint the config describes.
    """
    import muscriptor.transcription_model as upstream

    folder = _whole(pathlib.Path(directory))
    target = _torch_device(device)
    config = upstream._resolve_config(folder / "model.safetensors", folder / "config.json")
    model = upstream._build_model(target, config)
    model.eval()
    _load_weights(model, folder / "model.safetensors", target)
    model.to(target)
    return upstream.TranscriptionModel(model=model, tokenizer=upstream.MT3Tokenizer(), device=target)


def _load_weights(model, weights: pathlib.Path, device) -> None:
    """Copy the checkpoint in one tensor at a time, so the whole file is never a second copy."""
    from safetensors import safe_open

    destination = model.state_dict()
    with safe_open(weights, framework="pt", device="cpu") as source:
        for key in source.keys():  # noqa: SIM118 - safe_open exposes keys(), not iteration
            name = _remap(key)
            if name not in destination:
                raise KeyError(f"{weights.name} holds {key}, which this model has no parameter for")
            tensor = source.get_tensor(key)
            if tensor.shape != destination[name].shape:
                raise ValueError(f"{key} is {tuple(tensor.shape)}, the model wants {tuple(destination[name].shape)}")
            destination[name].copy_(tensor.to(device))
            del tensor


def _remap(key: str) -> str:
    """The released checkpoints store a single codebook under `emb.0` / `linears.0`."""
    if key.startswith(("emb.1.", "linears.1.")):
        raise ValueError("the checkpoint has more than one codebook; only a single stream is supported")
    if key.startswith("emb.0."):
        return "emb." + key[len("emb.0.") :]
    if key.startswith("linears.0."):
        return "linear." + key[len("linears.0.") :]
    return key


def extract(
    model,
    path: str | pathlib.Path,
    *,
    instruments: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Part]:
    """The notes of one audio file, one part per instrument the model kept.

    `instruments` narrows what may be decoded; an empty sequence leaves the choice to the model. A
    name the released tokenizer does not know is refused before the run, rather than dropped inside
    it.
    """
    chosen = _instruments(instruments)
    import torch
    from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

    notes: dict[str, list[Note]] = {}
    with torch.inference_mode():
        events = model.transcribe(
            path,
            instruments=chosen or None,
            use_sampling=False,
            batch_size=1,
            beam_size=1,
            prelude_forcing=True,
        )
        for event in events:
            if isinstance(event, ProgressEvent):
                if progress is not None:
                    progress(int(event.completed), max(1, int(event.total)))
            elif isinstance(event, NoteStartEvent):
                notes.setdefault(str(event.instrument), [])
            elif isinstance(event, NoteEndEvent):
                start = event.start_event
                instrument = str(start.instrument)
                if instrument not in INSTRUMENTS:
                    raise RuntimeError(
                        f"MuScriptor returned an instrument this program has no program for: {instrument}"
                    )
                notes.setdefault(instrument, []).append(
                    (float(start.start_time), float(event.end_time), float(start.pitch))
                )
            else:
                raise RuntimeError(f"MuScriptor returned an event this program does not read: {type(event).__name__}")
    return [Part(name, INSTRUMENTS[name], tuple(found)) for name, found in notes.items() if found]


def _instruments(instruments: Sequence[str] | None) -> list[str]:
    chosen = [name for name in (instruments or ()) if name]
    unknown = [name for name in chosen if name not in INSTRUMENTS]
    if unknown:
        raise ValueError(f"unknown instrument {unknown[0]!r}, expected one of {', '.join(INSTRUMENTS)}")
    return chosen


def read_config(directory: str | pathlib.Path) -> dict:
    """The checkpoint's own config, for a caller that wants to see what it loaded."""
    return json.loads((_whole(pathlib.Path(directory)) / "config.json").read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="namioto-cli muscriptor",
        description="Extract the notes of instruments with MuScriptor's checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="?", help="audio file to extract from; leave out to only fetch a model")
    parser.add_argument("-m", "--model", type=pathlib.Path, help="directory holding model.safetensors and config.json")
    parser.add_argument(
        "--size",
        choices=MODEL_SIZES,
        default="small",
        help="model size to use, downloaded into the data directory when it is not there yet",
    )
    parser.add_argument(
        "--device",
        choices=devices.RUN_CHOICES,
        default="cpu",
        help="run on the CPU, or on the GPU backend the settings name",
    )
    parser.add_argument(
        "--instrument",
        action="append",
        help="keep only this instrument; repeat to keep more, leave out to keep whatever the model hears",
    )
    parser.add_argument("--tempo", type=float, default=120.0, help="tempo used for the MIDI file")
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        help="snap to a beat grid this many subdivisions per quarter note",
    )
    parser.add_argument("--midi", type=pathlib.Path, help="also write the parts to this MIDI file")
    parser.add_argument("--json", action="store_true", help="print the parts as JSON")
    return parser.parse_args(argv)


def report_download(done: int, total: int) -> None:
    """One line per megabyte on stderr, so a fetch does not look like a hang."""
    megabytes = done // (1 << 20)
    if megabytes != getattr(report_download, "last", -1):
        report_download.last = megabytes
        where = f" of {total // (1 << 20)}" if total else ""
        print(f"  {megabytes}{where} MB", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        folder = resolve_model(args.model, args.size, progress=report_download)
    except (OSError, ValueError) as error:
        print(f"No model to use: {error}", file=sys.stderr)
        return 2
    if args.audio is None:
        print(f"{args.size} model in {folder}")
        return 0
    model = load(folder, device=devices.resolve(args.device))
    parts = extract(
        model,
        args.audio,
        instruments=args.instrument,
        progress=lambda done, total: print(f"  chunks {done}/{total}", file=sys.stderr),
    )
    grid = None
    if args.quantize:
        parts, unit, phase = quantized_parts(parts, args.tempo, args.quantize)
        grid = (unit, 480 // args.quantize)
        print(f"grid {unit * 1000:.1f} ms, phase {phase * 1000:.1f} ms", file=sys.stderr)
    if args.midi:
        save_parts(args.midi, parts, args.tempo, grid=grid)
    if args.json:
        payload = {
            "parts": [
                {
                    "instrument": part.name,
                    "program": part.program,
                    "notes": [{"onset": onset, "offset": oset, "pitch": pitch} for onset, oset, pitch in part.notes],
                }
                for part in parts
            ]
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        total = sum(len(part.notes) for part in parts)
        print(f"{len(parts)} parts, {total} notes")
        for part in parts:
            print(f"  {part.name:24} {len(part.notes):6} notes   program {part.program}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
