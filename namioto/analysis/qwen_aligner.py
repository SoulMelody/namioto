# SPDX-License-Identifier: AGPL-3.0-only
"""Forced alignment with Qwen3-ForcedAligner: a lyric and a voice in, times out.

TIFA aligns phonemes, so the languages a host can align are the languages it wrote a G2P for.
Qwen3-ForcedAligner reads the text itself: the model tokenizes the lyric, hears the audio and
answers one unit per word or character, so the host owns no phoneme set at all. What is left here
is two calls and one matching step.

    model = load("Qwen/Qwen3-ForcedAligner-0.6B")
    units = align(model, "vocal.wav", "きみのこえ", "Japanese")
    times = match_units(units, ["き", "み", "の", "こ", "え"])

Its sibling in the same package, Qwen3-ASR (`namioto.analysis.qwen_asr`), reads the words out of the
audio before anyone has typed them; the aligner here then puts a time on them.

The model is Apache-2.0 and ungated, but it is a torch model: `torch` and `qwen_asr` are reached
only inside `load` and `align`, which a spawned process calls, so opening the editor never pulls a
torch build in. Its weights are a separate work fetched on first use from the Hugging Face endpoint
the network settings name (`namioto.net.hf_endpoint`, the domestic mirror by default).

Two limits are the model's, not this module's: one call hears at most five minutes, and it was
trained on speech. The caller cuts long audio into windows (`alignment.chunk_seconds`) and runs the
alignment on a separated vocal when the source is a mix.
"""

from __future__ import annotations

import pathlib
import unicodedata
from collections.abc import Callable, Sequence
from typing import NamedTuple

from namioto.analysis.qwen_checkpoint import MISSING, Checkpoint

CHECKPOINT = Checkpoint(
    repo="Qwen/Qwen3-ForcedAligner-0.6B",
    # the revision this module was written against; a checkpoint that moved under the old one would
    # change results without a version of this program changing
    revision="c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    # the files `from_pretrained` reads; the weights are the only large one
    files=(
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ),
    directory="qwen3-forcedaligner-0.6b",
)
MODEL = CHECKPOINT.repo
REVISION = CHECKPOINT.revision
FILES = CHECKPOINT.files
# what the model calls each language, in the order the window offers them
LANGUAGES = (
    "Chinese",
    "Japanese",
    "English",
    "Cantonese",
    "Korean",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Spanish",
)
# the model's own ceiling: one call hears five minutes of audio, and the window asks for less
MAX_SECONDS = 300.0
# how far the matching may move the cursor to find a word: a different reading of one character,
# not a word that is missing altogether
MAX_SKIP = 2


class Unit(NamedTuple):
    """One thing the model put a time on: the text it read, and where it heard it."""

    text: str
    start: float
    end: float


def model_dir() -> pathlib.Path:
    """Where the weights are filed, whether or not they have been fetched yet."""
    return CHECKPOINT.model_dir()


def is_installed(directory: pathlib.Path | None = None) -> bool:
    return CHECKPOINT.is_installed(directory)


def resolve_model(
    explicit: str | pathlib.Path | None = None,
    *,
    download: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pathlib.Path:
    """The directory to load from: an explicit one, the one here, or a fetch of the pinned revision."""
    return CHECKPOINT.resolve(explicit, download=download, progress=progress)


def install(progress: Callable[[int, int], None] | None = None) -> pathlib.Path:
    """Fetch the pinned revision through the endpoint in force, one snapshot into `model_dir()`."""
    return CHECKPOINT.install(progress)


def _whole(directory: pathlib.Path) -> pathlib.Path:
    return CHECKPOINT.whole(directory)


def load(directory: str | pathlib.Path, device: str = "cpu"):
    """The aligner, on the device a run's key names; half precision only where it is a win.

    `device` is the resolved provider key, so anything that is not CUDA stays on the CPU — the
    WebGPU provider is an ONNX one and has nothing to do with this model.
    """
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as error:
        raise ModuleNotFoundError(MISSING) from error

    target = _whole(pathlib.Path(directory))
    if device == "webgpu":
        raise ValueError("Qwen3-ForcedAligner needs CUDA or the CPU; WebGPU is an ONNX Runtime provider")
    on_gpu = device == "cuda"
    return Qwen3ForcedAligner.from_pretrained(
        str(target),
        dtype=torch.bfloat16 if on_gpu else torch.float32,
        device_map="cuda" if on_gpu else "cpu",
    )


def align(
    model,
    audio: str | pathlib.Path,
    text: str,
    language: str,
) -> list[Unit]:
    """Every unit the model put a time on, in the order it read them.

    `audio` is a path, which the model resamples itself; `text` is the whole lyric of the window,
    spaces included, because the model's own tokenizer decides what a unit is. A language the model
    does not name is refused before the run rather than silently guessed.
    """
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}, expected one of {', '.join(LANGUAGES)}")
    if not text.strip():
        return []
    import torch

    with torch.inference_mode():
        results = model.align(audio=str(audio), text=text, language=language)
    return [
        Unit(str(item.text), float(item.start_time), float(item.end_time)) for item in (results[0] if results else ())
    ]


def normalize(text: str) -> str:
    """What the matching compares: letters and digits, with forms folded and punctuation dropped.

    The model's units and our words differ in spacing and punctuation, never in the characters
    themselves, so keeping only `isalnum()` characters is enough. Kana and kanji are letters, and a
    long vowel mark is one too, so nothing a lyric carries is lost.
    """
    return "".join(char for char in unicodedata.normalize("NFKC", text) if char.isalnum())


def match_units(units: Sequence[Unit], words: Sequence[str]) -> list[tuple[float, float] | None]:
    """The span each word was given, or None when the model read something else there.

    A word may be one unit (the usual case) or several; a unit may run past the word it started in,
    because the model's tokenizer is not our word list. The stream is consumed in order, so a word
    that does not appear where the cursor is moves the cursor over one character at a time until it
    does; a word that never appears keeps None, and the caller leaves that note's time alone rather
    than inventing one.
    """
    stream: list[tuple[str, float, float]] = [
        (char, unit.start, unit.end) for unit in units for char in normalize(unit.text)
    ]
    spans: list[tuple[float, float] | None] = []
    cursor = 0
    for word in words:
        wanted = normalize(word)
        if not wanted:
            spans.append(None)
            continue
        found = _find(stream, cursor, wanted)
        if found is None:
            spans.append(None)
            continue
        start_index, end_index = found
        spans.append((stream[start_index][1], stream[end_index][2]))
        cursor = end_index + 1
    return spans


def _find(stream: Sequence[tuple[str, float, float]], cursor: int, wanted: str) -> tuple[int, int] | None:
    """Where `wanted` sits at or after `cursor`, as the first and last index it covers.

    The search is bounded: a character the model read differently puts the cursor one or two places
    out, while a word that is genuinely absent must answer None instead of attaching the first
    distant character that happens to match.
    """
    first = cursor
    limit = min(len(stream) - len(wanted), cursor + MAX_SKIP)
    for start in range(first, limit + 1):
        if all(stream[start + offset][0] == char for offset, char in enumerate(wanted)):
            return start, start + len(wanted) - 1
    return None
