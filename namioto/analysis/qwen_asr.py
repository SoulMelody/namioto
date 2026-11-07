# SPDX-License-Identifier: AGPL-3.0-only
"""Speech recognition with Qwen3-ASR: a voice in, the words it sings out.

A lyric is normally typed, and a sung one is usually easier to correct than to write down, so the
recognition exists to fill the editor's lyric box rather than to replace it:

    model = load("Qwen/Qwen3-ASR-0.6B")
    text = recognize(model, "vocal.wav", "Japanese")

Two calls and one model, like the aligner beside it (`namioto.analysis.qwen_aligner`), which then
puts a time on the words this finds. The model answers one text for the whole file: it cuts long
audio into the chunks it can hear itself, so no windowing is done here.

The model is Apache-2.0 and ungated, but it is a torch model: `torch` and `qwen_asr` are reached only
inside `load` and `recognize`, which a spawned process calls, so opening the editor never pulls a
torch build in. Its weights are a separate work fetched on first use from the Hugging Face endpoint
the network settings name (`namioto.net.hf_endpoint`, the domestic mirror by default).
"""

from __future__ import annotations

import pathlib

from namioto.analysis.qwen_checkpoint import MISSING, Checkpoint

CHECKPOINT = Checkpoint(
    repo="Qwen/Qwen3-ASR-0.6B",
    # the revision this module was written against; a checkpoint that moved under the old one would
    # change results without a version of this program changing
    revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    # what `from_pretrained` reads: the processor's tokenizer, the weights, and the generation
    # settings the chat template needs
    files=(
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
    ),
    directory="qwen3-asr-0.6b",
)
MODEL = CHECKPOINT.repo
REVISION = CHECKPOINT.revision
# what the model calls each language; empty is not one of them, and means "identify it yourself"
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


def model_dir() -> pathlib.Path:
    """Where the weights are filed, whether or not they have been fetched yet."""
    return CHECKPOINT.model_dir()


def is_installed(directory: pathlib.Path | None = None) -> bool:
    return CHECKPOINT.is_installed(directory)


def resolve_model(explicit: str | pathlib.Path | None = None, *, progress=None) -> pathlib.Path:
    """The directory to load from: an explicit one, the one here, or a fetch of the pinned revision."""
    return CHECKPOINT.resolve(explicit, progress=progress)


def load(directory: str | pathlib.Path, device: str = "cpu"):
    """The recognizer, on the device a run names; half precision only where it is a win."""
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise ModuleNotFoundError(MISSING) from error

    target = CHECKPOINT.whole(pathlib.Path(directory))
    if device == "webgpu":
        raise ValueError("Qwen3-ASR needs CUDA or the CPU; WebGPU is an ONNX Runtime provider")
    on_gpu = device == "cuda"
    return Qwen3ASRModel.from_pretrained(
        str(target),
        dtype=torch.bfloat16 if on_gpu else torch.float32,
        device_map="cuda" if on_gpu else "cpu",
        max_inference_batch_size=1,
    )


def recognize(model, audio: str | pathlib.Path, language: str = "") -> str:
    """The words one audio file sings; an empty language lets the model identify it itself."""
    if language and language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}, expected one of {', '.join(LANGUAGES)}")
    results = model.transcribe(audio=str(audio), language=language or None)
    return str(results[0].text).strip() if results else ""
