# SPDX-License-Identifier: AGPL-3.0-only
"""Tempo (BPM) estimation with the TempoCNN model, using ONNX Runtime only.

The model is not shipped: it is downloaded from this project's `models` release on first use.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

from namioto.analysis import model_store

SAMPLE_RATE = 11025
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40
FMIN = 20
FMAX = 5000
PATCH_SIZE = 256
PATCH_HOP_SIZE = 128
BPM_OFFSET = 30
AGGREGATIONS = ("majority", "mean", "median")


@dataclass(frozen=True)
class LocalTempo:
    """One patch prediction, spanning the analysed audio it was computed from."""

    start: float
    end: float
    bpm: int
    probability: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class TempoSegment:
    start: float
    end: float
    bpm: int


@dataclass(frozen=True)
class TempoEstimate:
    bpm: float
    local: tuple[LocalTempo, ...]

    def segments(self) -> tuple[TempoSegment, ...]:
        """Merge neighbouring patches with the same BPM; boundaries sit between patch centers."""
        if not self.local:
            return ()
        starts = [0.0]
        bpms = [self.local[0].bpm]
        for previous, local in zip(self.local, self.local[1:], strict=False):
            if local.bpm == bpms[-1]:
                continue
            starts.append((previous.center + local.center) / 2)
            bpms.append(local.bpm)
        ends = [*starts[1:], self.local[-1].end]
        return tuple(TempoSegment(start, end, bpm) for start, end, bpm in zip(starts, ends, bpms, strict=True))


def default_model_path() -> Path:
    """The TempoCNN graph: the one in the data directory, downloaded there when it is not."""
    model = model_store.MODELS["tempocnn"]
    return model_store.resolve("tempocnn") / model.files[0]


def mel_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    # Symmetric Hann window to match Essentia's Windowing (librosa defaults to periodic)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=1,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        window=np.hanning(N_FFT),
    )
    return mel.T.astype(np.float32)


def patchify(mel: np.ndarray) -> np.ndarray:
    offsets = range(0, len(mel) - PATCH_SIZE + 1, PATCH_HOP_SIZE)
    patches = np.stack([mel[o : o + PATCH_SIZE] for o in offsets])
    scale = patches.std(axis=(1, 2), keepdims=True)
    scale[scale == 0] = 1
    patches = (patches - patches.mean(axis=(1, 2), keepdims=True)) / scale
    return patches.transpose(0, 2, 1)[:, :, :, None].astype(np.float32)


@lru_cache(maxsize=4)
def _session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def patch_times(count: int) -> list[tuple[float, float]]:
    frame = HOP_LENGTH / SAMPLE_RATE
    return [(i * PATCH_HOP_SIZE * frame, (i * PATCH_HOP_SIZE + PATCH_SIZE) * frame) for i in range(count)]


def aggregate(local_bpm: list[int], method: str = "majority") -> float:
    if method == "mean":
        return float(np.mean(local_bpm))
    if method == "median":
        return float(np.median(local_bpm))
    if method != "majority":
        raise ValueError(f"Unknown aggregation {method!r}, expected one of {AGGREGATIONS}")
    votes = Counter(local_bpm)
    best, best_votes = local_bpm[0], -1
    for bpm in local_bpm:
        if votes[bpm] > best_votes:
            best, best_votes = bpm, votes[bpm]
    return float(best)


def estimate_array(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    model: str | Path | None = None,
    aggregation: str = "majority",
) -> TempoEstimate:
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    mel = mel_spectrogram(audio)
    if len(mel) < PATCH_SIZE:
        raise ValueError(
            f"Audio too short: need at least {PATCH_SIZE} mel frames "
            f"(about {PATCH_SIZE * HOP_LENGTH / SAMPLE_RATE:.1f} s), got {len(mel)}"
        )

    session = _session(str(model or default_model_path()))
    predictions = session.run(None, {session.get_inputs()[0].name: patchify(mel)})[0]
    bpm = predictions.argmax(axis=1) + BPM_OFFSET
    probability = predictions.max(axis=1)

    local = tuple(
        LocalTempo(start, end, int(local_bpm), float(prob))
        for (start, end), local_bpm, prob in zip(patch_times(len(bpm)), bpm, probability, strict=True)
    )
    return TempoEstimate(aggregate([int(b) for b in bpm], aggregation), local)


def estimate(
    path: str | Path,
    *,
    model: str | Path | None = None,
    aggregation: str = "majority",
) -> TempoEstimate:
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return estimate_array(audio, sample_rate, model=model, aggregation=aggregation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namioto-cli tempocnn", description=__doc__)
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--model", help="TempoCNN ONNX model")
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATIONS,
        default="majority",
        help="global tempo aggregation (default: majority)",
    )
    parser.add_argument("--local", action="store_true", help="also print per-patch estimates")
    parser.add_argument("--json", action="store_true", help="print the estimate as JSON")
    args = parser.parse_args(argv)

    result = estimate(args.audio, model=args.model, aggregation=args.aggregation)

    if args.json:
        json.dump(
            {
                "bpm": result.bpm,
                "local": [vars(local) for local in result.local],
                "segments": [vars(segment) for segment in result.segments()],
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print(f"Song BPM: {result.bpm}")
    if args.local:
        for local in result.local:
            print(f"  {local.bpm} BPM (probability {local.probability:.4f}) at {local.center:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
