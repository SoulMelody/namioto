# SPDX-License-Identifier: AGPL-3.0-only
"""Tempo (BPM) estimation with WaveTone's own static analysis.

The absolute volume change from one frame to the next is read as a periodic signal: for each
candidate tempo its first two harmonics are summed, and the strongest is refined to a hundredth of
a BPM. Reversed from WaveTone 2.73's `awlib.dll` (`dfttempo_awd`); unlike the beat trackers it does
not pick a metrical relative of the beat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

FRAME = 1024
COARSE_BPM = np.arange(60, 240)
FINE_SPAN = 0.5
FINE_STEP = 0.01
SECOND_HARMONIC = 0.5


def volume_change(mono: np.ndarray, sample_rate: int, frame: int = FRAME) -> tuple[np.ndarray, float]:
    """The absolute change of the frame's peak volume, and the rate it is sampled at."""
    count = len(mono) // frame
    if count < 2:
        raise ValueError("Audio is too short to measure a tempo from")
    volume = np.abs(mono[: count * frame].reshape(count, frame)).max(axis=1)
    return np.abs(np.diff(volume)), sample_rate / frame


def score(envelope: np.ndarray, rate: float, bpms: np.ndarray) -> np.ndarray:
    """How strongly `envelope` repeats at each tempo: |DFT at f| + 0.5 |DFT at 2f|."""
    index = np.arange(len(envelope))
    out = np.empty(len(bpms))
    for position, bpm in enumerate(bpms):
        phase = 2.0 * np.pi * (bpm / 60.0) * index / rate
        first = np.hypot((envelope * np.cos(phase)).sum(), (envelope * np.sin(phase)).sum())
        second = np.hypot((envelope * np.cos(2.0 * phase)).sum(), (envelope * np.sin(2.0 * phase)).sum())
        out[position] = first + SECOND_HARMONIC * second
    return out


def estimate_array(mono: np.ndarray, sample_rate: int, *, frame: int = FRAME) -> float:
    envelope, rate = volume_change(mono, sample_rate, frame)
    if not envelope.any():
        raise ValueError("Found no volume change to measure a tempo from")
    coarse = COARSE_BPM[int(np.argmax(score(envelope, rate, COARSE_BPM)))]
    fine = np.round(np.arange(coarse - FINE_SPAN, coarse + FINE_SPAN + 1e-9, FINE_STEP), 2)
    return float(fine[int(np.argmax(score(envelope, rate, fine)))])


def estimate(path: str | Path, *, frame: int = FRAME) -> float:
    mono, sample_rate = librosa.load(path, sr=None, mono=True)
    return estimate_array(np.clip(mono, -1.0, 1.0), sample_rate, frame=frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namioto-cli wavetone", description=__doc__)
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--frame", type=int, default=FRAME, help="volume envelope frame in samples")
    parser.add_argument("--json", action="store_true", help="print the estimate as JSON")
    args = parser.parse_args(argv)

    bpm = estimate(args.audio, frame=args.frame)

    if args.json:
        json.dump({"bpm": bpm}, sys.stdout, indent=2)
        print()
        return 0

    print(f"Song BPM: {bpm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
