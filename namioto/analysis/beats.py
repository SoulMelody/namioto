# SPDX-License-Identifier: AGPL-3.0-only
"""Tempo (BPM) estimation by beat tracking: librosa's dynamic-programming tracker, a least-squares
fit over the beats it finds, and a grid refit that turns the fit into sub-BPM precision.

Kept alongside it, as the runner-up, is the TempoCNN implementation in `namioto.analysis.tempo`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 512
WINDOW_SECONDS = 12.0
WINDOW_HOP_SECONDS = 6.0
MIN_BEATS = 8
MIN_WINDOW_BEATS = 4
TOLERANCE = 0.01
REFINE_SPAN = 0.03
REFINE_STEP = 0.0005
REFINE_BINS = 200


@dataclass(frozen=True)
class LocalWindow:
    """The tempo fitted to one window of the audio; the spread of these is the drift."""

    start: float
    end: float
    bpm: float


@dataclass(frozen=True)
class BeatTempo:
    bpm: float
    beats: tuple[float, ...]
    local: tuple[LocalWindow, ...]
    residual: float

    @property
    def agreement(self) -> float:
        """Share of the windows within `TOLERANCE` of the global tempo; low means it drifts."""
        if not self.local:
            return 0.0
        return sum(abs(window.bpm - self.bpm) <= self.bpm * TOLERANCE for window in self.local) / len(self.local)


def fit_beats(beats: np.ndarray) -> tuple[float, float]:
    """Tempo from the slope of a straight line through the beat times, and its rms residual."""
    index = np.arange(len(beats))
    slope, intercept = np.polyfit(index, beats, 1)
    residual = float(np.sqrt(np.mean((beats - (slope * index + intercept)) ** 2)))
    return 60.0 / slope, residual


def grid_score(beats: np.ndarray, period: float) -> float:
    """Share of the beats that land in the busiest cell of a grid of `period`, its neighbours included."""
    counts, _ = np.histogram(np.mod(beats, period), bins=REFINE_BINS, range=(0.0, period))
    spread = counts + np.roll(counts, 1) + np.roll(counts, -1)
    return float(spread.max()) / len(beats)


def refine_grid(beats: np.ndarray, bpm: float) -> float:
    """The tempo near `bpm` on whose grid most of the detected beats actually sit.

    A least-squares fit spreads a handful of misplaced beats over the whole line; this picks the
    periodicity the beats agree on instead.
    """
    candidates = np.arange(bpm * (1.0 - REFINE_SPAN), bpm * (1.0 + REFINE_SPAN), REFINE_STEP)
    scores = [grid_score(beats, 60.0 / candidate) for candidate in candidates]
    return float(candidates[int(np.argmax(scores))])


def local_windows(
    beats: np.ndarray,
    window_seconds: float = WINDOW_SECONDS,
    window_hop_seconds: float = WINDOW_HOP_SECONDS,
) -> tuple[LocalWindow, ...]:
    windows = []
    for start in np.arange(0.0, beats[-1], window_hop_seconds):
        inside = beats[(beats >= start) & (beats < start + window_seconds)]
        if len(inside) >= MIN_WINDOW_BEATS:
            slope = np.polyfit(np.arange(len(inside)), inside, 1)[0]
            windows.append(LocalWindow(float(start), float(start + window_seconds), 60.0 / slope))
    return tuple(windows)


def estimate_array(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    window_seconds: float = WINDOW_SECONDS,
    window_hop_seconds: float = WINDOW_HOP_SECONDS,
) -> BeatTempo:
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    onset = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    _, detected = librosa.beat.beat_track(onset_envelope=onset, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units="time")
    beats = np.asarray(detected, dtype=float)
    if len(beats) < MIN_BEATS:
        raise ValueError(f"Found only {len(beats)} beats, which is too few to fit a tempo to")
    bpm, residual = fit_beats(beats)
    windows = local_windows(beats, window_seconds, window_hop_seconds)
    return BeatTempo(refine_grid(beats, bpm), tuple(beats.tolist()), windows, residual)


def estimate(
    path: str | Path,
    *,
    window_seconds: float = WINDOW_SECONDS,
    window_hop_seconds: float = WINDOW_HOP_SECONDS,
) -> BeatTempo:
    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return estimate_array(
        audio,
        sample_rate,
        window_seconds=window_seconds,
        window_hop_seconds=window_hop_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namioto-cli tempo", description=__doc__)
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--local", action="store_true", help="also print the per-window tempo")
    parser.add_argument("--json", action="store_true", help="print the estimate as JSON")
    args = parser.parse_args(argv)

    result = estimate(args.audio)

    if args.json:
        json.dump(
            {
                "bpm": result.bpm,
                "residual": result.residual,
                "agreement": result.agreement,
                "beats": result.beats,
                "windows": [vars(window) for window in result.local],
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print(f"Song BPM: {result.bpm:.3f}")
    print(
        f"  {len(result.beats)} beats, fit residual {1000 * result.residual:.1f} ms, "
        f"{result.agreement:.0%} of {len(result.local)} windows agree"
    )
    if args.local:
        for window in result.local:
            print(f"  {window.bpm:7.3f} BPM at {window.start:.1f}-{window.end:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
