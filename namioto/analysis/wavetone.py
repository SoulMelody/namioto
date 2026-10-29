# SPDX-License-Identifier: AGPL-3.0-only
"""Tempo (BPM) estimation with WaveTone's own static analysis.

The spectrum's rise from one frame to the next is read as a periodic signal: for each candidate
tempo its first two harmonics are summed, and the strongest is refined to a hundredth of a BPM.
WaveTone's `dfttempo_awd`; unlike the beat trackers it does not pick a metrical relative of the
beat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

HOP = 512
WINDOW = 2048
BLOCK = 1024
COARSE_BPM = np.arange(60, 240)
FINE_SPAN = 0.5
FINE_STEP = 0.01
SECOND_HARMONIC = 0.5


def onset_envelope(
    mono: np.ndarray, sample_rate: int, *, hop: int = HOP, window: int = WINDOW
) -> tuple[np.ndarray, float]:
    """How much each frame's spectrum grew over the one before it, and the rate it is sampled at.

    The rectified spectral flux of a Hann window: a plain amplitude envelope cannot see an onset
    that arrives at a constant loudness. Spectra are taken a block at a time, so a whole song's
    windows are never held at once.
    """
    frames_in_audio = (len(mono) - window) // hop + 1
    if frames_in_audio < 2:
        raise ValueError("Audio is too short to measure a tempo from")
    tapered = np.hanning(window)
    flux = []
    previous = None
    for start in range(0, frames_in_audio, BLOCK):
        windows = np.lib.stride_tricks.sliding_window_view(mono[start * hop :], window)[::hop][:BLOCK]
        spectrum = np.abs(np.fft.rfft(windows * tapered, axis=1))
        if previous is not None:
            spectrum = np.vstack([previous, spectrum])
        flux.append(np.maximum(np.diff(spectrum, axis=0), 0.0).sum(axis=1))
        previous = spectrum[-1]
    return np.concatenate(flux), sample_rate / hop


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


def estimate_array(mono: np.ndarray, sample_rate: int, *, hop: int = HOP, window: int = WINDOW) -> float:
    envelope, rate = onset_envelope(mono, sample_rate, hop=hop, window=window)
    if not envelope.any():
        raise ValueError("Found no onset to measure a tempo from")
    coarse = COARSE_BPM[int(np.argmax(score(envelope, rate, COARSE_BPM)))]
    fine = np.round(np.arange(coarse - FINE_SPAN, coarse + FINE_SPAN + 1e-9, FINE_STEP), 2)
    return float(fine[int(np.argmax(score(envelope, rate, fine)))])


def estimate(path: str | Path, *, hop: int = HOP, window: int = WINDOW) -> float:
    mono, sample_rate = librosa.load(path, sr=None, mono=True)
    return estimate_array(np.clip(mono, -1.0, 1.0), sample_rate, hop=hop, window=window)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namioto-cli wavetone", description=__doc__)
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--hop", type=int, default=HOP, help="onset envelope hop in samples")
    parser.add_argument("--window", type=int, default=WINDOW, help="onset envelope FFT window in samples")
    parser.add_argument("--json", action="store_true", help="print the estimate as JSON")
    args = parser.parse_args(argv)

    bpm = estimate(args.audio, hop=args.hop, window=args.window)

    if args.json:
        json.dump({"bpm": bpm}, sys.stdout, indent=2)
        print()
        return 0

    print(f"Song BPM: {bpm:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
