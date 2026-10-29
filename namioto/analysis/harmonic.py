# SPDX-License-Identifier: AGPL-3.0-only
"""How much of each spectrum frame is overtones rather than fundamentals.

A sung or played note does not fill one band: it puts energy at its own frequency and at every
overtone above it, and an overtone lands on a band that names a real note. Reading the table as it
stands, an A3 with a strong second partial looks like an A3 and an A4 sounding together, which is
the octave error this module exists to take out.

The estimator follows noteDigger's `_reduceHarmonic` (`core/app_analyser.js`, GPL-3.0, see NOTICE):
every note owns a template of its fundamental and the overtones above it, and one non-negative
least-squares fit per frame says how much of each template the frame holds. The fundamental is left
out of the reconstruction, so what comes back is the overtone energy alone - the part a caller may
subtract from the table it passed in.

Nothing here changes the table it was given. The estimate is a second table, `HarmonicSpectrum`, and
how much of it to subtract is the caller's decision (`subtract`). Only the parameters that shape the
fit key the cache, so changing how strongly the overtones are removed never runs it again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import nnls

if TYPE_CHECKING:
    from namioto.analysis.spectrum import NoteSpectrum

HARMONICS = 8
DECAY = 0.6
BLOCK_FRAMES = 256
# how much of the estimate the editor takes out when the switch is on: the fit is a model of the
# overtones, not a measurement of them, so removing all of it would eat the note's own attack too
STRENGTH = 0.5


@dataclass(frozen=True)
class HarmonicSpectrum:
    """The overtone energy of a note table, frame for frame, in the table's own units."""

    table: np.ndarray
    harmonics: int = HARMONICS
    decay: float = DECAY

    @property
    def frames(self) -> int:
        return self.table.shape[0]


def template(harmonics: int = HARMONICS, decay: float = DECAY) -> np.ndarray:
    """`(NOTE_COUNT, NOTE_COUNT)`: column `i` is note `i`'s overtones, its fundamental at 1.

    Overtone `h` sits `12 * log2(h)` semitones above the fundamental, which rarely lands on a band:
    its weight is split between the two bands it falls between, so a third partial at 19.02
    semitones reaches both neighbours in proportion. Its amplitude falls by `decay` per partial.
    """
    if harmonics < 1:
        raise ValueError("At least one harmonic is needed")
    if not 0.0 < decay <= 1.0:
        raise ValueError("The overtone decay must be in (0, 1]")

    # imported here, not at the top: `--help` must not drag the whole spectrum engine in
    from namioto.analysis.spectrum import NOTE_COUNT

    matrix = np.zeros((NOTE_COUNT, NOTE_COUNT), dtype=np.float64)
    for note in range(NOTE_COUNT):
        for order in range(1, harmonics + 1):
            position = note + 12.0 * math.log2(order)
            if position > NOTE_COUNT - 1:
                break
            low = math.floor(position)
            fraction = position - low
            weight = decay ** (order - 1)
            matrix[low, note] += weight * (1.0 - fraction)
            if fraction and low + 1 < NOTE_COUNT:
                matrix[low + 1, note] += weight * fraction
    return matrix


def _overtone_matrix(harmonics: int, decay: float) -> np.ndarray:
    """The templates with the fundamental taken out: what a fit's coefficients reconstruct."""
    matrix = template(harmonics, decay)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def reduce(
    table: np.ndarray,
    *,
    harmonics: int = HARMONICS,
    decay: float = DECAY,
    progress=None,
) -> HarmonicSpectrum:
    """Fit the overtone templates to every frame, and keep what they account for.

    Each frame is one non-negative least-squares problem against the same dictionary, so the result
    says how much of each note's overtone series the frame holds without ever claiming a negative
    amount of a partial. `progress` is called with (frames done, frames total) per block.
    """
    frames = np.asarray(table, dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError(f"Expected a (frames, notes) table, got {frames.shape}")
    from namioto.analysis.spectrum import NOTE_COUNT

    if frames.shape[1] != NOTE_COUNT:
        raise ValueError(f"Expected {NOTE_COUNT} note bands, got {frames.shape[1]}")
    dictionary = template(harmonics, decay)
    overtones = _overtone_matrix(harmonics, decay)

    out = np.zeros(frames.shape, dtype=np.float32)
    total = len(frames)
    for start in range(0, total, BLOCK_FRAMES):
        block = frames[start : start + BLOCK_FRAMES]
        for offset, frame in enumerate(block):
            coefficients, _ = nnls(dictionary, frame)
            out[start + offset] = overtones @ coefficients
        if progress is not None:
            progress(min(start + BLOCK_FRAMES, total), total)
    return HarmonicSpectrum(out, harmonics, decay)


def subtract(table: np.ndarray, harmonic: np.ndarray | None, strength: float = 0.5) -> np.ndarray:
    """`table` with `strength` of the overtone estimate taken out, never below zero.

    A strength of 0 returns the table unchanged, and 1 removes every partial the fit attributed to
    another note's overtones; the display default sits between the two because the fit is a model of
    the overtones, not a measurement of them.
    """
    amount = min(1.0, max(0.0, float(strength)))
    if harmonic is None or amount == 0.0:
        return table
    rows = min(len(table), len(harmonic))
    out = np.array(table, dtype=np.float32, copy=True)
    out[:rows] = np.maximum(0.0, out[:rows] - amount * harmonic[:rows])
    return out


def analyse(
    path: str | Path,
    spectrum: NoteSpectrum,
    *,
    harmonics: int = HARMONICS,
    decay: float = DECAY,
    cache: bool = True,
    progress=None,
) -> HarmonicSpectrum:
    """The overtone estimate for `spectrum`, reused when the audio and the fit agree.

    The key holds the audio content and the two parameters that shape the fit, and nothing about how
    the result is displayed, so the estimate survives a change of subtraction strength.
    """
    from namioto.analysis import spectrum_cache

    options = {"stage": "harmonic", "harmonics": harmonics, "decay": round(float(decay), 3)}
    key = spectrum_cache.cache_key(path, options) if cache else None
    if key is not None:
        cached = spectrum_cache.load_table(key)
        if cached is not None and cached.shape == spectrum.table.shape:
            if progress is not None:
                progress(1, 1)
            return HarmonicSpectrum(cached, harmonics, float(decay))
    result = reduce(spectrum.table, harmonics=harmonics, decay=decay, progress=progress)
    if key is not None:
        spectrum_cache.save_table(key, result.table)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="namioto-cli harmonic", description="Estimate the overtone energy of a note spectrum"
    )
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--harmonics", type=int, default=HARMONICS, help=f"partials per note (default: {HARMONICS})")
    parser.add_argument("--decay", type=float, default=DECAY, help=f"fall of each partial (default: {DECAY})")
    parser.add_argument("--backend", choices=("wavetone", "notedigger"), default="wavetone", help="spectrum backend")
    parser.add_argument("--t-num", type=float, default=40.0, help="analysis frames per second (default: 40)")
    parser.add_argument("--strength", type=float, default=0.5, help="how much of the estimate to remove (default: 0.5)")
    parser.add_argument("--bench", action="store_true", help="print per-stage timings")
    parser.add_argument("--no-cache", action="store_true", help="analyse without reading or writing the cache")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    from namioto.analysis.spectrum import analyse as analyse_spectrum

    timings: dict = {}
    total_start = time.perf_counter()
    spectrum = analyse_spectrum(
        args.audio, backend=args.backend, t_num=args.t_num, cache=not args.no_cache, timings=timings
    )
    start = time.perf_counter()
    harmonic = analyse(
        args.audio,
        spectrum,
        harmonics=args.harmonics,
        decay=args.decay,
        cache=not args.no_cache,
        progress=lambda done, total: print(f"  frames {done}/{total}", file=sys.stderr),
    )
    timings["harmonic"] = time.perf_counter() - start
    total = time.perf_counter() - total_start

    cleaned = subtract(spectrum.table, harmonic.table, args.strength)
    payload = {
        "frames": spectrum.frames,
        "harmonics": harmonic.harmonics,
        "decay": harmonic.decay,
        "strength": args.strength,
        "before": float(spectrum.table.mean()),
        "after": float(cleaned.mean()),
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print(
            f"{payload['frames']} frames, {payload['harmonics']} partials, decay {payload['decay']}: "
            f"mean energy {payload['before']:.4g} -> {payload['after']:.4g} at strength {args.strength}"
        )
    if args.bench:
        print(f"total {total:.2f} s")
        for stage, seconds in timings.items():
            print(f"  {stage:10} {seconds:7.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
