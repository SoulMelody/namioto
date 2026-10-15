# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free note-domain spectra on 84 bands (C1-B7), normalised for the editor.

WaveTone-style multirate correlation is the default; the original noteDigger
STFT remains available through backend="notedigger" or stft_notes().

The optional STFT algorithm and parameters are ported from noteDigger (GPL-3.0, see NOTICE):
- `dataProcess/fft_real.js` — real FFT with a symmetric Hann window
- `dataProcess/analyser.js` — `FreqTable`, the linear-to-note weight matrix, `normalize`, `autoFill`
- `core/app_analyser.js` — the STFT driver: frame geometry and channel selection

The note table is what the editor draws as the spectrum and what note extraction runs on; frame
values are `sqrt(energy / sigma)` where `sigma` is the standard deviation of all energies of the
song. The WaveTone-style path averages channel magnitudes before this normalisation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy import fft as sp_fft
from scipy import sparse

from namioto.analysis.choices import CHANNEL_MODES

NOTE_COUNT = 84
MIDI_OFFSET = 24
A4_INDEX = 45
SEMI_RANGE = 0.667
LEAK_RANGE = 1.0
OVERSAMPLE = 32


def freq_table(a4: float = 440.0) -> np.ndarray:
    """Fundamental frequency of every note band, C1 (index 0) to B7 (index 83)."""
    octave = a4 * 2 ** (np.arange(-9, 3) / 12)
    return np.concatenate([octave / 8, octave / 4, octave / 2, octave, octave * 2, octave * 4, octave * 8])


def note_label(index: int) -> str:
    return f"{('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')[index % 12]}{index // 12 + 1}"


def build_weights(
    freqs: np.ndarray,
    df: float,
    n_bins: int,
    *,
    semi_range: float = SEMI_RANGE,
    leak_range: float = LEAK_RANGE,
    oversample: int = OVERSAMPLE,
) -> sparse.csr_matrix:
    """Map linear FFT bins onto note bands: `weights @ energy = note energies`.

    Each row is a cosine window in the log-frequency domain (half width `semi_range` semitones),
    convolved with the leakage shape of the windowed FFT (radius `leak_range` bins) so that energy
    leaking out of a note band is still collected.
    """
    tap_step = df * leak_range / (oversample + 1)
    taps = (np.arange(2 * oversample + 1) - oversample) * tap_step
    leak = np.zeros(2 * oversample + 1)
    shape = (1 - np.cos(np.arange(1, oversample + 1) * (np.pi / (oversample + 1)))) * 0.5
    leak[:oversample] = shape
    leak[oversample] = 1.0
    leak[oversample + 1 :] = shape[::-1]

    leak_f = oversample * tap_step
    tuning = 2 ** (semi_range / 12)
    rows, cols, values = [], [], []
    for index, center in enumerate(freqs):
        start = max(0, math.ceil((center / tuning - leak_f) / df))
        end = min(n_bins, math.floor((center * tuning + leak_f) / df) + 1)
        if end <= start:
            continue
        bins = np.arange(start, end)
        freqs_of_taps = bins[:, None] * df + taps[None, :]

        distance = np.full(freqs_of_taps.shape, np.inf)
        positive = freqs_of_taps > 0
        distance[positive] = 12 * np.log2(freqs_of_taps[positive] / center)
        window = np.where(np.abs(distance) < semi_range, np.cos(distance * (np.pi / semi_range)) + 1.0, 0.0)

        rows.append(np.full(len(bins), index))
        cols.append(bins)
        values.append(window @ leak)

    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(freqs), n_bins),
    )


@dataclass(frozen=True)
class NoteSpectrum:
    """Note-domain spectrum: `table[frame, note]`, normalised and ready for display or analysis."""

    table: np.ndarray
    frame_ms: float
    sample_rate: int
    fft_points: int
    hop: int
    a4: float
    sigma: float
    backend: str = "notedigger"

    @property
    def frames(self) -> int:
        return self.table.shape[0]

    @property
    def duration(self) -> float:
        return self.frames * self.frame_ms / 1000

    def frame_at(self, seconds: float) -> int:
        return min(self.frames - 1, max(0, int(seconds * 1000 / self.frame_ms)))

    def time_at(self, frame: int) -> float:
        return frame * self.frame_ms / 1000

    def summary(self) -> dict:
        energy = self.table.mean(axis=0)
        loudest = np.argsort(energy)[::-1][:8]
        return {
            "frames": self.frames,
            "frame_ms": self.frame_ms,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "fft_points": self.fft_points,
            "hop": self.hop,
            "a4": self.a4,
            "sigma": self.sigma,
            "backend": self.backend,
            "peak_bands": [{"note": note_label(int(i)), "energy": float(energy[i])} for i in loudest],
        }


@dataclass(frozen=True)
class NoteSpan:
    note: int
    start_frame: int
    end_frame: int

    @property
    def midi(self) -> int:
        return self.note + MIDI_OFFSET


def as_channels(audio: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    array = np.asarray(audio, dtype=np.float32)
    return [array] if array.ndim == 1 else [np.ascontiguousarray(row) for row in array]


def stft_notes(
    audio: np.ndarray | list[np.ndarray],
    sample_rate: int,
    *,
    t_num: float = 40.0,
    fft_points: int = 8192,
    a4: float = 440.0,
    block_frames: int = 128,
    progress=None,
    timings: dict | None = None,
) -> NoteSpectrum:
    """Run the analysis on one or more channels and merge them into a single note table."""
    channels = as_channels(audio)
    length = min(len(channel) for channel in channels)
    hop = round(sample_rate / t_num)
    frames = 1 + (length - hop // 2) // hop
    if hop <= 0 or frames <= 0:
        raise ValueError(f"Audio too short: {length / sample_rate:.2f} s at {sample_rate} Hz is less than one frame")

    df = sample_rate / fft_points
    n_bins = fft_points // 2 + 1
    window = (1 - np.cos(2 * np.pi * np.arange(fft_points) / fft_points)).astype(np.float32)

    start_time = time.perf_counter()
    weights = build_weights(freq_table(a4), df, n_bins)
    if timings is not None:
        timings["weights"] = time.perf_counter() - start_time

    centers = hop // 2 + np.arange(frames) * hop
    pad = fft_points // 2
    energies = np.zeros((frames, NOTE_COUNT), dtype=np.float32)
    fft_seconds = reduce_seconds = 0.0

    for channel in channels:
        padded = np.pad(channel[:length], (pad, pad + fft_points))
        # a strided view avoids materialising an (block_frames, fft_points) index array per block
        frames_view = np.lib.stride_tricks.sliding_window_view(padded, fft_points)
        for start in range(0, frames, block_frames):
            index = centers[start : start + block_frames]
            block = frames_view[index]

            mark = time.perf_counter()
            spectrum = sp_fft.rfft(block * window, axis=1, workers=-1)
            power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
            fft_seconds += time.perf_counter() - mark

            mark = time.perf_counter()
            energies[start : start + len(index)] += (weights @ power.T).T
            reduce_seconds += time.perf_counter() - mark

            if progress is not None:
                progress(min(start + len(index), frames), frames)

    mark = time.perf_counter()
    sigma = float(energies.std())
    table = np.sqrt(np.maximum(energies, 0.0) / sigma).astype(np.float32) if sigma > 0 else np.zeros_like(energies)
    if timings is not None:
        timings["fft"] = fft_seconds
        timings["reduce"] = reduce_seconds
        timings["normalize"] = time.perf_counter() - mark

    return NoteSpectrum(
        table=table,
        frame_ms=1000 / t_num,
        sample_rate=sample_rate,
        fft_points=fft_points,
        hop=hop,
        a4=a4,
        sigma=sigma,
    )


def wave_notes(
    audio: np.ndarray | list[np.ndarray],
    sample_rate: int,
    *,
    t_num: float = 40.0,
    a4: float = 440.0,
    progress=None,
    timings: dict | None = None,
) -> NoteSpectrum:
    """Analyse variable-length frequency windows with WaveTone-style multirate correlation."""
    from namioto.analysis.wave_spectrum import amplitudes

    if sample_rate <= 0 or not math.isfinite(t_num) or t_num <= 0 or not math.isfinite(a4) or a4 <= 0:
        raise ValueError("Sample rate, frame rate and A4 must be positive and finite")
    selected = as_channels(audio)
    length = min(len(channel) for channel in selected)
    hop = round(sample_rate / t_num)
    if hop <= 0:
        raise ValueError("Frame rate exceeds the sample rate")
    frames = 1 + (length - hop // 2) // hop
    if frames <= 0:
        raise ValueError(f"Audio too short: {length / sample_rate:.2f} s at {sample_rate} Hz is less than one frame")
    centers = hop // 2 + np.arange(frames) * hop
    start = time.perf_counter()
    magnitude = amplitudes(
        [channel[:length] for channel in selected], sample_rate, freq_table(a4), centers, t_num, progress
    )
    if timings is not None:
        timings["correlate"] = time.perf_counter() - start
    start = time.perf_counter()
    energies = magnitude**2
    sigma = float(energies.std())
    table = (
        (magnitude / math.sqrt(sigma)).astype(np.float32) if sigma > 0 else np.zeros_like(magnitude, dtype=np.float32)
    )
    if timings is not None:
        timings["normalize"] = time.perf_counter() - start
    return NoteSpectrum(table, 1000 / t_num, sample_rate, 0, hop, a4, sigma, "wavetone")


def load_channels(path: str | Path, mode: str = "mono") -> tuple[list[np.ndarray], int]:
    """Decode `path` and select the channels to analyse, mirroring noteDigger's channel menu."""
    if mode not in CHANNEL_MODES:
        raise ValueError(f"Unknown channel mode {mode!r}, expected one of {CHANNEL_MODES}")
    data, sample_rate = librosa.load(path, sr=None, mono=False)
    if data.ndim == 1:
        data = data[None, :]
    left, right = data[0], data[-1]
    match mode:
        case "mono":
            return [data.mean(axis=0)], sample_rate
        case "left":
            return [left], sample_rate
        case "right":
            return [right], sample_rate
        case "sum":
            return [left + right], sample_rate
        case "side":
            return [left - right], sample_rate
        case _:
            return ([left] if data.shape[0] == 1 else [left, right]), sample_rate


def analyse(
    path: str | Path,
    *,
    channels: str = "mono",
    backend: str = "wavetone",
    cache: bool = True,
    t_num: float = 40.0,
    fft_points: int = 8192,
    a4: float = 440.0,
    progress=None,
    timings: dict | None = None,
) -> NoteSpectrum:
    """Decode and analyse audio, reusing content-addressed results unless cache=False."""
    from namioto.analysis import spectrum_cache

    if backend not in ("wavetone", "notedigger"):
        raise ValueError(f"Unknown spectrum backend: {backend!r}")
    if channels not in CHANNEL_MODES:
        raise ValueError(f"Unknown channel mode {channels!r}, expected one of {CHANNEL_MODES}")
    options = {"backend": backend, "channels": channels, "t_num": float(t_num), "a4": float(a4)}
    if backend == "notedigger":
        options["fft_points"] = fft_points
    start = time.perf_counter()
    key = spectrum_cache.cache_key(path, options) if cache else None
    cached = spectrum_cache.load(key) if key is not None else None
    if timings is not None and cache:
        timings["cache_read"] = time.perf_counter() - start
    if cached is not None:
        if progress is not None:
            progress(1, 1)
        return NoteSpectrum(**cached)
    start = time.perf_counter()
    selected, sample_rate = load_channels(path, channels)
    if timings is not None:
        timings["decode"] = time.perf_counter() - start
    if backend == "wavetone":
        result = wave_notes(selected, sample_rate, t_num=t_num, a4=a4, progress=progress, timings=timings)
    else:
        result = stft_notes(
            selected,
            sample_rate,
            t_num=t_num,
            fft_points=fft_points,
            a4=a4,
            progress=progress,
            timings=timings,
        )
    if key is not None:
        start = time.perf_counter()
        if spectrum_cache.cache_key(path, options) == key:
            spectrum_cache.save(key, result)
        if timings is not None:
            timings["cache_write"] = time.perf_counter() - start
    return result


def auto_fill(table: np.ndarray, threshold: float, from_frame: int = 0, to_frame: int | None = None) -> list[NoteSpan]:
    """Mark the note bands that stay above `threshold`, as (note, first frame, last frame + 1).

    Spans are ordered by start frame, then note.
    """
    stop = min(table.shape[0], to_frame if to_frame else table.shape[0])
    active = table[from_frame:stop] >= threshold
    if not active.any():
        return []

    edges = np.diff(np.pad(active.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    spans = []
    for note in range(active.shape[1]):
        rising = np.flatnonzero(edges[:, note] == 1)
        falling = np.flatnonzero(edges[:, note] == -1)
        spans.extend(
            NoteSpan(note=note, start_frame=int(start) + from_frame, end_frame=int(end) + from_frame)
            for start, end in zip(rising, falling, strict=True)
        )
    spans.sort(key=lambda span: (span.start_frame, span.note))
    return spans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="namioto-cli spectrum", description="Analyse audio into a note-domain spectrum"
    )
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("--channels", choices=CHANNEL_MODES, default="mono", help="which channels to analyse")
    parser.add_argument("--t-num", type=float, default=40.0, help="analysis frames per second (default: 40)")
    parser.add_argument(
        "--backend", choices=("wavetone", "notedigger"), default="wavetone", help="spectrum backend (default: wavetone)"
    )
    parser.add_argument(
        "--fft-points", type=int, default=8192, help="noteDigger FFT size (default: 8192; ignored by wavetone)"
    )
    parser.add_argument("--a4", type=float, default=440.0, help="frequency of A4 (default: 440)")
    parser.add_argument("--threshold", type=float, help="also report auto-filled notes above this value")
    parser.add_argument("--bench", action="store_true", help="print per-stage timings")
    parser.add_argument("--no-cache", action="store_true", help="analyse without reading or writing the spectrum cache")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    timings: dict = {}
    total_start = time.perf_counter()
    spectrum = analyse(
        args.audio,
        channels=args.channels,
        backend=args.backend,
        cache=not args.no_cache,
        t_num=args.t_num,
        fft_points=args.fft_points,
        a4=args.a4,
        timings=timings,
    )
    total = time.perf_counter() - total_start

    summary = spectrum.summary()
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        print()
    else:
        print(
            f"{summary['frames']} frames x {NOTE_COUNT} bands, {summary['frame_ms']:.0f} ms/frame, "
            f"{summary['duration']:.1f} s, sigma {summary['sigma']:.4g}"
        )
        print("peak bands: " + ", ".join(f"{band['note']} {band['energy']:.3f}" for band in summary["peak_bands"]))

    if args.threshold is not None:
        spans = auto_fill(spectrum.table, args.threshold)
        print(f"{len(spans)} note spans above {args.threshold}")

    if args.bench:
        print(f"total {total:.2f} s ({summary['duration'] / total:.1f}x realtime)")
        for stage, seconds in timings.items():
            print(f"  {stage:10} {seconds:7.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
