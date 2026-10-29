# SPDX-License-Identifier: AGPL-3.0-only
"""Tempo (BPM) estimation by beat tracking: librosa's dynamic-programming tracker, a least-squares
fit over the beats it finds, and a grid refit that turns the fit into sub-BPM precision.

Kept alongside it, as the runner-up, is the TempoCNN implementation in `namioto.analysis.tempo`.

`track` is the second path: it reads the onset envelope out of a note table the editor has already
computed, so a beat grid costs no second analysis of the audio. The onset envelope, the windowed
autocorrelation that gives a tempo per frame, and the downbeat search are ported from noteDigger
(`dataProcess/bpmEst.js`, GPL-3.0, see NOTICE).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
from scipy.signal import lfilter

if TYPE_CHECKING:
    from namioto.analysis.spectrum import NoteSpectrum

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

# the onset envelope's own settings: how much of the previous frame each frame keeps, how far a
# peak may stand above the 99th percentile before it is compressed, and the high-pass that takes
# the song's slow loudness changes out of it
ONSET_SMOOTHING = 0.8
OUTLIER_PERCENT = 0.99
OUTLIER_MARGIN = 1.3
DETREND_COEFFICIENT = 0.96
TEMPO_WINDOW_SECONDS = 8.0
TEMPO_HOP_SECONDS = 1.0
BPM_RANGE = (40.0, 200.0)
BEAT_STRENGTH_HALF_WINDOW = 2
METERS = (2, 3, 4)
# the downbeat search's own weights: a forgetting factor, how much an upbeat counts against a
# downbeat, the cost of changing meter, the pull towards 4/4, and how much history outweighs the
# local contrast
DOWNBEAT_ALPHA = 0.97
UPBEAT_WEIGHT = 0.2
METER_CHANGE_PENALTY = 0.6
COMMON_METER_BIAS = 0.02
HISTORY_WEIGHT = 0.25


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
    downbeats: tuple[float, ...] = ()
    per_frame_bpm: tuple[float, ...] = ()
    meter: int = 4

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


def _compress_outliers(values: np.ndarray, percent: float = OUTLIER_PERCENT, margin: float = OUTLIER_MARGIN) -> None:
    """Squash the few huge values in place, so one loud hit cannot flatten the rest of the envelope.

    Everything under the `percent` quantile is left alone; what stands above it is bent onto a cubic
    that leaves the quantile where it was and pulls the maximum down to `margin` times it.
    """
    ordered = np.sort(values)
    edge = ordered[min(len(ordered) - 1, int(len(ordered) * percent))]
    ceiling = edge * margin
    top = ordered[-1]
    if top <= ceiling or top <= edge:
        return
    span = top - edge
    target = ceiling - edge
    cubic = (span - 2 * target) / span**3
    quadratic = (3 * target - 2 * span) / span**2
    above = values > edge
    shifted = values[above] - edge
    values[above] = shifted * (cubic * shifted * shifted + quadratic * shifted + 1.0) + edge


def _detrend(values: np.ndarray, coefficient: float = DETREND_COEFFICIENT) -> None:
    """Take the slow drift out in place, with the high-pass applied forwards and backwards."""
    numerator, denominator = [1.0, -1.0], [1.0, -coefficient]
    values[:] = lfilter(numerator, denominator, values)
    values[:] = lfilter(numerator, denominator, values[::-1])[::-1]


def _normalise(values: np.ndarray) -> np.ndarray:
    """Shift the minimum to zero and divide by the standard deviation."""
    deviation = float(values.std())
    if deviation == 0.0:
        return np.zeros_like(values)
    return (values - values.min()) / deviation


def onset_envelope(table: np.ndarray) -> np.ndarray:
    """How much each frame's note table grew over the one before it.

    A note's attack raises its bands, a sustained note does not, so the rectified difference of the
    log table is the onset signal. Each frame is compared with a smoothed memory of the one before
    it (`ONSET_SMOOTHING`), which keeps a single noisy band from reading as an attack. The result is
    compressed, detrended and normalised, ready for autocorrelation.
    """
    frames = np.asarray(table, dtype=np.float64)
    if frames.ndim != 2 or len(frames) == 0:
        raise ValueError("A note table with at least one frame is needed")
    keep = 1.0 - ONSET_SMOOTHING
    # the memory starts at the first frame, not at silence: a file that opens on a note would
    # otherwise read that note as one enormous attack against a level it never had
    previous = np.log(frames[0] + 1e-6)
    flux = np.zeros(len(frames))
    for index in range(1, len(frames)):
        logged = np.log(frames[index] + 1e-6)
        difference = logged - previous
        flux[index] = difference[difference > 0.0].sum()
        previous = previous * keep + logged * ONSET_SMOOTHING
    _compress_outliers(flux)
    _detrend(flux)
    return _normalise(flux)


def tempo_curve(
    onset: np.ndarray,
    frame_rate: float,
    *,
    window_seconds: float = TEMPO_WINDOW_SECONDS,
    hop_seconds: float = TEMPO_HOP_SECONDS,
    bpm_range: tuple[float, float] = BPM_RANGE,
) -> np.ndarray:
    """One BPM per frame, from the autocorrelation of the window around it.

    The window is tapered and its autocorrelation read between the lags the BPM range allows; the
    best lag is refined by a parabola through its neighbours, so a tempo is not limited to whole
    frames. Frames before the first full window keep the first estimate.
    """
    if frame_rate <= 0.0 or len(onset) == 0:
        raise ValueError("A frame rate and at least one onset frame are needed")
    hop = max(1, round(frame_rate * hop_seconds))
    size = max(hop, round(frame_rate * window_seconds))
    low_lag = max(1, round(60.0 * frame_rate / bpm_range[1]))
    high_lag = max(low_lag + 2, round(60.0 * frame_rate / bpm_range[0]))
    window = np.hanning(size)
    curve = np.full(len(onset), 0.5 * (bpm_range[0] + bpm_range[1]))

    for start in range(0, len(onset), hop):
        segment = onset[start : start + size]
        if len(segment) < low_lag + 2:
            break
        padded = np.zeros(size)
        padded[: len(segment)] = segment * window[: len(segment)]
        spectrum = np.fft.rfft(padded, 2 * size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: high_lag + 2]
        lags = np.arange(low_lag, min(high_lag, len(correlation) - 2) + 1)
        if len(lags) == 0:
            break
        best = int(lags[int(np.argmax(correlation[lags]))])
        neighbours = correlation[best - 1], correlation[best], correlation[best + 1]
        curvature = neighbours[0] + neighbours[2] - 2.0 * neighbours[1]
        if curvature:
            best = best + 0.5 * (neighbours[0] - neighbours[2]) / curvature
        curve[start : start + hop] = 60.0 * frame_rate / max(1.0, best)
    return curve


def beat_strength(onset: np.ndarray, beats: np.ndarray, half_window: int = BEAT_STRENGTH_HALF_WINDOW) -> np.ndarray:
    """The strongest onset near each beat: how much a beat lands on something."""
    indices = np.clip(np.rint(np.asarray(beats, dtype=float)).astype(int), 0, len(onset) - 1)
    return np.array(
        [onset[max(0, index - half_window) : index + half_window + 1].max() for index in indices], dtype=np.float64
    )


def downbeats(
    strength: np.ndarray,
    meters: tuple[int, ...] = METERS,
    *,
    alpha: float = DOWNBEAT_ALPHA,
    upbeat_weight: float = UPBEAT_WEIGHT,
    change_penalty: float = METER_CHANGE_PENALTY,
    common_bias: float = COMMON_METER_BIAS,
    history_weight: float = HISTORY_WEIGHT,
) -> tuple[int, tuple[int, ...]]:
    """Which beats open a bar, and how many beats a bar has.

    Every state is one (meter, phase) pair and only advances to the next phase, or to phase 0 of
    another meter once the old bar has ended, so a bar is never cut in half. A state scores the
    contrast between the downbeats it has seen and the upbeats, with a forgetting factor so a
    change of feel can still win; the path with the best score names the bars.
    """
    values = np.asarray(strength, dtype=np.float64)
    if len(values) == 0:
        return 0, ()
    values = _normalise(values)
    count = len(values)
    # a beat with no onset at all is the file's own edge, not evidence about the bar: it is
    # advanced through but never counted as a downbeat or as an upbeat
    floor = 0.05 * float(np.median(values))
    states = [(meter, phase) for meter in meters for phase in range(meter)]

    # s and c are the downbeats' weighted sum and count, su and cu the upbeats', and a the score
    history: list[dict[tuple[int, int], tuple[float, float, float, float, float, tuple[int, int] | None]]] = []
    opening: dict[tuple[int, int], tuple[float, float, float, float, float, tuple[int, int] | None]] = {}
    first = float(values[0])
    seen = first > floor
    for meter, phase in states:
        if phase == 0:
            opening[(meter, phase)] = (
                first if seen else 0.0,
                1.0 if seen else 0.0,
                0.0,
                0.0,
                first if seen else 0.0,
                None,
            )
        else:
            opening[(meter, phase)] = (
                0.0,
                0.0,
                first if seen else 0.0,
                1.0 if seen else 0.0,
                -upbeat_weight * first if seen else 0.0,
                None,
            )
    history.append(opening)

    for index in range(1, count):
        strength_now = float(values[index])
        current: dict[tuple[int, int], tuple[float, float, float, float, float, tuple[int, int] | None]] = {}
        for meter, phase in states:
            best_score = -math.inf
            best: tuple[float, float, float, float, float, tuple[int, int] | None] | None = None
            for previous_meter, previous_phase in states:
                state = history[index - 1].get((previous_meter, previous_phase))
                if state is None:
                    continue
                switching = False
                if previous_meter == meter:
                    if phase != (previous_phase + 1) % meter:
                        continue
                elif previous_phase == previous_meter - 1 and phase == 0:
                    switching = True
                else:
                    continue
                downbeat = phase == 0
                observed = strength_now > floor
                count_downbeats = state[1] * alpha + (1.0 if downbeat and observed else 0.0)
                sum_upbeats = state[2] * alpha + (strength_now if not downbeat and observed else 0.0)
                sum_downbeats = state[0] * alpha + (strength_now if downbeat and observed else 0.0)
                count_upbeats = state[3] * alpha + (1.0 if not downbeat and observed else 0.0)
                average_downbeat = sum_downbeats / count_downbeats if count_downbeats > 0 else 0.0
                average_upbeat = sum_upbeats / count_upbeats if count_upbeats > 0 else 0.0
                contrast = average_downbeat - upbeat_weight * average_upbeat
                if switching:
                    contrast *= change_penalty
                    sum_upbeats = sum_upbeats * 1.05 + 4.0
                if 4 in meters and meter == 4:
                    contrast += common_bias
                score = state[4] * history_weight + contrast * (1.0 - history_weight)
                if score > best_score:
                    best_score = score
                    best = (
                        sum_downbeats,
                        count_downbeats,
                        sum_upbeats,
                        count_upbeats,
                        score,
                        (previous_meter, previous_phase),
                    )
            if best is not None:
                current[(meter, phase)] = best
        history.append(current)

    if not history[-1]:
        return 0, ()
    last = max(history[-1], key=lambda key: history[-1][key][4])
    # the meter the song ended in, not the one it opened in: a path may change meter at a bar line
    final_meter = last[0]
    indices = []
    cursor: tuple[int, int] | None = last
    for index in range(count - 1, -1, -1):
        if cursor is None:
            break
        _meter, phase = cursor
        if phase == 0:
            indices.append(index)
        cursor = history[index][cursor][5]
    indices.reverse()
    return final_meter, tuple(indices)


def track(
    spectrum: NoteSpectrum,
    *,
    window_seconds: float = TEMPO_WINDOW_SECONDS,
    hop_seconds: float = TEMPO_HOP_SECONDS,
    bpm_range: tuple[float, float] = BPM_RANGE,
) -> BeatTempo:
    """Tempo, beats and downbeats read from a note spectrum, without decoding the audio again."""
    frame_rate = 1000.0 / spectrum.frame_ms
    onset = onset_envelope(spectrum.table)
    if len(onset) < MIN_BEATS:
        raise ValueError(f"Found only {len(onset)} frames, which is too few to track beats over")
    curve = tempo_curve(onset, frame_rate, window_seconds=window_seconds, hop_seconds=hop_seconds, bpm_range=bpm_range)
    _, frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=frame_rate,
        hop_length=1,
        bpm=curve,
        units="frames",
        trim=False,
    )
    beat_frames = np.asarray(frames, dtype=int)
    if len(beat_frames) < MIN_BEATS:
        raise ValueError(f"Found only {len(beat_frames)} beats, which is too few to fit a tempo to")
    beats = beat_frames / frame_rate
    bpm, residual = fit_beats(beats)
    meter, indices = downbeats(beat_strength(onset, beat_frames))
    return BeatTempo(
        bpm=refine_grid(beats, bpm),
        beats=tuple(beats.tolist()),
        local=local_windows(beats, window_seconds, hop_seconds),
        residual=residual,
        downbeats=tuple(float(beats[index]) for index in indices),
        per_frame_bpm=tuple(curve.tolist()),
        meter=meter,
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
