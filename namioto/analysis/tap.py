# SPDX-License-Identifier: AGPL-3.0-only
"""Tempo from a human: tapping, and the histogram a tapped tempo is checked against.

Automatic detection is not always right. A rubato passage, a track with a drum machine buried
under reverb, a live recording with a wandering downbeat - all of them can come back with a tempo
that is plausible but wrong, and no amount of extra analysis settles it. Tapping four times does.

This module is Qt-free and librosa-free on purpose: it takes timestamps and returns numbers, so the
dialog that collects the taps and the CLI that checks a file both call the same arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence

MIN_BPM = 30.0
MAX_BPM = 300.0
MAX_INTERVAL = 2.0  # a gap longer than this is a missed beat, not a slow one
HISTOGRAM_LOW = 60.0
HISTOGRAM_HIGH = 240.0
HISTOGRAM_STEP = 1.0


def _intervals(times: Sequence[float], window: int | None = None) -> list[float]:
    """The usable gaps between consecutive taps: positive, and no longer than a missed beat."""
    gaps = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
    usable = [gap for gap in gaps if 0.0 < gap <= MAX_INTERVAL]
    if window is not None and window > 0:
        usable = usable[-window:]
    return usable


def tap_tempo(times: Sequence[float], *, window: int | None = None) -> float | None:
    """The tempo a run of taps implies, or None when there are not two taps to compare.

    The median gap is what makes a stray double-click or a skipped tap harmless: one outlier cannot
    drag the estimate the way it would drag a mean.
    """
    gaps = _intervals(times, window)
    if not gaps:
        return None
    ordered = sorted(gaps)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    if median <= 0.0:
        return None
    return min(MAX_BPM, max(MIN_BPM, 60.0 / median))


def tempo_histogram(
    beats: Sequence[float],
    *,
    low: float = HISTOGRAM_LOW,
    high: float = HISTOGRAM_HIGH,
    step: float = HISTOGRAM_STEP,
) -> tuple[list[float], list[float]]:
    """A tempo histogram over `low..high`, as `(centres, weights)`.

    Gaps outside the range are skipped rather than clamped: clamping would pile every out-of-range
    gap onto the first or last bucket and invent a peak that no gap supports.
    """
    if step <= 0.0 or high <= low:
        return [], []
    count = int(round((high - low) / step)) + 1
    centres = [low + index * step for index in range(count)]
    weights = [0.0] * count
    for gap in _intervals(beats):
        bpm = 60.0 / gap
        if not low <= bpm <= high:
            continue
        index = int(round((bpm - low) / step))
        weights[min(count - 1, max(0, index))] += 1.0 / gap
    return centres, weights


def refine_bpm(beats: Sequence[float], target: float, *, span: float = 2.0, step: float = 0.01) -> float:
    """The tempo near `target` whose grid the beats sit on most evenly.

    A tapped or detected tempo is a starting point, not an answer: the true tempo is the one that
    leaves the least residual between the beats and an evenly spaced grid, so the search walks a
    small window around `target` and keeps the flattest fit.
    """
    usable = sorted(float(beat) for beat in beats)
    if len(usable) < 2 or step <= 0.0:
        return float(target)
    start = usable[0]
    offsets = [beat - start for beat in usable]

    def residual(bpm: float) -> float:
        grid = 60.0 / bpm
        total = 0.0
        for offset in offsets:
            error = offset % grid
            total += min(error, grid - error) ** 2
        return total

    best_bpm, best_score = float(target), residual(float(target))
    span = max(0.0, float(span))
    steps = int(round(span / step))
    for direction in (-1, 1):
        for index in range(1, steps + 1):
            candidate = target + direction * index * step
            if candidate <= MIN_BPM or candidate > MAX_BPM:
                continue
            score = residual(candidate)
            if score < best_score:
                best_bpm, best_score = candidate, score
    return best_bpm


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(prog="namioto.analysis.tap", description="tempo from tap times")
    parser.add_argument("times", nargs="*", type=float, help="tap times in seconds")
    parser.add_argument("--target", type=float, help="refine a BPM instead of just measuring one")
    args = parser.parse_args(argv)
    tempo = tap_tempo(args.times)
    if tempo is None:
        print("need at least two taps")
        return 1
    print(f"{tempo:.3f} BPM")
    if args.target is not None:
        print(f"{refine_bpm(args.times, args.target):.3f} BPM after refining around {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
