# SPDX-License-Identifier: AGPL-3.0-only
"""The bar lengths of a score over time, and the one place a beat becomes a bar number.

A song does not keep one meter. A verse in 4/4 hands over to a bridge in 3/4, a shuffle passage
counts in 6, and the bar lines the editor draws have to follow all of them. This module is the
whole of that: a sparse list of meter changes, each naming the beat it takes effect on, and the
conversions every caller needs - `bar_at` for a beat, `beat_at` for a bar.

It is deliberately *not* part of `TempoMap`. A `TempoPoint` is `(beat, bpm)` and `TempoMap` only
converts beats to seconds; folding a bar length into the same point would make one entry mean two
things at once and muddy `rescaled` and `with_change`. WaveTone keeps the two tables apart for the
same reason (`var_121` tempo against `var_135` rhythm, with `createrhtable` deriving the second),
and the roll and the MIDI writer read both at their boundary rather than one owning the other.

Bar numbers are 1-based: `bar_offset` is the number the first bar is *called*, so a map starting
at beat 0 with the default offset calls that bar 1.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

DEFAULT_BEATS_PER_BAR = 4.0
DECIMALS = 4


@dataclass(frozen=True, order=True)
class MeterPoint:
    """A bar length that takes effect on a beat and holds until the next point."""

    beat: float
    beats_per_bar: float = DEFAULT_BEATS_PER_BAR
    bar_offset: int = 1


@dataclass(frozen=True)
class MeterMap:
    """Meter changes in beat order, the first of them on beat 0.

    `points` is kept normalised the way `TempoMap` keeps its own: sorted, one point per beat, and a
    point on beat 0 whatever the caller handed over. Each point's starting bar number is worked out
    once, here, so a lookup is a search and an addition rather than a walk.
    """

    points: tuple[MeterPoint, ...] = (MeterPoint(0.0),)

    def __post_init__(self) -> None:
        points: list[MeterPoint] = []
        for point in sorted(self.points, key=lambda item: item.beat):
            beat = round(float(point.beat), DECIMALS)
            beats = float(point.beats_per_bar)
            if beats <= 0.0:
                raise ValueError(f"beats_per_bar must be positive, got {point.beats_per_bar!r}")
            normalised = MeterPoint(beat, beats, int(point.bar_offset))
            if points and beat == points[-1].beat:
                points[-1] = normalised  # a later change on the same beat wins
            else:
                points.append(normalised)
        if not points:
            points = [MeterPoint(0.0)]
        if points[0].beat != 0.0:
            points.insert(0, MeterPoint(0.0, points[0].beats_per_bar, points[0].bar_offset))

        starts = [points[0].bar_offset]
        for previous, current in zip(points, points[1:], strict=False):
            whole = (current.beat - previous.beat) / previous.beats_per_bar
            starts.append(starts[-1] + int(round(whole)))
        object.__setattr__(self, "points", tuple(points))
        object.__setattr__(self, "_beats", tuple(point.beat for point in points))
        object.__setattr__(self, "_starts", tuple(starts))

    @classmethod
    def single(cls, beats_per_bar: float = DEFAULT_BEATS_PER_BAR, bar_offset: int = 1) -> MeterMap:
        """One meter for the whole score: what a project with no meter changes holds."""
        return cls((MeterPoint(0.0, beats_per_bar, bar_offset),))

    @classmethod
    def from_pairs(cls, pairs) -> MeterMap:
        """A map from `[[beat, beats_per_bar, bar_offset], ...]`, the shape the project file stores."""
        points = []
        for pair in pairs or ():
            try:
                beat, beats = pair[0], pair[1]
                offset = pair[2] if len(pair) > 2 else 1
            except (TypeError, ValueError, IndexError):
                continue
            try:
                points.append(MeterPoint(float(beat), float(beats), int(offset)))
            except (TypeError, ValueError):
                continue
        return cls(tuple(points))

    def to_pairs(self) -> list[list]:
        """The points as `[[beat, beats_per_bar, bar_offset], ...]`, ready for the project file."""
        return [[point.beat, point.beats_per_bar, point.bar_offset] for point in self.points]

    @classmethod
    def from_dict(cls, data) -> MeterMap:
        return cls.from_pairs((data or {}).get("points", ()))

    def to_dict(self) -> dict:
        return {"points": self.to_pairs()}

    @property
    def is_single(self) -> bool:
        """True when one meter covers the score, which the project file stores by omission."""
        return len(self.points) == 1

    def _index_at(self, beat: float) -> int:
        return max(0, bisect_right(self._beats, round(float(beat), DECIMALS)) - 1)

    def beats_per_bar_at(self, beat: float) -> float:
        """The bar length in force on `beat`."""
        return self.points[self._index_at(beat)].beats_per_bar

    def bar_at(self, beat: float) -> tuple[int, float]:
        """The bar `beat` falls in and how far into it, as `(number, beat_in_bar)`."""
        beat = float(beat)
        index = self._index_at(beat)
        point = self.points[index]
        bars = int((beat - point.beat) // point.beats_per_bar)
        return self._starts[index] + bars, beat - point.beat - bars * point.beats_per_bar

    def beat_at(self, bar: int, beat_in_bar: float = 0.0) -> float:
        """The beat a bar number starts on; a bar past the end keeps the last meter running."""
        bar = int(bar)
        index = 0
        for candidate, start in enumerate(self._starts):
            if start <= bar:
                index = candidate
            else:
                break
        point = self.points[index]
        return point.beat + ((bar - self._starts[index]) * point.beats_per_bar) + float(beat_in_bar)

    def bars_between(self, start_beat: float, end_beat: float) -> list[int]:
        """Every bar number the span `[start_beat, end_beat]` touches, in order."""
        first = self.bar_at(start_beat)[0]
        last = self.bar_at(end_beat)[0]
        return list(range(first, last + 1))

    def bar_lines(self, left: float, right: float) -> list[tuple[float, int]]:
        """Every bar line in `[left, right)` as `(beat, number)`, for the roll to draw."""
        left, right = float(left), float(right)
        if right <= left:
            return []
        lines: list[tuple[float, int]] = []
        for index, point in enumerate(self.points):
            end = self.points[index + 1].beat if index + 1 < len(self.points) else None
            bar = self._starts[index]
            beat = point.beat
            while end is None or beat < end:
                if beat >= right:
                    break
                if beat >= left:
                    lines.append((beat, bar))
                beat += point.beats_per_bar
                bar += 1
            if end is not None and end < left:
                continue
        return lines

    def with_change(self, point: MeterPoint) -> MeterMap:
        """This map with a meter taking effect on `point.beat`, replacing one already there."""
        target = round(float(point.beat), DECIMALS)
        points = [item for item in self.points if item.beat != target]
        points.append(MeterPoint(target, point.beats_per_bar, point.bar_offset))
        return MeterMap(tuple(points))

    def without(self, beat: float) -> MeterMap:
        """This map with the change on `beat` removed; the first point cannot go."""
        target = round(float(beat), DECIMALS)
        points = [point for point in self.points if point.beat != target or point.beat == 0.0]
        if len(points) == len(self.points):
            return self
        return MeterMap(tuple(points))

    def rescaled(self, ratio: float) -> MeterMap:
        """The whole map stretched by `ratio`, keeping every bar number where it was.

        Only the beats move: a bar is still a bar, so `beats_per_bar` and `bar_offset` stay put and
        the numbering a reader is used to survives a tempo change.
        """
        ratio = float(ratio)
        if ratio == 1.0:
            return self
        return MeterMap(
            tuple(MeterPoint(point.beat * ratio, point.beats_per_bar, point.bar_offset) for point in self.points)
        )

    def main(self, argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
        import argparse

        parser = argparse.ArgumentParser(prog="namioto.meter_map", description="show a meter map")
        parser.add_argument("--bars", type=int, default=12, help="how many bars to print")
        parser.add_argument("--beats-per-bar", type=float, default=DEFAULT_BEATS_PER_BAR)
        parser.add_argument("--offset", type=int, default=1)
        args = parser.parse_args(argv)
        meter = MeterMap.single(args.beats_per_bar, args.offset)
        for bar in range(args.offset, args.offset + args.bars):
            print(f"{bar:4d}  beat {meter.beat_at(bar):8.3f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(MeterMap().main())
