# SPDX-License-Identifier: AGPL-3.0-only
"""The tempo of a score over time, and the one place beats become seconds.

A song does not keep one tempo. A performance drifts, a rubato passage stretches, and a click
track is laid down at three speeds; the grid the editor draws has to follow all of them. This module
is the whole of that: a sparse list of tempo changes, each naming the beat it takes effect on, and
the two conversions every caller needs - `seconds_at` for a beat, `beats_at` for a second.

It is Qt-free and dependency-free on purpose. The roll draws with it, MIDI export converts with it,
the lyric mapping places its notes with it, and none of them keeps a second copy of a BPM. The file
format stores it beside the tempo section rather than inside it, because the tempo section is a
table of single values and this is not one.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass

DEFAULT_BPM = 120.0
MIN_BPM = 1.0
MAX_BPM = 1000.0
DECIMALS = 4


@dataclass(frozen=True, order=True)
class TempoPoint:
    """A tempo that takes effect on a beat and holds until the next point."""

    beat: float
    bpm: float


@dataclass(frozen=True)
class TempoMap:
    """Tempo changes in beat order, the first of them on beat 0.

    `points` is kept normalised: sorted, one point per beat, all tempos inside the allowed range,
    and a point on beat 0 whatever the caller handed over. The seconds each point starts at are
    worked out once, here, so a conversion is a search and a multiplication rather than a walk.
    """

    points: tuple[TempoPoint, ...] = (TempoPoint(0.0, DEFAULT_BPM),)

    def __post_init__(self) -> None:
        points: list[TempoPoint] = []
        for point in sorted(self.points, key=lambda item: item.beat):
            beat = max(0.0, round(float(point.beat), DECIMALS))
            bpm = min(MAX_BPM, max(MIN_BPM, round(float(point.bpm), DECIMALS)))
            if points and beat == points[-1].beat:
                points[-1] = TempoPoint(beat, bpm)  # a later change on the same beat wins
            else:
                points.append(TempoPoint(beat, bpm))
        if not points:
            points = [TempoPoint(0.0, DEFAULT_BPM)]
        if points[0].beat != 0.0:
            points.insert(0, TempoPoint(0.0, points[0].bpm))

        starts = [0.0]
        for previous, current in zip(points, points[1:], strict=False):
            starts.append(starts[-1] + (current.beat - previous.beat) * (60.0 / previous.bpm))
        object.__setattr__(self, "points", tuple(points))
        object.__setattr__(self, "_beats", tuple(point.beat for point in points))
        object.__setattr__(self, "_starts", tuple(starts))

    @classmethod
    def single(cls, bpm: float = DEFAULT_BPM) -> TempoMap:
        """One tempo for the whole score: what a project with no tempo changes holds."""
        return cls((TempoPoint(0.0, bpm),))

    @classmethod
    def from_pairs(cls, pairs) -> TempoMap:
        """A map from `[[beat, bpm], ...]`, the shape the project file stores."""
        points = []
        for pair in pairs or ():
            try:
                beat, bpm = pair
            except (TypeError, ValueError):
                continue
            points.append(TempoPoint(float(beat), float(bpm)))
        return cls(tuple(points))

    def to_pairs(self) -> list[list[float]]:
        return [[point.beat, point.bpm] for point in self.points]

    @property
    def bpm(self) -> float:
        """The tempo on beat 0: what a single-value control shows."""
        return self.points[0].bpm

    @property
    def is_single(self) -> bool:
        return len(self.points) == 1

    def _index_at_beat(self, beat: float) -> int:
        return max(0, bisect_right(self._beats, beat) - 1)

    def _index_at_seconds(self, seconds: float) -> int:
        return max(0, bisect_right(self._starts, seconds) - 1)

    def bpm_at(self, beat: float) -> float:
        return self.points[self._index_at_beat(beat)].bpm

    def seconds_per_beat(self, beat: float = 0.0) -> float:
        return 60.0 / self.bpm_at(beat)

    def seconds_at(self, beat: float) -> float:
        """When `beat` sounds, counted from the start of the score.

        A negative beat keeps the first tempo going backwards: the grid's offset slides by seconds,
        and that is a beat before the song begins.
        """
        if beat < 0.0:
            return beat * (60.0 / self.points[0].bpm)
        index = self._index_at_beat(beat)
        point = self.points[index]
        return self._starts[index] + (beat - point.beat) * (60.0 / point.bpm)

    def beats_at(self, seconds: float) -> float:
        """The beat that sounds at `seconds`; a negative time keeps the first tempo going backwards."""
        if seconds < 0.0:
            return seconds / (60.0 / self.points[0].bpm)
        index = self._index_at_seconds(seconds)
        point = self.points[index]
        return point.beat + (seconds - self._starts[index]) / (60.0 / point.bpm)

    def segments(self) -> Iterator[tuple[float, float | None, float]]:
        """(start beat, end beat, bpm) of every stretch of constant tempo; the last end is open."""
        for index, point in enumerate(self.points):
            end = self.points[index + 1].beat if index + 1 < len(self.points) else None
            yield point.beat, end, point.bpm

    def with_change(self, beat: float, bpm: float) -> TempoMap:
        """This map with a tempo taking effect on `beat`, replacing one already there."""
        target = max(0.0, round(float(beat), DECIMALS))
        points = [point for point in self.points if point.beat != target]
        points.append(TempoPoint(target, bpm))
        return TempoMap(tuple(points))

    def without(self, beat: float) -> TempoMap:
        """This map with the change on `beat` removed; the first point cannot go."""
        target = round(float(beat), DECIMALS)
        points = [point for point in self.points if point.beat != target or point.beat == 0.0]
        if len(points) == len(self.points):
            return self
        return TempoMap(tuple(points))

    def rescaled(self, bpm: float) -> TempoMap:
        """The whole map moved to another tempo: every point's beat and tempo scale together.

        Scaling both is what keeps the map describing the same song: twice the tempo means half the
        time for a beat, so a change that used to fall on beat 4 now falls on beat 8 and every note
        still sounds when it did.
        """
        ratio = max(MIN_BPM, float(bpm)) / self.bpm
        if ratio == 1.0:
            return self
        return TempoMap(
            tuple(TempoPoint(point.beat * ratio, point.bpm * ratio) for point in self.points)
        )
