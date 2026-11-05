# SPDX-License-Identifier: AGPL-3.0-only
"""Named points on the timeline: the bookmarks a long score is navigated by.

A marker is a beat and a name, nothing else. It is deliberately not a `Note` and not part of
`Document`: `Document`'s contract is notes and channels, and its `index_by_voice`, `collides` and
`replace_notes` are all about notes. Bookmarks are a separate sorted sequence that the project
file stores beside the notes, which keeps a marker from ever colliding with an edit.

The list is mutable and keeps itself in beat order, so `nearest` is a search rather than a walk.
A repeated beat keeps one marker: a bookmark is a place, and two names for one place is a mistake
the caller does not need to be told about.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

DECIMALS = 4


@dataclass(frozen=True, order=True)
class Marker:
    """A named point on the timeline, in beats."""

    beat: float
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or f"{self.beat:.2f}"


class MarkerList:
    """Markers in beat order; the same beat holds one marker, the last name winning."""

    def __init__(self, markers=()) -> None:
        self._markers: list[Marker] = []
        for marker in markers:
            self.add(marker.beat, marker.name)

    def __len__(self) -> int:
        return len(self._markers)

    def __iter__(self):
        return iter(self._markers)

    def __bool__(self) -> bool:
        return bool(self._markers)

    def __repr__(self) -> str:
        return f"MarkerList({self._markers!r})"

    @property
    def markers(self) -> tuple[Marker, ...]:
        return tuple(self._markers)

    @property
    def _beats(self) -> list[float]:
        return [marker.beat for marker in self._markers]

    def add(self, beat: float, name: str = "") -> Marker:
        """Put a marker on `beat`; a marker already there takes the new name and keeps its place."""
        marker = Marker(round(float(beat), DECIMALS), str(name))
        index = bisect_left(self._beats, marker.beat)
        if index < len(self._markers) and self._markers[index].beat == marker.beat:
            self._markers[index] = marker
        else:
            self._markers.insert(index, marker)
        return marker

    def remove(self, marker: Marker) -> None:
        self.remove_at(marker.beat)

    def remove_at(self, beat: float) -> None:
        target = round(float(beat), DECIMALS)
        self._markers = [marker for marker in self._markers if marker.beat != target]

    def clear(self) -> None:
        self._markers.clear()

    def nearest(self, beat: float) -> Marker | None:
        """The marker closest to `beat`; a tie goes to the earlier one."""
        if not self._markers:
            return None
        beat = float(beat)
        index = bisect_left(self._beats, beat)
        if index == 0:
            return self._markers[0]
        if index == len(self._markers):
            return self._markers[-1]
        before, after = self._markers[index - 1], self._markers[index]
        return before if beat - before.beat <= after.beat - beat else after

    def next_after(self, beat: float) -> Marker | None:
        """The first marker past `beat`, or None at the end."""
        beat = float(beat)
        for marker in self._markers:
            if marker.beat > beat:
                return marker
        return None

    def previous_before(self, beat: float) -> Marker | None:
        """The last marker before `beat`, or None at the start."""
        beat = float(beat)
        found = None
        for marker in self._markers:
            if marker.beat < beat:
                found = marker
            else:
                break
        return found

    def shift(self, from_beat: float, delta: float) -> None:
        """Move every marker on or after `from_beat` by `delta`, re-sorting as they move."""
        from_beat = float(from_beat)
        moved = [
            Marker(marker.beat + float(delta), marker.name) if marker.beat >= from_beat else marker
            for marker in self._markers
        ]
        self._markers = []
        for marker in moved:
            self.add(marker.beat, marker.name)

    def to_list(self) -> list[dict]:
        return [{"beat": marker.beat, "name": marker.name} for marker in self._markers]

    @classmethod
    def from_list(cls, data) -> MarkerList:
        markers = []
        for entry in data or ():
            if not isinstance(entry, dict):
                continue
            try:
                markers.append(Marker(float(entry["beat"]), str(entry.get("name", ""))))
            except (KeyError, TypeError, ValueError):
                continue
        return cls(markers)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(prog="namioto.markers", description="show a marker list")
    parser.add_argument("beats", nargs="*", type=float, help="beats to place markers on")
    args = parser.parse_args(argv)
    markers = MarkerList(Marker(beat) for beat in args.beats)
    for marker in markers:
        print(f"{marker.beat:8.3f}  {marker.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
