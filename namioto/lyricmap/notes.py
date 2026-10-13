# SPDX-License-Identifier: AGPL-3.0-only
"""The single voice the lyrics may map to: the target channel's notes, and the one path through them.

A lyric mapping needs one NOTE per instant. Where the target channel's notes overlap - any pitch
overlapping any other - the notes that stay are chosen as a maximal non-overlapping path through each
connected region of the overlap graph, and the rest are *filtered*: they are a blocking error the user
resolves by fixing the MIDI, never by confirming them away. Which path is chosen is the mapping's own
cost when it is given one; among equal costs the path that keeps more notes wins, then the one whose
note ids read smaller. Qt-free.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from namioto.document import OVERLAP_SLACK


class TimedNote(NamedTuple):
    """A target-channel note where the mapping reads it: seconds, pitch and stable id."""

    start: float
    end: float
    pitch: int
    id: int


class Note(Protocol):
    """What the stream reads of a note: its times, its pitch and its stable id.

    `document.Note` and `project.Note` both satisfy it; only `document.Note` has an `end` property,
    which the callers here pass as beats.
    """

    start: float
    end: float
    pitch: int
    id: int


@dataclass(frozen=True)
class Branch:
    """One candidate path through a conflict region, and what the mapping makes of it."""

    notes: tuple
    cost: float = 0.0


@dataclass(frozen=True)
class Resolved:
    """The target channel as a single voice: the notes that stay, and the ones filtered out.

    `branches` holds, per conflict region, every maximal path the UI may offer as a preview branch.
    """

    stream: tuple
    filtered: tuple
    branches: tuple[tuple[Branch, ...], ...] = ()


def ordered(notes: Iterable[Note]) -> list[Note]:
    """The notes in their one deterministic order: start, end, pitch, persistent id."""
    return sorted(notes, key=lambda note: (note.start, note.end, note.pitch, note.id))


def overlaps(one: Note, other: Note) -> bool:
    """Whether two notes are one instant: their spans cross by more than `OVERLAP_SLACK`, any pitch.

    Only touching or overlapping by no more than the slack is legal.
    """
    return one.start < other.end - OVERLAP_SLACK and other.start < one.end - OVERLAP_SLACK


def components(notes: Iterable[Note]) -> list[list[Note]]:
    """The connected regions of the overlap graph, each in `ordered` order.

    The notes are swept in start order: a note whose start is under the running end of the region
    overlaps the note that end belongs to, so it joins; any later start opens a new region, because
    every note before it ends no later than the running end.
    """
    regions: list[list[Note]] = []
    current: list[Note] = []
    reach = float("-inf")
    for note in ordered(notes):
        if current and note.start < reach - OVERLAP_SLACK:
            current.append(note)
            reach = max(reach, note.end)
            continue
        if current:
            regions.append(current)
        current = [note]
        reach = note.end
    if current:
        regions.append(current)
    return regions


def resolve(
    notes: Iterable[Note], cost: Callable[[tuple], float] | None = None, *, include_branches: bool = True
) -> Resolved:
    """Reduce the notes to one voice, choosing each region's path by `cost` and its tie-breaks.

    `cost(path)` is the mapping's own time cost for a candidate path; a caller that only wants a
    preview may leave it out, and the paths then tie, so the one keeping more notes and the smaller
    ids wins. Without `include_branches`, the default-cost path is selected by interval scheduling.
    """
    regions = components(notes)
    kept: list[Note] = []
    filtered: list[Note] = []
    branches: list[tuple[Branch, ...]] = []
    for region in regions:
        if len(region) == 1:
            kept.append(region[0])
            continue
        if cost is None and not include_branches:
            chosen = _best_path(region)
        else:
            priced = [Branch(path, 0.0 if cost is None else cost(path)) for path in _maximal_paths(region)]
            if not priced:
                continue
            chosen = _best(priced).notes
            if include_branches:
                branches.append(tuple(priced))
        kept.extend(chosen)
        chosen_ids = {id(note) for note in chosen}
        filtered.extend(note for note in region if id(note) not in chosen_ids)
    return Resolved(tuple(kept), tuple(filtered), tuple(branches))


def _best_path(region: list[Note]) -> tuple[Note, ...]:
    """Select the longest path with the smallest ids, retaining exhaustive semantics below the slack."""
    if any(note.end - note.start <= OVERLAP_SLACK for note in region):
        return _best([Branch(path) for path in _maximal_paths(region)]).notes
    starts = [note.start for note in region]
    size = len(region)
    counts = [0] * (size + 1)
    heads = [None] * (size + 1)
    links = [None] * size

    def preferred(left, right):
        while left is not None and right is not None:
            if region[left].id != region[right].id:
                return region[left].id < region[right].id
            left, right = links[left], links[right]
        return left is None

    for index in range(size - 1, -1, -1):
        following = bisect_left(starts, region[index].end - OVERLAP_SLACK, lo=index + 1)
        links[index] = heads[following]
        count = 1 + counts[following]
        if count > counts[index + 1] or (count == counts[index + 1] and preferred(index, heads[index + 1])):
            counts[index], heads[index] = count, index
        else:
            counts[index], heads[index] = counts[index + 1], heads[index + 1]
    found = []
    index = heads[0]
    while index is not None:
        found.append(region[index])
        index = links[index]
    return tuple(found)


def _best(branches: Sequence[Branch]) -> Branch:
    """The path with the smallest cost, then the most notes, then the smallest note ids."""
    return min(branches, key=lambda branch: (branch.cost, -len(branch.notes), tuple(_ids(branch.notes))))


def _ids(notes: Sequence[Note]) -> list[int]:
    return [note.id for note in notes]


def _maximal_paths(region: list[Note]) -> Iterator[tuple[Note, ...]]:
    """Every maximal set of pairwise non-overlapping notes in one conflict region.

    The region is small - a handful of notes crossing in time - so the search branches on each note
    in turn; a branch that leaves a note addable is not maximal and is dropped at the leaf.
    """
    seen: set[tuple[int, ...]] = set()

    def leaf(chosen: list[Note]) -> bool:
        return all(note in chosen or any(overlaps(note, other) for other in chosen) for note in region)

    def walk(available: list[Note], chosen: list[Note]) -> Iterator[tuple[Note, ...]]:
        if not available:
            if leaf(chosen):
                key = tuple(_ids(chosen))
                if key not in seen:
                    seen.add(key)
                    yield tuple(chosen)
            return
        first, rest = available[0], available[1:]
        free = [note for note in rest if not overlaps(note, first)]
        yield from walk(free, [*chosen, first])
        if any(overlaps(note, first) for note in rest):
            yield from walk(rest, chosen)

    yield from walk(list(region), [])
