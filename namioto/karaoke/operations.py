# SPDX-License-Identifier: AGPL-3.0-only
"""The authoritative mapping: one `match`, `merge` or `drop` per run of the line's Sounds.

A `SoundRef` names a Sound the way `natural_sounds` counts it: its line and its index within the
line. `Match` consumes `1..N` consecutive notes with one Sound, `Merge` consumes one note with `2..N`
consecutive Sounds, `Drop` consumes no note. Together the operations of a song are a partition - every
Sound is in exactly one, every note in exactly one `Match` or `Merge` - and that partition is the
mapping everything else is derived from. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from namioto.karaoke.model import KrcError

# the version of the mapping algorithm whose suggested operations a project may trust
MAPPING_VERSION = 8


@dataclass(frozen=True)
class SoundRef:
    """A Sound named by its line and its index in that line, as `natural_sounds` reads it."""

    line: int
    index: int

    def __str__(self) -> str:
        return f"{self.line}:{self.index}"


@dataclass(frozen=True)
class Match:
    """One Sound over `notes` consecutive notes, in the target NOTE stream's order."""

    sound: SoundRef
    notes: tuple[int, ...]
    confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.notes:
            raise KrcError("a match consumes at least one note")

    @property
    def sounds(self) -> tuple[SoundRef, ...]:
        return (self.sound,)

    @property
    def slots(self) -> int:
        return len(self.notes)

    @property
    def dot(self) -> int | None:
        """The `.N` to write back: omitted when the Sound takes one note, its note count else."""
        return None if len(self.notes) == 1 else len(self.notes)


@dataclass(frozen=True)
class Merge:
    """`sounds` consecutive Sounds sharing one note."""

    sounds: tuple[SoundRef, ...]
    note: int
    confirmed: bool = False

    def __post_init__(self) -> None:
        if len(self.sounds) < 2:
            raise KrcError("a merge consumes at least two sounds")

    @property
    def slots(self) -> int:
        return 1

    @property
    def dot(self) -> int | None:
        return 1


@dataclass(frozen=True)
class Drop:
    """One Sound that consumes no note."""

    sound: SoundRef
    confirmed: bool = False

    @property
    def sounds(self) -> tuple[SoundRef, ...]:
        return (self.sound,)

    @property
    def slots(self) -> int:
        return 0

    @property
    def dot(self) -> int | None:
        return 0


Operation = Match | Merge | Drop


def map_operation(operation: Operation, mapper) -> Operation:
    refs = tuple(mapper(ref) for ref in operation.sounds)
    return replace(operation, sounds=refs) if isinstance(operation, Merge) else replace(operation, sound=refs[0])


def remap_operation(operation: Operation, indices) -> Operation:
    return map_operation(operation, lambda ref: SoundRef(indices[ref.line], ref.index))


def keep_operations(operations: Sequence[Operation], version: int, current: int = MAPPING_VERSION) -> tuple:
    """The stored operations a mapping may keep: all while the algorithm is the same.

    A newer algorithm's suggestions are not trusted, but the user's confirmed operations are: they
    are kept and re-validated, and an inconsistent one is reported rather than migrated away.
    """
    if version == current:
        return tuple(operations)
    return tuple(operation for operation in operations if operation.confirmed)


def partition(operations: list[Operation], sounds_per_line: list[int]) -> list[list[Operation]]:
    """The operations sorted by line, one list per line, checked to cover every Sound exactly once.

    Raises `KrcError` when an operation leaves a Sound out, repeats one, crosses lines, holds
    non-consecutive Sounds, or does not follow the Sounds' order.
    """
    rows: list[list[Operation]] = [[] for _ in sounds_per_line]
    for operation in operations:
        refs = operation.sounds
        lines = {ref.line for ref in refs}
        indices = [ref.index for ref in refs]
        if len(lines) != 1:
            raise KrcError("an operation may not cross lines")
        line = lines.pop()
        if not 0 <= line < len(rows):
            raise KrcError(f"the mapping names a line that is not there: {line}")
        if indices != list(range(min(indices), min(indices) + len(indices))):
            raise KrcError(f"a {type(operation).__name__.lower()} names sounds of line {line} that do not run on")
        rows[line].append(operation)
    found: list[list[Operation]] = []
    for line, (here, count) in enumerate(zip(rows, sounds_per_line, strict=True)):
        ordered = sorted(here, key=lambda operation: min(ref.index for ref in operation.sounds))
        cursor = 0
        for operation in ordered:
            last = max(ref.index for ref in operation.sounds)
            if min(ref.index for ref in operation.sounds) != cursor:
                raise KrcError(f"the mapping does not cover the sounds of line {line} once, in order")
            cursor = last + 1
        if cursor != count:
            raise KrcError(f"the mapping leaves sounds of line {line} out")
        found.append(ordered)
    return found
