# SPDX-License-Identifier: AGPL-3.0-only
"""The one gate every lyric export passes through: the issues that block it, and the canonical `.krc`.

A button, a colour or a caught exception must never decide on its own whether a mapping may leave the
editor, so the KRC and the subtitle both ask this. It reports locatable problems - a filtered note, a
low-confidence operation, an anchor that no longer holds, a merge the format cannot write - and, when
nothing is wrong, hands back the canonical `.krc` the subtitle is generated from. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from namioto.karaoke.canonical import rebuild
from namioto.karaoke.model import KrcError
from namioto.karaoke.operations import Drop, Operation, SoundRef, partition
from namioto.karaoke.sounds import SoundLine
from namioto.lyricmap import problems as codes
from namioto.lyricmap.confidence import read
from namioto.lyricmap.notes import Note
from namioto.lyricmap.problems import MappingError, Problem
from namioto.lyricmap.raw import Raw, validate


@dataclass(frozen=True)
class Gate:
    """A verification's outcome: the problems found, and the canonical `.krc` when it opens."""

    problems: tuple[Problem, ...] = ()
    canonical: str = ""

    def open(self) -> bool:
        return not self.problems

    def counts(self) -> dict[str, int]:
        found: dict[str, int] = {}
        for problem in self.problems:
            found[problem.code] = found.get(problem.code, 0) + 1
        return found


def verify(
    text: str,
    lines: Sequence[SoundLine],
    raw: Sequence[Sequence[Raw]],
    notes: Sequence[Note],
    operations: Sequence[Operation],
    *,
    filtered: Sequence = (),
    flagged: Sequence[bool] = (),
    broken: bool = False,
    subtitle: bool = False,
    pending: bool = False,
) -> Gate:
    """Everything that blocks the mapping's export, and the canonical `.krc` if nothing does.

    `broken` says the lyrics could not even be normalised into Sounds; `filtered` is the notes a
    conflict kept out of the stream; `subtitle` adds the subtitle's own condition that at least one
    Sound is actually shown. `pending` blocks export until the current mapping is ready.
    A low-confidence operation blocks only while it is unconfirmed.
    """
    if pending:
        return Gate((Problem(codes.MAPPING_PENDING, "the lyric mapping is still being calculated"),))
    if broken:
        return Gate((Problem(codes.UNNORMALIZABLE_KRC, "the lyrics do not normalise into sounds"),))
    issues: list[Problem] = []
    blocked = False
    sounds = [sound for line in lines for sound in line.sounds]
    if not sounds:
        issues.append(Problem(codes.NO_LYRIC_SOUNDS, "the lyrics have no sounds"))
        blocked = True
    if not notes:
        issues.append(Problem(codes.NO_TARGET_NOTES, "the target channel has no notes"))
        blocked = True
    issues.extend(Problem(codes.FILTERED_NOTE, "a note is kept out by a conflict", (note.id,)) for note in filtered)
    try:
        validate(lines, raw)
    except MappingError as error:
        issues.append(Problem(error.code, str(error)))
        return Gate(tuple(issues))
    if blocked:
        return Gate(tuple(issues))
    confirmed = [operation for operation in operations if operation.confirmed]
    if confirmed:
        from namioto.lyricmap.solver import _anchors, _context

        try:
            _anchors(confirmed, _context(lines, raw, list(notes)))
        except MappingError as error:
            issues.append(Problem(codes.INVALID_ANCHOR, str(error)))
            blocked = True
    try:
        partition(list(operations), [len(line.sounds) for line in lines])
    except KrcError as error:
        code = codes.INVALID_ANCHOR if confirmed else codes.UNCOVERED_NOTE
        issues.append(Problem(code, str(error)))
        blocked = True
    if not blocked and operations and sounds and notes:
        readings = read(lines, raw, notes, operations, flagged)
        for operation, reading in zip(operations, readings, strict=True):
            if reading.low and not operation.confirmed:
                issues.append(
                    Problem(reading.code, "the mapping is not sure of this operation", tuple(operation.sounds))
                )
    if not blocked:
        canonical = ""
        try:
            canonical = rebuild(text, operations)
        except KrcError as error:
            code = codes.ROUND_TRIP_MISMATCH if "round_trip_mismatch" in str(error) else codes.UNWRITABLE_MERGE
            issues.append(Problem(code, str(error)))
        if subtitle and operations and all(isinstance(operation, Drop) for operation in operations):
            issues.append(Problem(codes.ROUND_TRIP_MISMATCH, "the subtitle would have nothing to show"))
        if not issues:
            return Gate((), canonical)
    return Gate(tuple(issues))


def faithful_gate(
    spans: Sequence[Sequence[tuple | None]],
    notes: Sequence[Note],
    *,
    filtered: Sequence = (),
    subtitle: bool = False,
    dropped: Sequence[SoundRef] = (),
    required_slots: int = 0,
) -> Gate:
    """The faithful subtitle gate checks NOTE coverage without solver operations.

    The KRC itself may always be copied, so this concerns the subtitle alone: the target channel must
    have notes, no conflict may have filtered one out, and every Sound it is to render must have a
    definite time. `required_slots` detects NOTE counts that were truncated for display. Explicit
    `.0` Sounds are excluded from the missing-time check.
    """
    issues: list[Problem] = []
    if not notes:
        issues.append(Problem(codes.NO_TARGET_NOTES, "the target channel has no notes"))
    issues.extend(Problem(codes.FILTERED_NOTE, "a note is kept out by a conflict", (note.id,)) for note in filtered)
    omitted = set(dropped)
    missing = any(
        span is None or span[0] is None or span[1] is None
        for line, row in enumerate(spans)
        for index, span in enumerate(row)
        if SoundRef(line, index) not in omitted
    )
    if missing or required_slots > len(notes):
        issues.append(Problem(codes.INCOMPLETE_ALIGNMENT, "a sound has unmatched NOTE slots"))
    if subtitle and all(span is None for row in spans for span in row):
        issues.append(Problem(codes.NO_LYRIC_SOUNDS, "the subtitle would have nothing to show"))
    return Gate(tuple(issues))
