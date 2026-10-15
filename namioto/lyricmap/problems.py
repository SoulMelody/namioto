# SPDX-License-Identifier: AGPL-3.0-only
"""The structured problems the lyric mapping reports, and the error that carries one.

A problem has a stable code the UI counts and locates, a message for a person, and the Sounds or
notes it concerns. The codes are the gate the exports read; the message and the colours are not.
`MappingError` is raised where a caller asked for a mapping the input cannot give.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# the stable codes the gate and the UI count
UNNORMALIZABLE_KRC = "unnormalizable_krc"
NO_LYRIC_SOUNDS = "no_lyric_sounds"
INCOMPLETE_ALIGNMENT = "incomplete_alignment"
NO_TARGET_NOTES = "no_target_notes"
FILTERED_NOTE = "filtered_note"
INVALID_ANCHOR = "invalid_anchor"
UNWRITABLE_MERGE = "unwritable_merge"
ROUND_TRIP_MISMATCH = "round_trip_mismatch"
UNCOVERED_NOTE = "uncovered_note"
MAPPING_PENDING = "mapping_pending"


@dataclass(frozen=True)
class Problem:
    """One locatable problem: its code, a message, and the refs it concerns."""

    code: str
    message: str = ""
    refs: tuple = field(default_factory=tuple)


class MappingError(Exception):
    """A mapping was refused; `.code` is the problem code that says why."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


__all__ = [
    "FILTERED_NOTE",
    "INCOMPLETE_ALIGNMENT",
    "INVALID_ANCHOR",
    "MAPPING_PENDING",
    "MappingError",
    "NO_LYRIC_SOUNDS",
    "NO_TARGET_NOTES",
    "Problem",
    "ROUND_TRIP_MISMATCH",
    "UNCOVERED_NOTE",
    "UNNORMALIZABLE_KRC",
    "UNWRITABLE_MERGE",
]
