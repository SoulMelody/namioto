# SPDX-License-Identifier: AGPL-3.0-only
"""Hand edits change literal KRC units and NOTE counts, never note identities or raw times.

Runs retain zero counts and unmatched slots. Moves preserve Sound order; only consecutive runs
may join, and only an edge Sound may leave a group. Qt-free.
"""

from collections.abc import Sequence
from dataclasses import replace

from namioto.karaoke.operations import NoteRun, SoundRef
from namioto.lyricmap.faithful import structure


def note_runs(text: str) -> tuple[NoteRun, ...]:
    return structure(text).runs


def operation_at(runs: Sequence[NoteRun], ref: SoundRef) -> NoteRun | None:
    return next((run for run in runs if ref in run.sounds), None)


def _replace(runs, old, new):
    return tuple(
        sorted(
            (*(run for run in runs if run not in old), *new), key=lambda run: (run.sounds[0].line, run.sounds[0].index)
        )
    )


def resize(runs: Sequence[NoteRun], ref: SoundRef, slots: int, *, left: bool = False) -> tuple[NoteRun, ...] | None:
    run = operation_at(runs, ref)
    if run is None or slots < 0:
        return None
    changed = replace(run, slots=slots)
    if not left:
        return _replace(runs, (run,), (changed,))
    previous = next((item for item in runs if item.sounds[-1] == SoundRef(ref.line, run.sounds[0].index - 1)), None)
    if previous is None or previous.slots + run.slots < slots:
        return None
    return _replace(runs, (previous, run), (replace(previous, slots=previous.slots + run.slots - slots), changed))


def move_boundary(runs: Sequence[NoteRun], left: SoundRef, right: SoundRef, slots: int) -> tuple[NoteRun, ...] | None:
    first, second = operation_at(runs, left), operation_at(runs, right)
    if first is None or second is None or first == second:
        return None
    if first.sounds[-1].line != second.sounds[0].line or first.sounds[-1].index + 1 != second.sounds[0].index:
        return None
    total = first.slots + second.slots
    if not 0 <= slots <= total:
        return None
    return _replace(runs, (first, second), (replace(first, slots=slots), replace(second, slots=total - slots)))


def detach(runs: Sequence[NoteRun], ref: SoundRef, slots: int = 1) -> tuple[NoteRun, ...] | None:
    run = operation_at(runs, ref)
    if run is None or ref not in (run.sounds[0], run.sounds[-1]) or slots < 0:
        return None
    remaining = tuple(sound for sound in run.sounds if sound != ref)
    replacements = (NoteRun((ref,), slots),)
    if remaining:
        replacements += (replace(run, sounds=remaining),)
    return _replace(runs, (run,), replacements)


def insert_sound(runs: Sequence[NoteRun], ref: SoundRef, target: NoteRun) -> tuple[NoteRun, ...] | None:
    source = operation_at(runs, ref)
    if source is None or source == target or target not in runs or ref not in (source.sounds[0], source.sounds[-1]):
        return None
    refs = tuple(sorted((*target.sounds, ref), key=lambda sound: (sound.line, sound.index)))
    if any(sound.line != ref.line for sound in refs) or [sound.index for sound in refs] != list(
        range(refs[0].index, refs[-1].index + 1)
    ):
        return None
    remaining = tuple(sound for sound in source.sounds if sound != ref)
    replacements = (replace(target, sounds=refs),)
    if remaining:
        replacements += (replace(source, sounds=remaining),)
    return _replace(runs, (source, target), replacements)
