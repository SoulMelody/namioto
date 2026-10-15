# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free ASS export from canonical KRC and encoded MIDI, shared by every entry point.

KRC channel wrappers select zero-based MIDI channels. Their NOTE slots are read faithfully after MIDI
serialization, so persistent and in-memory intermediate exports have identical timing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from namioto import midi
from namioto import settings as store
from namioto.karaoke import AssSettings, generate_ass
from namioto.karaoke.sounds import natural_sounds
from namioto.lyricmap import constraints, faithful
from namioto.lyricmap.channels import split
from namioto.lyricmap.notes import TimedNote, resolve
from namioto.lyricmap.verify import faithful_gate
from namioto.utils import resolved, write_bytes, write_text

ASS_FIELDS = next(section.fields for section in store.SECTIONS if section.name == "ass")


def validate_settings(settings: AssSettings) -> None:
    for field in ASS_FIELDS:
        value = getattr(settings, field.name)
        if field.name == "channel_colors":
            settings.channel_colours()
            continue
        if field.kind in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if field.kind == "int" and value != int(value):
                raise ValueError(f"{field.name} must be a whole number")
            if not field.low <= value <= field.high:
                raise ValueError(f"{field.name} must be between {field.low:g} and {field.high:g}")
        elif not isinstance(value, str):
            raise ValueError(f"{field.name} must be text")
        if field.name.endswith("_color"):
            if not value and field.name in ("overlay_blur_color", "base_blur_color"):
                continue
            colour = value.lstrip("#")
            if len(colour) != 6 or any(char not in "0123456789abcdefABCDEF" for char in colour):
                raise ValueError(f"{field.name} must be a six-digit RGB colour")
    if not settings.font or any(char in settings.font for char in ",\n\r"):
        raise ValueError("font must not be empty or contain commas or line breaks")
    if settings.fade_in_ms > settings.lead_time_ms:
        raise ValueError("fade_in_ms must not exceed lead_time_ms")


def build_ass(krc: str, data: bytes, *, settings: AssSettings | None = None, wavetone: bool = False) -> str:
    settings = settings or AssSettings()
    validate_settings(settings)
    lines = natural_sounds(krc)
    if not lines or not any(line.sounds for line in lines):
        raise ValueError("the lyrics have no sounds")
    imported = midi.read_bytes(data, wavetone=wavetone)
    if settings.guide_dot_duration_ms == AssSettings().guide_dot_duration_ms:
        duration = int(round(60 / imported.bpm * 4 * 1000))
        settings = replace(settings, guide_dot_duration_ms=duration if duration < 1800 else duration // 2)
    if imported.dropped or imported.left_out:
        raise ValueError("the MIDI contains incomplete notes or unsupported channels")
    spans = [[None] * len(line.sounds) for line in lines]
    for channel, part in split(krc).items():
        text = part.text
        notes = resolve(
            (
                TimedNote(note.start, note.start + note.duration, note.pitch, index)
                for index, note in enumerate(imported.notes)
                if note.channel == channel
            ),
            include_branches=False,
        )
        rows = faithful.read(text, notes.stream)
        dropped = tuple(ref for ref, (_run, slots) in constraints.read(text).items() if slots == 0)
        gate = faithful_gate(
            rows,
            notes.stream,
            filtered=notes.filtered,
            subtitle=True,
            dropped=dropped,
        )
        if not gate.open():
            raise ValueError(f"channel {channel}: " + "; ".join(problem.message for problem in gate.problems))
        for refs, row in zip(part.refs, rows, strict=True):
            for ref, span in zip(refs, row, strict=True):
                spans[ref.line][ref.index] = span
    return generate_ass(krc, spans, settings=settings)


def ass_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.suffix.lower() == ".ass" else target.with_name(target.name + ".ass")


def check_paths(outputs: Sequence[str | Path], inputs: Sequence[str | Path] = ()) -> None:
    paths = [resolved(path) for path in outputs]
    if len(set(paths)) != len(paths) or set(paths) & {resolved(path) for path in inputs}:
        raise ValueError("output paths must be distinct and must not overwrite input files")


def save_export(
    path: str | Path,
    krc: str,
    data: bytes,
    text: str,
    *,
    save_krc: str | Path | None = None,
    save_midi: str | Path | None = None,
) -> Path:
    target = ass_path(path)
    check_paths([target, *([save_krc] if save_krc else []), *([save_midi] if save_midi else [])])
    if save_krc:
        write_text(save_krc, krc)
    if save_midi:
        write_bytes(save_midi, data)
    return write_text(target, text)
