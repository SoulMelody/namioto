# SPDX-License-Identifier: AGPL-3.0-only
"""Retain concrete KRC syntax while verifying edits against the semantic writer.

Local edits reuse the original text. Edits that cannot be patched independently
replace affected lines; structural rewrites retain comments and blank lines near
their original lyrics. Qt-free.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class Source:
    text: str
    canonical: str
    line_numbers: tuple[int, ...]


def _visible(text: str) -> tuple[str, list[int]]:
    chars, positions = [], []
    comment = False
    for index, char in enumerate(text):
        if char == "\n":
            comment = False
        elif char == "#":
            comment = True
        if not comment and char not in "\r\f\xa0":
            chars.append(char)
            positions.append(index)
    return "".join(chars), positions


def _matches(text: str, expected: str) -> bool:
    from namioto.karaoke.model import KrcError
    from namioto.karaoke.parser import parse
    from namioto.karaoke.writer import _canonical

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _canonical(parse(text), dotted=True) == expected
    except KrcError:
        return False


def restore(source: Source, canonical: str) -> str:
    if canonical == source.canonical:
        return source.text
    visible, positions = _visible(source.text)
    locations = {}
    for block in SequenceMatcher(None, source.canonical, visible, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            locations[block.a + offset] = positions[block.b + offset]
    edits = []
    for kind, start, end, new_start, new_end in SequenceMatcher(
        None, source.canonical, canonical, autojunk=False
    ).get_opcodes():
        if kind == "equal":
            continue
        if start == end:
            if start and start - 1 in locations:
                low = high = locations[start - 1] + 1
            elif start in locations:
                low = high = locations[start]
            else:
                break
        elif all(index in locations for index in range(start, end)):
            low, high = locations[start], locations[end - 1] + 1
            if source.text[low:high] != source.canonical[start:end]:
                break
        else:
            break
        replacement = canonical[new_start:new_end]
        if "\r\n" in source.text:
            replacement = replacement.replace("\n", "\r\n")
        edits.append((low, high, replacement))
    else:
        result = source.text
        for low, high, text in reversed(edits):
            result = result[:low] + text + result[high:]
        if _matches(result, canonical):
            return result
    return _restore_lines(source, canonical)


def _ending(line: str) -> str:
    return "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""


def _restore_lines(source: Source, canonical: str) -> str:
    old = source.canonical.split("\n")
    new = canonical.split("\n") if canonical else []
    parts = source.text.split("\n")
    raw = [line + "\n" for line in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    physical = dict(enumerate(source.line_numbers))
    newline = "\r\n" if "\r\n" in source.text else "\n"
    opcodes = SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    result = list(raw)
    changes = []
    for kind, start, end, new_start, new_end in opcodes:
        if kind == "equal":
            continue
        replacements = new[new_start:new_end]
        for offset, index in enumerate(range(start, end)):
            row = physical[index]
            original = raw[row]
            comment = original[original.index("#") :].rstrip("\r\n") if "#" in original else ""
            text = replacements[offset] if offset < len(replacements) else ""
            if text == "---" and comment:
                text = comment + newline + text
                comment = ""
            changes.append((row, row + 1, [text + comment + _ending(original)] if text or comment else []))
        extra = replacements[end - start :]
        if extra:
            row = physical[end - 1] + 1 if end > start else physical.get(start, physical[len(old) - 1] + 1)
            text = newline.join(extra)
            if row and not _ending(raw[row - 1]):
                text = newline + text
            if row < len(raw) or source.text.endswith("\n"):
                text += newline
            changes.append((row, row, [text]))
    for start, end, replacement in sorted(changes, key=lambda change: (change[0], change[1]), reverse=True):
        result[start:end] = replacement
    text = "".join(result)
    if _matches(text, canonical):
        return text
    anchors = {}
    for _kind, start, end, new_start, new_end in opcodes:
        for index in range(start, end):
            anchors[index] = min(new_start + index - start, new_end)
    rows = {row: index for index, row in physical.items()}
    trivia: dict[int, list[str]] = {}
    for row, original in enumerate(raw):
        if "#" in original:
            text = original[original.index("#") :].rstrip("\r\n")
        elif not original.strip():
            text = original.rstrip("\r\n")
        else:
            continue
        index = next((rows[at] for at in range(row, len(raw)) if at in rows), len(old))
        trivia.setdefault(anchors.get(index, len(new)), []).append(text)
    lines = []
    for index in range(len(new) + 1):
        lines.extend(trivia.get(index, []))
        if index < len(new):
            lines.append(new[index])
    text = newline.join(lines)
    return text + newline if source.text.endswith("\n") and lines else text
