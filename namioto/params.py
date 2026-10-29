# SPDX-License-Identifier: AGPL-3.0-only
"""The primitive behind every parameter table: one `Field` per value, and the checks on it.

Qt-free on purpose. `namioto.settings` builds the app's own preferences from a table of these, and
`namioto.project` builds a document's values the same way; the align and transcription windows
describe their parameter files with the same `Field`, so a value is read back, checked and written
out by one implementation everywhere.

`coerce` brings a bad value back in line: a wrong type falls to the field's default, an out-of-range
one is clamped, and a number is snapped to its step. `build` turns a table of `Section`s into the
dataclass the rest of the program uses, and hangs the spec on the class so `get_value` / `set_value`
can find a field without being told where it came from.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Any

from namioto.utils import write_json

TEXT_LIMIT = 4096


@dataclass(frozen=True)
class Field:
    """One parameter: its default, how to read it back, and how a form shows it."""

    name: str
    kind: str  # bool, int, float, choice, device, text or style
    default: Any
    caption: str
    tooltip: str = ""
    low: float = 0.0
    high: float = 0.0
    step: float = 0.0  # values snap to it, 0 meaning they do not
    decimals: int = 3
    choices: tuple = ()
    labels: tuple[str, ...] = ()  # what to show for each choice, the choices themselves when empty
    suffix: str = ""
    # a document value that says more about the user than about the song, so the next document starts from it
    reuse: bool = False


@dataclass(frozen=True)
class Section:
    name: str  # the key the value sits under in the file
    title: str  # what a form calls the group
    fields: tuple[Field, ...]


def _python_type(kind: str) -> type:
    return {"bool": bool, "int": int, "float": float}.get(kind, str)


def build(sections: Sequence[Section]) -> type:
    """The dataclass a table describes, carrying the table so a value can be set by name later."""
    made = {
        section.name: make_dataclass(
            section.title,
            [(item.name, _python_type(item.kind), field(default=item.default)) for item in section.fields],
        )
        for section in sections
    }
    model = make_dataclass(
        "Settings",
        [(section.name, made[section.name], field(default_factory=made[section.name])) for section in sections],
    )
    model.__field_specs__ = specs(sections)
    return model


def specs(sections: Sequence[Section]) -> dict[tuple[str, str], Field]:
    """Every field of a table, keyed by the section and name it sits under."""
    return {(section.name, item.name): item for section in sections for item in section.fields}


def get_value(model: Any, section: str, name: str) -> Any:
    return getattr(getattr(model, section), name)


def set_value(model: Any, section: str, name: str, value: Any) -> None:
    """Put `value` on one field, first making it fit the field's type and range."""
    setattr(getattr(model, section), name, coerce(type(model).__field_specs__[(section, name)], value))


def coerce(spec: Field, value: Any) -> Any:
    """A file may hold anything at all, so every value is checked before it is used."""
    if spec.kind == "bool":
        return value if isinstance(value, bool) else spec.default
    if spec.kind in ("choice", "device"):
        # a bool would pass for 0 or 1 here, and the synthesiser would be sent a true instead of a number
        return value if not isinstance(value, bool) and value in spec.choices else spec.default
    if spec.kind in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return spec.default
        number = min(spec.high, max(spec.low, float(value)))
        number = round(number / spec.step) * spec.step if spec.step else number
        return int(round(number)) if spec.kind == "int" else round(number, spec.decimals)
    if value is None:  # a null is a missing text, not the word "None"
        return spec.default
    return str(value).strip()[:TEXT_LIMIT]


def defaults(fields: Sequence[Field]) -> dict[str, Any]:
    """The values a field table starts from."""
    return {item.name: item.default for item in fields}


def coerce_values(fields: Sequence[Field], values: Any) -> dict[str, Any]:
    """A file may hold anything at all, so every value goes through its field's own check."""
    if not isinstance(values, dict):
        return defaults(fields)
    return {item.name: coerce(item, values.get(item.name, item.default)) for item in fields}


def load_values(path: str | Path, fields: Sequence[Field]) -> dict[str, Any]:
    """What a dialog opens with: the file's values, or the defaults for anything unreadable."""
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults(fields)
    except (OSError, ValueError) as error:
        warnings.warn(f"{target} could not be read ({error}); defaults are in use", stacklevel=2)
        return defaults(fields)
    return coerce_values(fields, data)


def save_values(path: str | Path, fields: Sequence[Field], values: Any) -> Path:
    """Write one field table's values out whole, checked first."""
    return write_json(coerce_values(fields, values), path)


def clone(model: Any) -> Any:
    """A copy to edit, so a cancelled dialog leaves nothing behind."""
    return copy.deepcopy(model)
