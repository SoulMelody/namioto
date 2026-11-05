# SPDX-License-Identifier: AGPL-3.0-only
"""The window's own state between runs: where it sat, the last folder a file came from, and the
values worth carrying from the last project to the next one.

Qt-free on purpose. No preference lives here - a value the user can choose on is in
`namioto.settings` - and nothing here is a song's own: `project` is only the reusable values the user
changed themselves, and `namioto.project` decides which fields those are and checks them when it
reads them back. The window writes this file as it goes, so it is not made to be edited by hand.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from namioto.params import TEXT_LIMIT
from namioto.utils import config_dir, write_json


@dataclass
class State:
    geometry: str = ""
    last_audio_dir: str = ""
    project: dict[str, dict] = field(default_factory=dict)
    # the A-B loop the transport repeats, in seconds on the playhead's own clock. Session state, not
    # song data: a project file never carries it, and arming or moving it must not make the project
    # dirty - the same rule that keeps view and volume changes out of the dirty flag. None means the
    # loop is off. Beats were the other option; seconds keep it clear of `MeterMap`, which is song data.
    loop: tuple[float, float] | None = None


def default_path() -> Path:
    """`$NAMIOTO_STATE`, else the platform's config directory, beside the preferences."""
    from_env = os.environ.get("NAMIOTO_STATE")
    if from_env:
        return Path(from_env)
    return config_dir("state.json")


def _text(value) -> str:
    return str(value).strip()[:TEXT_LIMIT] if isinstance(value, str) else ""


def _project(value) -> dict[str, dict]:
    """The remembered project values: a section map of charts, anything else dropped."""
    if not isinstance(value, dict):
        return {}
    return {name: values for name, values in value.items() if isinstance(values, dict)}


def _loop(value) -> tuple[float, float] | None:
    """Two seconds, A before B: anything else means the loop is off rather than half on."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    start, end = (float(item) for item in value)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return (start, end)


def from_dict(data) -> State:
    if not isinstance(data, dict):
        return State()
    return State(
        geometry=_text(data.get("geometry")),
        last_audio_dir=_text(data.get("last_audio_dir")),
        project=_project(data.get("project")),
        loop=_loop(data.get("loop")),
    )


def load(path: str | Path | None = None) -> State:
    """Read the window's state, falling back to the defaults for anything missing or unusable."""
    target = Path(path) if path is not None else default_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return State()
    except (OSError, ValueError) as error:
        warnings.warn(f"{target} could not be read as window state ({error}); defaults are in use", stacklevel=2)
        return State()
    return from_dict(data)


def save(state: State, path: str | Path | None = None) -> Path:
    """Write the window's state out whole: a half-written file would be read as a broken one."""
    data = {
        "geometry": state.geometry,
        "last_audio_dir": state.last_audio_dir,
        "project": state.project,
        "loop": list(state.loop) if state.loop else None,
    }
    return write_json(data, Path(path) if path is not None else default_path())
