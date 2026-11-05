# SPDX-License-Identifier: AGPL-3.0-only
"""Small helpers shared across the Qt-free modules: the app's own directories, the identity of a
file under a path, and the atomic writers a settings or project file is written with.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import platformdirs

# The name the platform's own directories are filed under: every path the program keeps is built
# from it, so a renamed program takes its files with it.
APP_DIRECTORY = "namioto"


def data_dir(*parts: str) -> Path:
    """A path under the app's data root, where derived results and downloaded models belong."""
    return Path(platformdirs.user_data_dir(APP_DIRECTORY), *parts)


def config_dir(*parts: str) -> Path:
    """A path under the app's config root, where preferences belong."""
    return Path(platformdirs.user_config_dir(APP_DIRECTORY), *parts)


def runtime_dir(*parts: str) -> Path:
    """A path under the app's runtime root, where a socket or a lock file belongs.

    `$XDG_RUNTIME_DIR` when the session has one - it is cleared on logout, which is exactly right
    for something that only exists while the program runs - and the config directory otherwise.
    """
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) / APP_DIRECTORY if base else config_dir()
    return root.joinpath(*parts)


def resolved(path: str | Path) -> Path:
    """`path` made absolute, with `~` and symlinks taken out, so two names for one file compare equal."""
    return Path(path).expanduser().resolve()


def file_stamp(path: str | Path) -> tuple[int, int]:
    """Size and mtime, so the same path holding a different file is not the same file."""
    try:
        info = resolved(path).stat()
    except OSError:
        return (0, 0)
    return (info.st_size, info.st_mtime_ns)


def write_text(path: str | Path, text: str) -> Path:
    """Write a whole file at once, so a half-written one never exists to be read."""
    return write_bytes(path, text.encode("utf-8"))


def write_bytes(path: str | Path, data: bytes) -> Path:
    """Atomically replace a file with a complete binary export."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return target


def write_json(data: dict, path: str | Path) -> Path:
    """Write a whole file at once, with the layout both settings and projects use."""
    return write_text(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
