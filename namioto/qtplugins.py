# SPDX-License-Identifier: AGPL-3.0-only
"""The plugin directories of the machine's own Qt, which the Qt inside a virtual environment never reads.

PySide6 carries a Qt of its own, and the only directory on its plugin search path is the one inside
the wheel: the styles, platform themes and icon engines the distribution installs - Breeze,
Kvantum, qt6ct - stay invisible to it, and the window falls back to Fusion. `system_dirs()` finds
the distribution's Qt 6, and adding what it returns to the application's library paths is all it
takes to have those plugins again. A directory that is missing, or that Qt refuses, changes nothing:
the wheel's own plugins are still there and are still what get drawn with.

Qt-free on purpose: a library path is read only while `QApplication` is being built, so the caller
sets them before it exists, and which directories are worth adding is decided here, without widgets.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from namioto.utils import resolved

# on a machine that has both, `qtpaths` is Qt 5's and answers with the Qt 5 directory Qt 6 cannot
# load, so only the Qt 6 name is asked for; some distributions keep it out of the PATH
QT_PATHS = ("qtpaths6", "/usr/lib/qt6/bin/qtpaths6")
# where the distributions that answer neither way put Qt 6's plugins; Debian and Ubuntu use a
# multiarch triplet, which is globbed rather than guessed
CANDIDATES = ("usr/lib/qt6/plugins", "usr/lib64/qt6/plugins", "usr/lib/*-linux-gnu/qt6/plugins")
PLATFORMS = "platforms"
TIMEOUT_SECONDS = 5.0


def system_dirs() -> list[Path]:
    """The distribution's Qt 6 plugin directories, in the order they should be searched."""
    found: list[Path] = []
    for command in QT_PATHS:
        answer = _ask(command)
        if answer is not None:
            if _is_plugins(answer):
                _add(found, answer)
            break
    for candidate in _candidates():
        if _is_plugins(candidate):
            _add(found, candidate)
    return found


def _ask(command: str) -> Path | None:
    """What one Qt 6 `qtpaths` says its plugin directory is, when it can be run at all."""
    try:
        done = subprocess.run(
            [command, "--plugin-dir"], capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answer = done.stdout.strip()
    return Path(answer) if done.returncode == 0 and answer else None


def _candidates() -> list[Path]:
    """Every place a distribution may keep Qt 6's plugins, whether or not one is there."""
    found: list[Path] = []
    for pattern in CANDIDATES:
        found.extend(Path("/").glob(pattern))
    return found


def _is_plugins(path: Path) -> bool:
    """Whether a directory is one Qt searches: every plugin directory carries the platform plugins."""
    return (path / PLATFORMS).is_dir()


def _add(found: list[Path], path: Path) -> None:
    """Keep a directory once, whichever name reached it."""
    if not any(resolved(known) == resolved(path) for known in found):
        found.append(path)
