# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the plugin directories of the machine's own Qt.

`QApplication` is built once for the whole suite, so nothing here changes the paths of the running
application: what is checked is the list that would be added.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from namioto import qtplugins

DETECT = qtplugins.system_dirs


@pytest.fixture(autouse=True)
def real_system_dirs(monkeypatch):
    """Detection is what this module checks, so the suite-wide stand-in is put back."""
    monkeypatch.setattr(qtplugins, "system_dirs", DETECT)


def make_plugins(path: Path) -> Path:
    """A directory shaped like one Qt searches, which is what the plugins check looks at."""
    (path / qtplugins.PLATFORMS).mkdir(parents=True)
    return path


def answer(monkeypatch, output: str, returncode: int = 0) -> None:
    """Have `qtpaths6` say `output`, as the command does on the line it prints."""
    done = subprocess.CompletedProcess(args=["qtpaths6"], returncode=returncode, stdout=output, stderr="")
    monkeypatch.setattr(qtplugins.subprocess, "run", lambda *_args, **_kwargs: done)


def test_the_answer_of_qtpaths6_is_the_first_directory(monkeypatch, tmp_path) -> None:
    found = make_plugins(tmp_path / "qt6")
    answer(monkeypatch, f"{found}\n")
    monkeypatch.setattr(qtplugins, "_candidates", lambda: [make_plugins(tmp_path / "other")])
    assert qtplugins.system_dirs() == [found, tmp_path / "other"]


def test_a_qtpaths6_that_fails_falls_back_to_the_candidates(monkeypatch, tmp_path) -> None:
    answer(monkeypatch, "/nonexistent/plugins\n", returncode=1)
    found = make_plugins(tmp_path / "qt6")
    monkeypatch.setattr(qtplugins, "_candidates", lambda: [found])
    assert qtplugins.system_dirs() == [found]


def test_a_directory_without_the_platform_plugins_is_not_one(monkeypatch, tmp_path) -> None:
    plain = tmp_path / "not-plugins"
    plain.mkdir()
    answer(monkeypatch, f"{plain}\n")
    monkeypatch.setattr(qtplugins, "_candidates", lambda: [plain])
    assert qtplugins.system_dirs() == []


def test_the_same_directory_under_two_names_is_kept_once(monkeypatch, tmp_path) -> None:
    found = make_plugins(tmp_path / "qt6")
    link = tmp_path / "link"
    link.symlink_to(found)
    answer(monkeypatch, f"{found}\n")
    monkeypatch.setattr(qtplugins, "_candidates", lambda: [link])
    assert qtplugins.system_dirs() == [found]


def test_nothing_found_is_nothing_added(monkeypatch) -> None:
    answer(monkeypatch, "/nonexistent/plugins\n", returncode=1)
    monkeypatch.setattr(qtplugins, "_candidates", lambda: [])
    assert qtplugins.system_dirs() == []
