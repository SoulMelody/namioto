# SPDX-License-Identifier: AGPL-3.0-only
"""The task CLI keeps command arguments intact and loads no GUI or unrelated engines."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from namioto import cli


@pytest.mark.parametrize("command,module", cli.COMMANDS.items())
def test_dispatch_preserves_arguments_and_exit_status(monkeypatch, command, module):
    received = []
    arguments = ["song.wav", "--json", "--model", "a path with spaces", "--", "-input"]

    def run(argv):
        received.append(argv)
        return 7

    monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=run))
    assert cli.main([command, *arguments]) == 7
    assert received == [arguments]


@pytest.mark.parametrize("arguments", [[], ["unknown"]])
def test_missing_or_unknown_commands_fail(arguments):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("command", [None, *cli.COMMANDS])
def test_help_is_headless_and_loads_only_the_selected_engine(command):
    arguments = ["--help"] if command is None else [command, "--help"]
    script = """
import sys
from namioto import cli
try:
    cli.main(sys.argv[1:])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert not any(name.startswith(("PySide6", "namioto.ui")) for name in sys.modules)
selected = cli.COMMANDS.get(sys.argv[1])
assert all(module not in sys.modules for module in cli.COMMANDS.values() if module != selected)
"""
    result = subprocess.run([sys.executable, "-c", script, *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = "namioto-cli" if command is None else f"namioto-cli {command}"
    assert f"usage: {expected}" in result.stdout


def test_only_three_public_entry_points_are_registered():
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(project.read_text())["project"]["scripts"]
    assert scripts == {
        "namioto": "namioto.ui.app:main",
        "namioto-cli": "namioto.cli:main",
        "namioto-ctl": "namioto.ipc:main",
    }
