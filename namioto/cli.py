# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free task entry point; load only the task selected by the caller."""

from __future__ import annotations

import argparse
import importlib

COMMANDS = {
    "tempo": "namioto.analysis.beats",
    "wavetone": "namioto.analysis.wavetone",
    "tempocnn": "namioto.analysis.tempo",
    "spectrum": "namioto.analysis.spectrum",
    "game": "namioto.analysis.game",
    "align": "namioto.analysis.align",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="namioto-cli", description="Run Namioto tasks and automation without loading the GUI"
    )
    parser.add_argument("command", choices=COMMANDS, help="task to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="command arguments; use COMMAND --help for details")
    args = parser.parse_args(argv)
    return importlib.import_module(COMMANDS[args.command]).main(args.args)


if __name__ == "__main__":
    raise SystemExit(main())
