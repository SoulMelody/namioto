# SPDX-License-Identifier: AGPL-3.0-only
"""Build ASS from files or an IPC export snapshot, optionally rebuilding after changes.

Session watching polls the connected editor's export revision; it never discovers a new endpoint
or reacts to playback position. File watching follows paths across atomic replacements.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from namioto import ipc, subtitle
from namioto import settings as store
from namioto.karaoke import AssSettings
from namioto.utils import file_stamp

POLL_SECONDS = 0.1
DEBOUNCE_SECONDS = 0.3


def resolve_settings(arguments: argparse.Namespace, session: Mapping | None = None) -> AssSettings:
    settings = AssSettings(**session) if session is not None else AssSettings()
    overrides = {}
    for field in subtitle.ASS_FIELDS:
        value = getattr(arguments, field.name, None)
        if value is None:
            value = os.environ.get(f"NAMIOTO_ASS_{field.name.upper()}")
            if value is not None:
                convert = {"int": int, "float": float}.get(field.kind, str)
                try:
                    value = convert(value)
                except ValueError as error:
                    raise ValueError(f"NAMIOTO_ASS_{field.name.upper()} must be {field.kind}") from error
        if value is not None:
            overrides[field.name] = value
    settings = replace(settings, **overrides)
    subtitle.validate_settings(settings)
    return settings


def _attempt(build: Callable[[], Path]) -> bool:
    try:
        target = build()
    except ipc.IpcError as error:
        if error.code != "failed":
            raise
        print(f"namioto-cli ass: {error.message}", file=sys.stderr, flush=True)
        return False
    except (OSError, ValueError, EOFError) as error:
        print(f"namioto-cli ass: {error}", file=sys.stderr, flush=True)
        return False
    print(f"Exported {target}", flush=True)
    return True


def watch(build: Callable[[], Path], changed: Callable[[], object]) -> None:
    previous = changed()
    _attempt(build)
    due = None
    while True:
        time.sleep(POLL_SECONDS)
        current = changed()
        now = time.monotonic()
        if current != previous:
            previous = current
            due = now + DEBOUNCE_SECONDS
        if due is not None and now >= due:
            _attempt(build)
            due = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="namioto-cli ass", description="Build ASS from KRC and MIDI exports")
    parser.add_argument("--session", action="store_true", help="use the running editor's KRC and MIDI exports")
    parser.add_argument("--midi", type=Path, help="input MIDI file")
    parser.add_argument("--krc", type=Path, help="input KRC file")
    parser.add_argument("--out", type=Path, required=True, help="output ASS file")
    parser.add_argument("--watch", action="store_true", help="rebuild after changes; Ctrl+C stops watching")
    parser.add_argument("--save-krc", type=Path, help="persist the session's intermediate KRC export")
    parser.add_argument("--save-midi", type=Path, help="persist the session's intermediate MIDI export")
    parser.add_argument(
        "--wavetone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="read file input with WaveTone's leading bar removed (default: enabled)",
    )
    parser.add_argument("-e", "--endpoint", help=f"session address; overrides ${ipc.ENV_ENDPOINT}")
    parser.add_argument("--token", help=f"TCP session token; overrides ${ipc.ENV_TOKEN}")
    parser.add_argument("--run", help=f"session run file; overrides ${ipc.ENV_RUN}")
    parser.add_argument("-t", "--timeout", type=float, default=ipc.CONNECT_TIMEOUT, help="IPC timeout in seconds")
    for field in subtitle.ASS_FIELDS:
        convert = {"int": int, "float": float}.get(field.kind, str)
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=convert,
            help=f"{field.caption}; overrides $NAMIOTO_ASS_{field.name.upper()}",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.session:
        if args.midi or args.krc or args.wavetone is not None:
            parser.error("--session cannot be combined with --midi, --krc or --[no-]wavetone")
    else:
        if args.midi is None or args.krc is None:
            parser.error("provide both --midi and --krc, or use --session")
        if args.save_krc or args.save_midi or args.endpoint or args.token or args.run:
            parser.error("session connection and intermediate export options require --session")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be a finite positive number")
    try:
        target = subtitle.ass_path(args.out)
        outputs = [target, *([args.save_krc] if args.save_krc else []), *([args.save_midi] if args.save_midi else [])]
        subtitle.check_paths(outputs, [] if args.session else [args.krc, args.midi])
        if args.session:
            if args.run:
                os.environ[ipc.ENV_RUN] = args.run
            endpoint = ipc.resolve_endpoint(args.endpoint, token=args.token)
            with ipc.Client(endpoint, timeout=args.timeout, keep_events=False) as client:

                def build():
                    source = client.request("project.ass_source")
                    settings = resolve_settings(args, source["settings"])
                    data = base64.b64decode(source["midi"], validate=True)
                    text = subtitle.build_ass(source["krc"], data, settings=settings, wavetone=source["wavetone"])
                    return subtitle.save_export(
                        target, source["krc"], data, text, save_krc=args.save_krc, save_midi=args.save_midi
                    )

                if args.watch:
                    watch(build, lambda: client.request("state.get")["subtitle"]["revision"])
                else:
                    return 0 if _attempt(build) else 1
        else:
            settings = resolve_settings(args)
            wavetone = store.Settings().midi.wavetone if args.wavetone is None else args.wavetone

            def build():
                krc = args.krc.read_text(encoding="utf-8")
                data = args.midi.read_bytes()
                text = subtitle.build_ass(krc, data, settings=settings, wavetone=wavetone)
                return subtitle.save_export(target, krc, data, text)

            if args.watch:
                watch(build, lambda: (file_stamp(args.krc), file_stamp(args.midi)))
            else:
                return 0 if _attempt(build) else 1
    except KeyboardInterrupt:
        return 0
    except (ipc.IpcError, OSError, ValueError) as error:
        print(f"namioto-cli ass: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
