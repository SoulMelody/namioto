"""Package the frozen GUI as a Linux AppImage or a Windows portable ZIP."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent


def build_appimage(source: Path, output: Path, appimagetool: Path) -> Path:
    if not (source / "namioto").is_file():
        raise SystemExit("Build the frozen application first: bash build.sh app")
    appdir = ROOT / "build" / "Namioto.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    shutil.copytree(ROOT / "scripts" / "appimage", appdir)
    shutil.copytree(source, appdir / "namioto")
    (appdir / "AppRun").chmod(0o755)
    target = output / "namioto-linux-x86_64.AppImage"
    subprocess.run(
        [
            str(appimagetool.resolve()),
            "--appimage-extract-and-run",
            "--no-appstream",
            str(appdir),
            str(target),
        ],
        env={**os.environ, "ARCH": "x86_64"},
        check=True,
    )
    return target


def build_windows_zip(source: Path, output: Path) -> Path:
    """Zip the frozen application under one top-level directory, so extracting it once yields a
    folder to run `namioto/namioto.exe` from."""
    if not (source / "namioto.exe").is_file():
        raise SystemExit("Build the frozen application first: bash build.sh app")
    target = output / "namioto-windows-x86_64.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, str(PurePosixPath("namioto") / path.relative_to(source).as_posix()))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("appimage", "windows"))
    parser.add_argument("--app", type=Path, default=ROOT / "dist" / "namioto")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "release")
    parser.add_argument("--appimagetool", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.package == "appimage":
        if args.appimagetool is None:
            parser.error("--appimagetool is required for an AppImage")
        target = build_appimage(args.app, args.output, args.appimagetool)
    else:
        target = build_windows_zip(args.app, args.output)
    print(target)


if __name__ == "__main__":
    main()
