# SPDX-License-Identifier: AGPL-3.0-only
"""Optional headless libass integration with a fixed font and independent highlights."""

import shutil
import subprocess

import numpy as np
import pytest

from namioto.karaoke import AssSettings, generate_ass


def test_two_words_sweep_simultaneously_in_libass(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("fc-match"):
        pytest.skip("ffmpeg and fontconfig are required for the render check")
    font = subprocess.run(
        ["fc-match", "-f", "%{family}", "DejaVu Sans"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if font != "DejaVu Sans":
        pytest.skip("the fixed DejaVu Sans font is unavailable")
    filters = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, check=True).stdout
    if " ass " not in filters:
        pytest.skip("ffmpeg was built without libass")
    settings = AssSettings(
        font="DejaVu Sans",
        channel_colors={1: "FF0000", 2: "00FF00"},
        border=0,
        border_furi=0,
        blur=0,
        blur_scale=0,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    (tmp_path / "lyrics.ass").write_text(
        generate_ass("@1{AAAA }@2{BBBB}", [[(0, 2), (1, 3)]], settings=settings),
        encoding="utf-8",
    )
    counts = []
    for timestamp in (0.5, 1.5, 2.5):
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1920x1080:r=10:d=4",
                "-vf",
                "ass=lyrics.ass",
                "-ss",
                str(timestamp),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
        pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(1080, 1920, 3)
        red = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] < 60)
        green = (pixels[:, :, 1] > 180) & (pixels[:, :, 0] < 60) & (pixels[:, :, 2] < 60)
        counts.append((red.sum(), green.sum()))
    assert counts[0][0] > 500
    assert counts[0][1] == 0
    assert counts[1][0] > counts[0][0] + 500
    assert counts[1][1] > 500
    assert counts[2][0] > counts[1][0] + 500
    assert counts[2][1] > counts[1][1] + 500
