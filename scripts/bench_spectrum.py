#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark the note-domain spectrum analysis, to decide whether it needs Cython/GPU work.

Synthesises test audio of several lengths and runs the analysis in a few configurations, printing
per-stage timings and the peak RSS growth of each run. Every case runs in a fresh process so that
memory numbers are not polluted by earlier cases. Linux only (/proc for RSS).

    uv run scripts/bench_spectrum.py                # synthetic audio
    uv run scripts/bench_spectrum.py song.flac      # a real file
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile

SAMPLE_RATE = 44100


def rss_mb() -> float:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * 4096 / 2**20


def synthetic(seconds: float, stereo: bool) -> np.ndarray:
    rng = np.random.default_rng(1)
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    base = np.zeros_like(time, dtype=np.float32)
    for frequency in (55.0, 110.0, 164.81, 220.0, 329.63, 440.0):
        base += (0.3 / frequency**0.5) * np.sin(2 * np.pi * frequency * time)
    base += 0.03 * rng.standard_normal(len(time)).astype(np.float32)
    for beat in range(int(seconds)):
        start = int(beat * SAMPLE_RATE)
        base[start : start + 4410] += 0.3 * np.exp(-np.arange(4410) / 1500)
    mono = (base / np.abs(base).max() * 0.9).astype(np.float32)
    if not stereo:
        return mono
    return np.stack([mono, np.roll(mono, 997)], axis=1)


def run_case(path: str, mode: str, t_num: float, fft_points: int, backend: str = "wavetone") -> dict:
    from namioto.analysis.spectrum import analyse

    timings: dict = {}
    before = rss_mb()
    start = time.perf_counter()
    spectrum = analyse(path, channels=mode, backend=backend, t_num=t_num, fft_points=fft_points, timings=timings)
    total = time.perf_counter() - start
    return {
        "frames": spectrum.frames,
        "duration": spectrum.duration,
        "total": total,
        "compute": total - timings.get("decode", 0.0),
        "rss": max(rss_mb() - before, 0.0),
        "table_mb": spectrum.table.nbytes / 2**20,
        "stages": timings,
    }


def report(cases: list[tuple[str, str, float, int, str]], runner) -> None:
    header = f"{'case':52} {'frames':>7} {'total':>7} {'compute':>8} {'realtime':>9} {'rss':>7} {'table':>6}"
    print(header)
    print("-" * len(header))
    for path, mode, t_num, fft_points, backend in cases:
        result = runner(path, mode, t_num, fft_points, backend)
        stages = " ".join(f"{name}={value:.3f}" for name, value in result["stages"].items())
        label = f"{Path(path).name} {mode} {backend} t_num={t_num:g} fft={fft_points}"
        print(
            f"{label:52} {result['frames']:7d} "
            f"{result['total']:7.3f} {result['compute']:8.3f} {result['duration'] / result['compute']:8.0f}x "
            f"{result['rss']:6.0f}M {result['table_mb']:5.1f}M   {stages}"
        )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] != "--child":
        cases = [
            (sys.argv[1], "mono", 40.0, 8192, backend) for backend in ("wavetone", "notedigger", "cqt")
        ]
        report(cases, run_case)
        return 0

    with tempfile.TemporaryDirectory() as directory:
        cases = []
        for seconds, stereo in ((30.0, False), (180.0, False), (180.0, True), (600.0, False)):
            path = Path(directory) / f"synthetic_{int(seconds)}s{'_stereo' if stereo else ''}.wav"
            soundfile.write(str(path), synthetic(seconds, stereo), SAMPLE_RATE, subtype="PCM_16")
            cases.append((str(path), "both" if stereo else "mono", 40.0, 8192, "wavetone"))
        cases.append((cases[-1][0], "mono", 20.0, 8192, "wavetone"))
        cases.append((cases[-1][0], "mono", 40.0, 16384, "wavetone"))
        # the constant-Q path is the expensive one, so it is measured on the shortest file
        cases.append((cases[0][0], "mono", 40.0, 8192, "cqt"))

        def in_subprocess(path: str, mode: str, t_num: float, fft_points: int, backend: str) -> dict:
            process = subprocess.run(
                [sys.executable, __file__, "--child", path, mode, str(t_num), str(fft_points), backend],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(process.stdout.strip().splitlines()[-1])

        report(cases, in_subprocess)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        backend = sys.argv[6] if len(sys.argv) > 6 else "wavetone"
        print(json.dumps(run_case(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), backend)))
    else:
        raise SystemExit(main())
