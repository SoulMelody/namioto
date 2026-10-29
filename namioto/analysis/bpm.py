# SPDX-License-Identifier: AGPL-3.0-only
"""The tempo algorithms behind one switch: WaveTone's DFT over an onset-flux envelope (the default),
the librosa beat tracker and the TempoCNN model.

Each returns the same `BpmEstimate`, so the editor offers the number and where it came from without
knowing which algorithm ran. TempoCNN brings ONNX and its model with it, so it is imported only when
it is the chosen one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from namioto.analysis import beats, wavetone
from namioto.analysis.beats import WINDOW_HOP_SECONDS, WINDOW_SECONDS
from namioto.analysis.choices import ALGORITHMS

SOURCES = {
    "wavetone": "WaveTone onset-flux DFT",
    "librosa": "Librosa beat tracking and least-squares fit",
    "tempocnn": "TempoCNN model",
    "spectrum": "Note-spectrum onset and beat tracking",
}
DEFAULT = ALGORITHMS[0]


@dataclass(frozen=True)
class BpmEstimate:
    """A tempo and what is known about it: the windows it was fitted over, how many agree, the fit
    residual, and the algorithm it came from."""

    bpm: float
    algorithm: str
    windows: int = 0
    agreement: float = 0.0
    residual: float | None = None

    @property
    def source(self) -> str:
        return SOURCES[self.algorithm]


def from_spectrum(
    spectrum,
    *,
    window_seconds: float = 8.0,
    hop_seconds: float = 1.0,
) -> BpmEstimate:
    """Tempo from a note spectrum already on screen, so no second pass over the audio is needed."""
    result = beats.track(spectrum, window_seconds=window_seconds, hop_seconds=hop_seconds)
    return BpmEstimate(
        result.bpm,
        "spectrum",
        windows=len(result.local),
        agreement=result.agreement,
        residual=result.residual,
    )


def estimate(
    path: str | Path,
    algorithm: str = DEFAULT,
    *,
    window_seconds: float = WINDOW_SECONDS,
    window_hop_seconds: float = WINDOW_HOP_SECONDS,
) -> BpmEstimate:
    if algorithm == "wavetone":
        return BpmEstimate(wavetone.estimate(path), algorithm)
    if algorithm == "librosa":
        result = beats.estimate(path, window_seconds=window_seconds, window_hop_seconds=window_hop_seconds)
        return BpmEstimate(
            result.bpm,
            algorithm,
            windows=len(result.local),
            agreement=result.agreement,
            residual=result.residual,
        )
    if algorithm == "tempocnn":
        from namioto.analysis import devices, tempo

        problem = devices.validate()
        if problem is not None:
            raise RuntimeError(problem)
        result = tempo.estimate(path)
        agree = sum(1 for local in result.local if local.bpm == round(result.bpm))
        return BpmEstimate(
            result.bpm,
            algorithm,
            windows=len(result.local),
            agreement=agree / len(result.local) if result.local else 0.0,
        )
    raise ValueError(f"Unknown tempo algorithm {algorithm!r}, expected one of {ALGORITHMS}")
