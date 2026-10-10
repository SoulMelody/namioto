# SPDX-License-Identifier: AGPL-3.0-only
"""Spectrum rendering: the note table turned into a cached QImage, plus a background loader."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage

from namioto.analysis.spectrum import NoteSpectrum, analyse
from namioto.ui.loading import LoadingThread

LUT_SIZE = 384
BLUE_STEP = 90
RED_STEP = 240


def _ramp(value: float) -> tuple[float, float, float]:
    """Dark blue, then blue through green to red, then flat red; noteDigger's colour map."""
    if value <= BLUE_STEP:
        hue, lightness = 240.0, value / BLUE_STEP * 50.0
    elif value <= RED_STEP:
        hue = 240.0 - (value - BLUE_STEP) / (RED_STEP - BLUE_STEP) * 240.0
        lightness = 50.0
    else:
        hue, lightness = 0.0, 50.0
    return colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, 1.0)


def color_table() -> np.ndarray:
    """The ramp as 384 ARGB words in Qt's `Format_RGB32` layout."""
    values = np.arange(LUT_SIZE) / (LUT_SIZE - 1) * 255.0
    rgb = np.rint(np.array([_ramp(value) for value in values]) * 255.0).astype(np.uint32)
    return (0xFF << 24) | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


class SpectrumImage:
    """A `NoteSpectrum` as an image the roll can blit, cached per display parameter set.

    Row 0 is the highest note band, so it can be drawn straight down onto the pitch axis.
    """

    def __init__(self, spectrum: NoteSpectrum):
        self.spectrum = spectrum
        self._parameters: tuple[float, float] | None = None
        self._pixels: np.ndarray | None = None
        self._image: QImage | None = None

    def image(self, gain: float, contrast: float) -> QImage:
        parameters = (round(gain, 3), round(contrast, 3))
        if parameters != self._parameters or self._image is None:
            self._parameters = parameters
            self._render(gain, contrast)
        return self._image

    def _render(self, gain: float, contrast: float) -> None:
        amplitude = np.power(np.maximum(self.spectrum.table, 0.0), contrast) * gain
        index = np.clip(np.rint(amplitude * (LUT_SIZE - 1) / 255.0), 0, LUT_SIZE - 1).astype(np.uint32)
        rows = np.ascontiguousarray(color_table()[index].T[::-1])
        height, width = rows.shape
        self._pixels = rows
        self._image = QImage(rows.data, width, height, width * 4, QImage.Format.Format_RGB32)


class SpectrumLoader(LoadingThread):
    """Analyses a file off the GUI thread."""

    loaded = Signal(object)
    progress = Signal(int, int)

    def __init__(
        self,
        path: str | Path,
        channels: str = "mono",
        t_num: float = 40.0,
        fft_points: int = 8192,
        a4: float = 440.0,
        parent=None,
    ):
        super().__init__(path, parent)
        self.channels = channels
        self.t_num = t_num
        self.fft_points = fft_points
        self.a4 = a4

    def load(self) -> None:
        self.loaded.emit(
            analyse(
                self.path,
                channels=self.channels,
                t_num=self.t_num,
                fft_points=self.fft_points,
                a4=self.a4,
                progress=lambda done, total: self.progress.emit(done, total),
            )
        )
